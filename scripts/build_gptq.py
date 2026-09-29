"""Generate locked train calibration traces, then save dequantized BF16 GPTQ weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantsplit.data import manifest_hash, question_hash
from quantsplit.hf_backend import HFBackend, file_sha256, render_prompt, tokenizer_hash
from quantsplit.quantization import QUANT_CONFIG, quantize_qwen2
from quantsplit.sampling import probabilities, nucleus_mask, restrict, inverse_cdf, keyed_uniform


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lock", type=Path, default=Path("configs/execution_lock.json"))
    p.add_argument("--split-manifest", type=Path, default=Path("research/session02/split_manifest.json"))
    p.add_argument("--questions", type=Path, default=Path("data/gsm8k/calibration.jsonl"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--calibration-file", type=Path, help="Resume with completed calibration JSONL")
    args = p.parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("full-model GPTQ preparation requires a CUDA GPU")
    if args.output_dir.exists():
        raise FileExistsError("choose a fresh checkpoint directory")
    lock = json.loads(args.lock.read_text())
    manifest = json.loads(args.split_manifest.read_text())
    if manifest_hash(manifest) != lock["split_manifest_sha256"]:
        raise ValueError("split manifest mismatch")
    if file_sha256(args.questions) != lock["worklist_sha256"]["calibration"]:
        raise ValueError("calibration worklist file hash mismatch")
    allowed = {i["question_id"]: i["question_sha256"] for i in manifest["calibration"]}
    rows = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    if len(rows) != 128 or len({r["question_id"] for r in rows}) != 128:
        raise ValueError("exactly 128 unique locked calibration questions required")
    for row in rows:
        if (row.get("split") != "calibration" or "answer" in row or
                allowed.get(row["question_id"]) != question_hash(row["question"])):
            raise ValueError("calibration must be locked questions without answer fields")
    tokenizer = AutoTokenizer.from_pretrained(lock["model_id"], revision=lock["tokenizer_revision"],
                                               trust_remote_code=False)
    if tokenizer_hash(tokenizer) != lock["tokenizer_hash"]:
        raise ValueError("tokenizer mismatch")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(lock["model_id"], revision=lock["model_revision"],
                                               dtype=torch.bfloat16, attn_implementation="sdpa",
                                               trust_remote_code=False).to(args.device).eval()
    args.output_dir.mkdir(parents=True)
    calibration_path = args.output_dir / "calibration_tokens.jsonl"
    sequences = []
    resumed = None
    if args.calibration_file:
        resumed = [json.loads(line) for line in args.calibration_file.read_text().splitlines() if line.strip()]
        if len(resumed) != 128:
            raise ValueError("resume requires a complete 128-row calibration file")
    with calibration_path.open("x") as stream:
        for index, row in enumerate(rows):
            prompt, prompt_hash = render_prompt(tokenizer, row["question"])
            if len(prompt) >= 2048:
                raise ValueError("calibration prompt reaches token budget; no silent truncation")
            if resumed is not None:
                item = resumed[index]
                if (item["question_id"] != row["question_id"] or item["prompt_sha256"] != prompt_hash or
                        item["model_revision"] != lock["model_revision"] or item["seed"] != 42 or
                        item.get("temperature") != 0.6 or item.get("top_p") != 0.95 or
                        item.get("tokenizer_hash") != lock["tokenizer_hash"] or
                        item["token_ids"][:len(prompt)] != prompt or not len(prompt) < len(item["token_ids"]) <= 2048):
                    raise ValueError("calibration resume provenance mismatch")
            else:
                session = HFBackend(model).new_session()
                ids = list(prompt)
                for step in range(2048-len(prompt)):
                    probs = probabilities(session.next_logits(ids), 0.6)
                    token = inverse_cdf(restrict(probs, nucleus_mask(probs, .95)),
                                        keyed_uniform(row["question_id"],42,step))
                    ids.append(token)
                    if token in lock["eos_ids"]:
                        break
                item = {"question_id": row["question_id"], "prompt_sha256": prompt_hash,
                        "model_revision": lock["model_revision"], "seed": 42, "token_ids": ids,
                        "temperature": 0.6, "top_p": 0.95, "tokenizer_hash": lock["tokenizer_hash"]}
                del session
            sequences.append(item["token_ids"])
            stream.write(json.dumps(item) + "\n")
            stream.flush()
            print(f"calibration {index+1}/128", flush=True)
    report = quantize_qwen2(model, sequences)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    source = Path(__file__).resolve().parents[1] / "src/quantsplit"
    source_hashes = {str(p.relative_to(source)): file_sha256(p) for p in source.glob("*.py")}
    q_manifest = {"format": "dequantized_bfloat16", "method": "GPTQ", "bits": 3,
                  "group_size": 128, "symmetric": False,
                  "base_model_revision": lock["model_revision"], "tokenizer_hash": lock["tokenizer_hash"],
                  "split_manifest_sha256": lock["split_manifest_sha256"],
                  "calibration_tokens_sha256": file_sha256(calibration_path),
                  "quantization_code_revision": QUANT_CONFIG["driver"], "source_sha256": source_hashes,
                  "quantization_config": QUANT_CONFIG, "quantization_report": report,
                  "packages": {n: importlib.metadata.version(n) for n in ["torch","transformers","numpy"]},
                  "files_sha256": {p.name: file_sha256(p) for p in args.output_dir.iterdir()
                                   if p.is_file() and p.name != "manifest.json"}}
    (args.output_dir / "manifest.json").write_text(json.dumps(q_manifest, indent=2) + "\n")
    print("Saved dequantized GPTQ checkpoint and manifest. This is not an H1 result.")


if __name__ == "__main__":
    main()
