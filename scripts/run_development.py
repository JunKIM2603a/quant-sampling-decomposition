"""Development-only four-arm driver; Q must be a verified dequantized GPTQ checkpoint."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantsplit.data import manifest_hash, question_hash
from quantsplit.evaluation import score_gsm8k
from quantsplit.generation import Settings, generate_four
from quantsplit.hf_backend import (HFBackend, render_prompt, tokenizer_hash,
                                   verify_q_checkpoint, file_sha256)
from quantsplit.validation import validate_model_pair, validate_long_context_pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lock", type=Path, default=Path("configs/execution_lock.json"))
    p.add_argument("--split-manifest", type=Path, default=Path("research/session02/split_manifest.json"))
    p.add_argument("--questions", type=Path, default=Path("data/gsm8k/development.jsonl"))
    p.add_argument("--q-checkpoint", type=Path, required=True)
    p.add_argument("--q-manifest", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context-stress-only", action="store_true",
                   help="synthetic long-context check; no task outputs or accuracy")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("target-device validation requires an available CUDA GPU")
    if not 1 <= args.limit <= 128 or not 1 <= args.max_new_tokens <= 32768:
        raise ValueError("invalid development limits")
    if args.context_stress_only and args.max_new_tokens != 32768:
        raise ValueError("target context stress requires --max-new-tokens 32768")
    if any(path.exists() for path in [args.output,args.output.with_suffix(".manifest.json"),args.output.with_suffix(".validation.json"),args.output.with_suffix(".context.json")]):
        raise FileExistsError("choose a new output name; existing runs are not overwritten")
    lock = json.loads(args.lock.read_text())
    splits = json.loads(args.split_manifest.read_text())
    if manifest_hash(splits) != lock["split_manifest_sha256"]:
        raise ValueError("split manifest hash mismatch")
    if file_sha256(args.questions) != lock["worklist_sha256"]["development"]:
        raise ValueError("development worklist file hash mismatch")
    allowed = {item["question_id"]: item["question_sha256"] for item in splits["development"]}
    rows = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    if len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate question IDs")
    for row in rows:
        if (row.get("split") != "development" or row["question_id"] not in allowed or
                question_hash(row["question"]) != allowed[row["question_id"]]):
            raise ValueError("only locked development questions may run in this driver")
    if len(rows) < args.limit:
        raise ValueError("insufficient questions")
    q_manifest = json.loads(args.q_manifest.read_text())
    verify_q_checkpoint(args.q_checkpoint, q_manifest, lock)
    tokenizer = AutoTokenizer.from_pretrained(lock["model_id"], revision=lock["tokenizer_revision"],
                                               trust_remote_code=False)
    if tokenizer_hash(tokenizer) != lock["tokenizer_hash"]:
        raise ValueError("tokenizer hash mismatch")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    f = AutoModelForCausalLM.from_pretrained(lock["model_id"], revision=lock["model_revision"],
                                           dtype=torch.bfloat16, attn_implementation="sdpa",
                                           trust_remote_code=False).to(args.device).eval()
    q = AutoModelForCausalLM.from_pretrained(args.q_checkpoint, dtype=torch.bfloat16,
                                           attn_implementation="sdpa", trust_remote_code=False).to(args.device).eval()
    if getattr(q.config, "quantization_config", None):
        raise ValueError("packed quantized kernels are outside this dequantized-BF16 design")
    for key in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "max_position_embeddings", "model_type"]:
        if getattr(f.config, key) != getattr(q.config, key):
            raise ValueError(f"F/Q config mismatch: {key}")
    if any(w.dtype != torch.bfloat16 for model in [f, q] for w in model.parameters() if w.is_floating_point()):
        raise ValueError("all floating model parameters must be BF16")
    prompts = [render_prompt(tokenizer, row["question"])[0] for row in rows[:args.limit]]
    if any(len(prompt) + args.max_new_tokens > lock["model_context_limit"] for prompt in prompts):
        raise ValueError("prompt+cap exceeds model context; no silent truncation allowed")
    probe = prompts[0][:32]
    validation = validate_model_pair(f,q,probe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    validation_path = args.output.with_suffix(".validation.json")
    validation_path.write_text(json.dumps(validation,indent=2)+"\n")
    if not validation['passed']:
        print(json.dumps({"validation_report": str(validation_path), "validation": validation}, indent=2),
              flush=True)
        raise RuntimeError(f"implementation checks failed; see {validation_path}")
    source = Path(__file__).resolve().parents[1] / "src/quantsplit"
    provenance = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "lock_sha256": file_sha256(args.lock),
                  "q_manifest_sha256": file_sha256(args.q_manifest),
                  "runner_sha256": file_sha256(__file__),
                  "source_sha256": {p.name: file_sha256(p) for p in source.glob("*.py")},
                  "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                  "packages": {k: importlib.metadata.version(k) for k in ["torch", "transformers", "numpy"]}}
    if args.context_stress_only:
        # Cover the longest of all locked development prompts, not just the two smoke rows.
        stress_prompt = max((render_prompt(tokenizer, row["question"])[0] for row in rows), key=len)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        report = {**provenance, "kind": "synthetic_context_only", "completed": False,
                  "passed": False, "scientific_H1_result": False, "stage02_complete": False,
                  "prompt_tokens": len(stress_prompt), "generation_cap": args.max_new_tokens}
        try:
            report.update(validate_long_context_pair(f, q, stress_prompt,
                len(stress_prompt) + args.max_new_tokens,
                progress=lambda n, total: print(f"context {n}/{total}", flush=True)))
            torch.cuda.synchronize()
            report["completed"] = True
        except Exception as exc:
            report["error_type"], report["error"] = type(exc).__name__, str(exc)
            raise
        finally:
            report.update(seconds=time.perf_counter() - start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            args.output.with_suffix(".context.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise RuntimeError("long-context check failed; inspect saved .context.json; tolerances unchanged")
        print(json.dumps(report, indent=2))
        return
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    count = 0
    run_summaries = []
    with args.output.open("x") as out:
        for row, prompt in zip(rows[:args.limit], prompts):
            settings = Settings(args.max_new_tokens, eos_ids=tuple(lock["eos_ids"]),
                                diagnostic_ids=tuple(lock["reasoning_end_ids"]))
            results = generate_four(HFBackend(f), HFBackend(q), prompt, row["question_id"], args.seed, settings)
            for arm, result in results.items():
                text = tokenizer.decode(result.generated_ids, skip_special_tokens=False)
                score = score_gsm8k(text, row["answer"])
                record = {**asdict(result), **asdict(score), "generated_text": text,
                          "consumed_tokens": result.consumed_tokens,
                          "prompt_sha256": render_prompt(tokenizer, row["question"])[1],
                          "split": "development", "scientific_H1_result": False}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                count += result.consumed_tokens
                run_summaries.append({"question_id": row["question_id"], "arm": arm,
                    "prompt_tokens": len(prompt), "consumed_tokens": result.consumed_tokens,
                    "stop_reason": result.stop_reason,
                    "last_cache_length": len(prompt) + result.consumed_tokens - 1})
                print(f"{row['question_id']} {arm}: {result.consumed_tokens} tokens ({result.stop_reason})", flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    manifest = {**provenance, "kind": "development_only", "completed": True,
                "scientific_H1_result": False, "stage02_complete": False,
                "run_summaries": run_summaries,
                "full_generation_cap_reached": any(x["consumed_tokens"] == 32768 for x in run_summaries),
                "run_args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "output_sha256": file_sha256(args.output), "model_validation": validation,
                "seconds": seconds, "consumed_tokens": count, "effective_tokens_per_second": count/seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
