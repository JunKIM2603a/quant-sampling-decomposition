"""Pin tokenizer/data and make train-only worklists. Never generate test outputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from quantsplit.data import make_splits, manifest_hash
from quantsplit.evaluation import PARSER_VERSION
from quantsplit.hf_backend import render_prompt, tokenizer_hash
from quantsplit.sampling import RNG_VERSION
from quantsplit.assets import validate_asset_contract


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=Path("research/session02/upstream_metadata.json"))
    p.add_argument("--output-dir", type=Path, default=Path("data/gsm8k"))
    p.add_argument("--manifest", type=Path, default=Path("research/session02/split_manifest.json"))
    p.add_argument("--lock", type=Path, default=Path("configs/execution_lock.json"))
    args = p.parse_args()
    meta = json.loads(args.metadata.read_text())
    m, d = meta["model"], meta["dataset"]
    tokenizer = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"], trust_remote_code=False)
    dataset = load_dataset(d["id"], "main", revision=d["revision"])
    train, test = list(dataset["train"]), list(dataset["test"])
    if (len(train), len(test)) != (7473, 1319):
        raise ValueError("unexpected GSM8K split sizes")
    manifest = make_splits(train, test)
    manifest.update({"dataset_id": d["id"], "dataset_revision": d["revision"], "subset": "main"})
    worklists = {}
    for name in ["calibration", "development", "pilot"]:
        rows = []
        for item in manifest[name]:
            row = {**item, "split": name, "question": train[item["row_index"]]["question"]}
            if name != "calibration":
                row["answer"] = train[item["row_index"]]["answer"]
            rows.append(json.dumps(row, ensure_ascii=False))
        worklists[name] = "\n".join(rows) + "\n"
    eos = m["generation_config.json"]["eos_token_id"]
    eos_ids = [eos] if isinstance(eos, int) else eos
    if tokenizer.eos_token_id not in eos_ids:
        raise ValueError("tokenizer and generation EOS differ")
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    prompt, prompt_sha = render_prompt(tokenizer, "What is 2 + 3?")
    lock = {"status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY_FROZEN", "model_id": m["id"],
            "model_revision": m["revision"], "tokenizer_revision": m["revision"],
            "tokenizer_hash": tokenizer_hash(tokenizer), "dataset_revision": d["revision"],
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "split_manifest_sha256": manifest_hash(manifest), "rng_version": RNG_VERSION,
            "worklist_sha256": {name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                                for name, content in worklists.items()},
            "parser_version": PARSER_VERSION, "eos_ids": eos_ids,
            "reasoning_end_ids": think_end_ids,
            "reasoning_end_single_token": len(think_end_ids) == 1,
            "model_context_limit": m["config.json"]["max_position_embeddings"],
            "tokenizer_advertised_max_length": tokenizer.model_max_length,
            "context_policy": "No truncation; prompt + cap <= model limit; validate long context on target GPU",
            "prompt_probe_sha256": prompt_sha, "prompt_probe_ids": prompt,
            "model_vocab_size": m["config.json"]["vocab_size"],
            "tokenizer_length": len(tokenizer),
            "special_token_policy": "entire model vocabulary; EOS stops, reasoning-end does not stop",
            "packages": {name: importlib.metadata.version(name) for name in
                         ["torch", "transformers", "datasets", "numpy", "tokenizers", "huggingface-hub"]},
            "q_checkpoint_ready": False,
            "test_outputs_seen": False, "professor_PASS_verified": False,
            "user_authorized_assumed_approval": True}
    validate_asset_contract(args.manifest, args.lock, manifest, lock, worklists, args.output_dir)
    if not args.manifest.exists():
        write_json(args.manifest, manifest)
    if not args.lock.exists():
        write_json(args.lock, lock)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in worklists.items():
        path = args.output_dir / f"{name}.jsonl"
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
    print(json.dumps({"model_revision": m["revision"], "dataset_revision": d["revision"],
                      "split_manifest_sha256": manifest_hash(manifest),
                      "excluded_train_rows": len(manifest["excluded"]),
                      "eos_ids": eos_ids, "reasoning_end_ids": think_end_ids,
                      "test_outputs_generated": 0,
                      "recorded_files_preserved": True,
                      "runtime_packages": lock["packages"]}, indent=2))


if __name__ == "__main__":
    main()
