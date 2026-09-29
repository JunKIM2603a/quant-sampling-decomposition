"""Reuse locked F/Q assets for short diagnostics; never rewrite a checkpoint."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantsplit.cache_diagnostics import MODES, diagnose_model, diagnostic_mode
from quantsplit.data import manifest_hash, question_hash
from quantsplit.hf_backend import file_sha256, render_prompt, tokenizer_hash, verify_q_checkpoint
from quantsplit.validation import TOLERANCES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--q-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new report before any expensive work; keep errors and partial results.
    report = {"kind": "short_cache_diagnostic_only", "completed": False,
              "stage02_complete": False, "scientific_H1_result": False,
              "professor_PASS_verified": False, "tolerances": TOLERANCES,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_args": {k: str(v) for k, v in vars(args).items()}, "models": {}}
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    try:
        if not args.device.startswith("cuda:") or not torch.cuda.is_available():
            raise RuntimeError("target diagnostic requires an available CUDA GPU")
        root = Path(__file__).resolve().parents[1]
        lock_path = root / "configs/execution_lock.json"
        split_path = root / "research/session02/split_manifest.json"
        questions_path = root / "data/gsm8k/development.jsonl"
        lock, splits = json.loads(lock_path.read_text()), json.loads(split_path.read_text())
        if manifest_hash(splits) != lock["split_manifest_sha256"]:
            raise ValueError("split manifest hash mismatch")
        if file_sha256(questions_path) != lock["worklist_sha256"]["development"]:
            raise ValueError("development worklist hash mismatch")
        row = json.loads(questions_path.read_text().splitlines()[0])
        allowed = {r["question_id"]: r["question_sha256"] for r in splits["development"]}
        if row.get("split") != "development" or allowed.get(row["question_id"]) != question_hash(row["question"]):
            raise ValueError("first question is not locked development data")
        manifest_path = args.q_checkpoint / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        verify_q_checkpoint(args.q_checkpoint, manifest, lock)
        tokenizer = AutoTokenizer.from_pretrained(lock["model_id"],
            revision=lock["tokenizer_revision"], trust_remote_code=False)
        if tokenizer_hash(tokenizer) != lock["tokenizer_hash"]:
            raise ValueError("tokenizer hash mismatch")
        prompt, prompt_hash = render_prompt(tokenizer, row["question"])
        probe = prompt[:32]
        torch.cuda.set_device(args.device)
        report.update(
            device=args.device, gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
            packages={k: importlib.metadata.version(k) for k in ("torch", "transformers", "numpy")},
            lock_sha256=file_sha256(lock_path), q_manifest_sha256=file_sha256(manifest_path),
            q_manifest=manifest, q_checkpoint_hashes_verified=True,
            model_id=lock["model_id"], model_revision=lock["model_revision"],
            question_id=row["question_id"], full_prompt_sha256=prompt_hash,
            probe_token_ids=probe, continuation_token_ids=[probe[-1]] * 4,
            runner_sha256=file_sha256(__file__),
            source_sha256={p.name: file_sha256(p) for p in (root / "src/quantsplit").glob("*.py")},
        )
        save()
        reference_config = None
        for name in ("F", "Q"):
            report["last_operation"] = {"model": name, "operation": "load"}
            save()
            print(f"loading {name}; one model resident at a time", flush=True)
            kwargs = {"revision": lock["model_revision"]} if name == "F" else {}
            model = AutoModelForCausalLM.from_pretrained(
                lock["model_id"] if name == "F" else args.q_checkpoint,
                dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False,
                **kwargs).to(args.device).eval()
            if any(p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point()):
                raise ValueError("expected original BF16 parameters before diagnosis")
            config = {k: getattr(model.config, k, None) for k in ("model_type", "vocab_size", "hidden_size",
                "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
                "rope_theta", "rope_scaling", "use_sliding_window", "sliding_window")}
            if reference_config is not None and config != reference_config:
                raise ValueError("F/Q config mismatch")
            reference_config = config
            report["models"][name] = {"config": config, "modes": {}}
            for mode in MODES:
                report["last_operation"] = {"model": name, "operation": mode}
                save()
                if mode == "fp32_math":
                    # Promote the loaded BF16 values; do not reload/round different weights.
                    model.float()
                print(f"{name}: {mode}", flush=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with diagnostic_mode(mode) as settings:
                    result = diagnose_model(model, probe)
                torch.cuda.synchronize()
                result.update(settings=settings, parameter_dtype=str(next(model.parameters()).dtype),
                              seconds=time.perf_counter() - started,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated())
                report["models"][name]["modes"][mode] = result
                save()
                print(json.dumps({"model": name, "mode": mode, "maxima": result["maxima"]}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report["completed"] = True
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()
    print(f"Diagnostic saved: {args.output}; this does not complete stage 02", flush=True)


if __name__ == "__main__":
    main()
