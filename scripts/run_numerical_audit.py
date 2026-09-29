"""Collect the fixed development numerical audit; no automatic gate replacement."""
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
from quantsplit.cache_diagnostics import diagnostic_mode
from quantsplit.data import manifest_hash, question_hash
from quantsplit.hf_backend import file_sha256, render_prompt, tokenizer_hash, verify_q_checkpoint
from quantsplit.numerical_audit import collect_trace, make_trace, policy_metrics, summarize_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--q-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = {"kind": "development_numerical_audit_only", "completed": False,
              "stage02_complete": False, "scientific_H1_result": False,
              "professor_PASS_verified": False, "gate_replacement_active": False,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_args": {k: str(v) for k, v in vars(args).items()}, "checks": {}, "points": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    try:
        if not args.device.startswith("cuda:") or not torch.cuda.is_available():
            raise RuntimeError("numerical audit requires an available target CUDA GPU")
        plan_path = root / "configs/numerical_audit_v1.json"
        lock_path = root / "configs/execution_lock.json"
        question_path = root / "data/gsm8k/development.jsonl"
        plan, lock = json.loads(plan_path.read_text()), json.loads(lock_path.read_text())
        splits = json.loads((root / "research/session02/split_manifest.json").read_text())
        if manifest_hash(splits) != lock["split_manifest_sha256"]:
            raise ValueError("split manifest hash mismatch")
        if file_sha256(question_path) != lock["worklist_sha256"]["development"]:
            raise ValueError("development worklist hash mismatch")
        rows = [json.loads(s) for s in question_path.read_text().splitlines() if s.strip()][:8]
        if [r["question_id"] for r in rows] != plan["question_ids"]:
            raise ValueError("audit requires the fixed first eight development rows")
        allowed = {r["question_id"]: r["question_sha256"] for r in splits["development"]}
        for row in rows:
            if row.get("split") != "development" or allowed.get(row["question_id"]) != question_hash(row["question"]):
                raise ValueError("question does not match locked development split")
        q_manifest_path = args.q_checkpoint / "manifest.json"
        q_manifest = json.loads(q_manifest_path.read_text())
        verify_q_checkpoint(args.q_checkpoint, q_manifest, lock)
        tokenizer = AutoTokenizer.from_pretrained(lock["model_id"], revision=lock["tokenizer_revision"],
                                                   trust_remote_code=False)
        if tokenizer_hash(tokenizer) != lock["tokenizer_hash"]:
            raise ValueError("tokenizer hash mismatch")
        prompts = [render_prompt(tokenizer, row["question"])[0] for row in rows]
        if any(len(p) + plan["max_new_tokens"] > lock["model_context_limit"] for p in prompts):
            raise ValueError("full prompt plus audit continuation exceeds model context")
        torch.cuda.set_device(args.device)
        report.update(plan=plan, plan_sha256=file_sha256(plan_path), lock_sha256=file_sha256(lock_path),
            q_manifest_sha256=file_sha256(q_manifest_path), q_checkpoint_hashes_verified=True,
            model_id=lock["model_id"], model_revision=lock["model_revision"],
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
            runner_sha256=file_sha256(__file__),
            source_sha256={p.name: file_sha256(p) for p in (root / "src/quantsplit").glob("*.py")},
            packages={k: importlib.metadata.version(k) for k in ("torch", "transformers", "numpy")},
            device=args.device, gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)
        save()
        collected, reference_config = {}, None
        for name in ("F", "Q"):
            report["last_operation"] = f"load {name}"
            save()
            print(f"loading {name}; one model resident at a time", flush=True)
            kwargs = {"revision": lock["model_revision"]} if name == "F" else {}
            model = AutoModelForCausalLM.from_pretrained(
                lock["model_id"] if name == "F" else args.q_checkpoint,
                dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False,
                **kwargs).to(args.device).eval()
            if any(p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point()):
                raise ValueError("expected BF16 parameters before diagnostic promotion")
            config = {k: getattr(model.config, k, None) for k in ("model_type", "vocab_size", "hidden_size",
                "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
                "rope_theta", "rope_scaling", "use_sliding_window", "sliding_window")}
            if reference_config is not None and config != reference_config:
                raise ValueError("F/Q model config mismatch")
            reference_config = config
            report.setdefault("model_config", {})[name] = config
            if name == "F":
                traces = []
                with diagnostic_mode(plan["bf16_mode"]) as settings:
                    report["trace_settings"] = settings
                    for row, prompt in zip(rows, prompts):
                        print(f"trace {row['question_id']}", flush=True)
                        traces.append(make_trace(model, prompt, row["question_id"], plan, lock["eos_ids"]))
                        report["traces"] = traces
                        save()
            for dtype, mode in (("bf16", plan["bf16_mode"]), ("fp32", plan["fp32_mode"])):
                report["last_operation"] = f"{name} {dtype} replay"
                save()
                if dtype == "fp32":
                    model.float()
                key = f"{name}_{dtype}"
                collected[key], report["checks"][key] = {}, []
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with diagnostic_mode(mode) as settings:
                    report.setdefault("mode_settings", {})[key] = settings
                    for trace in traces:
                        print(f"{key}: {trace['question_id']}", flush=True)
                        points, check = collect_trace(model, trace)
                        collected[key][trace["question_id"]] = points
                        report["checks"][key].append({"question_id": trace["question_id"], **check,
                            "full_prefix_errors": {s: p["error"] for s, p in points.items()}})
                        save()
                torch.cuda.synchronize()
                report.setdefault("resources", {})[key] = {
                    "seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "parameter_dtype": str(next(model.parameters()).dtype)}
                save()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        special = sorted(set(lock["eos_ids"] + lock["reasoning_end_ids"]))
        for trace in traces:
            qid = trace["question_id"]
            for step in trace["probe_steps"]:
                paths = {label: {n: collected[f"{n}_{dtype}"][qid][step][path] for n in ("F", "Q")}
                         for label, dtype, path in (("bf16_cached", "bf16", "cached"),
                             ("bf16_full", "bf16", "full"), ("fp32_cached", "fp32", "cached"))}
                report["points"].append({"question_id": qid, "step": step,
                    "prefix_tokens": len(trace["prompt_ids"]) + step,
                    "policies": policy_metrics(paths, plan["temperature"], plan["top_p"], special)})
        report["summary"] = summarize_points(report["points"])
        report["same_schedule_checks_all_exact"] = all(c["same_schedule_exact"]
            for checks in report["checks"].values() for c in checks)
        limits = plan["fp32_cache_full_reference_limits"]
        report["fp32_comparison_markers_all_met"] = all(
            e["max_abs_logit"] <= limits["max_abs_logit"] and e["tv"] <= limits["tv"]
            for key, checks in report["checks"].items() if key.endswith("fp32")
            for c in checks for e in c["full_prefix_errors"].values())
        report["completed"] = True
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()
    print(f"Saved {args.output}; review required, stage 02 remains incomplete", flush=True)


if __name__ == "__main__":
    main()
