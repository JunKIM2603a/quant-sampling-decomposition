"""Read-only target preflight, usable even before torch has been installed."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new preflight output path")
    root = Path(__file__).resolve().parents[1]
    report = {"kind": "environment_preflight_only", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "scientific_H1_result": False, "stage02_complete": False,
              "professor_PASS_verified": False, "user_authorized_assumed_approval": True,
              "python": sys.version, "platform": platform.platform(), "device": args.device,
              "repository": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
              "working_tree": command(["git", "-C", str(root), "status", "--porcelain"]),
              "nvidia_smi": command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.free",
                                     "--format=csv,noheader"]),
              "disk_free_bytes": shutil.disk_usage(root).free, "packages": {}, "blockers": []}
    for name in ["torch", "transformers", "numpy", "datasets", "tokenizers", "huggingface-hub", "safetensors"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
            report["blockers"].append(f"missing package: {name}")
    for name, version in {"torch": "2.8.0", "transformers": "4.57.1", "datasets": "4.3.0"}.items():
        if (report["packages"].get(name) or "").split("+")[0] != version:
            report["blockers"].append(f"expected {name}=={version}; environment review required")
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        report["gpu_count"] = torch.cuda.device_count()
        if not args.device.startswith("cuda:") or not report["cuda_available"]:
            raise RuntimeError("requested CUDA GPU is unavailable")
        torch.cuda.set_device(args.device)
        props = torch.cuda.get_device_properties(args.device)
        report["selected_gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                                  "bf16_supported": torch.cuda.is_bf16_supported()}
        if "4090" not in props.name:
            report["blockers"].append("selected GPU is not RTX 4090; target evidence review required")
        if not report["selected_gpu"]["bf16_supported"]:
            report["blockers"].append("BF16 unsupported")
    except Exception as exc:
        report["blockers"].append(f"{type(exc).__name__}: {exc}")
    report["files_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ["configs/execution_lock.json", "protocols/quant_sampling_v0.2.1-development.json"]}
    report["ready_for_asset_preparation"] = not report["blockers"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_asset_preparation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
