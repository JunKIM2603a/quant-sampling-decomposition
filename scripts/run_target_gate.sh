#!/usr/bin/env bash
# Run from the repository root, inside the prepared quantsplit environment.
# No pilot/test execution. A zero exit code still requires evidence review.
set -euo pipefail
device="${1:-cuda:0}"
run_dir="${2:-runs/target-gate-$(date -u +%Y%m%dT%H%M%SZ)}"
q_dir="${3:-checkpoints/gptq-w3-g128}"
if [[ ! -f configs/execution_lock.json ]]; then
  echo 'Run this script from the repository root.' >&2
  exit 2
fi
if [[ -e "$run_dir" || -e "$run_dir.tar.gz" ]]; then
  echo 'Choose a new run directory; existing evidence is never overwritten.' >&2
  exit 2
fi
mkdir -p "$run_dir"
collect_evidence() {
  run_status=$?
  trap - EXIT
  printf '%s\n' "$run_status" > "$run_dir/exit_code.txt"
  # Only small reports/logs; never checkpoint weights or raw model outputs.
  python - "$run_dir" "$q_dir" <<'PY'
import pathlib, sys, tarfile
run, checkpoint = map(pathlib.Path, sys.argv[1:])
with tarfile.open(str(run) + '.tar.gz', 'x:gz') as archive:
    for path in sorted(run.iterdir()):
        if path.is_file() and path.suffix in {'.json', '.log', '.txt'}:
            archive.add(path, arcname='reports/' + path.name)
    if (checkpoint / 'manifest.json').is_file():
        archive.add(checkpoint / 'manifest.json', arcname='reports/q_manifest.json')
print('Evidence bundle:', str(run) + '.tar.gz')
PY
  exit "$run_status"
}
trap collect_evidence EXIT
python scripts/check_target_environment.py --device "$device" --output "$run_dir/preflight.json" 2>&1 | tee "$run_dir/preflight.log"
python -m unittest discover -s tests -v 2>&1 | tee "$run_dir/tests.log"
python scripts/verify_tiny_model.py --output "$run_dir/tiny_verification.json" 2>&1 | tee "$run_dir/tiny.log"
python scripts/prepare_assets.py 2>&1 | tee "$run_dir/assets.log"
if [[ ! -f "$q_dir/manifest.json" ]]; then
  python scripts/build_gptq.py --device "$device" --output-dir "$q_dir" 2>&1 | tee "$run_dir/gptq.log"
fi
python scripts/run_development.py --device "$device" --q-checkpoint "$q_dir" --q-manifest "$q_dir/manifest.json" --limit 2 --max-new-tokens 128 --output "$run_dir/dev-smoke.jsonl" 2>&1 | tee "$run_dir/dev-smoke.log"
python scripts/run_development.py --device "$device" --q-checkpoint "$q_dir" --q-manifest "$q_dir/manifest.json" --limit 2 --max-new-tokens 32768 --output "$run_dir/dev-fullcap.jsonl" 2>&1 | tee "$run_dir/dev-fullcap.log"
python scripts/run_development.py --device "$device" --q-checkpoint "$q_dir" --q-manifest "$q_dir/manifest.json" --limit 2 --max-new-tokens 32768 --context-stress-only --output "$run_dir/context-stress.jsonl" 2>&1 | tee "$run_dir/context-stress.log"
echo 'Device commands finished. Evidence review is required before closing stage 02. No H1 result.'
