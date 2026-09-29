# QuantSplit 02 — 실제 장비 실행 순서

## 최신 작업: 원본 검토 완료 → 고정 development 정책 수치 audit

`cache-diagnostic-gpu1-01.json` 원본을 검토했다. code/lock·40개 위치·28개 층 길이·실행 설정·Q manifest가 기록과 일치한다. 정밀도 의존성이 확인되지만 기존 BF16 관문은 실패다. 다음 검사는 수치 차이가 top-p·4-arm 분포에 미치는 영향을 F-Q 차이와 함께 보는 고정 development audit이다. [절차와 해석](06_numerical_audit.md)을 따른다.

```bash
conda activate quantsplit
git pull --ff-only
python scripts/run_numerical_audit.py --device cuda:1 --q-checkpoint checkpoints/gptq-w3-g128 --output runs/numerical-audit-gpu1-01.json
```

8문항에서 최대32토큰의 F/FF 공통 경로를 생성하고 F/Q BF16·FP32를 비교한다. 기존 checkpoint를 재사용한다. 결과 JSON을 첨부한다. 오류가 나면 JSON과 터미널 오류를 함께 전달하고, 재시도 시 새 출력 번호를 쓴다. 진단 수집 완료는 02 PASS가 아니다. 기존 오차 기준·BF16 연구 설정·본 runner 관문은 유지하며 smoke/full-cap은 아직 재개하지 않는다.

아래는 전체 실행 절차다.

사용자의 승인 가정 진행 지시가 적용된 작업이다. 같은 허가를 다시 요청할 필요는 없다. 현재 02는 실제 GPU 관문을 남긴 상태다. 아래 명령은 **사용자 RTX 4090 환경에서 수행할 명령이며 이 세션에서 실행한 기록이 아니다.**

## 0. 02-2에서 추가한 실행·로그 수집

환경이 준비되었으면 아래 명령 하나로 **장비 사전 확인 → CPU 회귀 검사 → 자산 대조 → calibration/GPTQ → 짧은 4-arm → full-cap 자유 생성 → 합성 긴 context**를 실행한다. 실패한 단계에서 중단하며 로그를 `runs/target-gate-01.tar.gz`에 묶는다. 이 파일을 현재 채팅에 첨부하면 다음 관문을 검토할 수 있다. 이미 같은 출력 경로가 있으면 `02`처럼 새 이름을 쓴다.

```bash
# RTX 4090 PC의 저장소 root, quantsplit 환경에서 실행
git pull --ff-only
bash scripts/run_target_gate.sh cuda:0 runs/target-gate-01 checkpoints/gptq-w3-g128
```

기존 완성 Q manifest가 있으면 재양자화하지 않고 뒤의 runner에서 checkpoint 해시를 검증한다. 부분 checkpoint 폴더가 있으면 덮어쓰지 않고 중단한다. 아래 2절의 완성 calibration 재사용 절차를 따른다. `.tar.gz`에는 작은 JSON·로그만 넣으며 가중치·원시 생성 JSONL은 넣지 않는다. 강제 종료/SIGKILL/디스크 장애 때에는 bundle이 없을 수 있으므로 남은 로그를 전달한다.

환경 준비 여부부터 확인하려면 다음 명령만 실행한다. 모델·데이터를 다운로드하지 않고, PyTorch가 없는 Python에서도 진단 파일을 남긴다. CUDA가 있으면 작은 BF16 행렬곱·SDPA·FP32 Cholesky를 실제 실행한다. exit code 2는 장비 또는 환경 관문 미충족이다. `ready_for_asset_preparation=true`는 02 완료가 아니다.

```bash
python scripts/check_target_environment.py --device cuda:0 --output runs/target-preflight-01.json
```

두 GPU를 함께 48GB로 사용하지 않는다. 이 관문은 지정한 한 GPU에서 실행한다. 두 번째 GPU의 독립 worker 검증과 실제 하루 가용 시간은 03 예산 결정 전에 확인한다.

## 1. 코드와 환경

저장소 root에서 실행한다. 기존 다른 프로젝트의 conda 환경을 바꾸지 않고 별도 환경을 사용한다.

```bash
git clone https://github.com/JunKIM2603a/quant-sampling-decomposition.git
cd quant-sampling-decomposition
conda create -n quantsplit python=3.12 -y
conda activate quantsplit
nvidia-smi
```

PyTorch 2.8.0의 CUDA 빌드를 현재 드라이버와 호환되는 공식 설치 명령으로 설치한다. [공식 이전 버전 설치표](https://pytorch.org/get-started/previous-versions/)를 확인한다. CPU 빌드를 설치하면 GPU runner가 중단한다. 그 뒤:

CUDA 12.6 빌드 설치 예(공식 v2.8.0 표, 2026-09-29 재확인). 현재 드라이버의 실행 호환성은 사전 확인 결과로 판단한다.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

### 사용자 로그의 CUDA 12.2 표시와 cu126 설치

2026-09-29 22:31:30 KST 사용자 제공 nvidia-smi: driver `535.183.01`, CUDA 표시 `12.2`, RTX 4090 두 장. 이 표시는 드라이버의 CUDA 지원 수준이며 conda 환경에 설치된 PyTorch runtime 버전이 아니다. CUDA 12.x의 minor-version compatibility 최소 Linux driver는 `525.60.13`이므로 현재 드라이버는 그 조건을 만족한다. **cu126을 시도할 수 있지만 모든 연산의 호환을 보장하지 않으므로 작은 실제 kernel 검사를 먼저 수행한다.** PTX JIT나 새 드라이버 기능을 요구하는 경로에는 제약이 있다.

현재 계획은 PyTorch 2.8.0/cu126을 유지하여 검사한다. 이 표시 차이만으로 드라이버·시스템 CUDA를 교체하거나 torch 2.5 등으로 내리지 않는다. torch 2.8.0 공식 설치표에는 cu122가 없다. 표시를 맞추기 위한 `cu122` URL을 만들지 않는다.

해당 시점 GPU 0은 5,793 MiB/28%, GPU 1은 789 MiB/0%였다. 다른 프로세스는 그대로 두고 먼저 `cuda:1`을 사용한다. GPU 1도 일부 메모리가 사용 중이므로 전용/유휴 GPU라고 기록하지 않는다. 인덱스는 `CUDA_VISIBLE_DEVICES`를 별도로 재매핑하지 않은 경우다.

```bash
conda activate quantsplit
git pull --ff-only
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[model,data]'
python scripts/check_target_environment.py --device cuda:1 --output runs/target-preflight-gpu1-01.json
```

각 설치가 성공한 뒤 다음 줄로 진행한다. `ready_for_asset_preparation=true`와 `kernel_checks.passed=true`이면 다음 통합 명령을 사용할 수 있다.

```bash
bash scripts/run_target_gate.sh cuda:1 runs/target-gate-gpu1-01 checkpoints/gptq-w3-g128
```

실패하면 preflight JSON과 오류를 검토하여 드라이버/라이브러리/실제 kernel 문제를 구분한다. 작은 연산 통과를 실제 1.5B/GPTQ/full-cap 통과로 바꾸지 않는다.

공식 근거(2026-09-29 확인): [nvidia-smi 문서](https://docs.nvidia.com/deploy/nvidia-smi/index.html), [CUDA 12.6 release notes](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html), [minor compatibility와 PTX 제약](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [PyTorch 공식 설치표](https://pytorch.org/get-started/previous-versions/).

### 개별 단계 실행 시

```bash
python -m pip install -e '.[model,data]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python -m unittest discover -s tests -v
python scripts/verify_tiny_model.py --output runs/tiny_verification.json
python scripts/prepare_assets.py
```

prepare_assets는 고정 revision의 tokenizer·GSM8K를 다운로드하고 train worklist를 재생성한다. checkpoint 가중치는 아직 다운로드하지 않는다. split hash가 `f650f57279971fee307f9a85ced0033d75347ab4c3af86d49c560c508a1c6ad8`인지 비교한다. **02-2 수정부터 기존 split/lock/worklist와 다르면 쓰기 전에 중단한다. 기존 lock을 새 환경 값으로 덮어쓰지 않는다.** 실제 package는 출력과 preflight/실행 manifest에 별도로 기록한다. 분할·prompt·tokenizer hash가 달라지면 먼저 원인을 해결한다.

## 2. F 생성 calibration과 GPTQ checkpoint

```bash
python scripts/build_gptq.py --device cuda:0 --output-dir checkpoints/gptq-w3-g128
```

이 명령은 고정 1.5B BF16 모델을 내려받고, calibration 128문항에 최대 2,048 total token을 생성한 뒤 GPTQ를 수행한다. 실제 처리 시간은 미측정이다. test 문항은 쓰지 않는다. 출력 directory가 이미 있으면 덮어쓰지 않는다.

생성물: `calibration_tokens.jsonl`, BF16 `*.safetensors`, 설정·tokenizer, `manifest.json`. manifest에는 원본 revision, calibration hash, 실제 quantization 설정, 구현 hash, 모든 결과 파일 hash가 들어간다. 메모리 부족·Cholesky 실패·NaN을 과학적 결과로 처리하지 않는다.

calibration 생성은 완료했고 양자화만 실패했다면 원인을 기록·수정한 뒤 새 output directory와 `--calibration-file 이전경로/calibration_tokens.jsonl`로 재시도할 수 있다. 부분적으로만 생성된 calibration 파일은 자동 이어 쓰지 않는다. 완성 128행의 provenance를 확인한다.

## 3. 개발 문항의 짧은 4-arm 검사

```bash
python scripts/run_development.py --device cuda:0 --q-checkpoint checkpoints/gptq-w3-g128 --q-manifest checkpoints/gptq-w3-g128/manifest.json --limit 2 --max-new-tokens 128 --output runs/dev-smoke.jsonl
```

F/Q 두 BF16 모델을 한 GPU에 올려 arm을 차례로 수행한다. runner는 먼저 실제 모델에서 F=Q·p=1 trace, cache/full-prefix, 독립 기준 분포를 검사한다. `dev-smoke.validation.json`이 실패면 생성 단계로 넘어가지 않는다. 생성 완료 시 `.manifest.json`에 코드 hash·GPU·VRAM peak·소비 토큰·시간을 기록한다. 실패한 부분 출력에는 완료 manifest가 없으므로 완성 결과로 집계하지 않는다. 에러 수정 후 새 파일 이름으로 재실행하며 실패 기록은 남긴다.

128토큰 제한의 accuracy/cap은 구현 smoke test용이며 H1·능력·파일럿 통계로 사용하지 않는다.

## 4. 실제 cap과 장비 관문

짧은 검사가 통과한 뒤 같은 development 표본에서 원래 cap을 점검한다.

```bash
python scripts/run_development.py --device cuda:0 --q-checkpoint checkpoints/gptq-w3-g128 --q-manifest checkpoints/gptq-w3-g128/manifest.json --limit 2 --max-new-tokens 32768 --output runs/dev-fullcap.jsonl
```

조기에 EOS가 나왔다면 32,768까지 cache가 늘어나는 경우를 검증한 것은 아니다. 새 manifest는 각 arm의 prompt/생성/cache 길이·종료 사유를 기록한다. 자유 생성 실효 처리량에는 sampler의 CPU 이동 비용을 포함한다.

아래 개발 전용 검사는 128개 development 문항 중 가장 긴 prompt를 반복해 **최대 prompt 길이+32,768**까지 두 모델의 독립 cache를 동시에 늘린다. 512토큰 chunk로 채운 뒤 마지막 두 번은 1토큰씩 처리한다. 마지막 위치에서 각 모델의 cache logits와 별도의 전체 prefix 계산을 비교한다. 기존 logit .25/TV .001 기준을 그대로 적용하며, 실패하면 기준을 완화하지 않고 검토한다.

```bash
python scripts/run_development.py --device cuda:0 --q-checkpoint checkpoints/gptq-w3-g128 --q-manifest checkpoints/gptq-w3-g128/manifest.json --max-new-tokens 32768 --context-stress-only --output runs/context-stress.jsonl
```

이 모드는 `.validation.json`과 `.context.json`만 남기고 생성 JSONL을 만들지 않는다. `.context.json`에는 실제 cache 길이·오차·VRAM peak·소요 시간·checkpoint/코드 hash가 들어간다. 합성 stress 시간은 자유 생성 처리량으로 사용하지 않는다. task 정답률·H1 결과가 아니며 모든 입력에서의 메모리 보증도 아니다. 합성 검사와 앞의 자연 생성 검사를 함께 검토한다.

남길 것: GPU/driver/package, F/Q checkpoint manifest, 구현 대조 결과, full-cap/context 검사, 실제 가용 GPU 시간. 실제 장비 전수 검사·환경 오차 검토가 끝나야 02 완료로 갱신한다. 그 후 03에서 128문항 pilot·정밀도·예산을 평가하고 확증 전 동결한다. 이 저장소의 현재 runner에는 test 실행 명령이 없으며, 여기서 H1 판정이나 확증을 수행하지 않는다.

현재 세션으로 가져올 작은 파일: `target-gate-01.tar.gz` 또는 preflight, Q `manifest.json`, `dev-smoke.validation.json`, 두 개발 실행의 `.manifest.json`, `context-stress.context.json`, 오류 로그. 하루 실제 사용 가능한 GPU별 시간을 함께 알려주면 03 예산을 계산할 수 있다. 대형 가중치·원시 trace는 Git에 올리지 않는다.
