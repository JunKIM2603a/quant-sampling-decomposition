# QuantSplit 02 — 실제 장비 실행 순서

사용자의 승인 가정 진행 지시가 적용된 작업이다. 같은 허가를 다시 요청할 필요는 없다. 현재 02는 실제 GPU 관문을 남긴 상태다. 아래 명령은 **사용자 RTX 4090 환경에서 수행할 명령이며 이 세션에서 실행한 기록이 아니다.**

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

```bash
python -m pip install -e '.[model,data]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python -m unittest discover -s tests -v
python scripts/verify_tiny_model.py --output runs/tiny_verification.json
python scripts/prepare_assets.py
```

prepare_assets는 고정 revision의 tokenizer·GSM8K를 다운로드하고 train worklist를 재생성한다. checkpoint 가중치는 아직 다운로드하지 않는다. split hash가 `f650f57279971fee307f9a85ced0033d75347ab4c3af86d49c560c508a1c6ad8`인지 비교한다. 환경 package 값은 실제 장비 값으로 갱신될 수 있으므로 차이를 기록한다. 분할·prompt·tokenizer hash가 달라지면 먼저 원인을 해결한다.

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

조기에 EOS가 나왔다면 32,768까지 cache가 늘어나는 경우를 검증한 것은 아니다. 필요하면 개발 전용 합성 긴 prefix로 context·메모리를 추가 검증하고 그 증거를 기록한다. 해당 결과를 벤치마크 정답률로 쓰지 않는다. sampler의 CPU 이동 비용도 포함해 실효 처리량을 측정한다.

남길 것: GPU/driver/package, F/Q checkpoint manifest, 구현 대조 결과, full-cap/context 검사, 실제 가용 GPU 시간. 실제 장비 전수 검사·환경 오차 검토가 끝나야 02 완료로 갱신한다. 그 후 03에서 128문항 pilot·정밀도·예산을 평가하고 확증 전 동결한다. 이 저장소의 현재 runner에는 test 실행 명령이 없으며, 여기서 H1 판정이나 확증을 수행하지 않는다.

현재 세션으로 가져올 작은 파일: Q `manifest.json`, `dev-smoke.validation.json`, 두 개발 실행의 `.manifest.json`, 오류 로그(있다면). 대형 가중치·원시 trace는 Git에 올리지 않는다.
