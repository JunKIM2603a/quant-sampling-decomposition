# QuantSplit 02 — 실제 장비 검증 대기 인계

최신 상태 2026-09-30 KST. **02 미완료. 사용자 GPU에서 F/Q cache 대조가 실패했다. 세부 JSON은 수신했고, 짧은 원인 분리 진단 결과를 기다린다. 아래 9/29 내용은 이력이다.**

사용자가 “승인 했다는 전제로 우선 진행해줘.”라고 명시했다. 이 권한으로 계속 진행한다. 교수님의 실제 PASS 확인은 여전히 미확인이며 승인 사실을 만들어 적지 않는다. 같은 진행 허가를 다시 묻지 않는다.

완료: 4-arm/독립 cache/FP32 sampler/RNG/parser/split 구현, tokenizer·데이터 revision 고정, 19개 core tests, 작은 실제 Qwen2 및 GPTQ CPU 검증, 실제 GPU용 calibration→GPTQ→개발 runner.

최신 미완료: 실제 Q manifest 직접 검토, F/Q 구현 검사 실패 진단, 긴 context/처리량/VRAM 검증, 하루 가용 시간. calibration 128개 생성과 GPTQ 저장은 사용자 로그에서 완료 확인했다. H1·파일럿·test 결과는 없다. seed 42/43·H1 수치·판정은 v0.2에서 바꾸지 않았다. 현재 개발 프로토콜 v0.2.1은 확증 동결이 아니다.

## 9/30 후속 validation JSON — 현재 우선 작업

F=Q·p=1 trace와 FF/QQ 독립 baseline 분포는 통과했다. F의 cache/full-prefix TV=0.0092865266은 기준 .001을 초과했다. Q는 max logit=.3125(기준 .25) 및 TV=.0266234534(기준 .001)로 실패했다. 이는 F와 Q 사이의 양자화 효과나 정답률 차이가 아니다.

1. 기존 checkpoint·실패 폴더·오차 기준을 보존한다. 재양자화는 필요하다고 확인되지 않았다.
2. [04_cache_diagnostic.md](../session02/04_cache_diagnostic.md)의 명령을 cuda:1에서 실행한다. 기본 BF16 / math SDPA / BF16 reduction 제한 / 같은 BF16 가중치의 FP32 진단을 같은 짧은 입력에서 비교한다.
3. `runs/cache-diagnostic-gpu1-01.json`을 받는다. checkpoint 파일 해시 검증과 Q manifest도 포함되므로 함께 검토할 수 있다. 신규 진단은 CPU 27 tests만 통과했고 실제 1.5B 실행은 대기다.
4. 수치 연산 경로와 cache 구현 문제를 구분한 뒤 근거 있는 수정을 결정한다. FP32 진단 통과가 BF16 연구 설정 변경이나 02 완료를 뜻하지 않는다. smoke/full-cap/context·파일럿·H1은 미실행, 03 인계는 비활성이다.

근거: [실패 수치 원문 JSON 내용](../session02/target_gpu_run01_validation.json), [로그 검토](../session02/target_gpu_run01_review.json). 교수 실제 PASS 미확인 상태를 유지한다.

이어서 사용할 세션: **QuantSplit 02-2 — 실제 GPU 검증·파일럿 준비**.

## 02-2에서 이어서 확인한 상태

- 시작 기준 main: `2963e90e5408c120fa8cd083ed816cfa8a269551`. 요청한 7개 문서를 읽었고 원격 파일 해시를 대조했다.
- 현재 assistant 환경은 사용자 PC가 아니며 GPU와 사용자 장비 연결 수단이 없다. 이 요청에 첨부된 새로운 장비 로그도 없다. 실제 교수 PASS는 미확인으로 유지한다.
- 고정 자산을 재생성하면서 lock을 덮어쓰던 결함을 고쳤다. 사전 대조 후 불일치이면 중단한다. 19개 기존+5개 추가 CPU 검사를 통과했다.
- `scripts/check_target_environment.py`, `scripts/run_target_gate.sh`, runner의 `--context-stress-only`를 추가했다. [변경·검증 범위](../session02/03_target_gpu_preparation.md), [실행 순서](../session02/02_runbook.md)를 읽는다.
- 다음 입력: 사용자 RTX 4090 PC의 `runs/target-gate-01.tar.gz` 또는 `target-preflight-01.json`/오류 로그. 하루 GPU별 실제 가용 시간도 필요하다.
- **02 완료 아님. 03으로 넘기지 않음.** 실제 checkpoint·짧은 대조·자연 생성 full-cap·합성 context·VRAM·처리량 증거를 확인한 뒤에만 조건부 인계를 활성화한다.

### 후속 장비 로그 수신

사용자가 2026-09-29 22:31:30 KST의 nvidia-smi를 제공했다. 두 RTX 4090과 driver 535.183.01/CUDA 표시 12.2를 확인했다. 이는 PyTorch 모델 실행 증거가 아니다. Python 3.12 conda 환경 생성은 성공했다. 공식 minor compatibility 조건을 검토하여 torch 2.8.0/cu126을 유지하고 작은 실제 BF16 matmul/SDPA·FP32 Cholesky 사전 검사를 추가했다. 로그상 여유가 더 큰 `cuda:1`을 우선 사용한다. 다음 입력은 `target-preflight-gpu1-01.json` 또는 실행 오류다. 02/03 및 실제 교수 PASS 상태는 변하지 않았다.

```text
이 세션은 ‘QuantSplit 02-2 — 실제 GPU 검증·파일럿 준비’야.
저장소: https://github.com/JunKIM2603a/quant-sampling-decomposition
나는 앞 세션에서 ‘승인 했다는 전제로 우선 진행해줘’라고 지시했어.
그 권한으로 이어서 진행하되 교수님의 실제 PASS를 확인한 것으로 기록하지 마.
최신 PROJECT_STATUS.md, PROJECT_INSTRUCTIONS.md, research/SESSION_PLAN.md,
research/handoffs/02_target_gpu_pending.md, research/session02/01_implementation.md,
research/session02/02_runbook.md와 protocols/quant_sampling_v0.2.1-development.json을 읽어줘.
코드·19개 core tests·작은 Qwen2/GPTQ CPU 검증과 분할은 완료됐지만,
실제 DeepSeek 1.5B/GPTQ·RTX 4090·32,768 cap 검증은 아직이야.
장비 연결이나 제공한 실행 로그로 남은 관문을 확인하고 오류가 있으면 수정해줘.
실제 장비 확인 전에 02 완료나 H1 결과라고 쓰지 마.
완료하면 Git을 갱신하고 03 파일럿·진행 판정·프로토콜 동결로 인계해줘.
```
