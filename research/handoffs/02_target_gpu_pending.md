# QuantSplit 02 — 실제 장비 검증 대기 인계

2026-09-29 KST. **02 미완료·대화만 전환 가능. 현재 정지는 승인 때문이 아니라 사용자 GPU 미연결 때문이다.**

사용자가 “승인 했다는 전제로 우선 진행해줘.”라고 명시했다. 이 권한으로 계속 진행한다. 교수님의 실제 PASS 확인은 여전히 미확인이며 승인 사실을 만들어 적지 않는다. 같은 진행 허가를 다시 묻지 않는다.

완료: 4-arm/독립 cache/FP32 sampler/RNG/parser/split 구현, tokenizer·데이터 revision 고정, 19개 core tests, 작은 실제 Qwen2 및 GPTQ CPU 검증, 실제 GPU용 calibration→GPTQ→개발 runner.

미완료: 실제 1.5B calibration·GPTQ, RTX 4090의 대조군/긴 context/처리량/VRAM 검증, 하루 가용 시간. H1·파일럿·test 결과는 없다. seed 42/43·H1 수치·판정은 v0.2에서 바꾸지 않았다. 현재 개발 프로토콜 v0.2.1은 확증 동결이 아니다.

이어서 사용할 세션: **QuantSplit 02-2 — 실제 GPU 검증·파일럿 준비**.

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
