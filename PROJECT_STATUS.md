# QuantSplit — 현재 연구 상태

최종 갱신: 2026-09-30 KST. 제출 가설 v0.2 / 개발 구현 v0.2.1. 과거 00·01 문서는 역사 기록을 포함한다.

| 항목 | 현재 상태 |
|---|---|
| 기준 저장소 | https://github.com/JunKIM2603a/quant-sampling-decomposition |
| 현재 단계 | **QuantSplit 02-2 — 실제 GPU 검증·파일럿 준비** |
| 진행 권한 | 사용자가 “승인 했다는 전제로 우선 진행해줘.”라고 지시. 승인 가정으로 진행 |
| 교수 실제 PASS | 미확인. 가정과 실제 승인 기록을 구분 |
| 01 산출물 | 문헌·신규성 범위·H1·경쟁 설명·판정·교수 제안서 완료 |
| 02 구현 | 4-arm, cache, sampler, 난수, parser, 분할, GPTQ 준비·개발 runner 구현 |
| 검증 | 기존 24개 + cache 진단 3개 = 27개 CPU tests 통과. 작은 Qwen2/GPTQ·합성 context 기록 보존. 실제 GPU 진단 8개 요약 수신; 원본 JSON 검토 대기 |
| 자산 | 모델/tokenizer·GSM8K revision 고정. calibration/development/pilot 128/128/128 분할 고정 |
| 실제 1.5B calibration/GPTQ | 사용자 로그에서 calibration 128/128 및 checkpoint/manifest 저장 완료 확인. 원본 Q manifest 직접 검토는 대기 |
| 사용자 RTX 4090 검증 | cuda:1, torch 2.8.0+cu126에서 BF16 matmul/SDPA·FP32 Cholesky 통과. 기본 BF16 실패 재현. 직접 cache 대조 0 차이, 세 BF16 모드 TV 실패, FP32 대조 오차 대폭 감소. identity·p=1·baseline 분포 통과 |
| 02 전체 | **미완료 — 원본 진단 보고서 검토·수치 검증 방법 정리 및 full-cap/context 검증 필요** |
| 파일럿·H1 결과·확증 | 미실행. 자체 test 모델 출력 없음 |
| 프로토콜 동결 | 아직 아님. 03에서 파일럿·정밀도·예산 후 확증 동결 |
| 이메일 | 초안만 작성, 미발송 |

## 현행 문서

1. [02 구현·검증·판정](research/session02/01_implementation.md), [실제 장비 실행 순서](research/session02/02_runbook.md)
2. [개발 프로토콜 v0.2.1](protocols/quant_sampling_v0.2.1-development.json), [execution lock](configs/execution_lock.json), [GPTQ 설정](configs/quantization.json)
3. [core 검증](research/session02/core_test_report.json), [작은 Qwen2/GPTQ 검증](research/session02/tiny_model_verification.json), [환경](research/session02/environment.json)
4. [분할 manifest](research/session02/split_manifest.json), [원본 메타데이터](research/session02/upstream_metadata.json)
5. [신규성 검토](research/session01/01_novelty_review.md), [H1·판정](research/session01/02_hypothesis_and_protocol.md), [경쟁 설명](research/session01/03_competing_explanations.md)
6. [교수 제안서](research/session01/04_proposal_ko.md), [이메일 초안](research/session01/05_advisor_email.md)
7. [현재 02 인계](research/handoffs/02_target_gpu_pending.md), [조건부 02→03](research/handoffs/02_to_03.md), [의사결정](research/DECISIONS.md)

## 현재 판단과 다음 행동

사용자가 `bc3e599`의 cache 진단을 cuda:1에서 완료한 터미널 출력을 제공했다. 기본 BF16은 이전 실패를 정확히 재현했고, 직접 Transformers cache 참조와의 차이는 모든 모드에서 0이다. BF16 세 모드는 TV 기준을 넘었으며 같은 weight 값의 FP32 계산에서는 F TV=5.97e-7, Q TV=3.84e-6으로 줄었다. **정밀도 의존적 수치 차이가 주요 설명이라는 강한 근거지만 기존 BF16 관문은 통과하지 못했다.** [GPU 진단 검토](research/session02/05_cache_diagnostic_review.md)를 따른다.

**다음 입력은 이미 저장된 `runs/cache-diagnostic-gpu1-01.json` 원본이다. 추가 실행·재양자화는 필요하지 않다.** step별 오차·실제 설정·hash·Q manifest를 확인한 뒤 cache 구현 검증과 수치 민감도 검사를 분리하는 개편안을 확정한다. 현재 기준·runner·프로토콜은 유지하며, 개편 시 변경 이유와 이미 본 데이터를 새 개발 버전에 기록한다. 교수 실제 PASS 미확인·02 미완료·03 비활성 상태를 유지한다.

H1은 1.5B GPTQ-W3/g128, T=.6, p=.95, GSM8K에서 QF가 QQ의 초과 토큰을 50% 넘게 복구하고 QQ 대비 정확도 손실이 3%p 미만이라는 기존 제출 가설을 유지한다. CPU 검증 수치는 이 H1의 실험 결과가 아니다. 정확한 판정은 v0.2 설계를 따른다.

현재 이어갈 대화: **QuantSplit 02-2 — 실제 GPU 검증·파일럿 준비**. 실제 장비 관문을 통과하면 **QuantSplit 03 — 파일럿·진행 판정·프로토콜 동결**.

## 보존된 이력과 위험

- 이전 저장소 `c21bccc704df3e1e487fd8241b87e702578695b2`의 관련 16파일을 원문 이관했다. 이관 커밋 `1ae2d88b47e1c48510ba2ad5f054ae1ec7a4110a`. [이관 기록](research/migration/2026-09-29_manifest.json).
- 01 문서 확정 커밋: `7ee774c4a9f07881a23d97bda0175adfc2d8ce48`.
- BF16 참조 mask는 진단용이고, BF16 모의 양자화는 실제 INT3 가속 증거가 아니다.
- 실제 GPTQ 품질·길이 증가 재현·3%p 비열등성 정밀도·마감 예산은 미확인이다. 10/12 핵심 발견을 보장하지 않는다.
- tokenizer 길이 16,384/model context 131,072 차이와 prompt+32,768 cap를 실제 장비에서 확인해야 한다.
- 두 모니터링 관련 선행 검색 논문은 초록만 확인했다. 최초성이나 동일 선행 부재를 보장하지 않는다.
