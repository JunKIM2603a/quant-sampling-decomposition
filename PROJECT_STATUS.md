# QuantSplit — 현재 연구 상태

최종 갱신: 2026-09-29 KST. 현행 기준: 제출 설계 v0.2. 이전 00단계 문서는 보존된 역사 기록이다.

| 항목 | 현재 상태 |
|---|---|
| 프로젝트 | QuantSplit — 저비트 추론의 확률·후보집합 분해 |
| 기준 저장소 | https://github.com/JunKIM2603a/quant-sampling-decomposition |
| 문서 이관 | 완료. 이전 저장소 c21bccc704df3e1e487fd8241b87e702578695b2의 관련 16파일을 원문 그대로 복사, blob SHA 모두 일치 |
| 이관 커밋 | 1ae2d88b47e1c48510ba2ad5f054ae1ec7a4110a |
| 현재 세션 | QuantSplit 01 — 신규성 검증·H1 확정·제안서 |
| 문헌·설계·제안서 작업 | 제출용 문서 확정 |
| 전체 01단계 | **미완료 — 교수님 PASS 및 수정사항 확인 대기** |
| 과학적 판단 | 제안서 제출 조건부 GO. 실험 HOLD |
| 프로토콜 | 0.2-proposal-final. 실행 revision 미확정, 확증 동결 아님 |
| 교수님 PASS | 미확인. 사용자도 이번 세션에서 미확인이라고 명시 |
| 이메일 | 초안만 작성, 발송하지 않음 |
| 모델 평가·calibration·파일럿·확증 | 미실행 |
| 실험 결과·핵심 발견 | 없음 |
| 장비·처리량·하루 가용 GPU 시간 | 연결·실측·확인 전 |

## 확정한 제출 주장

양자화 후 길이 증가와 일반 sampler 보정 자체에는 선행연구가 있다. 확인한 근접 자료에서, F/Q 확률 공급자와 top-p 후보집합 공급자를 교차한 네 자유 생성 정책의 길이·정답률·상호작용·작은 효과 상한을 함께 검증한 동일 설계를 확인하지 못했다. 이는 최초성 증명이 아니다.

H1: DeepSeek-R1-Distill-Qwen-1.5B, GPTQ W3/g128의 BF16 모의 실행, T=0.6, top-p=0.95, GSM8K에서 QF가 QQ의 초과 소비 토큰을 50% 넘게 복구하고 QQ 대비 정답률 손실이 3%p 미만이다. 원 현상·분모 적격 조건과 엄격한 구간 판정은 아래 현행 설계를 따른다. 이 수치는 관측값이 아니다.

## 현행 문서

1. [신규성 검토](research/session01/01_novelty_review.md)
2. [H1·판정·설계](research/session01/02_hypothesis_and_protocol.md), [JSON v0.2](protocols/quant_sampling_v0.2.json)
3. [경쟁 설명과 대조](research/session01/03_competing_explanations.md)
4. [교수님 제안서](research/session01/04_proposal_ko.md), [이메일 초안](research/session01/05_advisor_email.md)
5. [검색·접근 기록](research/session01/06_search_audit.json), [정밀도·예산](research/session01/07_resource_and_precision.md)
6. [현재 승인 대기 인계](research/handoffs/01_approval_pending.md), [조건부 01→02 인계](research/handoffs/01_to_02.md)
7. [이관 기록](research/migration/2026-09-29_manifest.json), [의사결정](research/DECISIONS.md)

## 다음 행동과 남은 위험

- 제안서를 교수님께 제출하고 PASS/수정 요청을 이 세션에 전달한다. 이 문서 생성은 제출이나 승인이 아니다.
- 승인 전에는 01 유지. 대화만 바꾸려면 ‘QuantSplit 01-2 — 교수 피드백·승인 기록’을 사용한다.
- PASS 확인 및 수정 반영 후에만 ‘QuantSplit 02 — 4-arm 구현·평가 검증’으로 전환한다.
- BF16 참조 후보집합은 진단용이다. 자연적 원인 비율이나 배포 가속으로 주장하지 않는다.
- H1 긍정 가능성·복구율 정밀도·실측 예산은 미확인이다. 1,319문제라고 정확도 3%p 비열등성이 자동 확보되지 않는다.
- 두 모니터링 관련 신규 검색 논문은 초록만 확인하고 본문 접근에 실패했다. 확인 범위 밖의 동일 선행 가능성은 남는다.
- MATH-500·다른 양자화·7B는 주 결과와 예산 확인 후 선택한다. 10/12 핵심 결과 확보를 보장하지 않는다.
