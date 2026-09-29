# QuantSplit — 저비트 추론의 확률·후보집합 분해

양자화 전후 모델의 확률 공급자와 top-p 후보집합 공급자를 교차해, 자유 생성의 소비 토큰·정답률·상호작용을 분석하는 연구입니다.

**현재: 제출용 신규성 검토·H1·제안서 정리 완료, 교수님 PASS 대기. 모델 실험은 실행하지 않았습니다.**

| 조건 | 확률 공급자 | 후보집합 공급자 |
|---|---|---|
| FF | BF16 | BF16 |
| QQ | 양자화 | 양자화 |
| QF | 양자화 | BF16 |
| FQ | BF16 | 양자화 |

각 조건의 현재 prefix를 두 모델이 동일하게 처리합니다. 효과는 특정 생성 정책의 개입 효과이며, 유일한 자연적 원인 비율 또는 배포 속도 개선이 아닙니다.

- 시작: [현재 상태](PROJECT_STATUS.md), [프로젝트 지침](PROJECT_INSTRUCTIONS.md), [세션 계획](research/SESSION_PLAN.md)
- 제출: [교수님 제안서](research/session01/04_proposal_ko.md), [이메일 초안](research/session01/05_advisor_email.md)
- 근거: [신규성 검토](research/session01/01_novelty_review.md), [검색 기록](research/session01/06_search_audit.json)
- 설계: [H1·판정 기준](research/session01/02_hypothesis_and_protocol.md), [프로토콜 v0.2](protocols/quant_sampling_v0.2.json), [경쟁 설명](research/session01/03_competing_explanations.md)
- 실행 전 검토: [정밀도·계산 예산](research/session01/07_resource_and_precision.md)
- 인계: [승인 대기](research/handoffs/01_approval_pending.md), [PASS 이후 02](research/handoffs/01_to_02.md)
- 기록: [결정 근거](research/DECISIONS.md), [이관 manifest](research/migration/2026-09-29_manifest.json)

이전 ai-research-chatgpt의 관련 자료 16개를 2026-09-29에 원문 이관했습니다. research/2026-09-29/와 protocol v0.1은 당시 초안 기록이며 현행 설계는 v0.2입니다.
