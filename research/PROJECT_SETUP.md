> 현재 상태(2026-09-29): 사용자 승인 가정 지시로 02 구현에 진입. 실제 GPU 검증 대기. 최신 [PROJECT_STATUS.md](../PROJECT_STATUS.md)와 [02 인계](handoffs/02_target_gpu_pending.md)를 따른다.

# QuantSplit — 프로젝트 구성

갱신: 2026-09-29 KST.

| 항목 | 확정 |
|---|---|
| 이름 | QuantSplit — 저비트 추론의 확률·후보집합 분해 |
| 영문 가제 | Separating Probability Distortion from Sampling-Support Changes in Quantized Reasoning Models |
| 전용 저장소 | https://github.com/JunKIM2603a/quant-sampling-decomposition |
| 저장소 설명 | 양자화 추론 모델의 확률 왜곡과 샘플링 후보 집합 변화가 생성 길이·정답률에 미치는 영향을 교차 개입으로 분석하는 연구 |
| 현재 세션 | QuantSplit 01 — 신규성 검증·H1 확정·제안서 |
| 현재 관문 | 교수님 PASS 및 수정사항 확인 |
| 승인 후 다음 세션 | QuantSplit 02 — 4-arm 구현·평가 검증 |

기존 ai-research-chatgpt의 관련 16개 문서를 [이관 manifest](migration/2026-09-29_manifest.json)에 기록한 commit에서 원문 복사하고 blob SHA로 검증했다. 원본 저장소를 삭제·변경하지 않았다. 전용 저장소의 기존 README는 이관 커밋에서 보존했고, 이후 현행 안내로 갱신했다.

프로젝트 지침은 [PROJECT_INSTRUCTIONS.md](../PROJECT_INSTRUCTIONS.md), 현재 상태는 [PROJECT_STATUS.md](../PROJECT_STATUS.md), 단계 운영은 [SESSION_PLAN.md](SESSION_PLAN.md)를 따른다. 새 세션은 [현재 승인 대기 인계](handoffs/01_approval_pending.md)를 읽는다. 역사 기록의 이전 저장소 URL·상태를 현재 상태로 오해하지 않는다.

교수 제안서 제출·PASS와 모델 실험 시작은 각각 별개다. 현재 문서는 제출 가능하지만 메시지 발송·승인·모델 실험은 이루어지지 않았다.
