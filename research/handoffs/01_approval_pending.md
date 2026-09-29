> 후속 상태(2026-09-29): 사용자가 승인 가정 진행을 지시하여 이 승인 대기 인계는 비활성화했다. 실제 교수 PASS 확인 기록은 아니다. 현재는 [02 실제 장비 인계](02_target_gpu_pending.md)를 따른다. 아래는 당시 기록이다.

# QuantSplit 01 — 승인 대기 인계

작성일: 2026-09-29 KST. **01단계 미완료. 문헌·설계·제안서 작업은 제출용으로 정리했으나 교수님 PASS가 없다.**

## 현재 결정

- 기준 저장소는 https://github.com/JunKIM2603a/quant-sampling-decomposition 이다. ai-research-chatgpt의 관련 16문서를 1ae2d88b47e1c48510ba2ad5f054ae1ec7a4110a에서 원문 이관하고 blob SHA 일치를 검증했다.
- 현행 제출 설계: 0.2-proposal-final. 확증 동결·실행 가능 manifest는 아직 아니다.
- 신규성: 동일 prefix에서 F/Q 확률·후보집합 공급자를 교차한 4-arm 자유 생성의 길이·정답률·상호작용·효과 상한으로 한정. 길이 증가·샘플러 튜닝·일반 점수/후보 분리는 선행 존재.
- H1: 1.5B GPTQ-W3/g128, BF16 모의 실행, T=0.6/p=0.95, GSM8K에서 R>0.50 AND 정확도 차이>−0.03. 상세 E 관문과 C_r 판정을 현행 프로토콜에서 읽는다.
- 제안서 제출은 조건부 GO, 모델 실험은 HOLD. 교수님께 이메일을 보내지 않았다.

## 읽을 파일

1. PROJECT_STATUS.md, PROJECT_INSTRUCTIONS.md, research/SESSION_PLAN.md
2. research/session01/01_novelty_review.md
3. research/session01/02_hypothesis_and_protocol.md, protocols/quant_sampling_v0.2.json
4. research/session01/03_competing_explanations.md, research/session01/07_resource_and_precision.md
5. research/session01/04_proposal_ko.md, research/session01/05_advisor_email.md
6. research/DECISIONS.md

## 남은 작업

교수님 PASS 여부, 승인일, 수정 요청을 사용자 또는 승인 기록에서 확인한다. 이메일 초안·제안서 파일의 존재나 사용자의 진행 요청을 교수 PASS로 해석하지 않는다. 수정이 있으면 v0.2 이후 새 버전과 변경 이유를 남긴다. 승인 기록과 수정 반영까지 마치면 01을 완료로 바꾸고 02로 넘어간다.

승인 전 허용된 다음 작업은 교수 피드백 반영·문헌 보완·설계 수정이다. 모델 평가·calibration 생성·파일럿·확증을 실행하지 않는다. 다른 연구의 AWQ 점검이나 결과를 이 저장소의 실험으로 가져오지 않는다.

## 대화를 먼저 옮길 때

세션 이름: **QuantSplit 01-2 — 교수 피드백·승인 기록**. 이는 같은 단계의 이어지는 대화다.

```text
이 세션은 ‘QuantSplit 01-2 — 교수 피드백·승인 기록’이야.
저장소: https://github.com/JunKIM2603a/quant-sampling-decomposition
PROJECT_STATUS.md, PROJECT_INSTRUCTIONS.md, research/SESSION_PLAN.md,
research/handoffs/01_approval_pending.md와 현행 v0.2 프로토콜을 읽고 이어가줘.
01의 문헌·H1·판정 기준·교수 제안서는 작성됐지만 단계 전체는 승인 대기야.
교수님 PASS 상태와 수정 요청: [여기에 실제 확인 내용 입력. 없으면 미확인]
미확인이면 모델 실험을 시작하지 말고 01을 유지해줘.
PASS가 확인되면 피드백을 반영하고 승인 근거·결정·인계를 Git에 기록한 뒤
‘QuantSplit 02 — 4-arm 구현·평가 검증’으로 전환할 시점을 알려줘.
```
