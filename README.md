# QuantSplit

저비트 추론 모델의 출력 길이 증가에서 확률 왜곡과 샘플링 후보집합 변화의 역할을 분리한다.

**현재: 사용자 승인 가정 지시로 02 구현을 진행. 19개 core tests와 작은 Qwen2/GPTQ CPU 검증 통과. 실제 1.5B/GPTQ·RTX 4090·긴 context 검증은 남아 있다. H1 결과는 아직 없다.**

| 정책 | 확률 공급자 | top-p 후보집합 공급자 |
|---|---|---|
| FF | BF16 F | BF16 F |
| QQ | 양자화 Q | 양자화 Q |
| QF | 양자화 Q | BF16 F |
| FQ | BF16 F | 양자화 Q |

각 정책이 자신의 토큰열을 만들고, 그 안에서 F/Q가 동일 prefix를 별도 KV cache로 처리한다. QF는 Q의 logits와 허용 토큰 간 상대확률을 유지한다. 재정규화된 절대확률은 변한다.

- [현재 상태](PROJECT_STATUS.md) · [프로젝트 지침](PROJECT_INSTRUCTIONS.md) · [세션 운영](research/SESSION_PLAN.md)
- [구현·검증](research/session02/01_implementation.md) · [실제 GPU 실행 순서](research/session02/02_runbook.md) · [현재 인계](research/handoffs/02_target_gpu_pending.md)
- [신규성](research/session01/01_novelty_review.md) · [H1·판정](research/session01/02_hypothesis_and_protocol.md) · [교수 제안서](research/session01/04_proposal_ko.md)
- [v0.2.1 개발 프로토콜](protocols/quant_sampling_v0.2.1-development.json) · [실행 lock](configs/execution_lock.json) · [결정 근거](research/DECISIONS.md)

CPU core 검사:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

실제 Transformers 작은 모델 검증과 GPU 실행은 runbook을 따른다. GPU가 없는 환경의 CPU 검증을 사용자 장비 실험으로 보고하지 않는다. `runs/`, `data/`, `checkpoints/`는 Git에 올리지 않으며 작은 manifest·검증 기록을 보존한다.

00단계 `research/2026-09-29/`와 v0.1은 역사 자료다. 제출 가설은 v0.2, 구현 상세와 승인 가정 지시는 v0.2.1을 따른다. 교수 실제 PASS와 확증 동결은 별도 사실로 기록한다.
