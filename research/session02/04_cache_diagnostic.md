# 실제 cache 대조 실패 — 짧은 원인 분리 진단

2026-09-30 KST 진단 준비 기록. **후속: 실제 GPU 8개 모드 요약을 받았고 정밀도 의존성 근거를 확보했다. 원본 진단 JSON 검토 대기·02 미완료.** 최신 판단은 [05_cache_diagnostic_review.md](05_cache_diagnostic_review.md)를 따른다. 아래 실행·검증 내용은 당시 준비 이력이며 H1·파일럿·확증 결과가 아니다. 사용자 승인 가정 진행 권한은 유효하며 교수님의 실제 PASS는 미확인이다.

## 확인한 실패

사용자가 `runs/target-gate-gpu1-01/dev-smoke.validation.json`의 cat 출력을 제공했다. JSON 내용은 [target_gpu_run01_validation.json](target_gpu_run01_validation.json)에 보존했다. 원본 파일 bytes/hash를 확보했다는 뜻은 아니다. 앞선 실행 로그의 환경은 cuda:1, RTX 4090, torch 2.8.0+cu126, Transformers 4.57.1이다.

| 항목 | F | Q | 사전 기준 |
|---|---:|---:|---:|
| cache/full-prefix 최대 logit 차이 | 0.21875 (통과) | 0.3125 (실패) | ≤0.25 |
| T=.6 분포 TV | 0.009286526618974375 (실패) | 0.02662345340131388 (실패) | ≤0.001 |

F=Q trace, p=1 trace, 독립 FF/QQ baseline 분포 대조는 통과했다. TV는 같은 모델·같은 토큰열에서 cached 계산과 full-prefix 재계산의 차이다. F-Q 양자화 효과나 정답률 %p 차이가 아니다. max logit와 max TV가 같은 step에서 발생했는지도 기존 집계 JSON만으로는 알 수 없다.

## 이번 진단

기존 runner와 같은 첫 development 문항의 prompt 앞 32토큰, 마지막 토큰 4회 반복으로 총 5지점을 검사한다. 자유 생성·정답 채점·pilot/test 열람은 하지 않는다. 기존 checkpoint의 전체 파일 해시와 잠긴 split/worklist/tokenizer를 확인한 뒤 읽는다. calibration과 GPTQ는 다시 실행하지 않는다.

F와 Q를 하나씩 올리고, 각 모델에서 아래 순서로 비교한다. 마지막 FP32는 이미 로드한 BF16 가중치를 승격한다. 저장된 가중치는 바꾸지 않는다.

| 모드 | 바뀌는 계산 조건 | 진단 목적 |
|---|---|---|
| `bf16_auto` | 기존 BF16 + 자동 SDPA, TF32 off | 기존 실패 수치 재현 여부 |
| `bf16_math` | math SDPA만 허용, attention 중간 계산의 저정밀 reduction 금지 | attention 경로 차이 |
| `bf16_math_full_reduction` | 위 조건 + BF16 GEMM reduced-precision reduction 금지 | GEMM 누산 경로 차이 |
| `fp32_math` | 같은 BF16 가중치 값을 FP32로 승격, FP32 activation/cache, math SDPA, TF32 off | 정밀도 의존성 확인 |

각 지점에서 repository cache vs full-prefix, repository cache vs 명시적 position/mask를 주는 직접 Transformers cache, fresh one-shot cache vs no-cache, 동일 full-prefix 반복을 비교한다. 모든 layer의 cache 길이와 logits 유한성도 검사한다. 직접 참조도 Transformers 내부를 공유하므로 둘의 일치가 공통 backend 오류까지 배제하지는 않는다. 자동 SDPA에서 실제 선택된 kernel을 profiler로 확인하는 진단은 아니다.

FP32 통과는 BF16 관문 통과를 대신하지 않으며, 연구의 BF16 activation/KV 설정을 FP32로 변경하는 결정이 아니다. 오차 기준·sampler·H1·기존 생성 코드·protocol은 그대로다. 모드별 `meets_existing_cache_tolerance`는 cache 대조에만 해당한다. `completed=true`는 진단 수집 완료이며 전체 구현 검사나 02 완료가 아니다.

## 사용자 장비에서 실행

저장소 root에서:

```bash
conda activate quantsplit
git pull --ff-only
python scripts/diagnose_cache.py --device cuda:1 --q-checkpoint checkpoints/gptq-w3-g128 --output runs/cache-diagnostic-gpu1-01.json
```

이 출력 JSON을 현재 채팅에 첨부한다. 이미 파일이 있으면 새 번호를 쓴다. Q manifest 원문, checkpoint 해시 검증 여부, 코드/lock 해시, 패키지·GPU·probe IDs, 각 모드 설정/오차가 포함된다. 실행 중 모드별로 저장하며 오류 시에도 완료된 모드와 오류 정보를 남긴다. 오류가 나면 JSON과 터미널 오류를 함께 전달한다. SIGKILL 등 강제 종료 시에는 마지막 저장분만 남을 수 있다.

첫 모드가 기존 실패를 재현하지 못하면 환경/입력/설정 차이를 먼저 확인한다. 직접 cache 참조와 불일치하면 위치·mask·wrapper를 우선 조사한다. 참조가 일치하고 연산 모드에 따라 차이가 변하면 수치 경로 원인이라는 근거가 되지만 자동으로 원인이 확정되는 것은 아니다. 결과 검토 후 근거 있는 수정 또는 추가 검사를 정하며, 통과하는 모드를 고르거나 기준을 완화하여 문제를 숨기지 않는다.

이번 실행은 짧은 진단이다. 결과를 검토하기 전 smoke/full-cap/context를 재개하지 않는다. 실제 수정 후 기존 전체 구현 검사·짧은 생성·32,768 cap·긴 context·VRAM/처리량 관문이 남는다. 이후에만 03 파일럿·진행 판정·동결 인계를 활성화한다.

## assistant 환경에서 확인한 범위

CPU torch 2.8.0+cpu / Transformers 4.57.1에서 `PYTHONPATH=src OMP_NUM_THREADS=2 python -m unittest discover -s tests -v`: **27 tests, 2.777초, 모두 통과**. 기존 24개에 실제 작은 Qwen2의 네 모드와 직접 cache 참조, 잘못된 repository logits 주입 검출, 예외 발생 시 backend 설정 복원/비유한값 거부 3개를 추가했다. GPU가 없는 CLI 실행은 JSON에 오류를 남기고 중단함을 확인했다. 실제 DeepSeek 1.5B/4090에서 이 신규 진단은 아직 실행하지 않았다.

## 공식 근거와 한계

- [PyTorch 2.8 수치 정확도](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html): 계산 묶음/형태에 따른 부동소수점 차이, BF16 GEMM reduced-precision reduction의 설정과 SDPA 정밀도. 이 일반적 사실만으로 이번 실패를 정상 오차라고 판정하지 않는다.
- [PyTorch 2.8 SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html): 자동 backend 선택, math backend 제어, backend별 수치 결과 차이.
- [Transformers 4.57.1 cache](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation): 전체 past+current attention mask 및 cache_position의 사용법. 설치된 4.57.1 Qwen2 구현도 대조했다.
