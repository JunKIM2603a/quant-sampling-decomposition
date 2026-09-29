# 실제 GPU cache 진단 검토

2026-09-30 KST. **정밀도에 의존하는 수치 차이의 근거를 확보했다. 기존 BF16 관문은 계속 실패이며 02는 미완료다.**

## 수신 증거

사용자가 `bc3e599`까지 fast-forward한 뒤 cuda:1에서 실행한 F/Q × 4모드 터미널 출력을 제공했다. 마지막 `Diagnostic saved` 메시지도 있다. 아래 수치는 사용자 제공 로그의 집계값이다. [기계 판독 기록](cache_diagnostic_gpu1_01_review.json)에 보존했다.

| 모드 | F 최대 logit 차이 | F 최대 TV | Q 최대 logit 차이 | Q 최대 TV |
|---|---:|---:|---:|---:|
| BF16 automatic SDPA | 0.21875 | 0.0092865266 | 0.3125 | 0.0266234534 |
| BF16 math SDPA | 0.328125 | 0.0113781842 | 0.25 | 0.0126859087 |
| BF16 math + full reduction | 0.25 | 0.0061440781 | 0.34375 | 0.0332893328 |
| 같은 BF16 가중치 값을 FP32로 승격 | 0.0000271797 | 0.0000005969 | 0.0000348091 | 0.0000038383 |

기존 기준은 max logit ≤.25와 TV ≤.001의 동시 충족이다. 세 BF16 모드 모두 F/Q의 TV 기준을 넘었다. FP32 모드의 수치는 이 기준 이내지만 BF16 실행이 아니므로 기존 BF16 검사를 통과시킨 것으로 기록하지 않는다.

8개 경우 모두 repository vs 직접 Transformers cache, fresh one-shot cache vs full-prefix, 같은 full-prefix 반복의 logit/TV 최대 차이가 0이다. 첫 BF16 모드는 앞선 validation JSON의 F/Q 실패 수치를 정확히 재현했다.

원본 `runs/cache-diagnostic-gpu1-01.json`은 아직 전달되지 않았다. 현재 첨부에는 각 step의 오차, cache 길이, 실제 backend 설정, 코드/lock hash, Q manifest가 없다. 저장 메시지까지 도달했다는 것은 코드상 해시·길이·유한성 검사를 마쳤다는 간접 근거지만, 해당 필드를 직접 검토했다고 쓰지 않는다. 원본 파일의 SHA256·실행 시각·GPU 이름·package 목록도 새로 지어내지 않는다.

## 해석

같은 BF16 weight 값에서 FP32 계산으로 바꾸면 cached/full-prefix 차이가 크게 줄었다. **incremental decode와 전체 prefix 계산의 정밀도 의존적 차이**가 주요 설명이라는 강한 근거다. 직접 Transformers cache와의 불일치, 반복 full-prefix의 비결정적 차이는 이 probe에서 관측되지 않았다.

math attention이나 BF16 reduction 설정 하나로 일관되게 개선되지는 않았다. 특히 Q의 full-reduction TV는 automatic보다 커졌다. 따라서 특정 attention kernel 하나의 문제로 단정하거나, F에서 가장 작은 오차가 나온 설정을 Q에도 적용하여 해결됐다고 할 수 없다.

PyTorch는 계산 형태가 달라지면 같은 수학적 연산이라도 부동소수점 결과가 같음을 보장하지 않으며 BF16 GEMM/SDPA의 정밀도 경로도 구분한다. 이는 이번 해석과 일치하지만, 특정 CUDA 연산을 원인으로 격리했다는 뜻은 아니다. FP32 승격은 activation·연산·KV precision을 함께 바꾸며 직접 참조도 Transformers 내부를 공유한다. 공통 backend 오류까지 완전히 배제하지 않는다.

현재 TV는 같은 모델 안의 cached/full-prefix 차이다. F-Q 양자화 효과, BF16-FP32 정책 차이, 생성 길이·정답률 영향은 아직 측정하지 않았다. 작은 next-token 차이가 긴 생성 전체에 미치는 영향을 이 로그로 상한화할 수 없다.

## 검증 방법 검토안 — 아직 적용하지 않음

기존 cache 검사는 구현 정확성과 BF16의 서로 다른 계산 형태 간 수치 일치성을 하나의 합격 조건에 묶었다. 두 목적을 분리하는 것이 다음 검토 방향이다.

1. **실제 BF16 cache 구현 검사:** 같은 prefix와 prefill/decode 순서에서 직접 Transformers 참조와 비교하고, cache 길이·위치·독립성과 identity/baseline 검사를 유지한다. 한 짧은 probe의 0 오차만으로 전체 경로를 보증하지 않는다.
2. **인과적 cache 계산의 정밀도 대조:** 같은 가중치 값의 고정밀도 cached/full-prefix 대조를 별도로 둔다. FP32 통과를 BF16 통과로 이름만 바꾸지 않는다.
3. **BF16 수치 민감도 기록:** 기존 cached/full-prefix 수치를 보존하고, 실제 top-p 후보집합과 교차 정책의 민감도를 development 데이터에서 별도 평가한다. 현재 로그만으로 양자화 신호에 비해 무시할 수준이라고 말하지 않는다.

이 문서는 개편 검토안이다. 현재 runner·TOLERANCES·v0.2.1 protocol은 변경하지 않았다. 개편을 채택할 경우 본 결과를 본 이력, 변경 이유, 시행 전 검사항목/표본/판정법을 새 개발 버전에 기록하고 그 버전의 실제 장비 검사를 수행해야 한다. 최초 BF16 실패 기록은 그대로 남긴다. 단순히 .001을 높이거나 FP32 생성으로 전환하는 방식은 이번 조치에 포함되지 않는다.

## 지금 필요한 입력과 후속 순서

**추가 GPU 실행 없이** 이미 생성된 `runs/cache-diagnostic-gpu1-01.json`을 첨부한다. 이 파일 안에 Q manifest가 포함되어 있으므로 체크포인트 원본 기록도 함께 검토할 수 있다. 파일 첨부가 어려우면 저장소 root에서 다음 출력 전체를 전달한다.

```bash
cat runs/cache-diagnostic-gpu1-01.json
```

원본 보고서에서 실제 설정·step별 패턴·provenance를 확인한 뒤 검증 개편안을 확정하거나 필요한 수정/추가 검사를 좁힌다. 현재 구현 오류로 확인되지 않은 부분을 추측으로 고치거나 같은 진단을 반복하지 않는다. 이후 짧은 4-arm, full-cap 자유 생성, 긴 context·VRAM·처리량 검증을 거쳐야 02 완료 여부를 판단할 수 있다. 03 파일럿·H1·확증은 미실행, 교수 실제 PASS도 미확인이다.

공식 근거(2026-09-30 확인): [PyTorch 2.8 numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html), [Transformers 4.57.1 cache](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation).
