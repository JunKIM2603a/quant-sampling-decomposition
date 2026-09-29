# 원본 GPU 진단 검토 완료와 정책 수치 민감도 audit v1

2026-09-30 KST. **원본 보고서 검토 완료. 실제 BF16 관문은 실패 상태이며 02 미완료다.** 다음 development 진단의 입력·절차를 고정했다. 이는 03 파일럿이나 확증 동결이 아니다.

## 이번에 확인한 것

원본 [cache-diagnostic-gpu1-01.json](evidence/cache-diagnostic-gpu1-01.json)을 byte 그대로 보존했다. 170,747 bytes, SHA256 `f8f084aa9ab42fc050468ca651a2ae3fd8f8c607f399316e5acab810d97a192e`. 보고 시각은 2026-09-29 20:44:38 UTC(9/30 05:44:38 KST), 실행 commit `bc3e599`, clean working tree, cuda:1/RTX 4090, torch 2.8.0+cu126, Transformers 4.57.1이다.

- runner·11개 source 파일·execution lock 해시가 기록된 코드/고정값과 일치한다. 8개 모드의 요약과 40개 step 값도 일치한다.
- 모든 28개 층에서 cache 길이는 32→36으로 입력과 일치했다. 첫 prefill(32토큰)의 cache/full-prefix 차이는 모두 0이고 차이는 이후 decode에서 나타났다. direct/fresh/repeat 대조 차이는 모든 위치에서 0, 유한성 검사도 모두 true다.
- 기본 BF16 TV는 F의 33/34/35/36 위치에서 .002609/.009287/.002277/.000713, Q는 .003050/.026623/.001021/.005307이다. 오차가 길이에 따라 단조 증가한다고 해석할 수 없다. 기존 실패는 그대로다.
- 실제 TF32·SDPA·reduction 설정이 모드 정의와 일치한다. 같은 BF16 값의 FP32 승격에서 TV가 크게 줄었다. **정밀도에 의존하는 계산 형태 차이**가 주요 설명이라는 판단을 유지한다. 특정 kernel 원인을 격리하거나 Transformers 공통 오류를 완전히 배제한 것은 아니다.

내장 Q manifest를 직접 검토했다. W3/g128·비대칭·BF16 모의 GPTQ 설정, model/tokenizer/split revision과 calibration 128개 기록이 일치한다. 대상은 28층×7개 projection=196개이며 대상 이름 집합과 before/after hash 변경 기록을 확인했다. 비대상 weight 불변은 manifest의 보고 내용이다. manifest를 생성 코드의 JSON 형식으로 재직렬화한 SHA256이 기록된 `8ce77f5a...caae85`와 일치한다.

사용자 장비에서 checkpoint 파일 해시 검증을 마쳤다는 `q_checkpoint_hashes_verified=true`를 확인했다. 가중치 파일 자체는 assistant 환경에 없으므로 독립적으로 weight bytes를 재해시하거나 실제 양자화 품질을 검증했다고 쓰지 않는다. 상세 [검토 JSON](cache_diagnostic_gpu1_01_review.json).

## 다음 검사의 목적과 고정 범위

현재 보고서에는 cache/full-prefix 차이만 있고, 그것이 F-Q 양자화 차이나 top-p 후보집합/교차 정책에 비해 어느 정도인지는 없다. 이 빈칸을 한 번의 제한된 development audit로 채운다. 기준값을 높여 통과시키지 않으며 기존 `run_development.py`와 v0.2.1 protocol은 유지한다.

고정 계획: [configs/numerical_audit_v1.json](../../configs/numerical_audit_v1.json). 실제 audit 결과를 보기 전에 입력·모드·관측 위치를 기록했다. 첫 development 문항의 앞 32토큰과 반복 토큰 4개를 이미 본 이력도 명시했다.

| 항목 | 고정값 |
|---|---|
| 표본 | 잠긴 development worklist의 첫 8문항: train/346, 5418, 4224, 419, 7008, 3885, 7212, 127 |
| 입력 | 전체 prompt, truncation 없음 |
| 공통 토큰열 | F의 BF16 automatic SDPA + FF 정책, T=.6, p=.95, seed42, 최대32 생성 토큰, EOS 종료 |
| 정책 관측 위치 | 생성 직전 step 0, 1, 8, 16, 31 및 실제 마지막 생성 직전 위치. EOS로 도달하지 못한 위치는 없음으로 기록 |
| cache 구현 대조 | 모든 decode step에서 별도 DynamicCache·명시적 position/mask를 사용하는 Transformers 직접 참조와 비교. 유한성과 모든 층 cache 길이도 검사 |
| 수치 대조 | 같은 prefix의 BF16 cached/full-prefix 및 BF16 cached/FP32 cached. FP32는 기존 BF16 weight 값을 승격하고 math SDPA·TF32 off |
| 대상 정책 | 동일 prefix에서 FF·QQ·QF·FQ 확률분포와 F/Q top-p 후보집합 |
| 출력 | per-point 원시 TV·4-arm 정책 TV·후보 수/Jaccard·변경 후보 확률 질량·EOS/think-end membership·F-Q 같은 prefix 차이, 코드/계획/자산 hash |

각 모델을 하나씩 GPU에 올린다. F와 Q는 같은 **F가 공급한 토큰열**을 처리하므로 이 결과는 4-arm의 독립 rollout 효과가 아니다. 생성 token IDs는 재현 목적으로 저장하지만 정답 채점·평균 길이·H1 판정은 수행하지 않는다. 최대 8×32=256개의 공급 토큰이며 actual count/EOS 종료를 보고한다. 32,768 cap 검사를 대신하지 않는다.

## 해석과 다음 결정

같은 계산 순서의 cache 참조는 0 logit 차이를 표지로 쓴다. FP32 cached/full-prefix에는 기존 .25/.001을 비교 표지로 함께 기록한다. 이 표지가 충족되어도 BF16 원 관문을 통과시킨 것이 아니며 새로운 자동 합격 기준도 아니다.

정책 민감도는 원래 F-Q 차이와 나란히 보고한다. 작은 분모로 나눈 비율은 만들지 않는다. max/mean/p95는 관측 위치의 기술통계이며 독립 표본 추론·효과 크기 CI가 아니다. F 경로 초반에 한정되므로 낮은 민감도가 나와도 모든 arm·장문에 대한 안정성을 보증하지 않는다.

결과 검토에서는 (1) 같은 순서 cache 참조의 불일치가 있으면 구현 원인 조사, (2) FP32 cache 대조가 크게 어긋나면 causal/position/backend 조사, (3) 정밀도·재계산 차이가 F-Q 정책 차이에 비해 크거나 후보집합을 크게 바꾸면 수치 안정화/설계 검토를 먼저 한다. 민감도가 작다는 판정에 필요한 경계는 임의로 이번 실패 숫자에 맞추지 않는다. 현재 audit에는 자동 GO나 H1 합격 임계값이 없다.

이 자료로 검증 개편의 타당성을 판단하고, 채택한다면 변경 이유·본 데이터·검사항목과 판정법을 명시한 새 개발 버전으로 적용한다. 최초 실패 기록을 보존한다. 아직 구현하지 않은 합격 기준을 미리 충족했다고 쓰지 않는다.

## 실행

저장소 root, 기존 quantsplit 환경에서:

```bash
conda activate quantsplit
git pull --ff-only
python scripts/run_numerical_audit.py --device cuda:1 --q-checkpoint checkpoints/gptq-w3-g128 --output runs/numerical-audit-gpu1-01.json
```

기존 checkpoint를 검증 후 재사용한다. calibration/GPTQ·설치·드라이버 변경은 하지 않는다. 이미 출력 파일이 있으면 새 번호를 쓴다. 결과 JSON을 첨부한다. 오류면 해당 JSON과 터미널 오류를 함께 전달한다. 진행 중 완료된 trace와 모델별 검사 정보를 저장하며, 강제 종료 시 마지막 저장분만 남을 수 있다. BF16 기준 실패 때문에 이 진단을 건너뛰지는 않지만 본 생성 runner의 기존 관문을 우회하지도 않는다.

이번 코드는 CPU 환경의 **30 tests(2.999초) 통과**로 검증했다. 작은 실제 Qwen2의 BF16/FP32 replay, EOS 조기 종료, 선택한 관측 위치 밖의 cache 오류 검출, 후보집합 교환 시 4-arm TV의 수작업 기대값을 확인했다. GPU 없음 CLI는 오류 JSON을 남기고 중단했다. 이 신규 audit의 실제 1.5B/4090 실행은 아직 없다.

03 파일럿·H1·확증은 미실행, 교수 실제 PASS는 미확인이다. 수치 검증 방법을 정리한 뒤 짧은 4-arm·full-cap·긴 context·VRAM/처리량 관문이 남는다.
