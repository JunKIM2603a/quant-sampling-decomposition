# QuantSplit 02-2 — 실제 GPU 실행 준비와 현재 한계

2026-09-29 KST. **실제 장비 검증 대기. 02 미완료, 03 인계 비활성.**

## 확인한 사실

- 최신 main `2963e90e5408c120fa8cd083ed816cfa8a269551`의 요청 문서 7개를 읽고 원격 58개 파일의 blob hash를 대조했다. 이후 같은 커밋을 실제 Git checkout으로 받았다.
- 현재 assistant 실행 환경에는 NVIDIA 장치와 nvidia-smi가 없고, 사용자 GPU에 연결할 도구/접속 정보가 없다. 이번 요청에는 새 장비 실행 로그가 첨부되지 않았다. 사용자 PC에서 명령을 실행했다고 기록하지 않는다.
- 사용자의 승인 가정 진행 권한을 적용했다. 교수님 실제 PASS는 미확인이다. 새 교수 승인 요청이나 이메일 발송은 하지 않았다.

## 수정과 이유

1. **고정값 덮어쓰기 수정.** 기존 `prepare_assets.py`는 재생성한 split/lock을 먼저 저장했다. 이후 runner는 새 lock과 새 worklist를 비교하므로 이전 고정값과의 차이를 놓칠 수 있었다. 이제 split·tokenizer·prompt·worklist 등을 기존 기록과 비교한 뒤, 일치하는 누락 파일만 생성한다. 패키지 차이는 runtime 보고서에 남기고 기존 lock은 보존한다.
2. **긴 context 관문 구현.** `--max-new-tokens 32768`을 지정해도 조기 EOS이면 실제 긴 cache 검사가 아니다. 별도의 `--context-stress-only`를 추가했다. 가장 긴 development prompt를 반복하여 prompt+32,768 길이의 F/Q cache를 동시에 유지하고 마지막 두 위치를 한 토큰씩 처리한다. 마지막 위치에서 full-prefix와 cache를 비교한다. 기존 오차 기준을 변경하지 않았다.
3. **실제 실행 증거 수집.** 사전 확인 스크립트는 GPU·driver·CUDA·패키지·Git 상태를 기록한다. 통합 shell은 실패 즉시 중단하고 작은 JSON/로그만 archive에 넣는다. 개발 manifest에 runner hash, 시각, 각 arm의 실제 생성/cache 길이와 종료 원인을 추가했다.

합성 context 검사는 자유 생성의 품질·처리량·H1 증거가 아니다. BF16 역양자화 checkpoint는 실제 INT3 kernel 가속 증거가 아니다. 이번에 H1·seed·분할·프로토콜·효과 기준을 바꾸지 않았다. 현행 개발 프로토콜 v0.2.1은 확증 동결이 아니다.

## 수행한 검사

| 검사 | 실제 결과 | 범위 |
|---|---|---|
| 기존 19개+추가 5개 tests | 24개 통과, 실패/skip 0 | torch 2.8.0+cpu, Transformers 4.57.1 |
| 새 context 함수 | 작은 무작위 Qwen2 쌍, BF16 CPU, 길이 67에서 통과 | 실제 1.5B/32,768 검증 아님 |
| 자산 재생성 | 고정 revision 캐시로 성공, 기존 split/lock/protocol byte 동일 | 자체 test 모델 출력 0 |
| 없는 GPU/패키지의 통합 shell | exit code 2, calibration 전에 중단, 보고서 묶음 생성 | 의도한 실패 경로 |
| 구문·diff 검사 | Python compile 및 bash syntax, diff whitespace 통과 | GPU 실행 보증 아님 |

기계 판독 기록과 해당 코드 hash: [session02_2_cpu_checks.json](session02_2_cpu_checks.json). 원래 작은 GPTQ 검증은 이전 기록을 보존했다. 이를 이번 실제 GPTQ 실행으로 다시 세지 않는다.

## 남은 관문과 다음 입력

사용자 RTX 4090 PC에서 [runbook](02_runbook.md)의 환경을 준비하고 `run_target_gate.sh`를 실행한다. 먼저 `check_target_environment.py`만 실행하여 환경 문제를 진단해도 된다. 같은 출력 이름을 재사용하지 않는다.

검토할 증거는 preflight, 실제 Q manifest, smoke 대조, full-cap 자연 생성 manifest, 합성 context 보고서와 오류 로그다. 모두 통과한 뒤 실제 GPU별 하루 가용 시간을 확인하고 `02_to_03.md`를 활성화한다. 대조 실패·OOM·넓은 실험 불확실성을 H1 반증으로 기록하지 않는다.
