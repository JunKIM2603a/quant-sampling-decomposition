# QuantSplit 02 — 구현과 현재 판정

2026-09-29 KST 구현 기록. **후속 9/30: 실제 1.5B/GPTQ 실행에서 cache 대조 실패. 02 전체 미완료.** 아래 CPU 결과는 최초 검증 이력이며, 최신 실제 GPU 수치·진단 판단은 [05_cache_diagnostic_review.md](05_cache_diagnostic_review.md)를 따른다. 기본 BF16 실패를 재현했고 FP32에서 오차가 크게 줄었으나 기존 BF16 관문은 실패 상태다. H1의 경험적 결과는 아직 없다.

## 진행 권한

사용자 지시: “승인 했다는 전제로 우선 진행해줘.” 이에 따라 기존 승인 대기를 해제하고 02의 구현·검증을 진행했다. 교수님의 실제 PASS를 확인했다는 뜻은 아니다. 실제 승인일·수정 요청은 미확인으로 유지하며 이 이유로 다시 작업 허가를 요청하지 않는다. 확증 전 03 동결 원칙은 유지한다.

현행 제출 가설·판정은 [v0.2 설계](../session01/02_hypothesis_and_protocol.md), 현행 실행 준비값은 [v0.2.1 개발 프로토콜](../../protocols/quant_sampling_v0.2.1-development.json)과 [execution lock](../../configs/execution_lock.json)이다. H1의 효과 기준·정답률 여유·주 seed 계획은 바꾸지 않았다.

## 구현

| 기능 | 구현과 결정 |
|---|---|
| FF/QQ/QF/FQ | 각 arm의 자체 prefix로 F/Q를 둘 다 계산. 첫 글자는 확률, 둘째는 mask 공급자 |
| cache | arm마다 F/Q의 DynamicCache를 새로 생성. cache 길이·처리한 prefix를 검증하고 신규 토큰만 전달 |
| top-p | FP32 softmax/누적합/재정규화. 확률 내림차순·동률 token ID 오름차순, 경계 포함. p=1은 model vocab 전체 |
| hybrid | mask 교환 후 재정규화 한 번. 추가 top-p 없음 |
| 난수 | `quantsplit-sha256-uniform-v1`: JSON `[version, question_id, seed, step]`의 SHA256 앞 53비트/2^53. arm·batch 순서와 무관 |
| CDF | token ID 순, FP32 누적합의 총합 반올림 보정. 53비트 uniform 비교만 FP64. 0 확률 토큰을 선택하지 않음 |
| 길이·종료 | 생성한 EOS를 길이에 포함. reasoning-end는 정지가 아님. cap 종료와 EOS 종료를 분리 |
| 진단 | arm별 실제 경로에서 mask 크기·Jaccard·F/Q 유지 질량·1-step TV·종료 토큰 membership의 step 평균. 전체 vocab 로그를 저장하지 않음 |
| parser | 마지막 `</think>` 뒤 마지막 boxed 수치. 정수·소수·분수, 올바른 천 단위 쉼표 허용. 단위·백분율·식은 실패. 뒤에 미완성 box가 있으면 실패 |
| 실패 | NaN/빈 support/0 mass/cache 오류를 실행 오류로 처리. 0 길이 오답으로 끼워 넣지 않음 |
| 개발 runner | manifest와 일치하는 development 128문항만 허용. 이 CLI에는 test 실행 경로가 없음 |

SHA256 uniform은 재현 가능한 키 기반 의사난수 규칙이다. 동일 토큰열을 강제하지 않는다. FP32 반올림 오차와 경계 동률의 정의는 위와 같이 명시하며, 다른 sampler의 정렬·난수 구현과 token trace까지 자동으로 같다고 가정하지 않는다.

## 고정한 자산과 분할

- 모델·tokenizer: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` @ `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- 데이터: `openai/gsm8k` main @ `740312add88f781978c0658806c59bc2815b9866`; train 7,473, test 1,319행을 읽어 질문 중복을 감사했다. 자체 test 모델 출력은 생성·열람하지 않았다.
- 정규화: NFKC → casefold → 공백 합치기. 질문 SHA256, 원본 행 순으로 정렬하여 calibration/development/pilot 각 128개를 배정했다. split manifest hash: `f650f57279971fee307f9a85ced0033d75347ab4c3af86d49c560c508a1c6ad8`.
- 중복: 정규화 exact 및 단어/문장부호 5-shingle Jaccard ≥0.8을 모든 train 후보에 적용. 이번 revision에서 제외 0건. 이 특정 휴리스틱에서의 결과이며 의미상 중복·사전학습 오염이 없다는 증거는 아니다.
- EOS `[151643]`, reasoning-end `[151649]`, reasoning-start `151648`. 공식 template에 `<think>`가 이미 있으므로 추가로 붙이지 않았다. probe의 BOS는 tokenizer template의 `151646`을 그대로 사용한다.
- tokenizer의 명시 길이 16,384와 모델 config 한도 131,072가 다르다. 자동 truncate를 하지 않으며 prompt+cap가 모델 한도를 넘으면 실패한다. **실제 cap 32,768과 메모리 검증은 아직 필요하다.**
- 모델 vocab 151,936과 tokenizer 길이 151,665가 다르다. 기존 full-model softmax와 동일한 전체 모델 vocab을 쓰며, 임의로 뒷부분 ID를 제거하지 않는다. 해당 토큰의 실제 발생 여부는 개발에서 확인한다.

원본 메타데이터와 실행 패키지 버전은 [upstream_metadata.json](upstream_metadata.json), [environment.json](environment.json)에 보존했다. worklist 전체 파일 hash도 lock에 있다. 정답 label이 모델 입력에 들어가지 않으며 calibration worklist에는 answer 필드 자체가 없다.

## GPTQ 준비

[코드 출처와 수정](../../third_party/QUANTIZATION_NOTICE.md), [상세 설정](../../configs/quantization.json).

COLM 선행연구 코드의 GPTQ/WeightQuantizer를 MIT 고지와 함께 사용한다. W3/g128, 비대칭, act-order/static groups, weight clipping(norm 2.4, grid 100, maxshrink .8), block128, damping .01을 개발 설정으로 명시했다. q/k/v → o → up/gate → down 순서로 실제 양자화된 앞 연산의 activation을 재수집한다. embedding·lm_head·norm·bias는 양자화하지 않는다. 작은 모델에서 비대상 weight가 그대로임을 확인했다.

128문항의 BF16 참조 생성으로 prompt+trace 최대 2,048토큰을 만든다(T=.6, p=.95, seed42). 가변 길이를 padding 없이 사용하며 layer kwargs/position embeddings를 개별 sequence별로 보존한다. GPTQ 코어의 damping 자동 증가를 제거해 Cholesky 실패는 명시적으로 중단한다. 기존 upstream 전체 pipeline이나 다른 calibration 집합의 정확 재현이라고 주장하지 않는다.

완성 checkpoint는 양자화·역양자화된 BF16 safetensors와 base revision, calibration token hash, quantization config/source hash, 파일 hash manifest를 가진다. GPTQ integer kernel 가속을 측정하는 구현이 아니다. 최초 CPU 검증 시 실제 checkpoint는 미생성이었다. 9/30 후속 사용자 로그에서 실제 calibration 128/128 및 checkpoint 저장을 확인했다. Q manifest 원본의 직접 검토는 대기다.

## 확인한 결과와 남은 관문

| 검사 | 이번 결과 | 증거 |
|---|---|---|
| 수치·경계·독립 경로·파서·분할 | 19개 통과 | [core_test_report.json](core_test_report.json) |
| 실제 Transformers 작은 Qwen2 | F=Q, p=1, 순서, cache/full-prefix, 독립 baseline 분포 통과 | [tiny_model_verification.json](tiny_model_verification.json) |
| 작은 모델 GPTQ | 대각 Hessian에서 RTN 일치, 가변 길이 처리, 14개 대상 weight 변경·나머지 보존, BF16 cache/4-arm 통과 | 같은 검증 JSON |
| 실제 DeepSeek 1.5B BF16/GPTQ (후속 9/30) | identity·p=1·baseline 통과, cache 대조 실패 | [실제 실패 수치](target_gpu_run01_validation.json) |
| RTX 4090의 VRAM/처리량·32,768 cap | 미측정 | 실제 장비 필요 |
| H1/파일럿/확증 | 미실행 | 03 이후 별도 |

작은 무작위 모델의 logits 최대 cache 오차는 FP32 약 `5.96e-8`, BF16은 검사한 짧은 prefix에서 0이었다. 이는 작은 검증 fixture의 관측값이며 학습된 1.5B의 보증이 아니다. 실제 모델 개발 검사는 max logit 오차≤.25, T=.6 확률 TV≤.001, 독립 baseline 확률 최대 오차≤2e-6을 사전에 코드에 정했다. 실패하면 원인과 수치를 남기고 검토한다. 결과를 본 뒤 기준을 조용히 완화하지 않는다. 실제 runner의 짧은 prefix 통과만으로 전체 cap 검증을 대신하지 않는다.

현재 환경은 CPU이며 사용자 RTX 4090과 연결되어 있지 않다. CPU PyTorch를 설치해 검증한 사실과 사용자 장비 실행을 구분한다. 모델 logits를 CPU NumPy sampler로 옮기는 진단용 구현이므로 처리량은 실제 runner로 측정해야 한다. 두 GPU는 독립 worker로 쓸 수 있으나 48GB 단일 GPU로 간주하지 않는다.

실행 순서는 [02_runbook.md](02_runbook.md), 현재 인계는 [02_target_gpu_pending.md](../handoffs/02_target_gpu_pending.md)를 따른다.

공식 API 근거(2026-09-29 확인): [Transformers 4.57.1 cache](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation), [Qwen2](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2). exact package와 model revision을 위 기록에 고정해 최신 문서 변화와 분리한다.
