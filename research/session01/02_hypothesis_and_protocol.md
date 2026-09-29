# QuantSplit — 제출용 연구 설계 v0.2

2026-09-29 KST. **제출 문안 확정 / 교수 승인 전 / 모델 실험 없음 / 확증 사전등록 미완료.** v0.1 초안을 이 문서와 protocols/quant_sampling_v0.2.json이 대체한다. H1과 판정 규칙을 지금 명시하되 실행 revision·split manifest·실측 예산까지 갖춘 확증 동결은 03단계에서 한다.

## 1. 질문과 대상

GPTQ-W3/g128로 양자화한 추론 모델의 출력 길이 증가 중, BF16 top-p 후보집합을 공급하는 정책 개입으로 복구 가능한 부분이 실질적으로 큰가? 정답률 손실이 허용 범위를 넘지 않는가?

| 항목 | 제출 설계 |
|---|---|
| F | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, BF16 |
| Q | 같은 원본의 GPTQ, weight-only W3/g128, 비대칭 group quantization |
| 실행 | 양자화·역양자화된 가중치를 BF16 연산으로 실행. activation/KV cache BF16. 실제 INT3 커널 속도 실험 아님 |
| 주 데이터 | openai/gsm8k, main, test 1,319문제 전체 |
| 디코딩 | T=0.6, top-p=0.95, top-k 없음, min-p=0, repetition penalty=1, 다른 penalty 없음 |
| 최대 소비량 | 신규 생성 32,768토큰. 입력을 포함한 context 허용 한도 검증 필수 |
| 주 조건 반복 | seed 42, 43. 기본안은 2개이며 결과를 보고 반복 수를 늘리지 않음 |
| 주 비교 | QQ 대비 QF. FF는 초과 길이 기준, FQ는 반대 방향 교환 및 상호작용 |
| 해석 범위 | 해당 모델·양자화·데이터·고정 토큰 예산 아래의 정책 개입 효과 |

새 task fine-tuning은 하지 않는다. GPTQ는 calibration activation X에 대해 층 출력 재구성 오차를 줄이는 기존 양자화 절차를 사용한다. downstream test를 목적함수나 양자화 설정 선택에 사용하지 않는다. damp/act-order/대상 layer 등 정확한 구현값은 02에서 명시해 03에서 고정한다.

## 2. 데이터와 prompt

- GSM8K train에서 calibration 128, development 128, pilot 128문제를 서로 겹치지 않게 고른다. 정규화 질문의 SHA-256으로 정렬하고 동률은 원본 row index로 처리한다. 앞 128/다음 128/다음 128 순서로 배정한다. 모델 평가 전 manifest를 저장한다.
- calibration은 해당 문제에서 F가 생성한 텍스트를 사용한다. 질문+생성 토큰으로 최대 2,048토큰의 calibration sequence를 구성하고 정답 label은 모델 입력에 넣지 않는다. 생성·양자화는 PASS 이후다. NuminaMath를 사용하는 근접 논문의 정확 재현과는 다르다.
- 공식 test와 train 후보의 정규화 exact/near duplicate는 모델 결과를 보기 전에 감사한다. 사전학습 오염 제거를 주장하지 않는다. 공식 test는 유지하며 중복 train 후보를 교체하고 이유·ID를 기록한다. near-duplicate 알고리즘과 임계값은 개발 단계에서 고정한다.
- user 메시지에 문제와 다음 지시를 넣고 system prompt는 쓰지 않는다: “Please reason step by step, and put your final answer within `\boxed{}`.” 모델 카드의 권고에 따라 assistant reasoning 시작 prefix를 적용하되 실제 chat template가 이미 넣었으면 중복 삽입하지 않는다. 렌더링한 prompt token IDs/hash를 고정한다.
- test 모델 출력은 03의 동결 전 생성·열람하지 않는다. 기존 문헌의 test 집계 수치를 읽은 사실은 데이터 열람 이력에 남긴다. 본 세션에서는 데이터 카드의 예시 질문은 읽었지만 자체 모델 출력은 보지 않았다.
- parser는 마지막 reasoning 종료 이후 최종 답변의 마지막 균형 잡힌 boxed 수치를 읽고, 정수·소수·분수를 결정적으로 정규화한다. 중간 reasoning의 숫자를 정답으로 취급하지 않는다. reasoning 종료 없음·최종 답 없음·추출 실패는 오답. cap에 도달해도 종료된 reasoning 뒤의 완결 boxed 답은 고정 예산 정답률로 채점하고 cap flag를 별도 보고한다. 단위/서식 세부 처리는 개발 표본으로 검증·동결한다.

## 3. 네 생성 정책

현재 arm의 prefix h에서 p_F와 p_Q는 각각 T=0.6을 적용한 full-vocabulary 확률이다. M_F와 M_Q는 해당 확률을 내림차순 정렬해 누적질량 0.95에 처음 도달하는 최소 집합이다. 동률은 token ID 오름차순이고 경계 토큰을 포함한다.

q_ab(v|h) = p_a(v|h) · 1[v∈M_b(h)] / Σ(u∈M_b(h)) p_a(u|h).

| arm | 확률 공급자 a | 후보집합 공급자 b |
|---|---|---|
| FF | F | F |
| QQ | Q | Q |
| QF | Q | F |
| FQ | F | Q |

각 arm은 자체 토큰열을 만든다. 한 arm 안에서만 F와 Q가 동일한 현재 prefix를 보고, 별도 KV cache에 그 arm에서 뽑은 토큰을 각각 추가한다. 다른 arm의 prefix/mask/cache를 재사용하지 않는다. 동일 prefix의 F/Q hidden state가 같아야 한다는 뜻은 아니다.

logits를 FP32로 올린 뒤 softmax·누적합·재정규화한다. hybrid에 top-p를 두 번 적용하지 않는다. QF에서 유지되는 것은 Q의 logits와 공통 허용 토큰 간 상대확률이며, mask 변경 후 절대확률은 재정규화된다. nonfinite 값·빈 mask·0 normalization mass는 실행 오류로 처리한다.

난수는 question ID·seed·생성 step에 키를 둔 공통 uniform으로 계획하고, token ID 순 CDF에서 뽑는다. 각 정책의 주변분포를 보존하기 위한 coupling이며 동일한 토큰 경로를 강제하지 않는다. batch 순서와 종료 시점이 다른 문항의 난수를 바꾸면 안 된다. 정확한 PRNG/수치 구현은 02에서 고정한다.

EOS는 tokenizer/generation config를 확인해 고정한다. reasoning 종료 기호와 실제 generation EOS를 구분한다. reasoning 종료만으로 생성을 멈추지 않는다. 추가 min-length나 stop-string을 arm마다 다르게 적용하지 않는다.

## 4. H1 및 추정량

L_ab: 정답 여부와 관계없이 모든 문제의 추론+최종답변 소비 생성 토큰 평균. EOS 토큰은 생성했으면 포함하고 prompt/padding은 제외한다. 먼저 문제 내 두 seed를 평균하고 문제에 같은 가중치를 준다. A_ab도 문제 내 seed별 정답 indicator 평균의 문제 평균이다.

Delta = L_QQ − L_FF; D = L_QQ − L_QF; R = D / Delta; A = A_QF − A_QQ.

**H1:** 주 조건에서 의미 있게 양의 초과 토큰이 확인될 때 **R > 0.50 AND A > −0.03**. 즉 QF가 초과 소비 토큰의 절반을 넘게 줄이고 QQ 대비 정답률 손실이 3%p 미만이다. 설명에서는 ‘50% 이상·3%p 이내’를 사용하지만 검정은 엄격한 경계 통과를 요구한다.

50%는 주요 복구라고 부를 크기, 20%는 작은 복구 상한, 3%p는 정답률 손실 허용치로 정한 **설계상 기준**이다. 문헌이 예측한 값이 아니다. 3%p는 ‘정확도가 동일하다’는 뜻도 아니며 QF가 FF 정확도를 회복한다는 뜻도 아니다. 원 현상 관문의 20% 길이 증가는 복구 상한 20%와 다른 수치다.

R은 자연적 매개효과 비율이 아니다. 0–1로 clip하지 않으며 음수·1 초과도 그대로 보고한다.

## 5. 추론과 판정 규칙

문제 단위 paired bootstrap 10,000회, analysis seed=20260929. 재표집할 때 해당 문제의 모든 arm·seed를 함께 유지한다. percentile 방식과 linear quantile interpolation을 사용한다. 주 판단은 동결한 전체 시행 완료 후 한 번 한다. 양측 95% 구간과 단측 95% 경계(q05/q95)를 구분해 보고한다. 고정된 test의 관측 평균과 문제 재표집 불확실성을 구분하며, 이 구간을 모든 문제·모든 seed에 대한 보장으로 해석하지 않는다.

비율의 불안정성을 피하려고 C_r = D − r·Delta를 사용한다. Delta>0일 때 R>r와 C_r>0은 동치다. 비율 CI와 다른 결과가 날 경우 미리 정한 **C_r의 판정**을 따른다.

비율 판정 가능 조건 E: (i) Delta의 양측 95% 하한(q025)>0, (ii) bootstrap Delta≤0 비율<2.5%, (iii) 평균 L_QQ/L_FF≥1.20. (i)/(ii)는 수치 경계·구간 일관성을 확인하는 중복 안전 조건이다. cap률은 별도로 표시하며 ‘자연 종료까지 필요한 길이’는 추정하지 않는다.

| 판정 | 필요한 증거 | 허용되는 결론 |
|---|---|---|
| H1 지지 | E 충족, C_0.50의 단측 95% 하한>0, A의 단측 95% 하한>−0.03 | 해당 정책에서 큰 복구와 정확도 비열등성 |
| 50% 복구 기준 반증 | E 충족, C_0.50의 단측 95% 상한<0 | 50% 복구 가설을 배제. 효과 없음은 아님 |
| 작은 복구 상한 | E 충족, C_0.20의 단측 95% 상한<0 | 이 개입의 20% 복구를 배제. 음의 효과일 수도 있음 |
| 정확도 기준 반증 | A의 단측 95% 상한<−0.03 | 3%p 이내 손실 조건에 맞지 않음. 길이 감소가 있어도 H1 미충족 |
| 중간 효과/부분 증거 | 예: R이 양수이나 50%에는 미달, 또는 정확도 구간이 기준을 가로지름 | 추정치와 CI를 보고하고 지지/반증이 가능한 구성요소만 말함 |
| 원 현상/분모 부족 | E 미충족 | R의 비율 추론을 하지 않고 Delta·D 및 CI 보고. H1 반증으로 바꾸지 않음 |
| 판정 불가 | 구간이 기준을 가로지르거나 구현/계획 완료 실패 | 유의하지 않음을 무효과·동등성으로 부르지 않음 |

H1은 두 구성요소 모두의 지지를 요구하는 intersection-union 판정으로 각각 단측 0.05를 사용한다. 반대로 두 실패 기준 중 하나로 통합 H1 반증을 선언할 때는 Bonferroni를 적용한다. C_0.50의 단측 97.5% 상한<0 또는 A의 단측 97.5% 상한<−0.03이면 통합 반증으로 판정한다. 표의 단측 95% 기준은 개별 endpoint의 결과 보고용이다. 작은 효과 상한은 사전 지정된 길이 endpoint의 별도 결론이며 전체 H1의 성공/실패 확률과 혼용하지 않는다.

선택적 동등성 분석은 D+0.20Delta의 단측 하한>0 AND D−0.20Delta의 단측 상한<0을 요구한다(E 충족). 이는 ±20% 이내라는 질문이며, upper R<20%만으로 동등성이라 하지 않는다. 확증 보조 family를 만들면 03에서 명시하고 Holm 보정을 적용한다. 지정되지 않은 보조 분석은 탐색으로 표시한다.

## 6. 전체 분해와 필수 보고

mask 효과(F 확률): L_FQ−L_FF; mask 효과(Q 확률): L_QQ−L_QF.

probability 효과(F mask): L_QF−L_FF; probability 효과(Q mask): L_QQ−L_FQ.

Interaction = L_QQ−L_QF−L_FQ+L_FF. 대칭적인 두 경로 평균은 기술 통계로만 사용할 수 있고, H1의 R을 대체하지 않는다. 네 arm의 L·A·cap률·완결답 비율, 모든 차이와 구간, seed별 결과, QF−FF 정확도 차이를 함께 공개한다.

Jaccard·mask 크기·Q/F 아래의 유지/제거 질량·EOS/reasoning-end 포함 여부를 online 기록한다. QQ와 QF의 같은 h에서, a=p_Q(M_Q), b=p_Q(M_F), c=p_Q(M_Q∩M_F)이면 1-step TV는 1−c/max(a,b)다. 이는 공통 부분의 min 확률 합으로 바로 얻는 항등식이며 새로운 정리로 주장하지 않는다. rollout 전체 길이의 상한은 아니다.

## 7. 파일럿 관문과 확증 동결

PASS 이후 train pilot 128문제에만 적용한다. FF 정확도≥50%, QQ≥25%, L_QQ/L_FF≥1.20, Delta의 단측 95% 하한>0, FF/QQ 각각 cap률<20%를 진행 관문으로 사용한다. 이는 실용적인 사전 기준이며 모델 품질 인증이 아니다. QF/FQ cap률도 공개하고 급증하면 길이 개선 해석을 보류한다.

추가로 구현 대조군 통과, 실제 처리량·VRAM 측정, 보수적 정밀도·시간 관문이 필요하다. QF가 좋아 보인다는 이유만으로 GO하지 않는다. GPTQ에서 원 현상이 없으면 AWQ-W3 파일럿 1회만 별도 기록할 수 있으나, 주 H1의 대상 변경은 교수 검토 및 새 버전이 필요하다. GPTQ의 실패를 지우지 않는다.

seed 1개 축소가 필요하면 test 출력 생성 전에 새 버전·이유·예상 정밀도를 기록한다. 본 v0.2의 두 seed 계획을 조용히 바꾸지 않는다. 정확한 모델/토크나이저/코드 revision, quantization 설정, split IDs/hash, parser, prompt, 난수, stop 정책, GPU-hours, 보조 family를 03에서 동결한다. 이 값이 비어 있는 현재 JSON을 실행 가능한 확증 프로토콜로 취급하지 않는다.

현재 승인 상태는 APPROVAL_PENDING이다. 모델 로딩·calibration 생성·파일럿·확증은 미실행이다. 문서·수식·정밀도 시나리오 검토만 했다.

공식 참조 문서(2026-09-29 확인): [모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [GSM8K 데이터 카드](https://huggingface.co/datasets/openai/gsm8k). 실행 revision은 아직 미동결이다.
