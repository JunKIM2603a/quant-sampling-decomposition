# 근거 대조표

확인일: 2026-09-29 KST. 논문이 주장하는 결과와 독립 재현은 다르다. 이 작업에서는 모델 실험을 재현하지 않았다. 핵심 자료의 방법·결과·부록을 읽고, 나머지는 표시한 수준까지 확인했다. 검색에서 안 나온 것이 연구 부재를 증명하지는 않는다.

| ID | 1차 자료 / 확인 수준 | 기존 연구가 중요하게 본 문제 | 사용한 방법·알려준 것 | 이번 연구와의 경계 |
|---|---|---|---|---|
| S01 | [Lotfi et al., Quantized Reasoning Models Think They Need to Think Longer, but They Do Not](https://arxiv.org/html/2606.00206v1), 2026 preprint. 본문 §3–7, 부록 A/C/D 확인 | 저비트 추론의 불필요한 길이 증가 | 공통 prefix의 KL 분석, 분기 표지 단어 logit 억제; 무작위·high/low-KL 단어 비교 | 현상과 단어 억제는 기존 기여. 후보 집합/확률의 2×2 교환은 확인되지 않음 |
| S02 | [Lee et al., ReSET](https://arxiv.org/pdf/2606.13233), 2026 preprint. §3–6, Table 5, 설정 확인 | NVFP4 추론의 정확도와 낮은 배치에서의 지연 | step-aware 온도 조정과 Blackwell용 커널; top-p/min-p 조정도 비교 | 단순 샘플러 튜닝은 새롭지 않음. 표 5는 제안 H1을 낙관하기 어렵게 하는 근거 |
| S03 | [Lian et al., Quantization Inflates Reasoning](https://arxiv.org/pdf/2606.25519), 2026 preprint. 설정, §7, 부록 모델표 확인 | 정확도 외에 늘어난 토큰이 만드는 비용 | T=0 생성의 길이·행동 분석, 반복 억제와 calibration/QAT 검토 | greedy에서도 증가하므로 top-p를 보편 원인으로 주장할 수 없음 |
| S04 | [Liu et al., Quantization Hurts Reasoning?](https://arxiv.org/html/2504.04823v2), COLM 2025. 설정·표·양자화 부록 확인 | 추론 모델에서 양자화 대상·비트 수의 영향 | 여러 양자화 방법·모델·과제, T=0.6/top-p=0.95 및 반복 평가 | 저비트 비교 자체는 기존 기여. 재현 기반과 기준선으로 사용 |
| S05 | [Ding et al., Min-k Sampling](https://arxiv.org/abs/2604.11012), 2026. arXiv 초록 및 ACL 2026 채택 표기 확인 | 온도에 민감한 확률 기반 truncation | logit의 국소 모양으로 후보 경계 설정 | 후보 선택/온도 분리가 전혀 새로운 발상이라는 주장은 불가. 양자화 기여 분해와는 질문이 다름 |
| S06 | [eapache/quant-experiments](https://github.com/eapache/quant-experiments), 공개 연구 저장소. README 확인 | 양자화 분포를 원본에 가깝게 보정할 수 있는가 | teacher-forced 분포 비교, 온도/top-p·bias·저랭크 보정의 held-out 평가 | **비심사 공개 실험**. 샘플러 보정은 선행 시도 있음. 자유 생성의 교차 교환·길이 결과는 검토 README에서 확인되지 않음 |
| S07 | [Jiang et al., Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010), 2025. 초록 확인 | 학습된 register 없이 ViT artifact 완화 | 고노름 활성값을 추가 토큰으로 이동 | test-time register 자체를 새 기여로 주장하면 안 됨 |
| S08 | [Parodi et al., Zero-Ablation Overstates Register Content Dependence](https://arxiv.org/html/2604.14433v1), CVPR 2026 HOW Workshop 표기. 본문·실험 목록 확인 | zero ablation의 해석 오류 | 평균·노이즈·이미지 간 register 교체 대조 | register 내용/존재 질문이 이미 연구됨. 적용 대상 변경만으로는 강한 신규성 불충분 |
| S09 | [Khodabandehlou & Krishnamachari, Compression-Aware Abstention](https://arxiv.org/pdf/2608.29934), 2026 preprint. 초록/문서 확인 | KV 압축이 근거를 제거할 때 환각 | survival mask 기반 답변/보류 지도 및 압축 캐시 평가 | 이 포괄적 H1은 충돌 위험이 매우 큼 |
| S10 | [Ross et al., How retriever redundancy and diversity impact RAG effectiveness](https://arxiv.org/abs/2608.13956), 2026 preprint. 초록 확인 | 문서 집합의 중복과 다양성 | fictional QA에서 duplicate/paraphrase/diverse 통제 | 중복 vs 다양성 비교만으로 새로운 H1이라 하기 어려움 |
| S11 | [Kim et al., Complexity- and Statistics-Guided Anomaly Detection in Time Series Foundation Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d06768735ce7d11ade5baef7099b4e05-Abstract-Conference.html), ICLR 2026. 공식 초록 확인 | 과도한 일반화와 정규화에 따른 이상탐지 실패 | 복잡도 기반 결합 및 평균·분산 정보 복구 | 정규화가 통계적 이상을 없앤다는 H1에 직접 선행 |
| S12 | [Adaptive Stopping for Multi-Turn LLM Reasoning](https://arxiv.org/abs/2604.01413), 2026 preprint. 초록 확인 | 여러 턴에서 보장과 비용을 함께 관리 | MiCP / 다중 턴 conformal stopping | 단순 위험 통제 조기종료는 좁혀야 할 주제 |
| S13 | [Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), 2020. 초록 확인 | 외부 지식과 생성 결합 | retrieval과 generation을 결합한 학습 | “검색기와 생성기를 함께 학습”은 새 H1이 아님 |

## 구현과 데이터 확인

| ID | 자료 | 확인한 범위 | 아직 확인하지 않은 것 |
|---|---|---|---|
| A01 | [Quantized-Reasoning-Models 공식 코드](https://github.com/ruikangliu/Quantized-Reasoning-Models) | GPTQ/AWQ 모의 양자화 및 INT4 실제 양자화 경로, 평가 설명 존재 | 현재 4090 환경에서의 설치·TP=1 실행·정확도 재현 |
| A02 | [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 공식 공개 모델 페이지 | 실제 다운로드·revision 고정·dual-cache 메모리 |
| A03 | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 공개 데이터 페이지 | 문제별 해시·중복 점검·로컬 parser |
| A04 | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 공개 데이터 페이지, train/test 구분 | 실제 split 해시·정규화·오염 감사 |

## 증거와 추론 구분

1. **확인된 사실**: S01–S04는 양자화·추론·디코딩을 다룬다. S02에는 top-p sweep이 있고 S03의 기본 설정은 greedy다.
2. **검토 의견**: 이 때문에 단순한 현상 재현과 샘플러 튜닝은 주가설로 약하다.
3. **새로 제안하는 질문**: 확률과 후보 집합을 교환하여 제한된 생성 정책에서 개입 효과를 측정한다.
4. **미검증**: H1 효과 크기, 재현 성공, GPU 처리량, 최종 게재 가치.

동일 표현의 논문을 못 찾았다는 판단에만 의존하지 않는다. S01–S06의 실제 비교 조건이 무엇인지 대조했고, 검색 결과가 관련 없는 일반 페이지로 치환된 질의는 신규성의 근거에서 제외했다. 미공개 논문·색인 지연·모든 분야의 유사 기법까지 배제하지는 못한다.
