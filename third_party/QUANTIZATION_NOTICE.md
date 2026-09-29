# 양자화 코드 출처

`src/quantsplit/gptq_core.py`는 ruikangliu/Quantized-Reasoning-Models의 MIT 코드에서 GPTQ, WeightQuantizer와 필요한 수치 함수를 추출·수정했다.

- 원본 commit: `bf947e29f52e3f666e3263efac149dae0ac18d00`
- [GPTQ 원본](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/methods/utils/gptq_utils.py)
- [quantizer 원본](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/methods/utils/quant_utils.py)
- MIT 고지는 [LICENSE.quantized-reasoning-models](LICENSE.quantized-reasoning-models)에 보존한다.
- 수정: 외부 utils/hadamard 의존성 제거, CPU일 때 CUDA synchronize 생략, 잘못된 zero_point 오류 출력 수정, damping 자동 증가 제거. Cholesky 실패는 설정 변경 없이 명시적 실패로 남긴다.
- 새로운 알고리즘이라고 주장하지 않는다. layerwise driver와 calibration 공급은 QuantSplit 코드다. upstream 전체 pipeline의 정확 재현을 주장하지 않는다.

