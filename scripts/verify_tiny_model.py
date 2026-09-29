"""Actual Transformers cache verification with random tiny Qwen2, no downloads."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM, TopPLogitsWarper
from quantsplit.generation import Settings, generate_four
from quantsplit.hf_backend import HFBackend, cache_errors, full_prefix_logits
from quantsplit.sampling import PolicyPair, probabilities
from quantsplit.quantization import quantize_qwen2
from quantsplit.gptq_core import GPTQ, WeightQuantizer
from quantsplit.validation import validate_model_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260929)
    torch.set_num_threads(2)
    config = Qwen2Config(vocab_size=31, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=128, attention_dropout=0.0)
    config._attn_implementation = "eager"
    f = Qwen2ForCausalLM(config).eval()
    q = Qwen2ForCausalLM(config).eval()
    prompt = [1, 2, 3]
    equal = generate_four(HFBackend(f), HFBackend(f), prompt, "tiny", 42, Settings(max_new_tokens=12))
    assert len({tuple(v.generated_ids) for v in equal.values()}) == 1
    untruncated = generate_four(HFBackend(f), HFBackend(q), prompt, "tiny", 42,
                                Settings(max_new_tokens=12, top_p=1))
    assert untruncated["FF"].generated_ids == untruncated["FQ"].generated_ids
    assert untruncated["QQ"].generated_ids == untruncated["QF"].generated_ids
    forward = generate_four(HFBackend(f), HFBackend(q), prompt, "tiny", 43, Settings(max_new_tokens=12))
    reverse = generate_four(HFBackend(f), HFBackend(q), prompt, "tiny", 43,
                            Settings(max_new_tokens=12), arm_order=("FQ", "QF", "QQ", "FF"))
    assert all(forward[a].generated_ids == reverse[a].generated_ids for a in forward)
    errors = {name: cache_errors(model, prompt, [4, 7, 9]) for name, model in [("F", f), ("Q", q)]}
    assert max(max(v) for v in errors.values()) < 1e-5
    lf, lq = full_prefix_logits(f, prompt), full_prefix_logits(q, prompt)
    pair = PolicyPair.from_logits(lf, lq)
    independent_errors = {}
    for arm, logits in [("FF", lf), ("QQ", lq)]:
        warped = TopPLogitsWarper(0.95)(torch.tensor([prompt]), torch.from_numpy(logits[None] / 0.6))
        baseline = torch.softmax(warped, dim=-1)[0].numpy()
        independent_errors[arm] = float(np.max(np.abs(baseline - pair.distribution(arm))))
        assert independent_errors[arm] < 1e-6
    # Probe cross-device dtype drift separately from strict FP32 logic checks.
    f = f.to(torch.bfloat16)
    bf16_errors = cache_errors(f, prompt, [4, 7, 9])
    assert max(bf16_errors) < 0.02
    # With a diagonal Hessian, GPTQ must reduce to independent RTN.
    linear = torch.nn.Linear(8, 3, bias=False)
    original = linear.weight.detach().clone()
    reference = WeightQuantizer()
    reference.configure(3, groupsize=4, sym=False, mse=False)
    reference.find_params(original)
    expected = reference.quantize(original)
    core = GPTQ(linear)
    core.H = torch.eye(8)
    core.quantizer = WeightQuantizer()
    core.quantizer.configure(3, groupsize=-1, sym=False, mse=False)
    core.fasterquant(blocksize=4, groupsize=4, static_groups=True, actorder=False)
    torch.testing.assert_close(linear.weight, expected, atol=1e-7, rtol=0)
    quantized = copy.deepcopy(f)
    calibration = [torch.randint(0,31,(length,)).tolist() for length in [19,21,23,25]]
    quant_report = quantize_qwen2(quantized, calibration)
    assert quant_report['target_count'] == 14 and quant_report['non_target_weights_unchanged']
    assert all(item['before'] != item['after'] for item in quant_report['weights'])
    quant_errors = cache_errors(quantized,prompt,[4,7,9])
    assert max(quant_errors) < .02
    quant_rollouts = generate_four(HFBackend(f), HFBackend(quantized), prompt, 'tiny-gptq',42,
                                   Settings(max_new_tokens=8))
    assert all(r.consumed_tokens == 8 for r in quant_rollouts.values())
    target_checks = validate_model_pair(f,quantized,prompt)
    assert target_checks['passed']
    result = {"kind": "random_tiny_qwen2_implementation_verification_not_research_result",
              "torch": torch.__version__, "transformers": transformers.__version__,
              "device": "cpu", "seed": 20260929, "passed": True,
              "checks": ["F_equals_Q", "top_p_one", "arm_order_invariance",
                         "cache_vs_full_prefix_FP32", "cache_vs_full_prefix_BF16",
                         "FF_QQ_transformers_TopPLogitsWarper_distribution",
                         "GPTQ_diagonal_H_equals_RTN", "GPTQ_variable_length_layerwise",
                         "GPTQ_only_14_target_weights_changed", "GPTQ_BF16_four_arm_cache"],
              "cache_max_abs_errors_fp32": errors, "cache_max_abs_errors_bf16": bf16_errors,
              "baseline_distribution_errors": independent_errors,
              "tiny_GPTQ_target_count": quant_report['target_count'],
              "tiny_GPTQ_cache_errors_bf16": quant_errors,
              "target_driver_validation_on_tiny_pair": target_checks,
              "tolerances": {"fp32_logits": 1e-5, "bf16_logits": 0.02, "distribution": 1e-6},
              "actual_1_5b_model_verified": False, "actual_GPTQ_verified": False,
              "GPU_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
