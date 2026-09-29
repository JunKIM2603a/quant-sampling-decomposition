import unittest
from unittest.mock import patch
import numpy as np


class CacheDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            raise unittest.SkipTest("optional torch/transformers not installed")
        torch.manual_seed(23)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, use_sliding_window=False)
        config._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()

    def test_tiny_real_backend_modes_and_explicit_cache_reference(self):
        import copy
        from quantsplit.cache_diagnostics import MODES, diagnose_model, diagnostic_mode
        model = copy.deepcopy(self.model)
        for mode in MODES:
            if mode == "fp32_math":
                model.float()
            with diagnostic_mode(mode):
                report = diagnose_model(model, [1, 2, 3])
            self.assertEqual([s["prefix_tokens"] for s in report["steps"]], [3, 4, 5, 6, 7])
            self.assertTrue(report["all_cache_lengths_correct"])
            self.assertTrue(report["all_logits_finite"])
            self.assertEqual(report["maxima"]["repository_vs_direct"]["max_abs_logit"], 0)
            self.assertEqual(report["maxima"]["full_repeat"]["max_abs_logit"], 0)
        self.assertLess(report["maxima"]["cached_vs_full"]["max_abs_logit"], 1e-5)

    def test_corrupt_repository_logits_are_detected_by_direct_reference(self):
        from quantsplit.cache_diagnostics import diagnose_model
        from quantsplit.hf_backend import HFSession
        original = HFSession.next_logits

        def corrupt(session, prefix):
            result = original(session, prefix).copy()
            result[0] += 1.0
            return result

        with patch.object(HFSession, "next_logits", corrupt):
            report = diagnose_model(self.model, [1, 2, 3])
        self.assertFalse(report["meets_existing_cache_tolerance"])
        self.assertGreater(report["maxima"]["repository_vs_direct"]["max_abs_logit"], .9)

    def test_backend_settings_restored_on_exception_and_nonfinite_rejected(self):
        import torch
        from quantsplit.cache_diagnostics import diagnostic_mode, difference

        def flags():
            return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                    torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
                    torch.backends.cuda.mem_efficient_sdp_enabled())

        before = flags()
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with diagnostic_mode("bf16_math_full_reduction"):
                self.assertFalse(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
                raise RuntimeError("deliberate")
        self.assertEqual(flags(), before)
        with self.assertRaises(ArithmeticError):
            difference(np.array([np.nan]), np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
