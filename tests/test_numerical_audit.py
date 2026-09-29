import copy
import unittest
from unittest.mock import patch
import numpy as np


class NumericalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            raise unittest.SkipTest("optional model dependencies not installed")
        torch.manual_seed(29)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, use_sliding_window=False)
        config._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(config).bfloat16().eval()
        cls.plan = {"max_new_tokens": 6, "probe_steps": [0, 1, 5],
                    "temperature": .6, "top_p": .95, "seed": 42}

    def test_real_model_trace_replay_and_early_eos_observation(self):
        from quantsplit.cache_diagnostics import diagnostic_mode
        from quantsplit.numerical_audit import collect_trace, make_trace
        model = copy.deepcopy(self.model)
        with diagnostic_mode("bf16_auto"):
            trace = make_trace(model, [1, 2, 3], "fixture", self.plan, [])
            repeat = make_trace(model, [1, 2, 3], "fixture", self.plan, [])
            points, checks = collect_trace(model, trace)
            early = make_trace(model, [1, 2, 3], "fixture", self.plan, list(range(64)))
            early_points, early_checks = collect_trace(model, early)
        self.assertEqual(trace, repeat)
        self.assertEqual(list(points), [0, 1, 5])
        self.assertEqual([p["prefix_tokens"] for p in points.values()], [3, 4, 8])
        self.assertEqual(checks["steps_checked"], 6)
        self.assertTrue(checks["same_schedule_exact"])
        self.assertEqual(early["stop_reason"], "eos")
        self.assertEqual(list(early_points), [0])
        self.assertEqual(early_checks["steps_checked"], 1)
        model.float()
        with diagnostic_mode("fp32_math"):
            fp_points, fp_checks = collect_trace(model, trace)
        self.assertTrue(fp_checks["same_schedule_exact"])
        self.assertLess(max(p["error"]["max_abs_logit"] for p in fp_points.values()), 1e-5)

    def test_cache_error_between_selected_probes_is_detected(self):
        from quantsplit.hf_backend import HFSession
        from quantsplit.numerical_audit import collect_trace
        original = HFSession.next_logits

        def corrupt(session, prefix):
            result = original(session, prefix).copy()
            if len(prefix) == 5:  # step 2, outside probe_steps
                result[0] += 1
            return result

        trace = {"prompt_ids": [1, 2, 3], "generated_ids": [4] * 6, "probe_steps": [0, 1, 5]}
        with patch.object(HFSession, "next_logits", corrupt):
            _, checks = collect_trace(self.model, trace)
        self.assertFalse(checks["same_schedule_exact"])
        self.assertGreater(checks["direct_max_abs_logit"], .9)

    def test_support_swap_has_expected_four_arm_sensitivity(self):
        from quantsplit.numerical_audit import policy_metrics, summarize_points
        f, q = np.array([3., 2., 0.]), np.array([2., 3., 0.])
        metrics = policy_metrics({"bf16_cached": {"F": f, "Q": q},
            "bf16_full": {"F": f, "Q": q}, "fp32_cached": {"F": q, "Q": q}},
            temperature=1, top_p=.6, special_ids=(2,))
        precision = metrics["bf16_cached_vs_fp32_cached"]
        self.assertEqual(precision["policy_tv"], {"FF": 1., "QQ": 0., "QF": 1., "FQ": 0.})
        self.assertEqual(precision["support"]["F"]["changed_tokens"], 2)
        self.assertEqual(precision["support"]["F"]["jaccard"], 0)
        self.assertEqual(set(metrics["bf16_cached_vs_full"]["policy_tv"].values()), {0.})
        summary = summarize_points([{"policies": metrics}])
        self.assertTrue(summary["review_required"])
        self.assertFalse(summary["automatic_stage02_pass"])
        self.assertEqual(summary["bf16_cached_vs_fp32_cached"]["policy_tv"]["QF"]["max"], 1.)


if __name__ == "__main__":
    unittest.main()
