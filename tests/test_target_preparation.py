import json
from pathlib import Path
import tempfile
import unittest

from quantsplit.assets import validate_asset_contract


class AssetContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = {"development": ["question-1"]}
        self.lock = {"tokenizer_hash": "recorded", "packages": {"torch": "2.8.0+cpu"}}
        self.mp, self.lp = self.root / "split.json", self.root / "lock.json"
        self.mp.write_text(json.dumps(self.manifest))
        self.lp.write_text(json.dumps(self.lock))

    def check(self, manifest=None, lock=None):
        validate_asset_contract(self.mp, self.lp, manifest or self.manifest, lock or self.lock,
                                {"development": "new worklist\n"}, self.root / "data")

    def test_runtime_package_differences_do_not_rewrite_locked_file(self):
        before = self.lp.read_bytes()
        self.check(lock={**self.lock, "packages": {"torch": "2.8.0+cu126"}})
        self.assertEqual(self.lp.read_bytes(), before)
        self.assertFalse((self.root / "data").exists())

    def test_changed_tokenizer_rejected_before_writes(self):
        before = self.lp.read_bytes()
        with self.assertRaisesRegex(ValueError, "execution lock differs"):
            self.check(lock={**self.lock, "tokenizer_hash": "changed"})
        self.assertEqual(self.lp.read_bytes(), before)
        self.assertFalse((self.root / "data").exists())

    def test_changed_split_rejected_before_writes(self):
        before = self.mp.read_bytes()
        with self.assertRaisesRegex(ValueError, "split differs"):
            self.check(manifest={"development": ["foreign-question"]})
        self.assertEqual(self.mp.read_bytes(), before)

    def test_changed_existing_worklist_is_not_overwritten(self):
        path = self.root / "data/development.jsonl"
        path.parent.mkdir()
        path.write_text("previous worklist\n")
        with self.assertRaisesRegex(ValueError, "worklist differs"):
            self.check()
        self.assertEqual(path.read_text(), "previous worklist\n")


class ContextProbeTests(unittest.TestCase):
    def test_real_tiny_backend_reaches_target_and_last_two_single_steps(self):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            self.skipTest("optional torch/transformers not installed")
        from quantsplit.validation import validate_long_context_pair
        torch.manual_seed(22)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=128, use_sliding_window=False)
        config._attn_implementation = "sdpa"
        f = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
        q = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
        seen = []
        result = validate_long_context_pair(f, q, [1, 2, 3], 67, chunk_size=16,
                                           progress=lambda n, total: seen.append(n))
        self.assertTrue(result["passed"], result)
        self.assertEqual(seen, [16, 32, 48, 64, 65, 66, 67])
        self.assertEqual([v["cache_length"] for v in result["cache"].values()], [67, 67])
        self.assertFalse(result["scientific_H1_result"])


if __name__ == "__main__":
    unittest.main()
