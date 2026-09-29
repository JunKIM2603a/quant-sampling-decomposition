"""Transformers 4.57.1 backend: one unpadded sequence per independent cache.

Model weights may be shared between sessions, but mutable KV caches never are.
This diagnostic implementation copies last-token logits to CPU for FP32 sampling.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


class HFBackend:
    def __init__(self, model):
        self.model = model.eval()

    def new_session(self):
        return HFSession(self.model)


class HFSession:
    def __init__(self, model):
        from transformers import DynamicCache
        self.model = model
        self.cache = DynamicCache(config=model.config)
        self.processed = ()

    def next_logits(self, prefix):
        import torch
        prefix = tuple(prefix)
        n = len(self.processed)
        if prefix[:n] != self.processed or len(prefix) <= n:
            raise ValueError("cache prefix must extend its own processed prefix")
        if self.cache.get_seq_length() != n:
            raise ValueError("KV length and prefix length disagree")
        device = self.model.get_input_embeddings().weight.device
        inputs = torch.tensor([prefix[n:]], dtype=torch.long, device=device)
        mask = torch.ones((1, len(prefix)), dtype=torch.long, device=device)
        positions = torch.arange(n, len(prefix), device=device)
        with torch.inference_mode():
            out = self.model(input_ids=inputs, attention_mask=mask,
                             cache_position=positions, past_key_values=self.cache,
                             use_cache=True, return_dict=True, logits_to_keep=1)
        self.cache = out.past_key_values
        self.processed = prefix
        if self.cache.get_seq_length() != len(prefix):
            raise ValueError("model returned unexpected cache length")
        return out.logits[0, -1].detach().float().cpu().numpy()


def full_prefix_logits(model, prefix):
    import torch
    device = model.get_input_embeddings().weight.device
    x = torch.tensor([prefix], device=device, dtype=torch.long)
    with torch.inference_mode():
        return model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                     return_dict=True, logits_to_keep=1).logits[0, -1].float().cpu().numpy()


def cache_errors(model, prompt, continuation):
    session, prefix, errors = HFBackend(model).new_session(), list(prompt), []
    for token in [None] + list(continuation):
        if token is not None:
            prefix.append(token)
        cached = session.next_logits(prefix)
        full = full_prefix_logits(model, prefix)
        errors.append(float(np.max(np.abs(cached - full))))
    return errors


def tokenizer_hash(tokenizer):
    value = {"vocab": sorted(tokenizer.get_vocab().items()),
             "chat_template": tokenizer.chat_template,
             "special_tokens": tokenizer.special_tokens_map}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def render_prompt(tokenizer, question):
    instruction = r"Please reason step by step, and put your final answer within \boxed{}."
    message = question.strip() + "\n\n" + instruction
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": message}],
                                              tokenize=False, add_generation_prompt=True)
    # Inspect the assistant suffix, not a possible literal <think> in the question.
    if "<｜Assistant｜>" not in rendered:
        raise ValueError("unexpected DeepSeek assistant template")
    suffix = rendered.rsplit("<｜Assistant｜>", 1)[1]
    if suffix.count("<think>") == 0:
        rendered += "<think>\n"
    elif suffix.count("<think>") != 1 or not suffix.rstrip().endswith("<think>"):
        raise ValueError("unexpected reasoning-start suffix")
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    return ids, hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_q_checkpoint(directory, manifest, lock):
    from .quantization import QUANT_CONFIG
    directory = Path(directory).resolve()
    expected = {"format": "dequantized_bfloat16", "method": "GPTQ", "bits": 3,
                "group_size": 128, "symmetric": False,
                "base_model_revision": lock["model_revision"],
                "tokenizer_hash": lock["tokenizer_hash"],
                "split_manifest_sha256": lock["split_manifest_sha256"]}
    for key, value in expected.items():
        if value is None or manifest.get(key) != value:
            raise ValueError(f"Q checkpoint provenance mismatch: {key}")
    for key in ["calibration_tokens_sha256", "quantization_code_revision", "quantization_config"]:
        if not manifest.get(key):
            raise ValueError(f"Q checkpoint manifest missing: {key}")
    if manifest["quantization_config"] != QUANT_CONFIG:
        raise ValueError("Q quantization config differs from development protocol")
    if manifest.get("source_sha256", {}).get("gptq_core.py") != file_sha256(Path(__file__).with_name("gptq_core.py")):
        raise ValueError("Q GPTQ core source differs from current implementation")
    files = manifest.get("files_sha256", {})
    weights = {p.name for p in directory.glob("*.safetensors")}
    if not weights or not weights.issubset(files) or "config.json" not in files:
        raise ValueError("manifest must hash every safetensors file and config.json")
    if files.get("calibration_tokens.jsonl") != manifest["calibration_tokens_sha256"]:
        raise ValueError("calibration token file not covered by manifest")
    if json.loads((directory / "config.json").read_text()).get("quantization_config"):
        raise ValueError("packed quantized checkpoint is outside the BF16 simulation design")
    for name, digest in files.items():
        path = (directory / name).resolve()
        if path.parent != directory or file_sha256(path) != digest:
            raise ValueError(f"checkpoint hash/path mismatch: {name}")
