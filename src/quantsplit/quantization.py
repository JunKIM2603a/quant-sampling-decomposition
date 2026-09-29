"""Variable-length, true-sequential Qwen2 GPTQ with captured layer kwargs."""
import hashlib
import torch
from .gptq_core import GPTQ, WeightQuantizer

QUANT_CONFIG = {
    "bits": 3, "group_size": 128, "symmetric": False,
    "damp_percent": 0.01, "damp_auto_increment": False,
    "act_order": True, "static_groups": True, "block_size": 128,
    "weight_clip": True, "clip_norm": 2.4, "clip_grid": 100, "clip_maxshrink": 0.8,
    "activation_dtype": "bfloat16", "kv_dtype": "bfloat16",
    "rotation": False, "quantize_embeddings": False, "quantize_lm_head": False,
    "upstream_revision": "bf947e29f52e3f666e3263efac149dae0ac18d00",
    "driver": "quantsplit-layerwise-captured-kwargs-v1"
}
GROUPS = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
          ("self_attn.o_proj",), ("mlp.up_proj", "mlp.gate_proj"), ("mlp.down_proj",))


def tensor_hash(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _move(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, tuple):
        return tuple(_move(x, device) for x in value)
    if isinstance(value, list):
        return [_move(x, device) for x in value]
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported layer kwarg: {type(value).__name__}")


@torch.inference_mode()
def quantize_qwen2(model, token_sequences, config=None):
    config = dict(QUANT_CONFIG if config is None else config)
    if model.config.model_type != "qwen2" or not token_sequences:
        raise ValueError("Qwen2 and nonempty calibration sequences required")
    model.eval()
    device = model.get_input_embeddings().weight.device
    before = {name: tensor_hash(value) for name, value in model.state_dict().items()}
    captured = []

    class CapturedFirstLayer(Exception):
        pass

    def capture(module, inputs, kwargs):
        captured.append((_move(inputs[0], "cpu"), _move(kwargs, "cpu")))
        raise CapturedFirstLayer()

    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for tokens in token_sequences:
            if not tokens:
                raise ValueError("empty calibration sequence")
            ids = torch.tensor([tokens], device=device, dtype=torch.long)
            try:
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            except CapturedFirstLayer:
                pass
    finally:
        handle.remove()
    if len(captured) != len(token_sequences):
        raise RuntimeError("calibration capture count mismatch")
    targets, changes = set(), []

    def replay(layer, item):
        hidden, kwargs = item
        output = layer(_move(hidden, device), **_move(kwargs, device))
        return output[0] if isinstance(output, tuple) else output

    for layer_index, layer in enumerate(model.model.layers):
        for names in GROUPS:
            cores, handles = {}, []
            try:
                for name in names:
                    module = layer.get_submodule(name)
                    if not isinstance(module, torch.nn.Linear):
                        raise TypeError("expected dense Linear; no packed kernels")
                    core = GPTQ(module)
                    core.quantizer = WeightQuantizer()
                    core.quantizer.configure(config["bits"], groupsize=-1,
                                              sym=config["symmetric"], mse=config["weight_clip"],
                                              norm=config["clip_norm"], grid=config["clip_grid"],
                                              maxshrink=config["clip_maxshrink"])
                    cores[name] = core
                    def collect(module, inp, out, core=core):
                        core.add_batch(inp[0].detach(), out.detach())
                    handles.append(module.register_forward_hook(collect))
                for item in captured:
                    replay(layer, item)
            finally:
                for hook in handles:
                    hook.remove()
            for name, core in cores.items():
                core.fasterquant(blocksize=config["block_size"], percdamp=config["damp_percent"],
                                 groupsize=config["group_size"], actorder=config["act_order"],
                                 static_groups=config["static_groups"])
                path = f"model.layers.{layer_index}.{name}.weight"
                weight = layer.get_submodule(name).weight
                if not torch.isfinite(weight).all():
                    raise ArithmeticError(f"nonfinite quantized weights: {path}")
                targets.add(path)
                changes.append({"name": path, "before": before[path], "after": tensor_hash(weight)})
                core.free()
            del cores
        captured = [(replay(layer, item).detach().cpu(), item[1]) for item in captured]
    untouched = [name for name, value in model.state_dict().items()
                 if name not in targets and tensor_hash(value) != before[name]]
    if untouched:
        raise RuntimeError(f"unexpected non-target mutations: {untouched}")
    return {"config": config, "target_count": len(targets), "weights": changes,
            "non_target_weights_unchanged": True, "calibration_sequences": len(token_sequences)}
