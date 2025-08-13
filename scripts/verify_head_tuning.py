import logging
import numpy as np
import os
import orbax.checkpoint as ocp
import orbax.checkpoint.args as ocp_args
from openpi.models import model

# Weight paths for full fine-tuning (not LoRA)
WEIGHT_PATHS = {
    "query"     : "PaliGemma/llm/layers/attn/q_einsum",
    "key_value" : "PaliGemma/llm/layers/attn/kv_einsum", 
    "output"    : "PaliGemma/llm/layers/attn/attn_vec_einsum",
}

# Absolute difference threshold to consider "changed/unchanged" (float32 domain)
CHANGE_DIFF_THRESH = 1e-8

def load_params_from_checkpoint(exp_dir: str, step: int) -> dict:
    """Load parameters using OpenPI's standard restore_params function."""
    params_path = os.path.abspath(f"{exp_dir}/{step}/params")
    print(f"Trying to load parameters from: {params_path}")
    
    try:
        # Restore as numpy arrays for convenient inspection using the standard API
        restored = model.restore_params(params_path, restore_type=np.ndarray)
        print(f"Successfully restored parameters, type: {type(restored)}")
        if hasattr(restored, 'keys'):
            print(f"Top-level keys: {list(restored.keys())}")
        return restored
    except Exception as e:
        print(f"restore_params failed: {e}")
        return {}

def get_weights(params: dict, path: str, weight_name: str = "w"):
    """Fetch the weight array by a slash-separated path; for full FT, read 'w'."""
    keys = path.split('/')
    w = params
    for k in keys:
        w = w[k]
    return w[weight_name]  # For full fine-tuning, we are interested in the 'w' tensor

def _get_leaf(params: dict, path: str, leaf_name: str):
    keys = path.split('/')
    node = params
    for k in keys:
        node = node[k]
    return node[leaf_name]

def _allclose_safe(a, b, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    return np.allclose(a32, b32, rtol=rtol, atol=atol)

def debug_pytree_structure(params: dict, max_depth: int = 3, current_depth: int = 0, prefix: str = ""):
    """Recursively print PyTree structure to help debug path issues."""
    if current_depth >= max_depth:
        return
    
    for key, value in params.items():
        current_path = f"{prefix}/{key}" if prefix else key
        
        if hasattr(value, 'value'):
            array_shape = getattr(value.value, 'shape', 'unknown')
            print(f"  {'  ' * current_depth}🍃 {current_path} -> shape: {array_shape}")
        elif isinstance(value, dict):
            print(f"  {'  ' * current_depth}📁 {current_path}/")
            debug_pytree_structure(value, max_depth, current_depth + 1, current_path)
        else:
            print(f"  {'  ' * current_depth}❓ {current_path} -> type: {type(value)}")

def _verify_full(params_base: dict, params_tuned: dict, trainable_heads: list[tuple[int, int]], untrained_head_to_check: tuple[int, int]) -> bool:
    print("\n=== Weight shape check (full) ===")
    for path_name, path in WEIGHT_PATHS.items():
        weights = get_weights(params_base, path, "w")
        print(f"{path_name}: {weights.shape}")
    
    print("\n--- Verification results (full) ---")
    all_ok = True

    print("Checking if trainable heads have changed...")
    for path_name, path in WEIGHT_PATHS.items():
        base_weights_full = get_weights(params_base, path, "w")
        tuned_weights_full = get_weights(params_tuned, path, "w")
        
        for layer_idx, head_idx in trainable_heads:
            if path_name == "key_value":
                if head_idx >= 2:
                    print(f"  [SKIP] {path_name} head ({layer_idx}, {head_idx}) - KV has only 2 heads")
                    continue
                base_head = base_weights_full[layer_idx, head_idx, 0]
                tuned_head = tuned_weights_full[layer_idx, head_idx, 0]
            else:
                if head_idx >= base_weights_full.shape[1]:
                    print(f"  [SKIP] {path_name} head ({layer_idx}, {head_idx}) - out of range")
                    continue
                base_head = base_weights_full[layer_idx, head_idx]
                tuned_head = tuned_weights_full[layer_idx, head_idx]
            
            diff = np.abs(np.asarray(base_head, dtype=np.float32) - np.asarray(tuned_head, dtype=np.float32)).max()
            if diff > CHANGE_DIFF_THRESH:
                print(f"  [PASS] {path_name} head ({layer_idx}, {head_idx}) changed. max diff: {diff:.6e}")
            else:
                print(f"  [FAIL] {path_name} head ({layer_idx}, {head_idx}) did not change")
                all_ok = False

    print("\nChecking if non-trainable heads remain unchanged...")
    layer_idx, head_idx = untrained_head_to_check
    for path_name, path in WEIGHT_PATHS.items():
        base_weights_full = get_weights(params_base, path, "w")
        tuned_weights_full = get_weights(params_tuned, path, "w")

        if path_name == "key_value":
            if head_idx >= 2:
                print(f"  [SKIP] untrained {path_name} head ({layer_idx}, {head_idx}) - KV has only 2 heads")
                continue
            base_head = base_weights_full[layer_idx, head_idx, 0]
            tuned_head = tuned_weights_full[layer_idx, head_idx, 0]
        else:
            if head_idx >= base_weights_full.shape[1]:
                print(f"  [SKIP] untrained {path_name} head ({layer_idx}, {head_idx}) - out of range")
                continue
            base_head = base_weights_full[layer_idx, head_idx]
            tuned_head = tuned_weights_full[layer_idx, head_idx]
        
        diff = np.abs(np.asarray(base_head, dtype=np.float32) - np.asarray(tuned_head, dtype=np.float32)).max()
        if diff <= CHANGE_DIFF_THRESH:
            print(f"  [PASS] untrained {path_name} head ({layer_idx}, {head_idx}) unchanged")
        else:
            print(f"  [FAIL] untrained {path_name} head ({layer_idx}, {head_idx}) unexpectedly changed. max diff: {diff:.6e}")
            all_ok = False

    print("\n=== Change magnitude analysis for all heads (full) ===")
    for path_name, path in WEIGHT_PATHS.items():
        print(f"\n{path_name} weight change analysis:")
        base_weights_full = get_weights(params_base, path, "w")
        tuned_weights_full = get_weights(params_tuned, path, "w")
        
        changes = []
        for layer_idx in range(base_weights_full.shape[0]):
            layer_changes = []
            for head_idx in range(base_weights_full.shape[1]):
                if path_name == "key_value":
                    base_head = base_weights_full[layer_idx, head_idx, 0]
                    tuned_head = tuned_weights_full[layer_idx, head_idx, 0]
                else:
                    base_head = base_weights_full[layer_idx, head_idx]
                    tuned_head = tuned_weights_full[layer_idx, head_idx]
                
                diff = np.abs(np.asarray(base_head, dtype=np.float32) - np.asarray(tuned_head, dtype=np.float32)).max()
                layer_changes.append(diff)
                is_trainable = (layer_idx, head_idx) in trainable_heads
                marker = "🎯" if is_trainable else "  "
                print(f"  {marker} Layer {layer_idx:2d}, Head {head_idx}: {diff:.6e}")
            changes.append(layer_changes)
        all_changes = [change for layer_changes in changes for change in layer_changes]
        print(f"  mean change: {np.mean(all_changes):.6e}")
        print(f"  max change: {np.max(all_changes):.6e}")
        print(f"  min change: {np.min(all_changes):.6e}")
    return all_ok

def _verify_lora(params_base: dict, params_tuned: dict, trainable_heads: list[tuple[int, int]], untrained_head_to_check: tuple[int, int]) -> bool:
    print("\n=== Weight shape check (LoRA) ===")
    for path_name, path in WEIGHT_PATHS.items():
        w_shape = _get_leaf(params_base, path, "w").shape
        print(f"{path_name} w: {w_shape}")
        if "q_einsum" in path or "attn_vec_einsum" in path:
            la_shape = _get_leaf(params_base, path, "lora_a").shape
            lb_shape = _get_leaf(params_base, path, "lora_b").shape
            print(f"{path_name} lora_a: {la_shape}, lora_b: {lb_shape}")
        elif "kv_einsum" in path:
            la_shape = _get_leaf(params_base, path, "lora_a").shape
            lb_shape = _get_leaf(params_base, path, "lora_b").shape
            print(f"{path_name} lora_a: {la_shape}, lora_b: {lb_shape} (KV has no head dimension, info only)")

    print("\n--- Verification results (LoRA) ---")
    all_ok = True

    print("Checking if base weight w remains unchanged...")
    for path_name, path in WEIGHT_PATHS.items():
        base_w = _get_leaf(params_base, path, "w")
        tuned_w = _get_leaf(params_tuned, path, "w")
        if not _allclose_safe(base_w, tuned_w):
            diff = np.abs(np.asarray(base_w, dtype=np.float32) - np.asarray(tuned_w, dtype=np.float32)).max()
            print(f"  [FAIL] {path_name} w changed, max diff: {diff:.6e}")
            all_ok = False
        else:
            print(f"  [PASS] {path_name} w unchanged")

    print("\nChecking if LoRA params of trainable heads have changed (q/attn_vec)...")
    for path_name, path in WEIGHT_PATHS.items():
        is_q_or_o = ("q_einsum" in path) or ("attn_vec_einsum" in path)
        is_kv = ("kv_einsum" in path)
        base_la = _get_leaf(params_base, path, "lora_a")
        base_lb = _get_leaf(params_base, path, "lora_b")
        tuned_la = _get_leaf(params_tuned, path, "lora_a")
        tuned_lb = _get_leaf(params_tuned, path, "lora_b")

        if is_q_or_o:
            for layer_idx, head_idx in trainable_heads:
                if layer_idx >= base_la.shape[0] or head_idx >= base_la.shape[1]:
                    print(f"  [SKIP] {path_name} head ({layer_idx}, {head_idx}) - out of range")
                    continue
                base_head_la = base_la[layer_idx, head_idx]
                tuned_head_la = tuned_la[layer_idx, head_idx]
                base_head_lb = base_lb[layer_idx, head_idx]
                tuned_head_lb = tuned_lb[layer_idx, head_idx]
                d1 = np.abs(np.asarray(base_head_la, dtype=np.float32) - np.asarray(tuned_head_la, dtype=np.float32)).max()
                d2 = np.abs(np.asarray(base_head_lb, dtype=np.float32) - np.asarray(tuned_head_lb, dtype=np.float32)).max()
                if (d1 > CHANGE_DIFF_THRESH) or (d2 > CHANGE_DIFF_THRESH):
                    print(f"  [PASS] {path_name} head ({layer_idx}, {head_idx}) LoRA changed. max|Δ|: la={d1:.6e}, lb={d2:.6e}")
                else:
                    print(f"  [FAIL] {path_name} head ({layer_idx}, {head_idx}) LoRA unchanged")
                    all_ok = False
        elif is_kv:
            layers_with_train = sorted({l for (l, _) in trainable_heads if l < base_la.shape[0]})
            for l in layers_with_train:
                base_layer_la = base_la[l]
                tuned_layer_la = tuned_la[l]
                base_layer_lb = base_lb[l]
                tuned_layer_lb = tuned_lb[l]
                d1 = np.abs(np.asarray(base_layer_la, dtype=np.float32) - np.asarray(tuned_layer_la, dtype=np.float32)).max()
                d2 = np.abs(np.asarray(base_layer_lb, dtype=np.float32) - np.asarray(tuned_layer_lb, dtype=np.float32)).max()
                print(f"  [INFO] KV layer {l} LoRA change: la_max|Δ|={d1:.6e}, lb_max|Δ|={d2:.6e}")

    print("\nChecking if LoRA params of an untrained head remain unchanged (q/attn_vec)...")
    layer_idx, head_idx = untrained_head_to_check
    for path_name, path in WEIGHT_PATHS.items():
        if not (("q_einsum" in path) or ("attn_vec_einsum" in path)):
            continue
        base_la = _get_leaf(params_base, path, "lora_a")
        base_lb = _get_leaf(params_base, path, "lora_b")
        tuned_la = _get_leaf(params_tuned, path, "lora_a")
        tuned_lb = _get_leaf(params_tuned, path, "lora_b")
        if layer_idx >= base_la.shape[0] or head_idx >= base_la.shape[1]:
            print(f"  [SKIP] untrained {path_name} head ({layer_idx}, {head_idx}) - out of range")
            continue
        base_head_la = base_la[layer_idx, head_idx]
        tuned_head_la = tuned_la[layer_idx, head_idx]
        base_head_lb = base_lb[layer_idx, head_idx]
        tuned_head_lb = tuned_lb[layer_idx, head_idx]
        d1 = np.abs(np.asarray(base_head_la, dtype=np.float32) - np.asarray(tuned_head_la, dtype=np.float32)).max()
        d2 = np.abs(np.asarray(base_head_lb, dtype=np.float32) - np.asarray(tuned_head_lb, dtype=np.float32)).max()
        if (d1 <= CHANGE_DIFF_THRESH) and (d2 <= CHANGE_DIFF_THRESH):
            print(f"  [PASS] untrained {path_name} head ({layer_idx}, {head_idx}) LoRA unchanged")
        else:
            print(f"  [FAIL] untrained {path_name} head ({layer_idx}, {head_idx}) LoRA unexpectedly changed. la={d1:.6e}, lb={d2:.6e}")
            all_ok = False

    print("\n=== Change magnitude analysis for all heads (LoRA, q/attn_vec only) ===")
    for path_name, path in WEIGHT_PATHS.items():
        if not (("q_einsum" in path) or ("attn_vec_einsum" in path)):
            continue
        print(f"\n{path_name} LoRA weight change analysis:")
        base_la = _get_leaf(params_base, path, "lora_a")
        base_lb = _get_leaf(params_base, path, "lora_b")
        tuned_la = _get_leaf(params_tuned, path, "lora_a")
        tuned_lb = _get_leaf(params_tuned, path, "lora_b")
        changes = []
        for layer_idx in range(base_la.shape[0]):
            layer_changes = []
            for head_idx in range(base_la.shape[1]):
                d1 = np.abs(np.asarray(base_la[layer_idx, head_idx], dtype=np.float32) - np.asarray(tuned_la[layer_idx, head_idx], dtype=np.float32)).max()
                d2 = np.abs(np.asarray(base_lb[layer_idx, head_idx], dtype=np.float32) - np.asarray(tuned_lb[layer_idx, head_idx], dtype=np.float32)).max()
                layer_changes.append(max(d1, d2))
                is_trainable = (layer_idx, head_idx) in trainable_heads
                marker = "🎯" if is_trainable else "  "
                print(f"  {marker} Layer {layer_idx:2d}, Head {head_idx}: max|Δ|={max(d1, d2):.6e}")
            changes.append(layer_changes)
        all_changes = [c for lc in changes for c in lc]
        print(f"  mean change: {np.mean(all_changes):.6e}")
        print(f"  max change: {np.max(all_changes):.6e}")
        print(f"  min change: {np.min(all_changes):.6e}")

    return all_ok

def main():
    # --- Configuration ---
    # Optional: set mode to "full" or "lora"
    mode = "lora"
    if mode == "lora":
        config_name = "pi0_fast_droid_h5_head_lora_tune_debug"
        exp_name = "head_lora_tune_SAV_pick_20_margin"
        base_step = 1000
        tuned_step = 9999
    else:
        config_name = "pi0_fast_droid_h5_head_tune_debug"
        exp_name = "head_tune_debug_mask"
        base_step = 1
        tuned_step = 3

    trainable_heads = [(6, 5), (9, 3), (2, 7), (2, 4), (6, 4), (8, 7), (5, 7), (7, 1), (6, 7), (6, 1), (7, 2), (6, 3), (5, 3), (1, 0), (10, 6), (11, 4), (9, 0), (8, 1), (3, 0), (12, 1)]
    untrained_head_to_check = (4, 1)

    exp_dir = f"./checkpoints/{config_name}/{exp_name}"
    print(f"Loading checkpoints from: {exp_dir}")

    params_base = load_params_from_checkpoint(exp_dir, base_step)
    params_tuned = load_params_from_checkpoint(exp_dir, tuned_step)
    
    if not params_base or not params_tuned:
        print("Failed to load checkpoints, exiting")
        return

    if mode == "lora":
        ok = _verify_lora(params_base, params_tuned, trainable_heads, untrained_head_to_check)
    else:
        ok = _verify_full(params_base, params_tuned, trainable_heads, untrained_head_to_check)

    print("\n--- Final summary ---")
    if ok:
        print("✅ Verification succeeded!")
    else:
        print("❌ Verification failed!")

if __name__ == "__main__":
    main() 