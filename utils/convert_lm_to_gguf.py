#!/usr/bin/env python3
"""
Convert VibeVoice-ASR LM (BitNet-trained) from SafeTensors to GGUF format.

Pipeline:
  1. Preprocess: quantize BitNet-trained weights to ternary FP32 (in-place)
  2. Convert to GGUF via convert-ms-to-gguf-bitnet.py
  3. Restore original safetensors files

Usage:
  python utils/convert_lm_to_gguf.py <model-directory> [outtype]

  outtype: f32 (default), f16, q8_0, etc.

Example:
  python utils/convert_lm_to_gguf.py models/lm-checkpoint/
  python utils/convert_lm_to_gguf.py models/lm-checkpoint/ f16
"""

import sys
import os
import json
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# --- BitNet weight preprocessing (inlined) ---

BITNET_WEIGHT_KEYWORDS = [
    'q_proj.weight',
    'k_proj.weight',
    'v_proj.weight',
    'o_proj.weight',
    'gate_proj.weight',
    'up_proj.weight',
    'down_proj.weight',
]


def quant_weight_fp16(weight):
    """Quantize a weight tensor to ternary {-1, 0, 1} scaled by 1/mean(|w|)."""
    weight = weight.to(torch.float)
    s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    new_weight = (weight * s).round().clamp(-1, 1) / s
    return new_weight


def preprocess_bitnet_safetensors(input_path: Path, output_path: Path):
    """Preprocess a SafeTensors file: quantize BitNet projection weights to ternary FP32.

    Also strips 'model.language_model.' prefix to 'model.' and filters out
    non-LM tensors (VAE/tokenizer) so convert-ms-to-gguf-bitnet.py can parse them.
    """
    tensors = {}

    with safe_open(str(input_path), framework='pt') as f:
        for name in f.keys():
            tensor = f.get_tensor(name)

            # Strip 'model.language_model.' -> 'model.' for compatibility
            out_name = name
            if name.startswith('model.language_model.'):
                out_name = 'model.' + name[len('model.language_model.'):]
            elif not name.startswith('model.') and not name.startswith('lm_head'):
                # Skip non-LM tensors (VAE, tokenizer, connectors)
                continue

            if any(keyword in out_name for keyword in BITNET_WEIGHT_KEYWORDS):
                print(f'  [preprocess] Quantizing {name} -> {out_name}')
                tensor = quant_weight_fp16(tensor)
            elif out_name != name:
                print(f'  [preprocess] Renaming {name} -> {out_name}')

            tensors[out_name] = tensor

    print(f'  [preprocess] Saving {len(tensors)} tensors to {output_path}')
    save_file(tensors, str(output_path))


# --- Main conversion logic ---

def run_command(command_list, cwd=None, check=True, env=None):
    print(f"Executing: {' '.join(map(str, command_list))}")
    try:
        process = subprocess.run(command_list, cwd=cwd, check=check, capture_output=False, text=True, env=env)
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(map(str, e.cmd))}")
        print(f"Return code: {e.returncode}")
        raise


def main():
    if len(sys.argv) < 2:
        script_name = Path(sys.argv[0]).name
        print(f"Usage: python {script_name} <model-directory> [outtype]")
        print(f"  outtype: f32, f16, q8_0, etc. (default: f32)")
        sys.exit(1)

    model_dir_arg = sys.argv[1]
    model_dir = Path(model_dir_arg).resolve()
    outtype = sys.argv[2] if len(sys.argv) > 2 else "f32"

    if not model_dir.is_dir():
        print(f"Error: Model directory '{model_dir}' not found.")
        sys.exit(1)

    utils_dir = Path(__file__).parent.resolve()
    project_root_dir = utils_dir.parent
    convert_script = utils_dir / "convert-ms-to-gguf-bitnet.py"

    if not convert_script.is_file():
        print(f"Error: Conversion script not found at '{convert_script}'")
        sys.exit(1)

    # Detect input files
    single_file = model_dir / "model.safetensors"
    if single_file.is_file():
        input_files = [single_file]
    else:
        input_files = sorted(list(model_dir.glob("model-*-of-*.safetensors")))

    if not input_files:
        print(f"Error: No safetensors files found in '{model_dir}'")
        sys.exit(1)

    gguf_output = model_dir / f"vibeasr-lm-{outtype}.gguf"
    backup_dir = model_dir / ".backup_originals"

    config_path = model_dir / "config.json"
    config_backup = None

    try:
        # Step 0: Prepare config.json for convert-ms-to-gguf-bitnet.py
        # If config.json has nested LM config (e.g. decoder_config), flatten it
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            if "decoder_config" in config and "hidden_size" not in config:
                print("Step 0: Flattening decoder_config into config.json for converter...")
                config_backup = model_dir / "config.json.bak"
                shutil.copy2(str(config_path), str(config_backup))
                lm_config = config["decoder_config"]
                with open(config_path, 'w') as f:
                    json.dump(lm_config, f, indent=2)

        # Step 1: Preprocess weights to ternary FP32 (in-place swap)
        print("Step 1: Preprocessing BitNet weights to ternary FP32...")
        backup_dir.mkdir(exist_ok=True)

        for f in input_files:
            preprocessed = f.parent / (f.stem + ".preprocessed" + f.suffix)
            preprocess_bitnet_safetensors(f, preprocessed)
            # Backup original, replace with preprocessed
            shutil.move(str(f), str(backup_dir / f.name))
            shutil.move(str(preprocessed), str(f))

        # Step 2: Convert to GGUF (ensure local gguf-py is used)
        print(f"\nStep 2: Converting to GGUF ({outtype})...")
        gguf_py_path = str((project_root_dir / "3rdparty" / "llama.cpp" / "gguf-py").resolve())
        env = os.environ.copy()
        env["PYTHONPATH"] = gguf_py_path + os.pathsep + env.get("PYTHONPATH", "")
        run_command([
            sys.executable, str(convert_script), str(model_dir),
            "--vocab-type", "bpe", "--outtype", outtype,
            "--concurrency", "1", "--outfile", str(gguf_output),
            "--pad-vocab", "--skip-unknown"
        ], env=env)
        print(f"\n✓ Done. Output: {gguf_output}")

    finally:
        # Restore original files
        if backup_dir.is_dir():
            for f in input_files:
                backup_path = backup_dir / f.name
                if backup_path.is_file():
                    if f.is_file():
                        os.remove(str(f))
                    shutil.move(str(backup_path), str(f))
            backup_dir.rmdir()
            print("Restored original safetensors files.")
        # Restore original config.json
        if config_backup and config_backup.is_file():
            shutil.move(str(config_backup), str(config_path))
            print("Restored original config.json.")


if __name__ == "__main__":
    main()
