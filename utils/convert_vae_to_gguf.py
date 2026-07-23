#!/usr/bin/env python3
"""
Convert VibeVoice-ASR VAE Encoder (acoustic + semantic) from SafeTensors to GGUF format.
Only extracts encoder weights, excluding language model and decoder weights.

Usage:
  python utils/convert_vae_to_gguf.py <model-directory> [--outtype f32|f16] [-o output.gguf]

Example:
  python utils/convert_vae_to_gguf.py models/vae-checkpoint/
  python utils/convert_vae_to_gguf.py models/vae-checkpoint/ --outtype f16
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add gguf-py to path
if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / '3rdparty' / 'llama.cpp' / 'gguf-py'))

import gguf


def load_safetensors_file(path: Path) -> dict[str, np.ndarray]:
    """Load tensors from a SafeTensors file."""
    print(f"Loading SafeTensors file: {path}")

    with open(path, 'rb') as fp:
        # Read header size
        header_size, = struct.unpack('<Q', fp.read(8))

        # Read header JSON
        header: dict[str, dict[str, Any]] = json.loads(fp.read(header_size))

        # Memory map the data section
        mapped = memoryview(mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ))
        byte_buf = mapped[8 + header_size:]

        tensors = {}
        for name, info in header.items():
            if name == '__metadata__':
                continue

            # Parse tensor info
            dtype_str = info['dtype']
            shape = info['shape']
            data_offsets = info['data_offsets']

            # Map dtype
            dtype_map = {
                'F32': np.float32,
                'F16': np.float16,
                'BF16': np.uint16,
                'I32': np.int32,
                'I64': np.int64,
            }
            dtype = dtype_map.get(dtype_str, np.float32)

            # Extract tensor data
            start, end = data_offsets
            tensor_data = byte_buf[start:end]
            tensor = np.frombuffer(tensor_data, dtype=dtype).reshape(shape)

            # Convert BF16 to FP32 if needed
            if dtype_str == 'BF16':
                tensor = bf16_to_fp32(tensor)

            tensors[name] = tensor

    return tensors


def bf16_to_fp32(bf16_arr: np.ndarray) -> np.ndarray:
    """Convert BF16 (as uint16) to FP32."""
    assert bf16_arr.dtype == np.uint16
    fp32_arr = bf16_arr.astype(np.uint32) << 16
    return fp32_arr.view(np.float32)


def pad_conv_weight_for_simd(tensor: np.ndarray, name: str) -> np.ndarray:
    """
    Pad convolution kernel for better SIMD utilization.
    Pads kernel dimension (ne[0]) to 4/8/16 based on size.
    For I8_S quantized weights, scale will be recalculated after padding.

    Args:
        tensor: Weight tensor with shape [..., kernel_size, ...]
        name: Tensor name for logging

    Returns:
        Padded tensor (or original if no padding needed)
    """
    # Only pad conv weights (not depthwise conv, handled separately)
    if 'conv.weight' not in name or 'conv.conv.conv.weight' in name:
        return tensor

    # Get kernel size (first dimension for conv weights)
    # Conv weight shape is typically [out_channels, in_channels, kernel_size]
    kernel_size = tensor.shape[-1]  # Last dimension is kernel

    # Determine padding target
    # Don't pad if already at optimal size (4, 8, 16) or larger
    if kernel_size == 4 or kernel_size == 8 or kernel_size == 16 or kernel_size > 16:
        return tensor
    elif kernel_size < 4:
        padded_size = 4
    elif kernel_size < 8:
        padded_size = 8
    else:  # kernel_size < 16
        padded_size = 16

    pad_amount = padded_size - kernel_size

    if pad_amount == 0:
        return tensor

    # Pad along the kernel dimension (last dimension)
    # Pad at the beginning (lowest index) to maintain alignment
    pad_width = [(0, 0)] * (tensor.ndim - 1) + [(pad_amount, 0)]
    padded_tensor = np.pad(tensor, pad_width, mode='constant', constant_values=0)

    print(f"  • Padded {name}: kernel {kernel_size} -> {padded_size} (shape {tensor.shape} -> {padded_tensor.shape})")

    return padded_tensor


def pad_depthwise_conv_weight_for_simd(tensor: np.ndarray, name: str) -> np.ndarray:
    """
    Pad depthwise convolution kernel for better SIMD utilization.
    Depthwise conv has shape [out_channels, 1, kernel_size]

    Args:
        tensor: Weight tensor
        name: Tensor name for logging

    Returns:
        Padded tensor (or original if no padding needed)
    """
    # Only pad depthwise conv weights
    if 'conv.conv.conv.weight' not in name:
        return tensor

    # Get kernel size (last dimension)
    kernel_size = tensor.shape[-1]

    # Determine padding target
    # Don't pad if already at optimal size (4, 8, 16) or larger
    if kernel_size == 4 or kernel_size == 8 or kernel_size == 16 or kernel_size > 16:
        return tensor
    elif kernel_size < 4:
        padded_size = 4
    elif kernel_size < 8:
        padded_size = 8
    else:  # kernel_size < 16
        padded_size = 16

    pad_amount = padded_size - kernel_size

    if pad_amount == 0:
        return tensor

    # Pad along kernel dimension (last dimension)
    # Pad at the beginning (lowest index) to maintain alignment
    pad_width = [(0, 0)] * (tensor.ndim - 1) + [(pad_amount, 0)]
    padded_tensor = np.pad(tensor, pad_width, mode='constant', constant_values=0)

    print(f"  • Padded depthwise {name}: kernel {kernel_size} -> {padded_size} (shape {tensor.shape} -> {padded_tensor.shape})")

    return padded_tensor


def filter_vae_encoder_tensors(all_tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Filter only acoustic and semantic encoder tensors and rename them."""
    vae_tensors = {}

    for name, tensor in all_tensors.items():
        # Only keep encoder tensors (not decoder, not language_model)
        new_name = None
        if name.startswith('model.acoustic_tokenizer.encoder.'):
            # Remove prefix to shorten name (GGML_MAX_NAME is 64 characters)
            new_name = 'acoustic.' + name.replace('model.acoustic_tokenizer.encoder.', '')
        elif name.startswith('model.semantic_tokenizer.encoder.'):
            new_name = 'semantic.' + name.replace('model.semantic_tokenizer.encoder.', '')
        # Add connector weights
        elif name.startswith('model.acoustic_connector.'):
            new_name = 'acoustic_connector.' + name.replace('model.acoustic_connector.', '')
        elif name.startswith('model.semantic_connector.'):
            new_name = 'semantic_connector.' + name.replace('model.semantic_connector.', '')

        if new_name:
            # Pad conv weights for better SIMD utilization
            if 'conv.conv.conv.weight' in new_name:
                # Depthwise conv
                tensor = pad_depthwise_conv_weight_for_simd(tensor, new_name)
            elif 'conv.weight' in new_name:
                # Regular conv
                tensor = pad_conv_weight_for_simd(tensor, new_name)

            # NOTE: GGUF stores dimensions in reverse order compared to numpy/PyTorch!
            # PyTorch linear weight [OC, IC] will be stored in GGUF as [IC, OC] automatically
            # due to dimension reversal, so NO transpose is needed!
            vae_tensors[new_name] = tensor
            print(f"  ✓ {new_name}: {tensor.shape} ({tensor.dtype})")

    return vae_tensors


def convert_to_gguf(
    model_dir: Path,
    output_path: Path,
    outtype: str = 'f32',
) -> None:
    """Convert VAE encoder weights to GGUF format."""

    print(f"\n{'='*80}")
    print(f"Converting VibeVoice-ASR VAE Encoder to GGUF")
    print(f"{'='*80}\n")

    print(f"Model directory: {model_dir}")
    print(f"Output file: {output_path}")
    print(f"Output type: {outtype}\n")

    # Find SafeTensors files
    safetensors_files = []
    single_file = model_dir / "model.safetensors"
    if single_file.is_file():
        safetensors_files.append(single_file)
    else:
        # Look for sharded files
        safetensors_files = sorted(list(model_dir.glob("model-*-of-*.safetensors")))

    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    print(f"Found {len(safetensors_files)} SafeTensors file(s):\n")
    for f in safetensors_files:
        print(f"  • {f.name}")
    print()

    # Load all tensors
    print("Loading tensors from SafeTensors files...\n")
    all_tensors = {}
    for safetensors_file in safetensors_files:
        tensors = load_safetensors_file(safetensors_file)
        all_tensors.update(tensors)

    print(f"\nTotal tensors loaded: {len(all_tensors)}\n")

    # Filter VAE encoder tensors
    print("Filtering VAE encoder tensors...\n")
    vae_tensors = filter_vae_encoder_tensors(all_tensors)

    print(f"\nVAE encoder tensors: {len(vae_tensors)}")

    # Count tensors by type
    acoustic_count = sum(1 for name in vae_tensors if name.startswith('acoustic.'))
    semantic_count = sum(1 for name in vae_tensors if name.startswith('semantic.'))
    acoustic_connector_count = sum(1 for name in vae_tensors if name.startswith('acoustic_connector.'))
    semantic_connector_count = sum(1 for name in vae_tensors if name.startswith('semantic_connector.'))

    print(f"  • Acoustic encoder: {acoustic_count} tensors")
    print(f"  • Semantic encoder: {semantic_count} tensors")
    print(f"  • Acoustic connector: {acoustic_connector_count} tensors")
    print(f"  • Semantic connector: {semantic_connector_count} tensors\n")

    # Load config.json for metadata
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}\n")
    else:
        config = {}
        print("Warning: config.json not found, using default values\n")

    # Create GGUF writer
    print(f"Creating GGUF file: {output_path}\n")
    gguf_writer = gguf.GGUFWriter(output_path, "vibeasr-vae")

    # Add metadata
    print("Writing metadata...")
    gguf_writer.add_string("general.name", "VibeASR VAE Encoder")
    gguf_writer.add_string("general.file_type", outtype)
    gguf_writer.add_uint32("general.version", 1)

    # VAE-specific metadata
    acoustic_config = config.get('acoustic_tokenizer', {})
    semantic_config = config.get('semantic_tokenizer', {})

    # Acoustic encoder config
    gguf_writer.add_uint32("vae.acoustic.output_dim",
                          acoustic_config.get('codebook_size', 64))
    gguf_writer.add_array("vae.acoustic.stage_depths",
                         acoustic_config.get('depths', [3, 3, 3, 3, 3, 3, 8]))
    gguf_writer.add_array("vae.acoustic.downsample_strides",
                         [8, 5, 5, 4, 2, 2])

    # Semantic encoder config
    gguf_writer.add_uint32("vae.semantic.output_dim",
                          semantic_config.get('codebook_size', 128))
    gguf_writer.add_array("vae.semantic.stage_depths",
                         semantic_config.get('depths', [3, 3, 3, 3, 3, 3, 8]))
    gguf_writer.add_array("vae.semantic.downsample_strides",
                         [8, 5, 5, 4, 2, 2])

    # Common config
    gguf_writer.add_float32("vae.rms_norm_eps", 1e-6)
    gguf_writer.add_uint32("vae.kernel_size", 7)
    gguf_writer.add_float32("vae.layer_scale_init_value", 1e-6)

    print("  ✓ Metadata written\n")

    # Prepare tensors (keep original dtype, will convert during write)
    print(f"Preparing tensors...")

    converted_tensors = {}
    total_params = 0

    for name, tensor in vae_tensors.items():
        # Keep original tensor, will convert to appropriate dtype when writing
        converted_tensors[name] = tensor
        total_params += tensor.size

    print(f"  ✓ {len(converted_tensors)} tensors prepared\n")

    # Write tensors with proper type conversion
    print("Writing tensors to GGUF file...")
    total_size_bytes = 0

    for name, tensor in converted_tensors.items():
        # IMPORTANT: GGML requires bias tensors to be 2D [1, N] for broadcasting with conv output
        # Conv output is [output_length, out_channels, batch_size]
        # Bias [1, out_channels] can broadcast properly
        if 'bias' in name and tensor.ndim == 1:
            original_shape = tensor.shape
            # Create a new 2D array [1, N] instead of reshaping
            # This ensures the data layout is correct (C-contiguous with shape [1, N])
            n = tensor.shape[0]
            tensor_2d = np.zeros((1, n), dtype=tensor.dtype)
            tensor_2d[0, :] = tensor[:]
            tensor = tensor_2d
            print(f"  • Reshaped bias {name}: {original_shape} -> {tensor.shape}")

        # Determine output dtype based on tensor type:
        # 1. Bias and normalization weights always stay F32 for numerical stability
        # 2. For i8_s: quantize weight tensors to per-tensor INT8 with scale
        # 3. For f32/f16: keep weights in specified format
        if 'bias' in name or 'norm' in name or 'gamma' in name:
            # Keep bias/norm in F32 for better numerical stability
            tensor_out = tensor.astype(np.float32)
            if tensor.dtype != np.float32:
                print(f"  • {name}: {tensor.dtype} -> F32 (bias/norm stability)")
            gguf_writer.add_tensor(name, tensor_out)
        elif outtype == 'i8_s' and 'weight' in name:
            # I8_S per-tensor quantization:
            # Layout: [int8_data (n_elements bytes)] [padding] [scale (float32)]
            # Total size: n_elements + 32 bytes (aligned)
            tensor_f32 = tensor.astype(np.float32).flatten()
            n_elements = tensor_f32.size

            # Compute per-tensor scale
            amax = np.max(np.abs(tensor_f32))
            if amax < 1e-8:
                amax = 1e-8
            scale = 127.0 / amax

            # Quantize to int8
            i8_values = np.round(tensor_f32 * scale).clip(-127, 127).astype(np.int8)

            # Pack: int8 data + padding + scale (total: n_elements + 32)
            packed = np.zeros(n_elements + 32, dtype=np.uint8)
            packed[:n_elements] = i8_values.view(np.uint8)
            # Store scale as float32 at offset n_elements
            scale_bytes = np.array([scale], dtype=np.float32).view(np.uint8)
            packed[n_elements:n_elements+4] = scale_bytes

            gguf_writer.add_tensor(name, packed, raw_shape=tensor.shape, raw_dtype=gguf.GGMLQuantizationType.I8_S)
            if tensor.size <= 1024:
                print(f"  • {name}: quantized to I8_S (scale={scale:.4f}, max={amax:.6f})")
        else:
            # F32 or F16 output
            if outtype == 'f32' or outtype == 'i8_s':
                # For i8_s mode, non-weight tensors stay F32
                tensor_out = tensor.astype(np.float32)
            else:
                tensor_out = tensor.astype(np.float16)
            gguf_writer.add_tensor(name, tensor_out)

        if 'bias' in name or 'norm' in name or 'gamma' in name or (outtype != 'i8_s' or 'weight' not in name):
            pass  # already added above
        total_size_bytes += tensor_out.nbytes

    print(f"  ✓ {len(converted_tensors)} tensors written\n")

    # Write file
    print("Finalizing GGUF file...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print(f"  ✓ GGUF file written\n")

    # Print summary
    print(f"{'='*80}")
    print(f"Conversion Summary")
    print(f"{'='*80}\n")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"Total tensor data: {total_size_bytes / (1024**2):.2f} MB")
    print(f"Number of tensors: {len(converted_tensors)}")
    print(f"  • Acoustic encoder: {acoustic_count} tensors")
    print(f"  • Semantic encoder: {semantic_count} tensors")
    print(f"  • Acoustic connector: {acoustic_connector_count} tensors")
    print(f"  • Semantic connector: {semantic_connector_count} tensors")
    print(f"Data type: {outtype}")
    print(f"\n✓ Conversion completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VibeVoice-ASR VAE Encoder from SafeTensors to GGUF format"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing SafeTensors model files"
    )
    parser.add_argument(
        "-o", "--outfile",
        type=Path,
        default=None,
        help="Output GGUF file path (default: model_dir/vibeasr-vae-encoder.gguf)"
    )
    parser.add_argument(
        "--outtype",
        type=str,
        default="f32",
        choices=["f32", "f16"],
        help="Output tensor data type (default: f32)"
    )

    args = parser.parse_args()

    # Validate model directory
    if not args.model_dir.is_dir():
        print(f"Error: Model directory '{args.model_dir}' not found")
        sys.exit(1)

    # Set default output path
    if args.outfile is None:
        args.outfile = args.model_dir / f"vibeasr-vae-encoder-{args.outtype}.gguf"

    # Convert
    try:
        convert_to_gguf(
            model_dir=args.model_dir,
            output_path=args.outfile,
            outtype=args.outtype,
        )
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
