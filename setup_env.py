"""
VibeASR.cpp Setup Script

Downloads pre-quantized GGUF models from HuggingFace and builds the project.

Usage:
    python setup_env.py
    python setup_env.py --hf-repo microsoft/VibeVoice-ASR-BitNet
    python setup_env.py --skip-download          # Build only
    python setup_env.py --skip-build             # Download only
"""

import subprocess
import signal
import sys
import os
import platform
import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("setup_env")

SUPPORTED_HF_MODELS = {
    "microsoft/VibeVoice-ASR-BitNet": {
        "model_name": "vibeasr",
    },
}

ARCH_ALIAS = {
    "AMD64": "x86_64",
    "x86": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}


def system_info():
    return platform.system(), ARCH_ALIAS.get(platform.machine(), platform.machine())


def run_command(command, shell=False, log_step=None):
    """Run a system command and ensure it succeeds."""
    if log_step:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / (log_step + ".log")
        with open(log_file, "w") as f:
            try:
                subprocess.run(command, shell=shell, check=True, stdout=f, stderr=f)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error: {e}, check details in {log_file}")
                sys.exit(1)
    else:
        try:
            subprocess.run(command, shell=shell, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error occurred while running command: {e}")
            sys.exit(1)


def setup_gguf():
    """Install the gguf Python package from the llama.cpp submodule."""
    gguf_py_dir = Path("3rdparty/llama.cpp/gguf-py")
    if gguf_py_dir.exists():
        logger.info("Installing gguf package from submodule...")
        run_command(
            [sys.executable, "-m", "pip", "install", str(gguf_py_dir)],
            log_step="install_gguf"
        )
    else:
        logger.warning("gguf-py not found. Did you clone with --recursive?")
        logger.warning("Run: git submodule update --init --recursive")
        sys.exit(1)


def compile_project():
    """Build the project with CMake."""
    os_name, arch = system_info()

    cmake_args = [
        "cmake", "-B", "build",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    # Use clang if available and functional
    clang_path = shutil.which("clang")
    clangpp_path = shutil.which("clang++")
    if clang_path and clangpp_path:
        # Verify clang++ can actually link (needs libstdc++)
        try:
            subprocess.run(
                [clangpp_path, "-x", "c++", "-", "-o", "/dev/null"],
                input=b"int main(){return 0;}",
                capture_output=True, check=True
            )
            cmake_args.extend([
                f"-DCMAKE_C_COMPILER={clang_path}",
                f"-DCMAKE_CXX_COMPILER={clangpp_path}",
            ])
        except (subprocess.CalledProcessError, OSError):
            logger.info("Clang found but not functional, using default compiler")

    logger.info("Configuring CMake...")
    run_command(cmake_args, log_step="cmake_configure")

    build_args = [
        "cmake", "--build", "build",
        "--config", "Release",
        "-j", str(args.threads),
    ]

    logger.info(f"Building with {args.threads} threads...")
    run_command(build_args, log_step="cmake_build")

    logger.info("Build complete! Binaries are in build/bin/")


def download_model():
    """Download pre-quantized GGUF models from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    if args.hf_repo not in SUPPORTED_HF_MODELS:
        logger.warning(f"Unknown repo: {args.hf_repo}, attempting download anyway...")
        model_name = args.hf_repo.split("/")[-1].lower()
    else:
        model_name = SUPPORTED_HF_MODELS[args.hf_repo]["model_name"]

    model_dir = Path(args.model_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading model from {args.hf_repo} to {model_dir}...")
    snapshot_download(
        repo_id=args.hf_repo,
        revision=args.hf_revision,
        local_dir=str(model_dir),
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
    )
    logger.info(f"Model downloaded to {model_dir}")


def signal_handler(sig, frame):
    print("\nInterrupted. Exiting...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.skip_build:
        setup_gguf()
        compile_project()

    if not args.skip_download:
        download_model()

    if not args.skip_build and not args.skip_download:
        model_name = SUPPORTED_HF_MODELS.get(args.hf_repo, {}).get("model_name", "vibeasr")
        model_dir = Path(args.model_dir) / model_name

        logger.info("=" * 60)
        logger.info("Setup complete! Try running:")
        logger.info("")
        logger.info(f"  ./build/bin/asr_infer \\")
        logger.info(f"      --vae-model {model_dir}/vibeasr-vae-encoder-i8_s.gguf \\")
        logger.info(f"      --lm-model {model_dir}/vibeasr-lm-i2_s-embed-q6_k.gguf \\")
        logger.info(f"      --audio <your_audio.wav> -t 4")
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VibeASR.cpp setup: build project and download models"
    )
    parser.add_argument(
        "--hf-repo", "-hr",
        type=str,
        default="microsoft/VibeVoice-ASR-BitNet",
        help="HuggingFace model repository (default: microsoft/VibeVoice-ASR-BitNet)"
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default="main",
        help="HuggingFace branch/revision to download (default: main)"
    )
    parser.add_argument(
        "--model-dir", "-md",
        type=str,
        default="models",
        help="Local directory to store models (default: models/)"
    )
    parser.add_argument(
        "--log-dir", "-ld",
        type=str,
        default="logs",
        help="Directory for build logs (default: logs/)"
    )
    parser.add_argument(
        "--threads", "-j",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of build threads (default: nproc)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download, only build"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip build, only download model"
    )

    args = parser.parse_args()
    main()
