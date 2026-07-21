#!/usr/bin/env python
"""
VibeASR.cpp Gradio Demo — Streaming ASR with multiple backends.
"""

import os
import re
import time
import tempfile
import subprocess
import argparse
from pathlib import Path
from typing import Tuple, List, Generator
from difflib import SequenceMatcher

import gradio as gr
import soundfile as sf
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_BIN = PROJECT_ROOT / "build" / "bin" / "asr_infer"
DEFAULT_SERVER_BIN = PROJECT_ROOT / "build" / "bin" / "asr_stream_server"

COMPRESS_RATIO = 3200
TARGET_SAMPLE_RATE = 24000
MAX_TOKENS = 4096

FIRST_CHUNK_DURATION = 5.0
DEFAULT_CHUNK_DURATION = 10.0
OVERLAP_DURATION = 2.5
MIN_SILENCE_MS = 300
SILENCE_THRESHOLD = 0.01

_runtime_config = {"bin_path": None, "server_bin_path": None}


def _set_bin_path(path: Path):
    _runtime_config["bin_path"] = path


def _set_server_bin_path(path: Path):
    _runtime_config["server_bin_path"] = path


def _get_bin_path() -> Path:
    return _runtime_config["bin_path"] or DEFAULT_BIN


def _get_server_bin_path() -> Path:
    return _runtime_config["server_bin_path"] or DEFAULT_SERVER_BIN


# ============================================================================
# ASR Server Process Manager
# ============================================================================

class ASRServerProcess:
    """Manages a persistent asr_server process for multi-chunk inference."""

    def __init__(self):
        self.process = None
        self._model_name = None
        self._n_threads = None
        self._greedy = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, model_name: str, n_threads: int, greedy: bool,
              temperature: float, top_p: float,
              hotwords: str = "", prompt_format: str = "text"):
        if (self.is_running() and self._model_name == model_name
                and self._n_threads == n_threads and self._greedy == greedy):
            if hotwords and hotwords.strip():
                self._send_command(f"CONTEXT:{hotwords.strip()}")
            return True

        self.stop()

        config = MODEL_CONFIGS[model_name]
        if config.get("backend") != "cpu":
            return False

        server_bin = str(_get_server_bin_path())
        vae_model = str(config["vae_model"])
        lm_model = str(config["lm_model"])

        if not Path(server_bin).exists():
            return False
        if not Path(vae_model).exists() or not Path(lm_model).exists():
            return False

        cmd = [
            server_bin,
            "--vae-model", vae_model,
            "--lm-model", lm_model,
            "-t", str(n_threads),
            "--max-tokens", str(MAX_TOKENS),
            "--prompt-format", prompt_format,
        ]

        if greedy:
            cmd.append("--greedy")
        else:
            cmd.extend(["--temperature", str(temperature)])
            cmd.extend(["--top-p", str(top_p)])

        if hotwords and hotwords.strip():
            cmd.extend(["--context", hotwords.strip()])

        print(f"🚀 Starting asr_server...")

        try:
            self.process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0,
            )
            ready_line = self.process.stdout.readline().decode().strip()
            if ready_line != "---READY---":
                self.stop()
                return False

            self._model_name = model_name
            self._n_threads = n_threads
            self._greedy = greedy
            print(f"✅ asr_server ready (PID: {self.process.pid})")
            return True

        except Exception as e:
            print(f"❌ Failed to start asr_server: {e}")
            self.process = None
            return False

    def process_chunk_streaming(self, audio_path: str):
        if not self.is_running():
            yield "[ERROR] Server not running"
            return

        self.process.stdin.write(f"{audio_path}\n".encode())
        self.process.stdin.flush()

        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            text = line.decode('utf-8', errors='replace')
            if text.strip() == "---END---":
                break
            if text.startswith("[ERROR]"):
                yield text.strip()
                break
            if text:
                yield text.rstrip('\n')

    def _send_command(self, cmd: str):
        if not self.is_running():
            return
        self.process.stdin.write(f"{cmd}\n".encode())
        self.process.stdin.flush()
        self.process.stdout.readline()

    def stop(self):
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.stdin.write(b"EXIT\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
            self._model_name = None

    def __del__(self):
        self.stop()


_asr_server = ASRServerProcess()

MODEL_CONFIGS = {
    "INT8+I2_S VibeASR-1.5B (CPU)": {
        "backend": "cpu",
        "vae_model": PROJECT_ROOT / "models" / "vibeasr" / "vibeasr-vae-encoder-i8_s.gguf",
        "lm_model": PROJECT_ROOT / "models" / "vibeasr" / "vibeasr-lm-i2_s-embed-q6_k.gguf",
        "prompt_format": "text",
    },
    "Nvidia Parakeet TDT 0.6B (GPU)": {
        "backend": "parakeet",
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
    },
    "OpenAI Whisper Large V3 (GPU)": {
        "backend": "whisper",
        "model_id": "openai/whisper-large-v3",
    },
}

_gpu_models = {}


def get_audio_duration(audio_path: str) -> float:
    try:
        return sf.info(audio_path).duration
    except Exception:
        return 0.0


# ============================================================================
# GPU Model Inference
# ============================================================================

def _load_whisper_model(model_id: str):
    if "whisper" not in _gpu_models:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)

        _gpu_models["whisper"] = {
            "model": model, "processor": processor,
            "device": device, "torch_dtype": torch_dtype,
        }
    return _gpu_models["whisper"]


def _load_parakeet_model(model_id: str):
    if "parakeet" not in _gpu_models:
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        model.eval()
        _gpu_models["parakeet"] = model
    return _gpu_models["parakeet"]


def run_whisper_inference(audio_path: str, model_id: str) -> Tuple[str, float]:
    """Returns (text, rtf)."""
    import torch
    import librosa

    ctx = _load_whisper_model(model_id)
    audio_duration = get_audio_duration(audio_path)
    audio_array, _ = librosa.load(audio_path, sr=16000)

    input_features = ctx["processor"](
        audio_array, sampling_rate=16000, return_tensors="pt",
    ).input_features.to(ctx["device"], dtype=ctx["torch_dtype"])

    t0 = time.time()
    with torch.no_grad():
        predicted_ids = ctx["model"].generate(
            input_features, language="en", return_timestamps=True,
        )
    inference_time = time.time() - t0

    transcription = ctx["processor"].batch_decode(predicted_ids, skip_special_tokens=True)
    text = transcription[0] if transcription else ""
    rtf = inference_time / audio_duration if audio_duration > 0 else 0.0
    return text, rtf


def run_parakeet_inference(audio_path: str, model_id: str) -> Tuple[str, float]:
    """Returns (text, rtf)."""
    model = _load_parakeet_model(model_id)
    audio_duration = get_audio_duration(audio_path)

    t0 = time.time()
    transcriptions = model.transcribe([audio_path])
    inference_time = time.time() - t0

    if isinstance(transcriptions, list) and len(transcriptions) > 0:
        if isinstance(transcriptions[0], str):
            text = transcriptions[0]
        else:
            text = transcriptions[0].text if hasattr(transcriptions[0], 'text') else str(transcriptions[0])
    else:
        text = str(transcriptions)

    rtf = inference_time / audio_duration if audio_duration > 0 else 0.0
    return text, rtf


# ============================================================================
# CPU Inference
# ============================================================================

def parse_stderr_timing(stderr_text: str) -> dict:
    timing = {}
    patterns = {
        "audio_duration_ms": r"Audio duration:\s+([\d.]+)\s*ms",
        "rtf": r"RTF:\s+([\d.]+)",
        "total_inference_ms": r"Total inference:\s+([\d.]+)\s*ms",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stderr_text)
        if match:
            timing[key] = float(match.group(1))

    compact_match = re.search(r"Audio:\s+([\d.]+)s\s*\|\s*Tokens:\s*(\d+)\s*\|\s*RTF:\s*([\d.]+)", stderr_text)
    if compact_match:
        if "audio_duration_ms" not in timing:
            timing["audio_duration_ms"] = float(compact_match.group(1)) * 1000.0
        if "rtf" not in timing:
            timing["rtf"] = float(compact_match.group(3))

    return timing


def run_asr_inference(
    audio_path: str, model_name: str, n_threads: int,
    greedy: bool, temperature: float, top_p: float,
    bin_path: str, hotwords: str = "",
) -> Tuple[str, str, dict]:
    """Run asr_infer binary. Returns (stdout, stderr, timing_dict)."""
    config = MODEL_CONFIGS[model_name]
    vae_model = str(config["vae_model"])
    lm_model = str(config["lm_model"])

    if not Path(vae_model).exists():
        return "", f"❌ VAE model not found: {vae_model}", {}
    if not Path(lm_model).exists():
        return "", f"❌ LM model not found: {lm_model}", {}
    if not Path(bin_path).exists():
        return "", f"❌ asr_infer binary not found: {bin_path}", {}

    cmd = [
        str(bin_path),
        "--vae-model", vae_model,
        "--lm-model", lm_model,
        "--audio", audio_path,
        "-t", str(n_threads),
        "--max-tokens", str(MAX_TOKENS),
    ]

    if greedy:
        cmd.append("--greedy")
    else:
        cmd.extend(["--temperature", str(temperature)])
        cmd.extend(["--top-p", str(top_p)])

    if hotwords and hotwords.strip():
        cmd.extend(["--context", hotwords.strip()])

    prompt_format = config.get("prompt_format", "text")
    cmd.extend(["--prompt-format", prompt_format])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        timing = parse_stderr_timing(stderr)
        return stdout, stderr, timing
    except subprocess.TimeoutExpired:
        return "", "❌ Inference timed out (>10 minutes)", {}
    except Exception as e:
        return "", f"❌ Error: {str(e)}", {}


# ============================================================================
# Streaming Transcription
# ============================================================================

def transcribe_streaming(
    audio_input, model_name: str, n_threads: int,
    greedy: bool, temperature: float, top_p: float,
    hotwords: str = "", mode: str = "Online",
) -> Generator[str, None, None]:
    """Streaming transcription generator. Yields transcription text progressively."""
    if audio_input is None:
        yield "❌ Please provide audio input!"
        return

    # Handle audio input
    if isinstance(audio_input, str):
        audio_path = audio_input
    elif isinstance(audio_input, tuple):
        sample_rate, audio_data = audio_input
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_path = temp_file.name
        temp_file.close()
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        sf.write(audio_path, audio_data, sample_rate, subtype='PCM_16')
    else:
        yield "❌ Invalid audio input format!"
        return

    # Convert to standard PCM WAV
    try:
        audio_data, sample_rate = sf.read(audio_path)
        wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_path = wav_tmp.name
        wav_tmp.close()
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        sf.write(wav_path, audio_data, sample_rate, subtype='PCM_16')
        audio_path = wav_path
    except Exception as e:
        yield f"❌ Failed to convert audio: {str(e)}"
        return

    total_audio_duration = get_audio_duration(audio_path)
    config = MODEL_CONFIGS[model_name]
    backend = config.get("backend", "cpu")

    # GPU backends
    if backend == "whisper":
        try:
            yield "🔄 Loading Whisper model..."
            text, rtf = run_whisper_inference(audio_path, config["model_id"])
            yield f"{text}\n\n---\n📊 RTF: {rtf:.4f} | Audio: {total_audio_duration:.1f}s"
        except Exception as e:
            yield f"❌ Whisper error: {str(e)}"
        return

    elif backend == "parakeet":
        try:
            yield "🔄 Loading Parakeet model..."
            text, rtf = run_parakeet_inference(audio_path, config["model_id"])
            yield f"{text}\n\n---\n📊 RTF: {rtf:.4f} | Audio: {total_audio_duration:.1f}s"
        except Exception as e:
            yield f"❌ Parakeet error: {str(e)}"
        return

    elif backend != "cpu":
        yield f"❌ Unknown backend: {backend}"
        return

    # VibeASR.cpp CPU inference
    bin_path = str(_get_bin_path())
    prompt_format = config.get("prompt_format", "text")

    streaming_enabled = (mode == "Online")
    if not streaming_enabled or total_audio_duration <= FIRST_CHUNK_DURATION + 2.0:
        # Offline: process entire audio at once
        stdout, stderr, timing = run_asr_inference(
            audio_path=audio_path, model_name=model_name,
            n_threads=n_threads, greedy=greedy,
            temperature=temperature, top_p=top_p,
            bin_path=bin_path, hotwords=hotwords,
        )

        if not stdout and "❌" in stderr:
            yield stderr
            return
        if not stdout and not timing:
            yield "❌ Inference produced no output."
            return

        rtf = timing.get("rtf", 0.0)
        transcription = stdout if stdout else "(No output)"
        yield f"{transcription}\n\n---\n📊 RTF: {rtf:.4f} | Audio: {total_audio_duration:.1f}s"
        return

    # Online streaming mode
    DRAFT_DUR = FIRST_CHUNK_DURATION
    CHUNK_DUR = DEFAULT_CHUNK_DURATION
    OVERLAP_DUR = OVERLAP_DURATION

    use_server = _get_server_bin_path().exists()
    if use_server:
        server_started = _asr_server.start(
            model_name=model_name, n_threads=n_threads, greedy=greedy,
            temperature=temperature, top_p=top_p,
            hotwords=hotwords, prompt_format=prompt_format,
        )
        if not server_started:
            use_server = False

    audio_data_full, sr = sf.read(audio_path)
    if len(audio_data_full.shape) > 1:
        audio_data_full = audio_data_full.mean(axis=1)
    if sr != TARGET_SAMPLE_RATE:
        try:
            import librosa
            audio_data_full = librosa.resample(audio_data_full, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        except ImportError:
            ratio = TARGET_SAMPLE_RATE / sr
            new_len = int(len(audio_data_full) * ratio)
            audio_data_full = np.interp(
                np.linspace(0, len(audio_data_full) - 1, new_len),
                np.arange(len(audio_data_full)), audio_data_full)

    total_samples = len(audio_data_full)
    sample_rate = TARGET_SAMPLE_RATE
    sim_start = time.time()

    def _find_silence_boundary(target_s, search_range=1.0):
        window_samples = int(MIN_SILENCE_MS / 1000.0 * sample_rate)
        search_samples = int(search_range * sample_rate)
        target_sample = int(target_s * sample_rate)

        best_pos = target_sample
        best_energy = float('inf')
        start = max(0, target_sample - search_samples)
        end = min(total_samples, target_sample + search_samples)

        for pos in range(start, end - window_samples, window_samples // 2):
            window = audio_data_full[pos:pos + window_samples]
            rms = np.sqrt(np.mean(window ** 2))
            if rms < best_energy:
                best_energy = rms
                best_pos = pos + window_samples // 2

        if best_energy < SILENCE_THRESHOLD:
            return best_pos / sample_rate
        return target_s

    def _write_chunk(start_s, end_s):
        s = int(start_s * sample_rate)
        e = min(int(end_s * sample_rate), total_samples)
        chunk = audio_data_full[s:e]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, chunk, sample_rate, subtype='PCM_16')
        tmp.close()
        return tmp.name

    def _tokenize_for_display(text):
        """Split text: English words as whole units, CJK chars individually."""
        tokens = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch.isascii() and (ch.isalpha() or ch == "'"):
                word_start = i
                while i < len(text) and text[i].isascii() and (text[i].isalpha() or text[i] == "'"):
                    i += 1
                tokens.append(text[word_start:i])
            else:
                tokens.append(ch)
                i += 1
        return tokens

    def _infer_streaming(chunk_path):
        if use_server:
            for t in _asr_server.process_chunk_streaming(chunk_path):
                if t.startswith("[ERROR]"):
                    break
                yield t
        else:
            stdout, _, _ = run_asr_inference(
                audio_path=chunk_path, model_name=model_name,
                n_threads=n_threads, greedy=greedy,
                temperature=temperature, top_p=top_p,
                bin_path=bin_path, hotwords=hotwords,
            )
            if stdout:
                for token in _tokenize_for_display(stdout.strip()):
                    yield token
        try:
            os.unlink(chunk_path)
        except OSError:
            pass

    def _infer(chunk_path):
        return "".join(_infer_streaming(chunk_path))

    def _lcs_dedup(prev_text, new_text, overlap_ratio):
        if not prev_text or not new_text or overlap_ratio <= 0:
            return new_text
        tail_len = min(len(prev_text), int(len(new_text) * overlap_ratio) + 20)
        prev_tail = prev_text[-tail_len:]
        head_len = int(len(new_text) * overlap_ratio) + 20
        new_head = new_text[:head_len]

        matcher = SequenceMatcher(None, prev_tail, new_head)
        match = matcher.find_longest_match(0, len(prev_tail), 0, len(new_head))

        if match.size >= 3:
            return new_text[match.b + match.size:]
        return new_text[int(len(new_text) * overlap_ratio):]

    def _wait_until(target_time_s):
        while True:
            elapsed = time.time() - sim_start
            if elapsed >= target_time_s:
                break
            time.sleep(min(target_time_s - elapsed, 0.1))

    def _typewriter_yield(base_text, new_text, chunk_dur):
        if not new_text:
            return
        display_tokens = _tokenize_for_display(new_text)
        total_tokens = len(display_tokens)
        if total_tokens == 0:
            return

        target_interval = 0.15
        num_yields = max(1, int(chunk_dur / target_interval))
        batch = max(1, total_tokens // num_yields)

        for i in range(0, total_tokens, batch):
            accumulated = "".join(display_tokens[:i + batch])
            yield base_text + accumulated
            time.sleep(target_interval)

    accumulated_text = ""
    all_timings = []

    yield "🔄 Processing..."

    # Phase 1: Draft
    draft_end = min(DRAFT_DUR, total_audio_duration)
    _wait_until(draft_end)
    draft_path = _write_chunk(0, draft_end)
    t0 = time.time()
    draft_text = _infer(draft_path)
    draft_time = time.time() - t0
    all_timings.append(draft_time / draft_end)

    if draft_text:
        for text in _typewriter_yield("", draft_text, draft_end):
            yield text
        accumulated_text = draft_text
        yield accumulated_text

    # Phase 2: Refine (first full chunk replaces draft)
    refine_end = min(CHUNK_DUR, total_audio_duration)
    if refine_end > draft_end:
        _wait_until(refine_end)
        refine_path = _write_chunk(0, refine_end)
        t0 = time.time()
        refine_text = _infer(refine_path)
        refine_time = time.time() - t0
        all_timings.append(refine_time / refine_end)

        if refine_text:
            if len(refine_text) > len(draft_text):
                new_part = refine_text[len(draft_text):]
                for text in _typewriter_yield(draft_text, new_part, refine_end - draft_end):
                    yield text
            accumulated_text = refine_text
            yield accumulated_text

    # Phase 3: Subsequent chunks with VAD + overlap + LCS
    pos = refine_end
    while pos < total_audio_duration:
        raw_end = min(pos + CHUNK_DUR, total_audio_duration)
        if raw_end < total_audio_duration:
            chunk_end = _find_silence_boundary(raw_end)
        else:
            chunk_end = raw_end
        _wait_until(chunk_end)

        overlap_start = max(0, pos - OVERLAP_DUR)
        chunk_path = _write_chunk(overlap_start, chunk_end)
        t0 = time.time()
        chunk_text = _infer(chunk_path)
        chunk_time = time.time() - t0
        chunk_dur = chunk_end - pos
        all_timings.append(chunk_time / chunk_dur if chunk_dur > 0 else 0)

        if chunk_text and prompt_format == "text":
            overlap_ratio = OVERLAP_DUR / (chunk_end - overlap_start) if (chunk_end - overlap_start) > 0 else 0
            new_text = _lcs_dedup(accumulated_text, chunk_text, overlap_ratio)
            if chunk_end < total_audio_duration and new_text and new_text[-1] in '.。!！?？;；':
                new_text = new_text.rstrip('.。!！?？;；')
            for text in _typewriter_yield(accumulated_text, new_text, chunk_dur):
                yield text
            accumulated_text += new_text
        elif chunk_text:
            accumulated_text = _merge_json_outputs(accumulated_text, chunk_text)

        yield accumulated_text
        pos = chunk_end

    yield accumulated_text


def _merge_json_outputs(existing: str, new_chunk: str) -> str:
    existing = existing.strip()
    new_chunk = new_chunk.strip()
    if (existing.startswith("[") and existing.endswith("]") and
            new_chunk.startswith("[") and new_chunk.endswith("]")):
        return existing[:-1] + "," + new_chunk[1:]
    return existing + "\n" + new_chunk


# ============================================================================
# Gradio Interface
# ============================================================================

def create_gradio_interface():
    with gr.Blocks(title="VibeASR.cpp Demo") as demo:
        gr.Markdown("# 🎙️ VibeASR.cpp Demo")

        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value=list(MODEL_CONFIGS.keys())[0],
                    label="Model",
                )

                n_threads_slider = gr.Slider(
                    minimum=1, maximum=16, value=4, step=1,
                    label="Threads",
                )

                mode_radio = gr.Radio(
                    choices=["Online", "Offline"],
                    value="Offline",
                    label="Mode",
                )

                greedy_checkbox = gr.Checkbox(
                    value=False, label="Greedy Decoding",
                )

                with gr.Column(visible=True) as sampling_params:
                    temperature_slider = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                        label="Temperature",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                        label="Top-p",
                    )

            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="Audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                    format="wav",
                )

                hotwords_input = gr.Textbox(
                    label="Hotwords (Optional)",
                    placeholder="One per line, e.g.:\nVibeVoice\nMicrosoft",
                    lines=2, max_lines=5,
                )

                transcribe_button = gr.Button(
                    "🚀 Transcribe", variant="primary", size="lg"
                )

                transcription_output = gr.Textbox(
                    label="Transcription",
                    lines=15, max_lines=50,
                    autoscroll=True, interactive=False,
                )

        # Event handlers
        greedy_checkbox.change(
            fn=lambda x: gr.update(visible=not x),
            inputs=[greedy_checkbox],
            outputs=[sampling_params],
        )

        transcribe_button.click(
            fn=transcribe_streaming,
            inputs=[
                audio_input, model_selector, n_threads_slider,
                greedy_checkbox, temperature_slider, top_p_slider,
                hotwords_input, mode_radio,
            ],
            outputs=[transcription_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="VibeASR.cpp Gradio Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--bin", type=str, default=None)
    parser.add_argument("--server-bin", type=str, default=None)
    args = parser.parse_args()

    bin_path = Path(args.bin) if args.bin else DEFAULT_BIN
    _set_bin_path(bin_path)
    server_bin_path = Path(args.server_bin) if args.server_bin else DEFAULT_SERVER_BIN
    _set_server_bin_path(server_bin_path)

    effective_bin = _get_bin_path()
    effective_server = _get_server_bin_path()

    if not effective_bin.exists():
        print(f"⚠️  asr_infer not found: {effective_bin}")
    if effective_server.exists():
        print(f"✅ asr_server found: {effective_server}")
    else:
        print(f"⚠️  asr_server not found, falling back to asr_infer")

    print(f"\n🚀 Starting on http://{args.host}:{args.port}")
    print(f"   Streaming: draft={FIRST_CHUNK_DURATION}s, chunk={DEFAULT_CHUNK_DURATION}s\n")

    demo = create_gradio_interface()
    demo.launch(
        server_name=args.host, server_port=args.port,
        share=args.share, show_error=True,
    )


if __name__ == "__main__":
    main()
