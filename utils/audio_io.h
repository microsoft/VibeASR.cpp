/**
 * Audio I/O utilities for VibeASR.cpp
 *
 * Provides audio file loading with automatic resampling to target sample rate.
 * Supports WAV (via dr_wav.h) and MP3 (via dr_mp3.h).
 */

#ifndef AUDIO_IO_H
#define AUDIO_IO_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

namespace audio_io {

// Linear interpolation resampler (sufficient for 16kHz → 24kHz or vice versa)
static std::vector<float> resample_linear(
    const float * input,
    size_t input_len,
    int src_rate,
    int dst_rate) {

    if (src_rate == dst_rate) {
        return std::vector<float>(input, input + input_len);
    }

    double ratio = (double)src_rate / (double)dst_rate;
    size_t output_len = (size_t)((double)input_len / ratio);
    std::vector<float> output(output_len);

    for (size_t i = 0; i < output_len; i++) {
        double src_pos = (double)i * ratio;
        size_t idx = (size_t)src_pos;
        double frac = src_pos - (double)idx;

        if (idx + 1 < input_len) {
            output[i] = (float)((1.0 - frac) * input[idx] + frac * input[idx + 1]);
        } else if (idx < input_len) {
            output[i] = input[idx];
        } else {
            output[i] = 0.0f;
        }
    }

    return output;
}

// Audio normalization (RMS-based, matching Python AudioNormalizer)
// Computes: scalar = 10^(target_dB_FS/20) / (rms + eps); audio *= scalar
static void normalize_audio(float * audio, size_t n_samples, float target_dB_FS = -25.0f) {
    if (n_samples == 0) return;

    const float eps = 1e-6f;

    // Compute RMS
    double sum_sq = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        sum_sq += (double)audio[i] * (double)audio[i];
    }
    float rms = sqrtf((float)(sum_sq / n_samples));

    if (rms < eps) return;  // silence

    // Calculate scalar: target_linear / (rms + eps)
    float target_linear = powf(10.0f, target_dB_FS / 20.0f);
    float scalar = target_linear / (rms + eps);

    // Apply scalar
    for (size_t i = 0; i < n_samples; i++) {
        audio[i] *= scalar;
    }
}

struct AudioData {
    std::vector<float> samples;  // mono, float32, at target sample rate
    int sample_rate;
    float duration_sec;
};

// Load WAV file and resample to target rate
// Returns true on success
static bool load_wav(
    const std::string & path,
    int target_sample_rate,
    bool do_normalize,
    AudioData & out) {

    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), NULL)) {
        fprintf(stderr, "[audio_io] Error: Failed to open WAV file: %s\n", path.c_str());
        return false;
    }

    // Read all frames as float32
    size_t total_frames = wav.totalPCMFrameCount;
    uint32_t channels = wav.channels;
    uint32_t file_sr = wav.sampleRate;

    std::vector<float> raw(total_frames * channels);
    size_t frames_read = drwav_read_pcm_frames_f32(&wav, total_frames, raw.data());
    drwav_uninit(&wav);

    if (frames_read == 0) {
        fprintf(stderr, "[audio_io] Error: No audio data in file: %s\n", path.c_str());
        return false;
    }

    // Convert to mono (average channels)
    std::vector<float> mono(frames_read);
    if (channels == 1) {
        memcpy(mono.data(), raw.data(), frames_read * sizeof(float));
    } else {
        for (size_t i = 0; i < frames_read; i++) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < channels; c++) {
                sum += raw[i * channels + c];
            }
            mono[i] = sum / (float)channels;
        }
    }

    // Resample if needed
    if ((int)file_sr != target_sample_rate) {
        fprintf(stderr, "[audio_io] Resampling from %u Hz to %d Hz\n", file_sr, target_sample_rate);
        out.samples = resample_linear(mono.data(), mono.size(), (int)file_sr, target_sample_rate);
    } else {
        out.samples = std::move(mono);
    }

    // Normalize if requested
    if (do_normalize) {
        normalize_audio(out.samples.data(), out.samples.size());
    }

    out.sample_rate = target_sample_rate;
    out.duration_sec = (float)out.samples.size() / (float)target_sample_rate;

    fprintf(stderr, "[audio_io] Loaded: %s (%.2fs, %zu samples @ %d Hz)\n",
            path.c_str(), out.duration_sec, out.samples.size(), target_sample_rate);

    return true;
}

// Load MP3 file and resample to target rate
// Returns true on success
static bool load_mp3(
    const std::string & path,
    int target_sample_rate,
    bool do_normalize,
    AudioData & out) {

    drmp3_config config;
    drmp3_uint64 total_frames;
    float * raw = drmp3_open_file_and_read_pcm_frames_f32(
        path.c_str(), &config, &total_frames, NULL);

    if (!raw || total_frames == 0) {
        fprintf(stderr, "[audio_io] Error: Failed to open MP3 file: %s\n", path.c_str());
        if (raw) drmp3_free(raw, NULL);
        return false;
    }

    uint32_t channels = config.channels;
    uint32_t file_sr = config.sampleRate;

    // Convert to mono (average channels)
    std::vector<float> mono(total_frames);
    if (channels == 1) {
        memcpy(mono.data(), raw, total_frames * sizeof(float));
    } else {
        for (size_t i = 0; i < total_frames; i++) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < channels; c++) {
                sum += raw[i * channels + c];
            }
            mono[i] = sum / (float)channels;
        }
    }

    drmp3_free(raw, NULL);

    // Resample if needed
    if ((int)file_sr != target_sample_rate) {
        fprintf(stderr, "[audio_io] Resampling from %u Hz to %d Hz\n", file_sr, target_sample_rate);
        out.samples = resample_linear(mono.data(), mono.size(), (int)file_sr, target_sample_rate);
    } else {
        out.samples = std::move(mono);
    }

    // Normalize if requested
    if (do_normalize) {
        normalize_audio(out.samples.data(), out.samples.size());
    }

    out.sample_rate = target_sample_rate;
    out.duration_sec = (float)out.samples.size() / (float)target_sample_rate;

    fprintf(stderr, "[audio_io] Loaded: %s (%.2fs, %zu samples @ %d Hz)\n",
            path.c_str(), out.duration_sec, out.samples.size(), target_sample_rate);

    return true;
}

// Get file extension in lowercase
static std::string get_extension(const std::string & path) {
    size_t dot_pos = path.rfind('.');
    if (dot_pos == std::string::npos) return "";
    std::string ext = path.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

// Load audio file (WAV or MP3) with automatic format detection by extension
// Returns true on success
static bool load_audio(
    const std::string & path,
    int target_sample_rate,
    bool do_normalize,
    AudioData & out) {

    std::string ext = get_extension(path);

    if (ext == ".wav" || ext == ".wave") {
        return load_wav(path, target_sample_rate, do_normalize, out);
    } else if (ext == ".mp3") {
        return load_mp3(path, target_sample_rate, do_normalize, out);
    } else {
        fprintf(stderr, "[audio_io] Error: Unsupported audio format '%s' (supported: .wav, .mp3)\n",
                ext.c_str());
        return false;
    }
}

}  // namespace audio_io

#endif // AUDIO_IO_H
