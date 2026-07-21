/**
 * VibeASR.cpp - Standalone ASR Inference
 *
 * Integrates VAE encoder (acoustic + semantic) and LM (Qwen2.5-1.5B BitNet)
 * into a single executable for end-to-end speech recognition.
 *
 * Usage:
 *   ./asr_infer --vae-model <vae.gguf> --lm-model <lm.gguf> --audio <wav> [options]
 */

#include "vae.h"
#include "llama.h"
#include "ggml.h"

#include "../utils/audio_io.h"
#include "../utils/prompt_builder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

//
// Configuration
//

struct asr_params {
    std::string vae_model_path;
    std::string lm_model_path;
    std::string audio_path;
    std::string context_info;      // Hotwords/context info for improved recognition
    std::string prompt_format = "text";  // "text" for plain text, "json" for JSON with keys

    int n_threads       = 4;
    int n_ctx           = 16384;
    int n_batch         = 2048;
    int max_tokens      = 16384;
    int target_sr       = 24000;   // Target sample rate for VAE
    int compress_ratio  = 3200;    // Speech token compression ratio

    bool greedy         = false;
    bool normalize      = true;
    float temperature   = 0.7f;
    float top_p         = 0.9f;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --vae-model <path>   Path to VAE encoder GGUF model (required)\n");
    fprintf(stderr, "  --lm-model <path>    Path to LM GGUF model (required)\n");
    fprintf(stderr, "  --audio <path>       Path to input WAV file (required)\n");
    fprintf(stderr, "  -t <n>               Number of threads (default: 4)\n");
    fprintf(stderr, "  -c <n>               Context size (default: 16384)\n");
    fprintf(stderr, "  -b <n>               Batch size (default: 2048)\n");
    fprintf(stderr, "  --max-tokens <n>     Maximum tokens to generate (default: 16384)\n");
    fprintf(stderr, "  --greedy             Use greedy decoding (default: sampling)\n");
    fprintf(stderr, "  --temperature <f>    Sampling temperature (default: 0.7)\n");
    fprintf(stderr, "  --top-p <f>          Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  --sample-rate <n>    Target sample rate (default: 24000)\n");
    fprintf(stderr, "  --compress-ratio <n> Speech compression ratio (default: 3200)\n");
    fprintf(stderr, "  --context <text>     Hotwords/context info to improve recognition accuracy\n");
    fprintf(stderr, "  --prompt-format <s>  Prompt format: 'text' (plain text) or 'json' (with keys) (default: text)\n");
    fprintf(stderr, "  --no-normalize       Disable audio normalization\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s --vae-model models/vibeasr-vae-encoder-i8_s.gguf \\\n", prog);
    fprintf(stderr, "     --lm-model models/vibeasr-lm-i2_s.gguf \\\n");
    fprintf(stderr, "     --audio eval/audio/podcast_en_0.wav -t 4 --greedy\n");
}

static bool parse_args(int argc, char ** argv, asr_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--vae-model" && i + 1 < argc) {
            params.vae_model_path = argv[++i];
        } else if (arg == "--lm-model" && i + 1 < argc) {
            params.lm_model_path = argv[++i];
        } else if (arg == "--audio" && i + 1 < argc) {
            params.audio_path = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            params.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "-b" && i + 1 < argc) {
            params.n_batch = std::stoi(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            params.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--greedy") {
            params.greedy = true;
        } else if (arg == "--temperature" && i + 1 < argc) {
            params.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            params.top_p = std::stof(argv[++i]);
        } else if (arg == "--sample-rate" && i + 1 < argc) {
            params.target_sr = std::stoi(argv[++i]);
        } else if (arg == "--compress-ratio" && i + 1 < argc) {
            params.compress_ratio = std::stoi(argv[++i]);
        } else if (arg == "--context" && i + 1 < argc) {
            params.context_info = argv[++i];
        } else if (arg == "--prompt-format" && i + 1 < argc) {
            params.prompt_format = argv[++i];
        } else if (arg == "--no-normalize") {
            params.normalize = false;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Error: Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (params.vae_model_path.empty() || params.lm_model_path.empty() || params.audio_path.empty()) {
        fprintf(stderr, "Error: --vae-model, --lm-model, and --audio are required\n\n");
        print_usage(argv[0]);
        return false;
    }

    return true;
}

//
// Timing utilities
//

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

//
// Main inference pipeline
//

int main(int argc, char ** argv) {
    // Parse arguments
    asr_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    fprintf(stderr, "========================================\n");
    fprintf(stderr, " VibeASR.cpp - Standalone ASR Inference\n");
    fprintf(stderr, "========================================\n\n");
    fprintf(stderr, "Configuration:\n");
    fprintf(stderr, "  VAE model:  %s\n", params.vae_model_path.c_str());
    fprintf(stderr, "  LM model:   %s\n", params.lm_model_path.c_str());
    fprintf(stderr, "  Audio:      %s\n", params.audio_path.c_str());
    fprintf(stderr, "  Threads:    %d\n", params.n_threads);
    fprintf(stderr, "  Context:    %d\n", params.n_ctx);
    fprintf(stderr, "  Max tokens: %d\n", params.max_tokens);
    fprintf(stderr, "  Decoding:   %s\n", params.greedy ? "greedy" : "sampling");
    fprintf(stderr, "\n");

    double total_start = get_time_ms();

    // ========================================
    // Step 1: Load audio
    // ========================================
    fprintf(stderr, "[Step 1] Loading audio...\n");
    double t0 = get_time_ms();

    audio_io::AudioData audio;
    if (!audio_io::load_wav(params.audio_path, params.target_sr, params.normalize, audio)) {
        fprintf(stderr, "Error: Failed to load audio file\n");
        return 1;
    }

    double audio_load_time = get_time_ms() - t0;
    fprintf(stderr, "  Audio loaded: %.2fs duration, %zu samples\n\n",
            audio.duration_sec, audio.samples.size());

    // ========================================
    // Step 2: Load VAE model
    // ========================================
    fprintf(stderr, "[Step 2] Loading VAE model...\n");
    t0 = get_time_ms();

    struct vae_model_params vae_mparams = vae_model_default_params();
    vae_mparams.n_threads = params.n_threads;

    vae_model_t * vae_model = vae_load_model_from_file(params.vae_model_path.c_str(), vae_mparams);
    if (!vae_model) {
        fprintf(stderr, "Error: Failed to load VAE model\n");
        return 1;
    }

    struct vae_context_params vae_cparams = vae_context_default_params();
    vae_cparams.n_threads = params.n_threads;

    vae_context_t * vae_ctx = vae_new_context_with_model(vae_model, vae_cparams);
    if (!vae_ctx) {
        fprintf(stderr, "Error: Failed to create VAE context\n");
        vae_free_model(vae_model);
        return 1;
    }

    int acoustic_dim = vae_model_acoustic_dim(vae_model);
    int semantic_dim = vae_model_semantic_dim(vae_model);
    fprintf(stderr, "  VAE loaded: acoustic_dim=%d, semantic_dim=%d\n\n",
            acoustic_dim, semantic_dim);

    double vae_load_time = get_time_ms() - t0;

    // ========================================
    // Step 3: Load LM model
    // ========================================
    fprintf(stderr, "[Step 3] Loading LM model...\n");
    t0 = get_time_ms();

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    llama_log_set([](enum ggml_log_level, const char *, void *){}, nullptr);

    llama_model_params lm_mparams = llama_model_default_params();
    lm_mparams.n_gpu_layers = 0;  // CPU only

    llama_model * lm_model = llama_load_model_from_file(params.lm_model_path.c_str(), lm_mparams);
    if (!lm_model) {
        fprintf(stderr, "Error: Failed to load LM model\n");
        vae_free(vae_ctx);
        vae_free_model(vae_model);
        return 1;
    }

    llama_context_params lm_cparams = llama_context_default_params();
    lm_cparams.n_ctx = params.n_ctx;
    lm_cparams.n_batch = params.n_batch;
    lm_cparams.n_threads = params.n_threads;
    lm_cparams.n_threads_batch = params.n_threads;

    llama_context * lm_ctx = llama_new_context_with_model(lm_model, lm_cparams);
    if (!lm_ctx) {
        fprintf(stderr, "Error: Failed to create LM context\n");
        llama_free_model(lm_model);
        vae_free(vae_ctx);
        vae_free_model(vae_model);
        return 1;
    }

    int n_embd = llama_n_embd(lm_model);
    int n_vocab = llama_n_vocab(lm_model);
    fprintf(stderr, "  LM loaded: n_embd=%d, n_vocab=%d\n\n", n_embd, n_vocab);

    double lm_load_time = get_time_ms() - t0;

    // ========================================
    // Step 4: VAE Acoustic Encode
    // ========================================
    fprintf(stderr, "[Step 4] VAE acoustic encoding...\n");

    int32_t n_samples = (int32_t)audio.samples.size();
    int32_t expected_frames = (n_samples + params.compress_ratio - 1) / params.compress_ratio;

    std::vector<float> acoustic_features(expected_frames * acoustic_dim);
    float acoustic_time_ms = 0.0f;

    int32_t acoustic_frames = vae_encode_acoustic_with_timing(
        vae_ctx, audio.samples.data(), n_samples,
        acoustic_features.data(), &acoustic_time_ms);

    if (acoustic_frames < 0) {
        fprintf(stderr, "Error: VAE acoustic encoding failed\n");
        goto cleanup;
    }

    acoustic_features.resize(acoustic_frames * acoustic_dim);
    fprintf(stderr, "  Acoustic: %d frames, dim=%d, time=%.1fms\n",
            acoustic_frames, acoustic_dim, acoustic_time_ms);

    // ========================================
    // Step 5: VAE Semantic Encode
    // ========================================
    {
        fprintf(stderr, "[Step 5] VAE semantic encoding...\n");

        std::vector<float> semantic_features(expected_frames * semantic_dim);
        float semantic_time_ms = 0.0f;

        int32_t semantic_frames = vae_encode_semantic_with_timing(
            vae_ctx, audio.samples.data(), n_samples,
            semantic_features.data(), &semantic_time_ms);

        if (semantic_frames < 0) {
            fprintf(stderr, "Error: VAE semantic encoding failed\n");
            goto cleanup;
        }

        semantic_features.resize(semantic_frames * semantic_dim);
        fprintf(stderr, "  Semantic: %d frames, dim=%d, time=%.1fms\n\n",
                semantic_frames, semantic_dim, semantic_time_ms);

        // Use the minimum frame count
        int n_frames = std::min(acoustic_frames, semantic_frames);

        // ========================================
        // Step 6: Build prompt tokens
        // ========================================
        fprintf(stderr, "[Step 6] Building prompt...\n");
        t0 = get_time_ms();

        prompt_builder::PromptTokens prompt = prompt_builder::build_prompt(
            lm_model, n_samples, params.compress_ratio, audio.duration_sec,
            params.context_info, params.prompt_format);

        if (prompt.tokens.empty()) {
            fprintf(stderr, "Error: Failed to build prompt\n");
            goto cleanup;
        }

        double prompt_build_time = get_time_ms() - t0;

        // ========================================
        // Step 7: LM Prefill (segmented: token ID + embedding + token ID)
        // ========================================
        fprintf(stderr, "[Step 7] LM prefill (segmented)...\n");
        t0 = get_time_ms();

        int n_prompt_tokens = (int)prompt.tokens.size();

        // Clear KV cache
        llama_kv_cache_clear(lm_ctx);

        // Segmented prefill: text tokens use token ID mode (llama.cpp handles
        // quantized embedding dequantize), speech region uses embedding mode
        {
            int pos = prompt_builder::prefill_segmented(
                lm_model, lm_ctx, prompt,
                acoustic_features.data(), acoustic_dim,
                semantic_features.data(), semantic_dim,
                n_frames, params.n_batch);

            if (pos < 0) {
                fprintf(stderr, "Error: Segmented prefill failed\n");
                goto cleanup;
            }
        }

        double prefill_time = get_time_ms() - t0;
        fprintf(stderr, "  Prefill: %d tokens, time=%.1fms (%.1f tok/s)\n\n",
                n_prompt_tokens, prefill_time,
                n_prompt_tokens / (prefill_time / 1000.0));

        // ========================================
        // Step 8: Autoregressive decode
        // ========================================
        fprintf(stderr, "[Step 8] Decoding...\n");
        t0 = get_time_ms();

        // Set up sampler
        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());

        if (params.greedy) {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temperature));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));
        }

        std::vector<llama_token> generated_tokens;
        int cur_pos = n_prompt_tokens;
        int n_decoded = 0;

        // Sample first token from prefill logits
        llama_token new_token = llama_sampler_sample(smpl, lm_ctx, -1);
        llama_sampler_accept(smpl, new_token);

        generated_tokens.push_back(new_token);
        n_decoded++;

        // EOG token IDs (canonical from Python tokenizer)
        // The GGUF metadata may have incorrect EOS ID due to vocab mapping issues
        const llama_token EOG_IM_END = 151645;       // <|im_end|>
        const llama_token EOG_ENDOFTEXT = 151643;    // <|endoftext|>

        // Decode loop
        while (n_decoded < params.max_tokens) {
            if (new_token == EOG_IM_END || new_token == EOG_ENDOFTEXT) {
                break;
            }

            llama_batch batch = llama_batch_get_one(&new_token, 1, cur_pos, 0);
            int ret = llama_decode(lm_ctx, batch);
            if (ret != 0) {
                fprintf(stderr, "Error: llama_decode failed (ret=%d)\n", ret);
                break;
            }

            cur_pos++;
            new_token = llama_sampler_sample(smpl, lm_ctx, -1);
            llama_sampler_accept(smpl, new_token);
            generated_tokens.push_back(new_token);
            n_decoded++;
        }

        double decode_time = get_time_ms() - t0;
        double per_token_ms = (n_decoded > 0) ? decode_time / n_decoded : 0.0;

        fprintf(stderr, "  Decode: %d tokens, time=%.1fms (%.2f ms/tok, %.1f tok/s)\n\n",
                n_decoded, decode_time, per_token_ms,
                (n_decoded > 0) ? 1000.0 / per_token_ms : 0.0);

        llama_sampler_free(smpl);

        // ========================================
        // Step 9: Detokenize and output
        // ========================================
        fprintf(stderr, "[Step 9] Post-processing output...\n\n");

        // Skip the assistant header tokens that the model generates itself
        // (since we don't include generation prompt, model generates <|im_start|>assistant\n first)
        // These are special tokens that won't appear in detokenize(special=false) output,
        // but "assistant\n" (token 77091 + 198) would. We skip all tokens before the
        // actual content by finding where the assistant header ends.
        int content_start = 0;
        {
            // Look for the pattern: <|im_start|>(151644) + "assistant"(77091) + "\n"(198)
            // Skip these tokens from the generated output
            llama_token im_start_id = 151644;  // Canonical ID from Python tokenizer
            if (generated_tokens.size() >= 3 &&
                generated_tokens[0] == im_start_id) {
                // Skip <|im_start|>assistant\n (typically 3 tokens)
                // Find where the header ends (first non-header token after im_start)
                content_start = 1;  // skip <|im_start|>
                // Skip "assistant" token
                std::string piece = prompt_builder::token_to_piece(lm_model, generated_tokens[1], true);
                if (piece == "assistant") {
                    content_start = 2;
                    // Skip "\n" token
                    if (content_start < (int)generated_tokens.size()) {
                        piece = prompt_builder::token_to_piece(lm_model, generated_tokens[content_start], true);
                        if (piece == "\n") {
                            content_start = 3;
                        }
                    }
                }
                fprintf(stderr, "  Skipped %d assistant header tokens\n", content_start);
            }
        }

        // Remove trailing <|im_end|> or <|endoftext|> if present
        int content_end = (int)generated_tokens.size();
        if (content_end > content_start) {
            llama_token last_tok = generated_tokens[content_end - 1];
            if (last_tok == EOG_IM_END || last_tok == EOG_ENDOFTEXT) {
                content_end--;
            }
        }

        std::vector<llama_token> content_tokens(
            generated_tokens.begin() + content_start,
            generated_tokens.begin() + content_end);
        std::string output_text = prompt_builder::detokenize(lm_model, content_tokens, false);

        double total_time = get_time_ms() - total_start;
        double rtf = (total_time / 1000.0) / audio.duration_sec;

        // Print timing summary
        fprintf(stderr, "========================================\n");
        fprintf(stderr, " Timing Summary\n");
        fprintf(stderr, "========================================\n");
        fprintf(stderr, "  Audio loading:        %8.1f ms\n", audio_load_time);
        fprintf(stderr, "  VAE model loading:    %8.1f ms\n", vae_load_time);
        fprintf(stderr, "  LM model loading:     %8.1f ms\n", lm_load_time);
        fprintf(stderr, "  VAE acoustic encode:  %8.1f ms\n", (double)acoustic_time_ms);
        fprintf(stderr, "  VAE semantic encode:  %8.1f ms\n", (double)semantic_time_ms);
        fprintf(stderr, "  Prompt build:         %8.1f ms\n", prompt_build_time);
        fprintf(stderr, "  LM prefill:           %8.1f ms (%d tokens)\n", prefill_time, n_prompt_tokens);
        fprintf(stderr, "  LM decode:            %8.1f ms (%d tokens, %.2f ms/tok)\n",
                decode_time, n_decoded, per_token_ms);
        fprintf(stderr, "  ─────────────────────────────────────\n");
        fprintf(stderr, "  Total inference:      %8.1f ms\n", total_time);
        fprintf(stderr, "  Audio duration:       %8.1f ms\n", audio.duration_sec * 1000.0);
        fprintf(stderr, "  RTF:                  %8.4f\n", rtf);
        fprintf(stderr, "========================================\n\n");

        // Parse and display transcription segments
        std::string trimmed = output_text;
        {
            size_t s = trimmed.find_first_not_of(" \t\n\r");
            size_t e = trimmed.find_last_not_of(" \t\n\r");
            if (s != std::string::npos && e != std::string::npos)
                trimmed = trimmed.substr(s, e - s + 1);
        }

        // Print formatted transcription to stdout
        fprintf(stdout, "\n");
        if (!trimmed.empty() && trimmed[0] == '[') {
            // Parse simple JSON array of segments: [{"Start":..,"End":..,"Speaker":..,"Content":".."},...]
            size_t pos = 0;
            while ((pos = trimmed.find("{\"", pos)) != std::string::npos) {
                // Extract fields using simple string search
                auto extract_num = [&](const char * key) -> std::string {
                    std::string k = std::string("\"") + key + "\":";
                    size_t kp = trimmed.find(k, pos);
                    if (kp == std::string::npos) return "";
                    size_t vstart = kp + k.size();
                    size_t vend = trimmed.find_first_of(",}", vstart);
                    return trimmed.substr(vstart, vend - vstart);
                };
                auto extract_str = [&](const char * key) -> std::string {
                    std::string k = std::string("\"") + key + "\":\"";
                    size_t kp = trimmed.find(k, pos);
                    if (kp == std::string::npos) return "";
                    size_t vstart = kp + k.size();
                    size_t vend = vstart;
                    while (vend < trimmed.size() && !(trimmed[vend] == '"' && trimmed[vend-1] != '\\'))
                        vend++;
                    return trimmed.substr(vstart, vend - vstart);
                };

                std::string start_t = extract_num("Start");
                std::string end_t = extract_num("End");
                std::string speaker = extract_num("Speaker");
                std::string content = extract_str("Content");

                if (!content.empty()) {
                    if (!speaker.empty()) {
                        fprintf(stdout, "[%s - %s] Speaker %s: %s\n",
                                start_t.c_str(), end_t.c_str(), speaker.c_str(), content.c_str());
                    } else {
                        fprintf(stdout, "[%s - %s] %s\n",
                                start_t.c_str(), end_t.c_str(), content.c_str());
                    }
                }

                pos = trimmed.find("}", pos);
                if (pos != std::string::npos) pos++;
            }
        } else {
            fprintf(stdout, "%s\n", output_text.c_str());
        }

        // Print timing summary to stderr
        fprintf(stderr, "\n  Audio: %.2fs | Tokens: %d | RTF: %.4f\n",
                audio.duration_sec, n_decoded, rtf);
        fprintf(stderr, "  VAE: %.1fms (A:%.1f + S:%.1f) | Prefill: %.1fms | Decode: %.1fms (%.1f ms/tok)\n",
                (double)acoustic_time_ms + (double)semantic_time_ms,
                (double)acoustic_time_ms, (double)semantic_time_ms,
                prefill_time, decode_time, per_token_ms);
    }

cleanup:
    // Cleanup
    if (lm_ctx) llama_free(lm_ctx);
    if (lm_model) llama_free_model(lm_model);
    llama_backend_free();

    if (vae_ctx) vae_free(vae_ctx);
    if (vae_model) vae_free_model(vae_model);

    return 0;
}
