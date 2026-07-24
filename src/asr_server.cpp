/**
 * VibeASR.cpp - Streaming ASR Server
 *
 * Persistent server that loads models once and processes multiple audio chunks
 * via stdin/stdout with token-level streaming output.
 *
 * Protocol:
 *   stdin:  Audio file path (one per line), "EXIT" to quit
 *   stdout: Token-by-token text (one token per line), "---END---" marks chunk done
 *   Ready signal: "---READY---" on stdout after model loading
 */

#include "vae.h"
#include "llama.h"
#include "ggml.h"

#include "../utils/audio_io.h"
#include "../utils/prompt_builder.h"

#include "time_compat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

struct server_params {
    std::string vae_model_path;
    std::string lm_model_path;
    std::string context_info;
    std::string prompt_format = "text";

    int n_threads       = 4;
    int n_ctx           = 16384;
    int n_batch         = 2048;
    int max_tokens      = 16384;
    int target_sr       = 24000;
    int compress_ratio  = 3200;

    bool greedy         = false;
    bool normalize      = true;
    bool token_stream   = true;
    float temperature   = 0.7f;
    float top_p         = 0.9f;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Streaming ASR server - loads models once, processes audio via stdin.\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --vae-model <path>   Path to VAE encoder GGUF model (required)\n");
    fprintf(stderr, "  --lm-model <path>    Path to LM GGUF model (required)\n");
    fprintf(stderr, "  -t <n>               Number of threads (default: 4)\n");
    fprintf(stderr, "  -c <n>               Context size (default: 16384)\n");
    fprintf(stderr, "  -b <n>               Batch size (default: 2048)\n");
    fprintf(stderr, "  --max-tokens <n>     Maximum tokens to generate (default: 16384)\n");
    fprintf(stderr, "  --greedy             Use greedy decoding (default: sampling)\n");
    fprintf(stderr, "  --temperature <f>    Sampling temperature (default: 0.7)\n");
    fprintf(stderr, "  --top-p <f>          Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  --sample-rate <n>    Target sample rate (default: 24000)\n");
    fprintf(stderr, "  --compress-ratio <n> Speech compression ratio (default: 3200)\n");
    fprintf(stderr, "  --context <text>     Hotwords/context info\n");
    fprintf(stderr, "  --prompt-format <s>  'text' or 'json' (default: text)\n");
    fprintf(stderr, "  --no-normalize       Disable audio normalization\n");
    fprintf(stderr, "  --no-token-stream    Disable token-level streaming\n");
    fprintf(stderr, "\nProtocol:\n");
    fprintf(stderr, "  Send audio file paths via stdin (one per line)\n");
    fprintf(stderr, "  Receive transcription via stdout, terminated by '---END---'\n");
    fprintf(stderr, "  Send 'EXIT' to terminate the server\n");
}

static bool parse_args(int argc, char ** argv, server_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--vae-model" && i + 1 < argc) {
            params.vae_model_path = argv[++i];
        } else if (arg == "--lm-model" && i + 1 < argc) {
            params.lm_model_path = argv[++i];
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
        } else if (arg == "--no-token-stream") {
            params.token_stream = false;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Error: Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (params.vae_model_path.empty() || params.lm_model_path.empty()) {
        fprintf(stderr, "Error: --vae-model and --lm-model are required\n\n");
        print_usage(argv[0]);
        return false;
    }

    return true;
}

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int process_chunk(
    const std::string & audio_path,
    const server_params & params,
    vae_context_t * vae_ctx,
    vae_model_t * vae_model,
    llama_model * lm_model,
    llama_context * lm_ctx) {

    double chunk_start = get_time_ms();

    // Load audio
    audio_io::AudioData audio;
    if (!audio_io::load_wav(audio_path, params.target_sr, params.normalize, audio)) {
        fprintf(stdout, "[ERROR] Failed to load audio: %s\n---END---\n", audio_path.c_str());
        fflush(stdout);
        return -1;
    }

    int acoustic_dim = vae_model_acoustic_dim(vae_model);
    int semantic_dim = vae_model_semantic_dim(vae_model);
    int32_t n_samples = (int32_t)audio.samples.size();
    int32_t expected_frames = (n_samples + params.compress_ratio - 1) / params.compress_ratio;

    // VAE Acoustic Encode
    std::vector<float> acoustic_features(expected_frames * acoustic_dim);
    float acoustic_time_ms = 0.0f;
    int32_t acoustic_frames = vae_encode_acoustic_with_timing(
        vae_ctx, audio.samples.data(), n_samples,
        acoustic_features.data(), &acoustic_time_ms);

    if (acoustic_frames < 0) {
        fprintf(stdout, "[ERROR] VAE acoustic encoding failed\n---END---\n");
        fflush(stdout);
        return -1;
    }
    acoustic_features.resize(acoustic_frames * acoustic_dim);

    // VAE Semantic Encode
    std::vector<float> semantic_features(expected_frames * semantic_dim);
    float semantic_time_ms = 0.0f;
    int32_t semantic_frames = vae_encode_semantic_with_timing(
        vae_ctx, audio.samples.data(), n_samples,
        semantic_features.data(), &semantic_time_ms);

    if (semantic_frames < 0) {
        fprintf(stdout, "[ERROR] VAE semantic encoding failed\n---END---\n");
        fflush(stdout);
        return -1;
    }
    semantic_features.resize(semantic_frames * semantic_dim);

    int n_frames = std::min(acoustic_frames, semantic_frames);

    // Build prompt
    prompt_builder::PromptTokens prompt = prompt_builder::build_prompt(
        lm_model, n_samples, params.compress_ratio, audio.duration_sec,
        params.context_info, params.prompt_format);

    if (prompt.tokens.empty()) {
        fprintf(stdout, "[ERROR] Failed to build prompt\n---END---\n");
        fflush(stdout);
        return -1;
    }

    int n_prompt_tokens = (int)prompt.tokens.size();

    // Clear KV cache
    llama_kv_cache_clear(lm_ctx);

    // Segmented prefill
    int pos = prompt_builder::prefill_segmented(
        lm_model, lm_ctx, prompt,
        acoustic_features.data(), acoustic_dim,
        semantic_features.data(), semantic_dim,
        n_frames, params.n_batch);

    if (pos < 0) {
        fprintf(stdout, "[ERROR] Segmented prefill failed\n---END---\n");
        fflush(stdout);
        return -1;
    }

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

    const llama_token EOG_IM_END = 151645;
    const llama_token EOG_ENDOFTEXT = 151643;
    const llama_token im_start_id = 151644;

    // Autoregressive decode
    std::vector<llama_token> generated_tokens;
    int cur_pos = n_prompt_tokens;
    int n_decoded = 0;

    // Sample first token from prefill logits
    llama_token new_token = llama_sampler_sample(smpl, lm_ctx, -1);
    llama_sampler_accept(smpl, new_token);
    generated_tokens.push_back(new_token);
    n_decoded++;

    // Skip assistant header: <|im_start|> assistant \n (3 tokens)
    int header_skip_remaining = (new_token == im_start_id) ? 2 : 0;

    // Decode loop
    while (n_decoded < params.max_tokens) {
        if (new_token == EOG_IM_END || new_token == EOG_ENDOFTEXT) {
            break;
        }

        llama_batch batch = llama_batch_get_one(&new_token, 1, cur_pos, 0);
        if (llama_decode(lm_ctx, batch) != 0) {
            break;
        }

        cur_pos++;
        new_token = llama_sampler_sample(smpl, lm_ctx, -1);
        llama_sampler_accept(smpl, new_token);
        generated_tokens.push_back(new_token);
        n_decoded++;

        if (header_skip_remaining > 0) {
            header_skip_remaining--;
            continue;
        }

        // Token-level streaming: one token per line
        if (params.token_stream && new_token != EOG_IM_END && new_token != EOG_ENDOFTEXT) {
            std::string piece = prompt_builder::token_to_piece(lm_model, new_token, false);
            if (!piece.empty()) {
                fprintf(stdout, "%s\n", piece.c_str());
                fflush(stdout);
            }
        }
    }

    llama_sampler_free(smpl);

    // Non-streaming mode: output all at once
    if (!params.token_stream) {
        int content_start = 0;
        if (generated_tokens.size() >= 3 && generated_tokens[0] == im_start_id) {
            content_start = 1;
            std::string piece = prompt_builder::token_to_piece(lm_model, generated_tokens[1], true);
            if (piece == "assistant") {
                content_start = 2;
                if (content_start < (int)generated_tokens.size()) {
                    piece = prompt_builder::token_to_piece(lm_model, generated_tokens[content_start], true);
                    if (piece == "\n") content_start = 3;
                }
            }
        }

        int content_end = (int)generated_tokens.size();
        if (content_end > content_start) {
            llama_token last_tok = generated_tokens[content_end - 1];
            if (last_tok == EOG_IM_END || last_tok == EOG_ENDOFTEXT) content_end--;
        }

        std::vector<llama_token> content_tokens(
            generated_tokens.begin() + content_start,
            generated_tokens.begin() + content_end);
        std::string output_text = prompt_builder::detokenize(lm_model, content_tokens, false);
        fprintf(stdout, "%s", output_text.c_str());
    }

    fprintf(stdout, "\n---END---\n");
    fflush(stdout);

    // Timing to stderr
    double total_time = get_time_ms() - chunk_start;
    double rtf = (total_time / 1000.0) / audio.duration_sec;
    fprintf(stderr, "  Audio: %.2fs | Tokens: %d | RTF: %.4f | VAE: %.1fms (A:%.1f+S:%.1f) | Total: %.1fms\n",
            audio.duration_sec, n_decoded, rtf,
            (double)acoustic_time_ms + (double)semantic_time_ms,
            (double)acoustic_time_ms, (double)semantic_time_ms, total_time);

    return 0;
}

int main(int argc, char ** argv) {
    server_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    fprintf(stderr, "========================================\n");
    fprintf(stderr, " VibeASR.cpp - Streaming ASR Server\n");
    fprintf(stderr, "========================================\n\n");
    fprintf(stderr, "  VAE model:    %s\n", params.vae_model_path.c_str());
    fprintf(stderr, "  LM model:     %s\n", params.lm_model_path.c_str());
    fprintf(stderr, "  Threads:      %d\n", params.n_threads);
    fprintf(stderr, "  Token stream: %s\n", params.token_stream ? "on" : "off");
    fprintf(stderr, "  Decoding:     %s\n\n", params.greedy ? "greedy" : "sampling");

    // Load VAE model
    fprintf(stderr, "[Init] Loading VAE model...\n");
    double t0 = get_time_ms();

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

    // Load LM model
    fprintf(stderr, "[Init] Loading LM model...\n");

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    llama_log_set([](enum ggml_log_level, const char *, void *){}, nullptr);

    llama_model_params lm_mparams = llama_model_default_params();
    lm_mparams.n_gpu_layers = 0;
#ifdef _WIN32
    lm_mparams.use_mmap = false;  // MinGW/Windows lacks PrefetchVirtualMemory; mmap load fails
#endif

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

    double load_time = get_time_ms() - t0;
    fprintf(stderr, "[Init] Models loaded in %.1f ms. Ready.\n\n", load_time);

    // Signal readiness
    fprintf(stdout, "---READY---\n");
    fflush(stdout);

    // Server loop
    char line_buf[4096];
    int chunk_count = 0;

    while (fgets(line_buf, sizeof(line_buf), stdin) != nullptr) {
        size_t len = strlen(line_buf);
        while (len > 0 && (line_buf[len-1] == '\n' || line_buf[len-1] == '\r')) {
            line_buf[--len] = '\0';
        }
        if (len == 0) continue;

        std::string input(line_buf);

        if (input == "EXIT" || input == "exit" || input == "quit") {
            fprintf(stderr, "[Server] Exit command received.\n");
            break;
        }

        if (input.rfind("CONTEXT:", 0) == 0) {
            params.context_info = input.substr(8);
            fprintf(stderr, "[Server] Context updated.\n");
            fprintf(stdout, "---ACK---\n");
            fflush(stdout);
            continue;
        }

        if (input.rfind("FORMAT:", 0) == 0) {
            params.prompt_format = input.substr(7);
            fprintf(stderr, "[Server] Format updated: %s\n", params.prompt_format.c_str());
            fprintf(stdout, "---ACK---\n");
            fflush(stdout);
            continue;
        }

        chunk_count++;
        fprintf(stderr, "[Server] Chunk %d: %s\n", chunk_count, input.c_str());
        process_chunk(input, params, vae_ctx, vae_model, lm_model, lm_ctx);
    }

    // Cleanup
    fprintf(stderr, "[Server] Processed %d chunks. Shutting down.\n", chunk_count);
    llama_free(lm_ctx);
    llama_free_model(lm_model);
    llama_backend_free();
    vae_free(vae_ctx);
    vae_free_model(vae_model);

    return 0;
}
