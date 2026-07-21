/**
 * Prompt builder for VibeASR.cpp
 *
 * Constructs the chat-template prompt and assembles the final embedding sequence
 * by combining text embeddings with VAE speech features.
 *
 * Prompt format (Qwen2.5 chat template, aligned with Python apply_chat_template):
 *   system_tokens = encode(apply_chat_template([{system}], tokenize=False))
 *   user_tokens = apply_chat_template([{user}], tokenize=True)
 *   full_tokens = system_tokens + user_tokens
 *
 * Which expands to:
 *   <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n
 *   <|im_start|>user\n<|speech_start|><|speech_pad|>*N<|speech_end|>\n{user_suffix}<|im_end|>\n
 *
 * NOTE: No generation prompt (<|im_start|>assistant\n) — the model generates it.
 */

#ifndef PROMPT_BUILDER_H
#define PROMPT_BUILDER_H

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace prompt_builder {

static const char * SYSTEM_PROMPT =
    "You are a helpful assistant that transcribes audio input into text output in JSON format.";

// Tokenize a string using llama.cpp tokenizer
static std::vector<llama_token> tokenize(
    const llama_model * model,
    const std::string & text,
    bool add_special,
    bool parse_special) {

    // First call to get required size
    int n = llama_tokenize(model, text.c_str(), (int)text.size(), nullptr, 0, add_special, parse_special);
    if (n < 0) {
        n = -n;  // negative means buffer too small, abs value is required size
    }

    std::vector<llama_token> tokens(n);
    int actual = llama_tokenize(model, text.c_str(), (int)text.size(), tokens.data(), n, add_special, parse_special);
    if (actual < 0) {
        fprintf(stderr, "[prompt_builder] Error: tokenization failed\n");
        return {};
    }
    tokens.resize(actual);
    return tokens;
}

// Detokenize a single token to string
static std::string token_to_piece(const llama_model * model, llama_token token, bool special = true) {
    char buf[256];
    int n = llama_token_to_piece(model, token, buf, sizeof(buf), 0, special);
    if (n < 0) {
        // Buffer too small
        std::vector<char> big_buf(-n);
        n = llama_token_to_piece(model, token, big_buf.data(), -n, 0, special);
        if (n > 0) {
            return std::string(big_buf.data(), n);
        }
        return "";
    }
    return std::string(buf, n);
}

// Detokenize a sequence of tokens
static std::string detokenize(const llama_model * model, const std::vector<llama_token> & tokens, bool special = false) {
    std::string result;
    for (auto token : tokens) {
        result += token_to_piece(model, token, special);
    }
    return result;
}

// Find a special token ID by its text representation
static llama_token find_token_by_text(const llama_model * model, const std::string & text) {
    int n_vocab = llama_n_vocab(model);
    for (int i = 0; i < n_vocab; i++) {
        const char * token_text = llama_token_get_text(model, i);
        if (token_text && std::string(token_text) == text) {
            return i;
        }
    }
    return -1;
}

struct PromptTokens {
    std::vector<llama_token> tokens;       // Full token sequence
    int speech_pad_start;                   // Index of first speech_pad token
    int speech_pad_count;                   // Number of speech_pad tokens
    llama_token speech_pad_id;              // The speech_pad token ID
};

// Build the full prompt token sequence
// n_audio_samples: number of audio samples at target_sample_rate
// speech_tok_compress_ratio: downsample ratio (default 3200 for 24kHz, or audio_samples/frames)
//
// Strategy: Build prompt by manually assembling token IDs, using llama_tokenize
// only for plain text segments. Special tokens (<|im_start|>, <|im_end|>, speech tokens)
// are inserted by their known IDs directly, because the GGUF tokenizer may not
// recognize them via parse_special (depends on how the model was converted).
// prompt_format: "text" = plain text output, "json" = JSON with Start/End/Speaker/Content keys
static PromptTokens build_prompt(
    const llama_model * model,
    int n_audio_samples,
    int speech_tok_compress_ratio,
    float audio_duration_sec,
    const std::string & context_info = "",
    const std::string & prompt_format = "text") {

    PromptTokens result = {};

    // Calculate number of speech frames
    int vae_tok_len = (int)ceil((double)n_audio_samples / (double)speech_tok_compress_ratio);

    // Use the canonical token IDs from the Python HuggingFace tokenizer.
    // These correspond to the positions in the embedding table that were trained.
    // NOTE: The GGUF vocab may have different text-to-ID mappings due to conversion issues,
    // but the embedding table positions are always consistent with the training tokenizer.
    //
    // Qwen2.5 + VibeVoice token ID layout (from HuggingFace tokenizer):
    //   151643 = <|endoftext|>
    //   151644 = <|im_start|>
    //   151645 = <|im_end|>
    //   151646 = <|speech_start|> (= <|object_ref_start|> in Qwen2.5)
    //   151647 = <|speech_end|>   (= <|object_ref_end|> in Qwen2.5)
    //   151648 = <|speech_pad|>   (= <|box_start|> in Qwen2.5)
    llama_token im_start_id     = 151644;
    llama_token im_end_id       = 151645;
    llama_token speech_start_id = 151646;
    llama_token speech_end_id   = 151647;
    llama_token speech_pad_id   = 151648;

    result.speech_pad_id = speech_pad_id;


    // Build prompt by assembling token IDs directly
    // Format (matching Python apply_chat_template):
    //   <|im_start|> + tokenize("system\n{SYSTEM_PROMPT}") + <|im_end|> + tokenize("\n")
    //   <|im_start|> + tokenize("user\n") + <|speech_start|> + <|speech_pad|>*N + <|speech_end|> + tokenize("\n{suffix}") + <|im_end|> + tokenize("\n")

    // Tokenize plain text segments (no special tokens in these strings)
    std::vector<llama_token> system_content = tokenize(model, std::string("system\n") + SYSTEM_PROMPT, false, false);
    std::vector<llama_token> newline_token  = tokenize(model, "\n", false, false);
    std::vector<llama_token> user_prefix    = tokenize(model, "user\n", false, false);

    // Build user suffix based on prompt_format and optional context_info (hotwords)
    // "text" format (1.5B model, plain text output):
    //   Without context: "This is a X.XX seconds audio, please transcribe it."
    //   With context:    "This is a X.XX seconds audio, with extra info: {context}\n\nPlease transcribe it."
    // "json" format (7B model, JSON output with keys):
    //   Without context: "This is a X.XX seconds audio, please transcribe it with these keys: Start, End, Speaker, Content"
    //   With context:    "This is a X.XX seconds audio, with extra info: {context}\n\nPlease transcribe it with these keys: Start, End, Speaker, Content"
    std::string suffix_str;
    char duration_buf[64];
    snprintf(duration_buf, sizeof(duration_buf), "%.2f", audio_duration_sec);

    std::string transcribe_instruction;
    if (prompt_format == "json") {
        transcribe_instruction = "please transcribe it with these keys: Start, End, Speaker, Content";
    } else {
        transcribe_instruction = "please transcribe it.";
    }

    if (!context_info.empty()) {
        suffix_str = std::string("\nThis is a ") + duration_buf +
                     " seconds audio, with extra info: " + context_info +
                     "\n\n" + (prompt_format == "json" ?
                         "Please transcribe it with these keys: Start, End, Speaker, Content" :
                         "Please transcribe it.");
    } else {
        suffix_str = std::string("\nThis is a ") + duration_buf +
                     " seconds audio, " + transcribe_instruction;
    }
    std::vector<llama_token> user_suffix = tokenize(model, suffix_str, false, false);

    // Assemble system part: <|im_start|> system\n{SYSTEM_PROMPT} <|im_end|> \n
    result.tokens.push_back(im_start_id);
    result.tokens.insert(result.tokens.end(), system_content.begin(), system_content.end());
    result.tokens.push_back(im_end_id);
    result.tokens.insert(result.tokens.end(), newline_token.begin(), newline_token.end());

    // Assemble user part: <|im_start|> user\n <|speech_start|> <|speech_pad|>*N <|speech_end|> \n{suffix} <|im_end|> \n
    result.tokens.push_back(im_start_id);
    result.tokens.insert(result.tokens.end(), user_prefix.begin(), user_prefix.end());
    result.tokens.push_back(speech_start_id);

    result.speech_pad_start = (int)result.tokens.size();
    result.speech_pad_count = vae_tok_len;
    for (int i = 0; i < vae_tok_len; i++) {
        result.tokens.push_back(speech_pad_id);
    }

    result.tokens.push_back(speech_end_id);
    result.tokens.insert(result.tokens.end(), user_suffix.begin(), user_suffix.end());
    result.tokens.push_back(im_end_id);
    result.tokens.insert(result.tokens.end(), newline_token.begin(), newline_token.end());

    fprintf(stderr, "  Prompt built: %zu tokens (speech_pad: %d frames at [%d, %d))\n",
            result.tokens.size(), vae_tok_len,
            result.speech_pad_start, result.speech_pad_start + result.speech_pad_count);

    return result;
}

// Build VAE speech features for the speech_pad region.
// Returns: flat float array of shape [n_frames, n_embd] (acoustic + semantic element-wise sum)
static std::vector<float> build_speech_embeddings(
    int n_embd,
    const float * acoustic_features,  // [n_frames, acoustic_dim]
    int acoustic_dim,
    const float * semantic_features,  // [n_frames, semantic_dim]
    int semantic_dim,
    int n_frames) {

    std::vector<float> speech_emb(n_frames * n_embd, 0.0f);

    for (int f = 0; f < n_frames; f++) {
        float * dst = speech_emb.data() + f * n_embd;
        const float * a_src = acoustic_features + f * acoustic_dim;
        const float * s_src = semantic_features + f * semantic_dim;

        if (acoustic_dim == n_embd && semantic_dim == n_embd) {
            // Element-wise addition (matching Python: acoustic_features + semantic_features)
            for (int d = 0; d < n_embd; d++) {
                dst[d] = a_src[d] + s_src[d];
            }
        } else if (acoustic_dim + semantic_dim == n_embd) {
            // Concatenate
            memcpy(dst, a_src, acoustic_dim * sizeof(float));
            memcpy(dst + acoustic_dim, s_src, semantic_dim * sizeof(float));
        } else if (acoustic_dim == n_embd) {
            memcpy(dst, a_src, n_embd * sizeof(float));
        }
    }

    return speech_emb;
}

// Prefill the LM context with the prompt using segmented decode:
//   - Text segments use token ID mode (llama.cpp handles embedding lookup + dequantize)
//   - Speech segment uses embedding mode (VAE features injected directly)
//
// This avoids manual embedding table dequantization and works with any quantization format.
static int prefill_segmented(
    llama_model * model,
    llama_context * ctx,
    const PromptTokens & prompt,
    const float * acoustic_features,
    int acoustic_dim,
    const float * semantic_features,
    int semantic_dim,
    int n_frames,
    int n_batch) {

    int n_embd = llama_n_embd(model);
    int n_tokens = (int)prompt.tokens.size();
    int pos = 0;

    // Segment 1: text tokens before speech_pad (token ID mode)
    int seg1_len = prompt.speech_pad_start;
    if (seg1_len > 0) {
        std::vector<llama_token> seg1_tokens(prompt.tokens.begin(),
                                              prompt.tokens.begin() + seg1_len);
        int processed = 0;
        while (processed < seg1_len) {
            int batch_len = std::min(n_batch, seg1_len - processed);
            llama_batch batch = llama_batch_init(batch_len, 0, 1);
            batch.n_tokens = batch_len;
            for (int i = 0; i < batch_len; i++) {
                batch.token[i] = seg1_tokens[processed + i];
                batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 0;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "[prompt_builder] Error: prefill seg1 failed\n");
                llama_batch_free(batch);
                return -1;
            }
            llama_batch_free(batch);
            processed += batch_len;
            pos += batch_len;
        }
    }

    // Segment 2: speech_pad region (embedding mode with VAE features)
    int seg2_len = std::min(prompt.speech_pad_count, n_frames);
    if (seg2_len > 0) {
        std::vector<float> speech_emb = build_speech_embeddings(
            n_embd, acoustic_features, acoustic_dim,
            semantic_features, semantic_dim, seg2_len);

        int processed = 0;
        while (processed < seg2_len) {
            int batch_len = std::min(n_batch, seg2_len - processed);
            llama_batch batch = llama_batch_init(batch_len, n_embd, 1);
            batch.n_tokens = batch_len;
            batch.token = nullptr;  // embedding mode

            float * original_embd = batch.embd;
            batch.embd = speech_emb.data() + processed * n_embd;

            for (int i = 0; i < batch_len; i++) {
                batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 0;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "[prompt_builder] Error: prefill seg2 failed\n");
                batch.embd = original_embd;
                llama_batch_free(batch);
                return -1;
            }
            batch.embd = original_embd;
            llama_batch_free(batch);
            processed += batch_len;
            pos += batch_len;
        }
    }

    // Segment 3: text tokens after speech_pad (token ID mode)
    int seg3_start = prompt.speech_pad_start + prompt.speech_pad_count;
    int seg3_len = n_tokens - seg3_start;
    if (seg3_len > 0) {
        std::vector<llama_token> seg3_tokens(prompt.tokens.begin() + seg3_start,
                                              prompt.tokens.end());
        int processed = 0;
        while (processed < seg3_len) {
            int batch_len = std::min(n_batch, seg3_len - processed);
            llama_batch batch = llama_batch_init(batch_len, 0, 1);
            batch.n_tokens = batch_len;
            for (int i = 0; i < batch_len; i++) {
                batch.token[i] = seg3_tokens[processed + i];
                batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                // Request logits only for the very last token of the entire prompt
                batch.logits[i] = (processed + i == seg3_len - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "[prompt_builder] Error: prefill seg3 failed\n");
                llama_batch_free(batch);
                return -1;
            }
            llama_batch_free(batch);
            processed += batch_len;
            pos += batch_len;
        }
    }

    return pos;  // total tokens processed
}

}  // namespace prompt_builder

#endif // PROMPT_BUILDER_H
