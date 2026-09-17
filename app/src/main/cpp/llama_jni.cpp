// JNI bridge to llama.cpp, so the app can read GGUF.
//
// GGUF is the point. More open models are published in that format than in every other
// on-device format combined, and neither LiteRT-LM nor ExecuTorch reads it. This is CPU:
// llama.cpp does have GPU backends on Android, but they are build-time options with their
// own device caveats, and shipping one that has never run on the target would be the same
// kind of claim this app exists to avoid making.
//
// Written against the llama.cpp public C API at tag b6100, which is what CMakeLists.txt
// pins. That API has churned across releases -- `llama_load_model_from_file` and
// `llama_new_context_with_model` are both deprecated in favour of the names used here --
// so the pin is load-bearing rather than tidiness.

#include <jni.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

namespace {

// Stop reasons, mirrored by LlamaCppNative.Stop on the Kotlin side.
constexpr jint STOP_COMPLETE = 0;      // hit an end-of-generation token
constexpr jint STOP_LIMIT = 1;         // ran out of max_tokens
constexpr jint STOP_CANCELLED = 2;     // the callback asked to stop
constexpr jint STOP_CONTEXT_FULL = 3;  // no KV slot left
constexpr jint STOP_ERROR = 4;

// One decoded token, as text. `llama_token_to_piece` does not null-terminate and reports
// the length it needed, so a fixed buffer has to handle the negative-return retry.
std::string token_text(const llama_vocab *vocab, llama_token token) {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n >= 0) {
        return std::string(buf, static_cast<size_t>(n));
    }
    std::vector<char> big(static_cast<size_t>(-n));
    n = llama_token_to_piece(vocab, token, big.data(), static_cast<int32_t>(big.size()), 0, true);
    if (n < 0) {
        return {};
    }
    return std::string(big.data(), static_cast<size_t>(n));
}

/**
 * Turns a C++ exception into a Java one.
 *
 * llama.cpp throws -- `llama_tokenize` does it for a vocabulary that cannot represent the
 * input, and the model loader does it in several places. An exception that escapes a JNI
 * function is not caught by the JVM; it reaches `std::terminate` and aborts the process,
 * which on a phone is a silent disappearance with no Java stack trace to explain it. So
 * every entry point below that can reach llama.cpp funnels through here instead.
 *
 * Found by running the whole surface against a deliberately awkward model: a
 * `std::out_of_range` from the tokenizer took the whole JVM down.
 */
void rethrow_as_java(JNIEnv *env, const char *where, const std::string &what) {
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls != nullptr) {
        env->ThrowNew(cls, (std::string(where) + ": " + what).c_str());
    }
}

std::string to_string(JNIEnv *env, jstring s) {
    if (s == nullptr) {
        return {};
    }
    const char *chars = env->GetStringUTFChars(s, nullptr);
    std::string out(chars == nullptr ? "" : chars);
    if (chars != nullptr) {
        env->ReleaseStringUTFChars(s, chars);
    }
    return out;
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_init(JNIEnv *, jobject) {
    llama_backend_init();
}

JNIEXPORT jlong JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_loadModel(JNIEnv *env, jobject, jstring path) {
    const std::string file = to_string(env, path);
    llama_model_params params = llama_model_default_params();
    // CPU only. Offloading would need a GPU backend compiled in, and none is.
    params.n_gpu_layers = 0;
    try {
        return reinterpret_cast<jlong>(llama_model_load_from_file(file.c_str(), params));
    } catch (const std::exception &e) {
        rethrow_as_java(env, "loadModel", e.what());
        return 0;
    }
}

JNIEXPORT jlong JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_newContext(
        JNIEnv *env, jobject, jlong model_ptr, jint n_ctx, jint n_threads) {
    auto *model = reinterpret_cast<llama_model *>(model_ptr);
    if (model == nullptr) {
        return 0;
    }
    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(n_ctx);
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    try {
        return reinterpret_cast<jlong>(llama_init_from_model(model, params));
    } catch (const std::exception &e) {
        rethrow_as_java(env, "newContext", e.what());
        return 0;
    }
}

JNIEXPORT jint JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_contextSize(JNIEnv *, jobject, jlong ctx_ptr) {
    auto *ctx = reinterpret_cast<llama_context *>(ctx_ptr);
    return ctx == nullptr ? 0 : static_cast<jint>(llama_n_ctx(ctx));
}

/** The context the model was trained for. Asking for more than this degrades quality. */
JNIEXPORT jint JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_trainedContextSize(
        JNIEnv *, jobject, jlong model_ptr) {
    auto *model = reinterpret_cast<llama_model *>(model_ptr);
    return model == nullptr ? 0 : llama_model_n_ctx_train(model);
}

/** Architecture, parameter count and quantisation, as llama.cpp itself describes them. */
JNIEXPORT jstring JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_describeModel(
        JNIEnv *env, jobject, jlong model_ptr) {
    auto *model = reinterpret_cast<llama_model *>(model_ptr);
    if (model == nullptr) {
        return env->NewStringUTF("");
    }
    char buf[256] = {0};
    const int32_t n = llama_model_desc(model, buf, sizeof(buf));
    return env->NewStringUTF(n > 0 ? buf : "");
}

/** Wipes the KV cache, which is what starting a new conversation means here. */
JNIEXPORT void JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_clearMemory(JNIEnv *, jobject, jlong ctx_ptr) {
    auto *ctx = reinterpret_cast<llama_context *>(ctx_ptr);
    if (ctx != nullptr) {
        llama_memory_clear(llama_get_memory(ctx), true);
    }
}

/**
 * Formats a conversation with the model's own chat template.
 *
 * The template comes out of the GGUF metadata, so a Qwen file is formatted as Qwen and a
 * Gemma file as Gemma without the app knowing which it downloaded. `llama_chat_apply_template`
 * is not a Jinja interpreter -- it matches the template text against a list of known
 * families -- so an unrecognised one returns -1, and that becomes null here rather than a
 * silently wrong prompt.
 */
JNIEXPORT jstring JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_formatChat(
        JNIEnv *env, jobject, jlong model_ptr, jobjectArray roles, jobjectArray contents,
        jboolean add_assistant) {
    auto *model = reinterpret_cast<llama_model *>(model_ptr);
    if (model == nullptr || roles == nullptr || contents == nullptr) {
        return nullptr;
    }
    const jsize count = env->GetArrayLength(roles);
    if (count != env->GetArrayLength(contents)) {
        return nullptr;
    }

    // The llama_chat_message array points into these, so they have to outlive the call.
    std::vector<std::string> owned;
    owned.reserve(static_cast<size_t>(count) * 2);
    std::vector<llama_chat_message> messages;
    messages.reserve(static_cast<size_t>(count));
    size_t total = 0;
    for (jsize i = 0; i < count; i++) {
        auto role = reinterpret_cast<jstring>(env->GetObjectArrayElement(roles, i));
        auto content = reinterpret_cast<jstring>(env->GetObjectArrayElement(contents, i));
        owned.push_back(to_string(env, role));
        owned.push_back(to_string(env, content));
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(content);
        total += owned[owned.size() - 2].size() + owned.back().size();
    }
    for (jsize i = 0; i < count; i++) {
        messages.push_back({owned[static_cast<size_t>(i) * 2].c_str(),
                            owned[static_cast<size_t>(i) * 2 + 1].c_str()});
    }

    // nullptr here would silently mean "chatml"; the model's own template is the point.
    const char *tmpl = llama_model_chat_template(model, nullptr);

    // Upstream recommends twice the message length; the retry below covers templates with
    // more scaffolding than that.
    std::vector<char> buf(total * 2 + 512);
    int32_t n = 0;
    try {
        n = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                      add_assistant == JNI_TRUE, buf.data(),
                                      static_cast<int32_t>(buf.size()));
        if (n > static_cast<int32_t>(buf.size())) {
            buf.resize(static_cast<size_t>(n));
            n = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                          add_assistant == JNI_TRUE, buf.data(),
                                          static_cast<int32_t>(buf.size()));
        }
    } catch (const std::exception &e) {
        rethrow_as_java(env, "formatChat", e.what());
        return nullptr;
    }
    if (n < 0) {
        return nullptr;
    }
    return env->NewStringUTF(std::string(buf.data(), static_cast<size_t>(n)).c_str());
}

/**
 * Decodes `prompt` and streams the reply, one token's text at a time.
 *
 * `prompt` is only the part of the conversation that is not in the KV cache yet: the
 * caller diffs the formatted transcript against the previous one and passes the suffix,
 * so a fifth turn costs five turns' worth of decode rather than fifteen. Positions
 * continue from the cache automatically, which is what `llama_batch_get_one` does when it
 * leaves `pos` null.
 *
 * The callback returns false to stop, which is how cancellation reaches a loop that is
 * otherwise entirely inside native code.
 *
 * @param add_special whether to prepend BOS -- true only on the first turn of a
 *   conversation, since the cache already holds it afterwards.
 * @return one of the STOP_* codes.
 */
JNIEXPORT jint JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_generate(
        JNIEnv *env, jobject, jlong model_ptr, jlong ctx_ptr, jstring prompt,
        jboolean add_special, jint max_tokens, jfloat temperature, jint top_k, jfloat top_p,
        jint seed, jobject callback) {
    auto *model = reinterpret_cast<llama_model *>(model_ptr);
    auto *ctx = reinterpret_cast<llama_context *>(ctx_ptr);
    if (model == nullptr || ctx == nullptr) {
        return STOP_ERROR;
    }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    jclass cb_class = env->GetObjectClass(callback);
    jmethodID on_token = env->GetMethodID(cb_class, "onToken", "(Ljava/lang/String;)Z");
    if (on_token == nullptr) {
        return STOP_ERROR;
    }

    const std::string text = to_string(env, prompt);
    const auto text_len = static_cast<int32_t>(text.size());

    std::vector<llama_token> tokens;
    try {
        // A negative return is how llama_tokenize reports the buffer size it needs.
        const int32_t needed = -llama_tokenize(vocab, text.c_str(), text_len, nullptr, 0,
                                               add_special == JNI_TRUE, true);
        if (needed <= 0) {
            return STOP_ERROR;
        }
        tokens.resize(static_cast<size_t>(needed));
        llama_tokenize(vocab, text.c_str(), text_len, tokens.data(),
                       static_cast<int32_t>(tokens.size()), add_special == JNI_TRUE, true);
    } catch (const std::exception &e) {
        rethrow_as_java(env, "tokenize", e.what());
        return STOP_ERROR;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sampler_params);
    if (top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    }
    if (top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(static_cast<uint32_t>(seed)));

    // Held outside the loop because the batch points at it and must outlive the decode.
    llama_token next = 0;
    jint stop = STOP_LIMIT;

    try {
        // Prefill, in chunks no larger than the context's batch size.
        //
        // This is not an optimisation, it is the difference between working and aborting.
        // `llama_decode` asserts that a batch fits in `n_batch`, and an assert is not an
        // exception -- it calls abort(), so the app disappears with no Java stack trace and
        // nothing to catch. `n_batch` is itself clamped to `n_ctx`, so choosing a 1024-token
        // window in Settings and then pasting something longer than that is enough to hit
        // it. Found by feeding a long prompt into a small context on purpose.
        const auto n_batch = static_cast<size_t>(llama_n_batch(ctx));
        const auto n_ctx_max = static_cast<size_t>(llama_n_ctx(ctx));
        const llama_pos last = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        const size_t cached = last < 0 ? 0 : static_cast<size_t>(last) + 1;

        // No room for the prompt plus at least one token of reply. Said now rather than
        // discovered as a failed decode half way through prefill.
        if (cached + tokens.size() >= n_ctx_max) {
            llama_sampler_free(sampler);
            return STOP_CONTEXT_FULL;
        }

        for (size_t offset = 0; offset < tokens.size(); offset += n_batch) {
            const auto chunk = static_cast<int32_t>(std::min(n_batch, tokens.size() - offset));
            // With a null `logits` field only the batch's last token produces any, which is
            // exactly what is wanted: the intermediate chunks just fill the cache.
            llama_batch batch = llama_batch_get_one(tokens.data() + offset, chunk);
            const int32_t decoded = llama_decode(ctx, batch);
            if (decoded != 0) {
                llama_sampler_free(sampler);
                // 1 means there was no KV slot for the batch, which is the context filling
                // up rather than a fault -- worth telling apart, the fix is different.
                return decoded == 1 ? STOP_CONTEXT_FULL : STOP_ERROR;
            }
        }

        for (jint produced = 0; produced < max_tokens; produced++) {
            // `llama_sampler_sample` accepts the token itself; calling accept again would
            // advance any stateful sampler in the chain twice.
            next = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab, next)) {
                stop = STOP_COMPLETE;
                break;
            }

            const std::string piece = token_text(vocab, next);
            jstring out = env->NewStringUTF(piece.c_str());
            const jboolean keep_going = env->CallBooleanMethod(callback, on_token, out);
            env->DeleteLocalRef(out);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                stop = STOP_ERROR;
                break;
            }
            if (keep_going == JNI_FALSE) {
                stop = STOP_CANCELLED;
                break;
            }

            // Feeding the token just emitted is what lets the next iteration sample from
            // a state that includes it.
            llama_batch batch = llama_batch_get_one(&next, 1);
            const int32_t decoded = llama_decode(ctx, batch);
            if (decoded != 0) {
                stop = decoded == 1 ? STOP_CONTEXT_FULL : STOP_ERROR;
                break;
            }
        }
    } catch (const std::exception &e) {
        llama_sampler_free(sampler);
        rethrow_as_java(env, "generate", e.what());
        return STOP_ERROR;
    }

    llama_sampler_free(sampler);
    return stop;
}

JNIEXPORT void JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_freeContext(JNIEnv *, jobject, jlong ctx_ptr) {
    if (ctx_ptr != 0) {
        llama_free(reinterpret_cast<llama_context *>(ctx_ptr));
    }
}

JNIEXPORT void JNICALL
Java_dev_neuroforge_runtime_LlamaCppNative_freeModel(JNIEnv *, jobject, jlong model_ptr) {
    if (model_ptr != 0) {
        llama_model_free(reinterpret_cast<llama_model *>(model_ptr));
    }
}

}  // extern "C"
