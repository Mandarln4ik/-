"""Writes a tiny but structurally valid llama GGUF, so the JNI paths can be run for real.
A 32-wide, two-layer llama with a vocabulary of 24 pieces plus the 256 byte fallbacks. It
cannot say anything sensible -- the weights are noise -- but it exercises exactly the code
under test: GGUF parsing, the chat template, tokenisation, the KV cache and the decode loop.
"""
import sys
sys.path.insert(0, sys.argv[2])  # gguf-py, from the llama.cpp checkout CMake fetched
import numpy as np
import gguf

N_EMBD, N_LAYER, N_HEAD, N_FF, N_CTX = 32, 2, 2, 64, 256
# Byte tokens matter: SentencePiece falls back to them for anything the vocabulary does
# not cover, and a vocabulary without them makes llama_tokenize throw.
VOCAB = (["<unk>", "<s>", "</s>"] + [chr(ord("a") + i) for i in range(21)]
         + ["<0x%02X>" % b for b in range(256)])

w = gguf.GGUFWriter(sys.argv[1], "llama")
w.add_name("tiny")
w.add_context_length(N_CTX)
w.add_embedding_length(N_EMBD)
w.add_block_count(N_LAYER)
w.add_feed_forward_length(N_FF)
w.add_head_count(N_HEAD)
w.add_head_count_kv(N_HEAD)
w.add_rope_dimension_count(N_EMBD // N_HEAD)
w.add_layer_norm_rms_eps(1e-5)
w.add_file_type(gguf.LlamaFileType.ALL_F32)

w.add_tokenizer_model("llama")
w.add_token_list(VOCAB)
w.add_token_scores([0.0] * len(VOCAB))
def token_type(t):
    if t.startswith("<0x"):
        return gguf.TokenType.BYTE
    return gguf.TokenType.CONTROL if t.startswith("<") else gguf.TokenType.NORMAL
w.add_token_types([token_type(t) for t in VOCAB])
w.add_bos_token_id(1)
w.add_eos_token_id(2)
# ChatML, so llama_chat_apply_template has something it recognises to detect.
w.add_chat_template(
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + "
    "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

rng = np.random.default_rng(0)
def t(name, shape):
    w.add_tensor(name, rng.standard_normal(shape).astype(np.float32) * 0.02)

t("token_embd.weight", (len(VOCAB), N_EMBD))
t("output_norm.weight", (N_EMBD,))
t("output.weight", (len(VOCAB), N_EMBD))
for i in range(N_LAYER):
    t(f"blk.{i}.attn_norm.weight", (N_EMBD,))
    t(f"blk.{i}.attn_q.weight", (N_EMBD, N_EMBD))
    t(f"blk.{i}.attn_k.weight", (N_EMBD, N_EMBD))
    t(f"blk.{i}.attn_v.weight", (N_EMBD, N_EMBD))
    t(f"blk.{i}.attn_output.weight", (N_EMBD, N_EMBD))
    t(f"blk.{i}.ffn_norm.weight", (N_EMBD,))
    t(f"blk.{i}.ffn_gate.weight", (N_FF, N_EMBD))
    t(f"blk.{i}.ffn_up.weight", (N_FF, N_EMBD))
    t(f"blk.{i}.ffn_down.weight", (N_EMBD, N_FF))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", sys.argv[1])
