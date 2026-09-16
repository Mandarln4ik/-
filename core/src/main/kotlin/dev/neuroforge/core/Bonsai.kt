package dev.neuroforge.core

/**
 * The exact tensor contract of the Bonsai Image 4B LiteRT graphs.
 *
 * Every constant and every layout below is read out of the model's own reference host loop
 * (`python/generate.py` in the `litert-samples` release), not inferred from the
 * architecture's family. That distinction matters here more than usual: this pipeline
 * previously assumed the FLUX defaults — 16 latent channels, 77 prompt tokens, a hardcoded
 * time-shift of 3.0, a `timestep` scaled by 1000 — and every one of those was wrong. None
 * of them would have crashed. A graph handed a correctly-shaped tensor containing the wrong
 * numbers returns a correctly-shaped image containing noise, so the failure mode of
 * guessing here is a model that looks broken rather than a pipeline that looks broken.
 *
 * ```
 *  prompt ─tokenize(256)─▶ [ text encoder ] ─▶ embeds
 *                                                │
 *  seed ─▶ noise (1024, 128) ──▶ [ DiT × steps ] ┘   lat, embeds, σ, img_ids, txt_ids
 *                                     │
 *                          affine + unpatchify
 *                                     ▼
 *                            (1, 32, 64, 64) ─▶ [ VAE ] ─▶ (1, 3, 512, 512)
 * ```
 */
object Bonsai {

  /** Prompt length the text encoder was converted at; shorter prompts are padded to it. */
  const val SEQ = 256

  /** Image tokens the DiT sees: a 32×32 patch grid over a 64×64 latent. */
  const val TOKENS = 1024

  /** Side of the patch grid, i.e. `sqrt(TOKENS)`. */
  const val LAT_GRID = 32

  /** Width of one packed token: 32 latent channels × a 2×2 patch. */
  const val PACKED_CHANNELS = 128

  /** Output side, fixed at conversion time. */
  const val IMAGE_SIZE = 512

  /** Step count the model was distilled for. More steps also work. */
  const val DEFAULT_STEPS = 4

  /** Position-id width: `[batch, h, w, seq]` for every token, image and text alike. */
  const val ID_WIDTH = 4

  /** The geometry, in the vocabulary the rest of this module uses. */
  val latent: LatentSpec get() = LatentSpec.BONSAI

  /**
   * Position ids for the image tokens: `[0, h, w, 0]` per token, row-major over the grid.
   *
   * Float, not int — these are RoPE coordinates that the graph consumes as float32, and an
   * int tensor is a type error rather than an implicit conversion.
   */
  fun imagePositionIds(grid: Int = LAT_GRID): FloatArray {
    val out = FloatArray(grid * grid * ID_WIDTH)
    var k = 0
    for (h in 0 until grid) {
      for (w in 0 until grid) {
        out[k] = 0f
        out[k + 1] = h.toFloat()
        out[k + 2] = w.toFloat()
        out[k + 3] = 0f
        k += ID_WIDTH
      }
    }
    return out
  }

  /** Position ids for the prompt tokens: `[0, 0, 0, i]`, so text advances along its own axis. */
  fun textPositionIds(seq: Int = SEQ): FloatArray {
    val out = FloatArray(seq * ID_WIDTH)
    for (i in 0 until seq) out[i * ID_WIDTH + 3] = i.toFloat()
    return out
  }

  /**
   * Undoes the latent normalisation the DiT works in, per packed channel.
   *
   * These are the VAE's BatchNorm running statistics, shipped in `pipeline_meta.json`
   * because the DiT was trained on a whitened latent and the VAE decoder expects the raw
   * one. Skipping it produces a washed-out, low-contrast image — not an error.
   *
   * Applied *before* [LatentPacking.unpatchify], on the 128-wide packed axis, which is the
   * order the reference uses and the only order in which a length-128 vector lines up with
   * anything.
   *
   * @param tokens `TOKENS × PACKED_CHANNELS`, modified in place and returned.
   */
  fun denormalize(tokens: FloatArray, scale: FloatArray, shift: FloatArray): FloatArray {
    val width = scale.size
    require(shift.size == width) {
      "latent_bn_scale has $width entries but latent_bn_shift has ${shift.size}"
    }
    require(width > 0 && tokens.size % width == 0) {
      "latent has ${tokens.size} elements, which is not a whole number of $width-wide tokens"
    }
    var i = 0
    while (i < tokens.size) {
      for (m in 0 until width) {
        tokens[i + m] = tokens[i + m] * scale[m] + shift[m]
      }
      i += width
    }
    return tokens
  }

  /**
   * Turns the DiT's packed output into the tensor the VAE decoder takes.
   *
   * The whole of `unpatchify()` in the reference, in one call and in its order: the
   * per-packed-channel affine first, then the 2×2 patch unfold. Both steps are individually
   * plausible in either order and only one is right — `bn_scale` is 128 long, which lines up
   * with the packed axis and nothing else — so they are kept together rather than left for
   * a caller to sequence.
   *
   * @param tokens `TOKENS × PACKED_CHANNELS`; consumed and overwritten.
   * @return the latent flattened CHW, ready for the decoder.
   */
  fun toVaeLatent(
    tokens: FloatArray,
    scale: FloatArray,
    shift: FloatArray,
    spec: LatentSpec = latent,
    latentSide: Int = IMAGE_SIZE / latent.vaeScale,
  ): FloatArray = LatentPacking.unpatchify(
    denormalize(tokens, scale, shift), spec, latentSide, latentSide,
  )

  /**
   * Wraps a prompt in the chat template the text encoder's tokenizer applies.
   *
   * The encoder is a pruned Qwen3-4B, and the reference calls `apply_chat_template(...,
   * add_generation_prompt=True, enable_thinking=False)` rather than tokenizing the bare
   * prompt. The markers are not decoration: the model's prompt-conditioning was trained on
   * text in this exact shape, and feeding it a naked sentence shifts every embedding.
   *
   * Spelled out here because rendering the repository's Jinja template on a phone is not
   * worth a template engine — but it is not a guess. CI reads the repository's own
   * `chat_template.jinja` on every commit and prints it beside this rendering. The two
   * lines that produce it are:
   *
   * ```jinja
   * {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
   * ...
   * {{- '<|im_start|>assistant\n' }}
   * {%- if enable_thinking is defined and enable_thinking is false %}
   *     {{- '<think>\n\n</think>\n\n' }}
   * ```
   *
   * which is character for character what this returns for a single user turn. A future
   * re-release that changes the template shows up in that log rather than as an image that
   * quietly answers a slightly different prompt.
   */
  fun chatPrompt(prompt: String): String =
    "<|im_start|>user\n$prompt<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
}
