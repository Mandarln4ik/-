package dev.neuroforge.core

/**
 * Works out which of a graph's input buffers each argument belongs in.
 *
 * ### Why this is not just "argument 0 goes in buffer 0"
 *
 * The runtime hands back input buffers in the converter's tensor order, which is not
 * necessarily the order the model's `forward()` declared its arguments. The reference host
 * loop for the Bonsai graphs opens with a warning about precisely this, and recovers the
 * argument order by parsing the `serving_default_args_<n>` tensor names. The Kotlin
 * `CompiledModel` API hands back buffers, not names, so that recovery is not available
 * here.
 *
 * What *is* available is how many elements each buffer holds. For the diffusion transformer
 * that is enough to be decisive: its five inputs are 131072, 1, 4096, 1024 and the
 * embedding's own length — all different. So the binding is derived rather than assumed,
 * and positional order is used only where the counts cannot tell two arguments apart.
 *
 * Getting this wrong is not a crash. Every one of those buffers is float32 and the graph
 * will happily run with the position ids where the latent goes; it returns a correctly
 * shaped image of noise.
 */
object GraphBinding {

  /** How a binding was arrived at, so the log can say which one the run used. */
  enum class Basis {
    /** Every argument matched exactly one buffer by element count. */
    ELEMENT_COUNT,

    /** Two arguments have the same length, so the declared order is all there is to go on. */
    POSITIONAL,
  }

  data class Binding(val slots: IntArray, val basis: Basis) {
    /** Buffer index for argument [argument]. */
    operator fun get(argument: Int): Int = slots[argument]

    override fun equals(other: Any?): Boolean =
      this === other || (other is Binding && slots.contentEquals(other.slots) && basis == other.basis)

    override fun hashCode(): Int = 31 * slots.contentHashCode() + basis.hashCode()
  }

  /**
   * Binds arguments of the given [expected] element counts to buffers of the given
   * [declared] counts.
   *
   * @return the binding, or null when [declared] cannot satisfy [expected] at all — a
   *   different arity, or a length the graph has no buffer for. That is a real mismatch
   *   between this pipeline and the converted graph, and is worth failing on: the
   *   alternative is a run that produces an image nobody can interpret.
   */
  fun bind(declared: IntArray, expected: IntArray): Binding? {
    if (declared.size != expected.size) return null

    // Ambiguity is a property of the expectation, not of a particular assignment: if two
    // arguments want the same length, no count-based rule can separate them.
    val distinct = expected.toSet().size == expected.size
    if (distinct) {
      val slots = IntArray(expected.size) { -1 }
      val taken = BooleanArray(declared.size)
      for (arg in expected.indices) {
        val hit = declared.indices.firstOrNull { !taken[it] && declared[it] == expected[arg] }
          ?: return null
        slots[arg] = hit
        taken[hit] = true
      }
      return Binding(slots, Basis.ELEMENT_COUNT)
    }

    // Fall back to the declared order, but only if it is consistent with the lengths — a
    // positional guess that contradicts what the buffers hold is not worth making.
    for (arg in expected.indices) if (declared[arg] != expected[arg]) return null
    return Binding(IntArray(expected.size) { it }, Basis.POSITIONAL)
  }

  /** A description of the mismatch, for an error message someone can act on. */
  fun describeMismatch(label: String, declared: IntArray, expected: Map<String, Int>): String =
    "$label declares inputs of ${declared.toList()} elements; this pipeline feeds " +
      expected.entries.joinToString(", ") { "${it.key}=${it.value}" } +
      ". The graph is not the one this pipeline was written against — check the model files."
}
