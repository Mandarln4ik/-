package dev.neuroforge.pipeline

import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * `pipeline_meta.json` — the constants the graphs cannot carry themselves.
 *
 * The two vectors are the VAE's BatchNorm running statistics. The diffusion transformer
 * works in a whitened latent space and the decoder expects the raw one, so the latent has
 * to be put back through `x * scale + shift` on its 128-wide packed axis before it is
 * decoded. Skipping that step is not an error at any layer: the decoder accepts the tensor
 * and returns a washed-out, low-contrast image.
 *
 * Read from the file rather than baked in, because they are checkpoint constants: a
 * re-release with a retrained VAE changes them, and a hardcoded copy would then quietly
 * describe the wrong model.
 */
data class PipelineMeta(
  val latentBnScale: FloatArray,
  val latentBnShift: FloatArray,
  val files: Map<String, String>,
) {
  override fun equals(other: Any?): Boolean =
    this === other || (
      other is PipelineMeta &&
        latentBnScale.contentEquals(other.latentBnScale) &&
        latentBnShift.contentEquals(other.latentBnShift) &&
        files == other.files
      )

  override fun hashCode(): Int =
    31 * (31 * latentBnScale.contentHashCode() + latentBnShift.contentHashCode()) + files.hashCode()

  companion object {

    fun read(file: File): PipelineMeta {
      require(file.isFile) {
        "pipeline_meta.json is missing from ${file.parentFile?.name}. It ships with the " +
          "checkpoint and carries the latent normalisation the decoder needs; without it " +
          "an image still decodes, just wrongly."
      }
      return parse(file.readText())
    }

    fun parse(json: String): PipelineMeta {
      val root = JSONObject(json)
      val scale = floats(root, "latent_bn_scale")
      val shift = floats(root, "latent_bn_shift")
      require(scale.size == shift.size) {
        "latent_bn_scale has ${scale.size} entries, latent_bn_shift has ${shift.size}"
      }
      val files = root.optJSONObject("files")?.let { obj ->
        obj.keys().asSequence().associateWith { obj.getString(it) }
      }.orEmpty()
      return PipelineMeta(scale, shift, files)
    }

    private fun floats(root: JSONObject, key: String): FloatArray {
      val array: JSONArray = root.optJSONArray(key)
        ?: error("pipeline_meta.json has no '$key'")
      return FloatArray(array.length()) { array.getDouble(it).toFloat() }
    }
  }
}
