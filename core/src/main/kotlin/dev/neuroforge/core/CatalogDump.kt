package dev.neuroforge.core

/**
 * Prints the catalogue as tab-separated rows so CI can check every URL really resolves.
 *
 * This exists because of a failure that happened twice: a file name in the catalogue was a
 * plausible guess, could not be verified from the authoring environment (Hugging Face is
 * unreachable there), and 404'd on someone's phone at the moment they pressed Download.
 *
 * CI has unrestricted network access, so the check belongs there. A catalogue entry is a
 * claim about someone else's repository, and the only honest way to hold that claim is to
 * test it on every commit rather than at the point of use.
 *
 * Output columns: model id, local file name, URL, repository (blank for direct downloads).
 */
fun main() {
  ModelCatalog.all.forEach { spec ->
    spec.files.forEach { file ->
      val repo = (file.source as? ModelSource.HuggingFace)?.repoId.orEmpty()
      println(listOf(spec.id, file.fileName, file.downloadUrl(), repo).joinToString("\t"))
    }
  }
}
