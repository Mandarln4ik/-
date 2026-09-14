package dev.neuroforge

import android.app.Application
import dev.neuroforge.models.ModelRepository
import dev.neuroforge.pipeline.GenerationPipeline

/**
 * Process-wide singletons.
 *
 * Hand-wired rather than injected: three objects with a plain dependency chain do not need
 * a container, and a reader tracing "where does the model file come from" gets there in one
 * hop instead of through generated code.
 */
class NeuroForgeApp : Application() {

  val repository: ModelRepository by lazy { ModelRepository(this) }
  val pipeline: GenerationPipeline by lazy { GenerationPipeline(this, repository) }
}
