package dev.neuroforge.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import dev.neuroforge.NeuroForgeApp
import dev.neuroforge.core.AcceleratorPolicy
import dev.neuroforge.core.BenchmarkReport
import dev.neuroforge.core.ModelCatalog
import dev.neuroforge.core.ModelSpec
import dev.neuroforge.core.OutputTarget
import dev.neuroforge.core.RenderPlan
import dev.neuroforge.core.UpscaleStrategy
import dev.neuroforge.models.DownloadProgress
import dev.neuroforge.models.ModelState
import dev.neuroforge.pipeline.GenerationProgress
import dev.neuroforge.pipeline.GenerationRequest
import dev.neuroforge.pipeline.GenerationResult
import dev.neuroforge.runtime.AcceleratorProbe
import dev.neuroforge.runtime.Benchmark
import dev.neuroforge.runtime.DeviceReport
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class UiState(
  val device: DeviceReport? = null,
  val modelStates: Map<String, Map<String, ModelState>> = emptyMap(),
  val downloads: Map<String, DownloadProgress> = emptyMap(),
  val benchmark: BenchmarkReport? = null,
  val benchmarkRunning: Boolean = false,
  val generation: GenerationProgress? = null,
  val result: GenerationResult? = null,
  val plan: RenderPlan? = null,
  val diskUsage: Long = 0L,
  val error: String? = null,
)

class AppViewModel(private val app: NeuroForgeApp) : ViewModel() {

  private val _state = MutableStateFlow(UiState())
  val state: StateFlow<UiState> = _state.asStateFlow()

  // Prompt settings live outside UiState: they change on every keystroke, and folding them
  // into the same object would recompose the device panel and the model list too.
  val prompt = MutableStateFlow("a red fox sitting in fresh snow at sunrise, cinematic light")
  val seed = MutableStateFlow(7L)
  val steps = MutableStateFlow(4)
  val target = MutableStateFlow(OutputTarget.SQUARE_4K)
  val strategy = MutableStateFlow(UpscaleStrategy.FAST)

  /**
   * Which accelerators a run may use.
   *
   * Persisted because it is a deliberate choice about how the app behaves, not a per-run
   * knob: someone who set "NPU only" to find out whether their APU works wants that answer
   * to survive the app being killed mid-download.
   */
  val policy = MutableStateFlow(loadPolicy())

  private var generationJob: Job? = null
  private val downloadJobs = mutableMapOf<String, Job>()

  init {
    refreshDevice()
    refreshModels()
    refreshPlan()
  }

  fun refreshDevice() {
    viewModelScope.launch {
      _state.update { it.copy(device = AcceleratorProbe.probe(app)) }
    }
  }

  /**
   * Re-reads what is on disk.
   *
   * Runs off the main thread and lands in state, rather than being called from composition:
   * sizing the model directory touches the filesystem, and recomposition is not the place
   * for that however cheap a single call looks.
   */
  fun refreshModels() {
    viewModelScope.launch(Dispatchers.IO) {
      val states = ModelCatalog.all.associate { it.id to app.repository.stateOf(it) }
      val usage = app.repository.diskUsage()
      _state.update { it.copy(modelStates = states, diskUsage = usage) }
    }
  }

  /** Recomputes the route to the target so the UI can show the cost before anything runs. */
  fun refreshPlan() {
    _state.update {
      it.copy(plan = RenderPlan.of(512, 512, target.value, strategy.value))
    }
  }

  fun setTarget(t: OutputTarget) { target.value = t; refreshPlan() }
  fun setStrategy(s: UpscaleStrategy) { strategy.value = s; refreshPlan() }

  fun download(spec: ModelSpec) {
    if (downloadJobs[spec.id]?.isActive == true) return
    downloadJobs[spec.id] = viewModelScope.launch {
      try {
        spec.files.forEach { file ->
          app.repository.download(spec, file).collect { progress ->
            _state.update { it.copy(downloads = it.downloads + (spec.id to progress)) }
          }
        }
        _state.update { it.copy(downloads = it.downloads - spec.id) }
        refreshModels()
      } catch (e: Throwable) {
        _state.update {
          it.copy(downloads = it.downloads - spec.id, error = "${spec.displayName}: ${e.message}")
        }
      }
    }
  }

  fun cancelDownload(spec: ModelSpec) {
    downloadJobs.remove(spec.id)?.cancel()
    _state.update { it.copy(downloads = it.downloads - spec.id) }
  }

  fun deleteModel(spec: ModelSpec) {
    app.repository.delete(spec)
    refreshModels()
  }

  /**
   * Runs the smoke-test model on every accelerator.
   *
   * Uses the small benchmark model rather than a pipeline stage on purpose: it answers "does
   * the NPU work here, and by how much" in seconds, without a multi-gigabyte download in the
   * way, and it is the only model in the catalogue every accelerator will accept.
   */
  fun runBenchmark() {
    if (_state.value.benchmarkRunning) return
    val spec = ModelCatalog.BENCHMARK_MOBILENET
    val file = app.repository.fileFor(spec, spec.files.first())
    if (!file.isFile) {
      _state.update { it.copy(error = "Download the benchmark model first (${spec.displayName}).") }
      return
    }
    viewModelScope.launch {
      _state.update { it.copy(benchmarkRunning = true, error = null) }
      try {
        val report = Benchmark.compare(app, spec, file)
        _state.update { it.copy(benchmark = report, benchmarkRunning = false) }
        // A benchmark that reached the NPU is proof the probe's static answer was right.
        val verified = report.results.filter { r -> r.ok }.map { r -> r.accelerator }
        _state.update { s -> s.copy(device = s.device?.copy(verifiedAccelerators = verified)) }
      } catch (e: Throwable) {
        _state.update { it.copy(benchmarkRunning = false, error = e.message) }
      }
    }
  }

  fun generate() {
    if (generationJob?.isActive == true) return
    generationJob = viewModelScope.launch {
      _state.update { it.copy(error = null, result = null) }
      try {
        val result = app.pipeline.generate(
          GenerationRequest(
            prompt = prompt.value,
            seed = seed.value,
            steps = steps.value,
            target = target.value,
            strategy = strategy.value,
            policy = policy.value,
          )
        ) { progress -> _state.update { it.copy(generation = progress) } }
        _state.update { it.copy(result = result, generation = null) }
      } catch (e: Throwable) {
        _state.update { it.copy(error = e.message ?: e.javaClass.simpleName, generation = null) }
      }
    }
  }

  fun cancelGeneration() {
    generationJob?.cancel()
    _state.update { it.copy(generation = null) }
  }

  fun dismissError() = _state.update { it.copy(error = null) }

  fun randomizeSeed() { seed.value = (Math.random() * Long.MAX_VALUE).toLong() }

  fun setPolicy(next: AcceleratorPolicy) {
    policy.value = next
    prefs().edit().putString(KEY_POLICY, next.name).apply()
  }

  /**
   * Resolved on each call rather than held in a field.
   *
   * A property initialiser that reads another property only works if the other one is
   * declared above it — and [policy] is initialised from [loadPolicy], so a field here
   * silently depended on declaration order. It did not survive contact: moving the field
   * below `policy` left its delegate null and crashed the app on launch. The framework
   * caches SharedPreferences instances, so calling for it is cheap and the ordering
   * hazard simply stops existing.
   */
  private fun prefs(): android.content.SharedPreferences =
    app.getSharedPreferences(PREFS_NAME, android.content.Context.MODE_PRIVATE)

  private fun loadPolicy(): AcceleratorPolicy =
    AcceleratorPolicy.fromStoredName(prefs().getString(KEY_POLICY, null))


  /** Writes the result as a lossless PNG; a 4K render is not something to re-encode as JPEG. */
  fun saveResult(bitmap: Bitmap) {
    // Encoding a 4096x4096 PNG takes seconds; doing it on the main thread would freeze the
    // UI at exactly the moment the user is admiring the result.
    viewModelScope.launch(Dispatchers.IO) {
      try {
        val dir = File(app.filesDir, "renders").apply { mkdirs() }
        val out = File(dir, "neuroforge_${System.currentTimeMillis()}.png")
        FileOutputStream(out).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        _state.update { it.copy(error = "Saved to ${out.name}") }
      } catch (e: Throwable) {
        _state.update { it.copy(error = "Save failed: ${e.message}") }
      }
    }
  }


  companion object {
    private const val PREFS_NAME = "neuroforge"
    private const val KEY_POLICY = "accelerator_policy"

    fun factory(app: NeuroForgeApp) = viewModelFactory {
      initializer { AppViewModel(app) }
    }
  }
}
