package dev.neuroforge.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import dev.neuroforge.NeuroForgeApp
import dev.neuroforge.core.AcceleratorPolicy
import dev.neuroforge.core.Chat
import dev.neuroforge.core.ChatKind
import dev.neuroforge.core.ChatMessage
import dev.neuroforge.core.Speaker
import dev.neuroforge.core.modelsFor
import dev.neuroforge.core.BenchmarkReport
import dev.neuroforge.core.ModelCatalog
import dev.neuroforge.core.ModelFile
import dev.neuroforge.core.ModelRole
import dev.neuroforge.core.ModelSource
import dev.neuroforge.core.ModelSpec
import dev.neuroforge.core.OutputTarget
import dev.neuroforge.core.RenderPlan
import dev.neuroforge.core.UpscaleStrategy
import dev.neuroforge.models.DownloadProgress
import dev.neuroforge.models.ChatStore
import dev.neuroforge.models.HfBrowser
import dev.neuroforge.models.ModelState
import dev.neuroforge.models.RepoFile
import dev.neuroforge.runtime.LlmEngine
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
  val benchmarkModel: String? = null,
  val generation: GenerationProgress? = null,
  val result: GenerationResult? = null,
  val plan: RenderPlan? = null,
  val diskUsage: Long = 0L,

  val chats: List<Chat> = emptyList(),
  val activeChatId: String? = null,
  val replying: Boolean = false,

  val browseRepo: String = "",
  val browseFiles: List<RepoFile> = emptyList(),
  val browsing: Boolean = false,

  val error: String? = null,
) {
  val activeChat: Chat? get() = chats.firstOrNull { it.id == activeChatId }
}

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
    loadChats()
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

  /**
   * Downloads everything a model needs, as one action.
   *
   * A model is a bundle - weights, a decoder, a tokenizer, metadata - and asking someone to
   * press four buttons and know which four is a worse interface than one button that
   * reports which part it is on.
   */
  fun download(spec: ModelSpec) {
    if (downloadJobs[spec.id]?.isActive == true) return
    downloadJobs[spec.id] = viewModelScope.launch {
      try {
        spec.files.forEachIndexed { index, file ->
          app.repository.download(spec, file).collect { progress ->
            val labelled = progress.copy(
              fileName = if (spec.files.size > 1) {
                "${file.fileName} (${index + 1}/${spec.files.size})"
              } else {
                file.fileName
              }
            )
            _state.update { it.copy(downloads = it.downloads + (spec.id to labelled)) }
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
    // Any downloaded graph with a declared input shape will do. Pinning this to one model
    // is what made the feature unreachable when that model's download failed - and the
    // upscaler is arguably the better subject anyway, since it is the graph the NPU is
    // actually meant to run.
    val spec = benchmarkable().firstOrNull { app.repository.isReady(it) }
    if (spec == null) {
      _state.update {
        it.copy(
          error = "Download a model with a fixed input first — the smoke test or the " +
            "upscaler, either works.",
        )
      }
      return
    }
    val file = app.repository.fileFor(spec, spec.files.first())
    viewModelScope.launch {
      _state.update { it.copy(benchmarkRunning = true, error = null) }
      try {
        val report = Benchmark.compare(app, spec, file)
        _state.update { it.copy(benchmarkModel = spec.displayName) }
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

  /** Models the benchmark can drive, most appropriate first. */
  fun benchmarkable(): List<ModelSpec> =
    listOf(ModelCatalog.BENCHMARK_MOBILENET, ModelCatalog.UPSCALER_ESRGAN_X4)
      .filter { it.inputShape != null }

  fun cancelGeneration() {
    generationJob?.cancel()
    _state.update { it.copy(generation = null) }
  }

  // ---- chats ------------------------------------------------------------------------

  private val chatStore by lazy { ChatStore(app) }
  private val browser by lazy { HfBrowser() }
  private var llm: LlmEngine? = null
  private var loadedLlmModelId: String? = null
  private var replyJob: Job? = null

  private fun loadChats() {
    viewModelScope.launch {
      val stored = chatStore.load()
      _state.update { it.copy(chats = stored) }
    }
  }

  private fun persistChats() {
    viewModelScope.launch { chatStore.save(_state.value.chats) }
  }

  /** Models that can drive a chat of this kind and are already downloaded. */
  fun availableModels(kind: ChatKind): List<ModelSpec> =
    modelsFor(kind).filter { app.repository.isReady(it) }

  fun createChat(kind: ChatKind, modelId: String) {
    val chat = Chat(
      id = "chat-${System.currentTimeMillis()}",
      kind = kind,
      modelId = modelId,
      title = Chat.DEFAULT_TITLE,
      createdAt = System.currentTimeMillis(),
    )
    _state.update { it.copy(chats = listOf(chat) + it.chats, activeChatId = chat.id) }
    persistChats()
  }

  fun openChat(id: String?) = _state.update { it.copy(activeChatId = id) }

  fun deleteChat(id: String) {
    _state.update { s ->
      s.copy(
        chats = s.chats.filterNot { it.id == id },
        activeChatId = if (s.activeChatId == id) null else s.activeChatId,
      )
    }
    persistChats()
  }

  private fun appendMessage(chatId: String, message: ChatMessage) {
    _state.update { s ->
      s.copy(
        chats = s.chats.map { chat ->
          if (chat.id != chatId) chat
          else chat.copy(
            messages = chat.messages + message,
            title = chat.copy(messages = chat.messages + message).derivedTitle(),
          )
        }
      )
    }
  }

  /** Replaces the last model message, so a streamed reply grows in place. */
  private fun updateLastModelMessage(chatId: String, text: String) {
    _state.update { s ->
      s.copy(
        chats = s.chats.map { chat ->
          if (chat.id != chatId || chat.messages.isEmpty()) chat
          else {
            val last = chat.messages.last()
            if (last.speaker != Speaker.MODEL) chat
            else chat.copy(messages = chat.messages.dropLast(1) + last.copy(text = text))
          }
        }
      )
    }
  }

  fun send(text: String) {
    val chat = _state.value.activeChat ?: return
    if (text.isBlank() || _state.value.replying) return

    appendMessage(chat.id, ChatMessage(Speaker.USER, text, timestamp = System.currentTimeMillis()))

    replyJob = viewModelScope.launch {
      _state.update { it.copy(replying = true, error = null) }
      try {
        when (chat.kind) {
          ChatKind.TEXT -> replyWithText(chat, text)
          ChatKind.IMAGE -> replyWithImage(chat, text)
          ChatKind.MUSIC -> error(
            ChatKind.MUSIC.unsupportedReason ?: "Music generation is not available."
          )
        }
      } catch (e: Throwable) {
        _state.update { it.copy(error = e.message ?: e.javaClass.simpleName) }
      } finally {
        _state.update { it.copy(replying = false) }
        persistChats()
      }
    }
  }

  private suspend fun replyWithText(chat: Chat, prompt: String) {
    val spec = ModelCatalog.byId(chat.modelId) ?: error("Model ${chat.modelId} is gone")
    val file = app.repository.fileFor(spec, spec.files.first())

    // The engine holds the conversation, so it is reloaded only when the model changes.
    if (llm == null || loadedLlmModelId != chat.modelId) {
      llm?.close()
      llm = LlmEngine.load(app, file, policy.value)
      loadedLlmModelId = chat.modelId
      llm?.startConversation(
        "You are a concise assistant running entirely on this phone. Keep answers short."
      )
    }
    val engine = llm ?: error("Text engine unavailable")

    val started = System.currentTimeMillis()
    appendMessage(chat.id, ChatMessage(Speaker.MODEL, "", accelerator = engine.accelerator))
    val builder = StringBuilder()
    engine.send(prompt).collect { chunk ->
      builder.append(chunk)
      updateLastModelMessage(chat.id, builder.toString())
    }
    _state.update { s ->
      s.copy(
        chats = s.chats.map { c ->
          if (c.id != chat.id || c.messages.isEmpty()) c
          else c.copy(
            messages = c.messages.dropLast(1) +
              c.messages.last().copy(millis = System.currentTimeMillis() - started)
          )
        }
      )
    }
  }

  private suspend fun replyWithImage(chat: Chat, prompt: String) {
    val result = app.pipeline.generate(
      GenerationRequest(
        prompt = prompt,
        seed = seed.value,
        steps = steps.value,
        target = target.value,
        strategy = strategy.value,
        policy = policy.value,
      )
    ) { progress -> _state.update { it.copy(generation = progress) } }

    val saved = writePng(result.bitmap)
    appendMessage(
      chat.id,
      ChatMessage(
        speaker = Speaker.MODEL,
        text = "${result.bitmap.width}x${result.bitmap.height}",
        imagePath = saved.absolutePath,
        accelerator = result.timings.lastOrNull()?.accelerator,
        millis = result.totalMillis,
        timestamp = System.currentTimeMillis(),
      ),
    )
    _state.update { it.copy(generation = null, result = result) }
  }

  fun cancelReply() {
    replyJob?.cancel()
    _state.update { it.copy(replying = false, generation = null) }
  }

  // ---- browsing Hugging Face ----------------------------------------------------------

  fun setBrowseRepo(value: String) = _state.update { it.copy(browseRepo = value) }

  /**
   * Lists a repository the user named.
   *
   * This is the answer to every download failure in this app so far: rather than trusting
   * a file name someone wrote down, show what is actually there and let it be picked.
   */
  fun browse() {
    val repo = _state.value.browseRepo.trim()
    if (repo.isBlank() || _state.value.browsing) return
    viewModelScope.launch {
      _state.update { it.copy(browsing = true, error = null, browseFiles = emptyList()) }
      try {
        val files = browser.list(repo)
        _state.update { it.copy(browseFiles = files) }
        if (files.none { it.isModel }) {
          _state.update {
            it.copy(error = "No .tflite or .litertlm files in $repo — nothing here to run.")
          }
        }
      } catch (e: Throwable) {
        _state.update { it.copy(error = e.message ?: "Could not list $repo") }
      } finally {
        _state.update { it.copy(browsing = false) }
      }
    }
  }

  /** Downloads one file from a browsed repository into its own model directory. */
  fun downloadFromRepo(repoId: String, file: RepoFile) {
    val spec = customSpec(repoId, file)
    download(spec)
  }

  private fun customSpec(repoId: String, file: RepoFile): ModelSpec {
    val name = file.path.substringAfterLast('/')
    return ModelSpec(
      id = "custom.${repoId.replace('/', '.')}.${name.substringBeforeLast('.')}",
      displayName = "$repoId — $name",
      role = if (name.endsWith(".litertlm", true)) ModelRole.TEXT_CHAT else ModelRole.BENCHMARK,
      accelerators = listOf(Accel.NPU, Accel.GPU, Accel.CPU),
      files = listOf(
        ModelFile(
          fileName = name,
          source = ModelSource.HuggingFace(repoId, file.path),
          sizeBytes = file.sizeBytes,
        )
      ),
      notes = "Added by you from $repoId.",
    )
  }

  private fun writePng(bitmap: Bitmap): File {
    val dir = File(app.filesDir, "renders").apply { mkdirs() }
    val out = File(dir, "neuroforge_${System.currentTimeMillis()}.png")
    FileOutputStream(out).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
    return out
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
