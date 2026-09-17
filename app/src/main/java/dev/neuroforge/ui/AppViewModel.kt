package dev.neuroforge.ui

import android.app.Application
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import dev.neuroforge.NeuroForgeApp
import dev.neuroforge.core.Accel
import dev.neuroforge.core.Bonsai
import dev.neuroforge.core.AcceleratorPolicy
import dev.neuroforge.core.Chat
import dev.neuroforge.core.ChatKind
import dev.neuroforge.core.ChatMessage
import dev.neuroforge.core.Speaker
import dev.neuroforge.core.modelsFor
import dev.neuroforge.core.BenchmarkReport
import dev.neuroforge.core.ContextBudget
import dev.neuroforge.core.ReplyStats
import dev.neuroforge.core.StopReason
import dev.neuroforge.core.TextBackend
import dev.neuroforge.core.backendFor
import dev.neuroforge.core.contextUsed
import dev.neuroforge.core.estimateTokens
import dev.neuroforge.core.mostRecentFirst
import dev.neuroforge.core.splitThinking
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
import dev.neuroforge.runtime.TextEngine
import dev.neuroforge.runtime.TextEngines
import dev.neuroforge.runtime.LlmLoadProgress
import dev.neuroforge.runtime.SamplingOptions
import dev.neuroforge.pipeline.GenerationProgress
import dev.neuroforge.pipeline.GenerationRequest
import dev.neuroforge.pipeline.GenerationResult
import dev.neuroforge.runtime.AcceleratorProbe
import dev.neuroforge.runtime.Benchmark
import dev.neuroforge.runtime.DeviceReport
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.CancellationException
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
  val plan: RenderPlan? = null,
  val diskUsage: Long = 0L,

  val chats: List<Chat> = emptyList(),
  val activeChatId: String? = null,
  val replying: Boolean = false,

  /** Where the text engine has got to while loading; null once it is ready or idle. */
  val llmLoad: LlmLoadProgress? = null,

  /** Context window the loaded engine opened, for the gauge above the input. */
  val llmContextTokens: Int = 0,

  val browseRepo: String = "",
  val browseFiles: List<RepoFile> = emptyList(),
  val browsing: Boolean = false,

  val error: String? = null,
) {
  val activeChat: Chat? get() = chats.firstOrNull { it.id == activeChatId }

  /** Conversations in the order to list them: the one being worked on first. */
  val orderedChats: List<Chat> get() = chats.mostRecentFirst()

  /**
   * How much of the context window the active conversation has spent.
   *
   * Null until a model has actually loaded, because the limit comes from the engine rather
   * than from the setting: a model can refuse the requested window and open a smaller one,
   * and a gauge drawn against a number the engine did not accept would be fiction.
   */
  val contextBudget: ContextBudget?
    get() {
      val chat = activeChat ?: return null
      if (chat.kind != ChatKind.TEXT || llmContextTokens <= 0) return null
      val parts = chat.messages.flatMap { listOfNotNull(it.text, it.thinking) }
      return ContextBudget(contextUsed(SYSTEM_PROMPT, parts), llmContextTokens)
    }

  companion object {
    /** Kept here so the context estimate counts the same prompt the engine was given. */
    const val SYSTEM_PROMPT =
      "You are a concise assistant running entirely on this phone. Keep answers short."
  }
}

class AppViewModel(private val app: NeuroForgeApp) : ViewModel() {

  private val _state = MutableStateFlow(UiState())
  val state: StateFlow<UiState> = _state.asStateFlow()

  // Generation settings live outside UiState: they change on every drag of a slider, and
  // folding them into the same object would recompose the device panel and the model list
  // too. The prompt itself is the chat's draft, not a setting.
  val seed = MutableStateFlow(7L)
  // The count the model was step-distilled for. Named rather than typed so the UI default
  // and the pipeline default cannot drift apart.
  val steps = MutableStateFlow(Bonsai.DEFAULT_STEPS)
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

  /**
   * Which text runtime to prefer.
   *
   * This filters what the model list offers. It does not decide what opens a given file —
   * that follows the extension, because a `.gguf` cannot be read by LiteRT-LM whatever is
   * selected here.
   */
  val textBackend = MutableStateFlow(TextBackend.fromStoredName(prefs().getString(KEY_BACKEND, null)))

  // ---- Inference settings -------------------------------------------------------------
  // These four are what LiteRT-LM actually exposes: `maxNumTokens` on the engine and the
  // three sampler fields on the conversation. There is deliberately no KV-cache dtype among
  // them; the Settings screen says why it cannot be one.
  val contextTokens = MutableStateFlow(prefs().getInt(KEY_CONTEXT, 2048))
  val temperature = MutableStateFlow(prefs().getFloat(KEY_TEMPERATURE, 0.7f).toDouble())
  val topK = MutableStateFlow(prefs().getInt(KEY_TOP_K, 40))
  val topP = MutableStateFlow(prefs().getFloat(KEY_TOP_P, 0.9f).toDouble())

  /**
   * Hugging Face access token, for gated and private repositories.
   *
   * Kept in the same preferences file as everything else, which is app-private storage.
   * That is not a secret store — it is the same protection the rest of the app's state
   * gets, and a read token for a public model hub is the right thing at that level.
   */
  val hfToken = MutableStateFlow(prefs().getString(KEY_HF_TOKEN, "").orEmpty())

  private var generationJob: Job? = null
  private val downloadJobs = mutableMapOf<String, Job>()

  /**
   * Collaborators, declared above [init] — deliberately, and this is load-bearing.
   *
   * `viewModelScope` runs on `Dispatchers.Main.immediate`, so a `launch` from `init` on the
   * main thread does **not** defer: the body executes synchronously up to its first
   * suspension point. `loadChats()` reaches [chatStore] before `load()` can suspend, and if
   * the delegate for a `by lazy` further down the file has not been assigned yet, that read
   * is a NullPointerException on a null `Lazy` — the crash v1.1.0 shipped with, in the same
   * class, from the same cause.
   *
   * Kotlin assigns property initialisers in declaration order and gives no warning when one
   * runs early. So anything an `init` path can touch belongs here, above it.
   */
  private val chatStore by lazy { ChatStore(app) }

  /** Reads the token through a supplier rather than copying it, so a later paste reaches it. */
  private val browser by lazy { HfBrowser { hfToken.value.takeIf { t -> t.isNotBlank() } } }

  init {
    applyToken(hfToken.value)
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

  /**
   * Downloads every graph a text-to-4K run needs, one after another.
   *
   * Four separate buttons for one capability is a quiz about which four; this is the answer.
   * Sequential rather than parallel because the four together are about four gigabytes, and
   * four concurrent streams on a phone connection finish later than four consecutive ones
   * while making the progress bar useless.
   */
  fun downloadImagePipeline() {
    if (downloadJobs[IMAGE_PIPELINE_JOB]?.isActive == true) return
    downloadJobs[IMAGE_PIPELINE_JOB] = viewModelScope.launch {
      for (spec in ModelCatalog.textTo4kPipeline) {
        if (app.repository.isReady(spec)) continue
        try {
          spec.files.forEachIndexed { index, file ->
            app.repository.download(spec, file).collect { progress ->
              val labelled = progress.copy(
                fileName = "${file.fileName} (${index + 1}/${spec.files.size})"
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
          // One missing graph makes the pipeline unusable, so there is nothing to gain by
          // downloading the rest; stopping here also leaves the error on screen.
          break
        }
      }
      downloadJobs.remove(IMAGE_PIPELINE_JOB)
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

  /** Models the benchmark can drive, most appropriate first. */
  fun benchmarkable(): List<ModelSpec> =
    listOf(ModelCatalog.BENCHMARK_MOBILENET, ModelCatalog.UPSCALER_ESRGAN_X4)
      .filter { it.inputShape != null }

  fun cancelGeneration() {
    generationJob?.cancel()
    _state.update { it.copy(generation = null) }
  }

  // ---- chats ------------------------------------------------------------------------

  private var llm: TextEngine? = null
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
  /**
   * Downloaded models that can drive a chat of this kind.
   *
   * Text chats are additionally filtered to the preferred runtime, so the list offers what
   * the chosen backend reads. A model whose format belongs to a different backend is not
   * hidden from the Models tab - it is only left out of the "new chat" list, where picking
   * it would fail at load rather than at the click.
   */
  fun availableModels(kind: ChatKind): List<ModelSpec> =
    modelsFor(kind)
      .filter { app.repository.isReady(it) }
      .filter { spec ->
        kind != ChatKind.TEXT ||
          spec.files.any { backendFor(it.fileName) == textBackend.value }
      }

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
  private fun updateLastModelMessage(chatId: String, text: String, thinking: String? = null) {
    _state.update { s ->
      s.copy(
        chats = s.chats.map { chat ->
          if (chat.id != chatId || chat.messages.isEmpty()) chat
          else {
            val last = chat.messages.last()
            if (last.speaker != Speaker.MODEL) chat
            else chat.copy(
              messages = chat.messages.dropLast(1) + last.copy(text = text, thinking = thinking)
            )
          }
        }
      )
    }
  }

  /** Writes the telemetry onto the reply that has just finished streaming. */
  private fun finishLastModelMessage(
    chatId: String,
    answer: String,
    thinking: String?,
    stats: ReplyStats,
    millis: Long,
  ) {
    _state.update { s ->
      s.copy(
        chats = s.chats.map { c ->
          if (c.id != chatId || c.messages.isEmpty()) c
          else c.copy(
            messages = c.messages.dropLast(1) + c.messages.last().copy(
              text = answer, thinking = thinking, stats = stats, millis = millis,
            )
          )
        }
      )
    }
  }

  /**
   * Forgets a conversation.
   *
   * Also drops the engine when the deleted chat was the one it was loaded for: the engine
   * owns the KV cache for that conversation, and keeping it alive would have the next chat
   * on the same model inherit turns that no longer exist anywhere in the UI.
   */
  fun deleteChat(id: String) {
    val chat = _state.value.chats.firstOrNull { it.id == id }
    if (chat != null && chat.modelId == loadedLlmModelId) reloadEngine()
    _state.update { s ->
      s.copy(
        chats = s.chats.filterNot { it.id == id },
        activeChatId = if (s.activeChatId == id) null else s.activeChatId,
      )
    }
    persistChats()
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
      llm = null
      loadedLlmModelId = null
      try {
        llm = TextEngines.load(
          context = app,
          modelFile = file,
          policy = policy.value,
          contextTokens = contextTokens.value,
          sampling = SamplingOptions(temperature.value, topK.value, topP.value),
          // ExecuTorch keeps the tokenizer outside the model, so it is a second file in
          // the same spec; the other backends carry theirs inside and ignore this.
          tokenizerFile = spec.files
            .firstOrNull { it.fileName.endsWith(".json") || it.fileName.endsWith(".model") }
            ?.let { app.repository.fileFor(spec, it) },
        ) { progress -> _state.update { it.copy(llmLoad = progress) } }
      } finally {
        _state.update { it.copy(llmLoad = null) }
      }
      loadedLlmModelId = chat.modelId
      llm?.startConversation(
        UiState.SYSTEM_PROMPT,
        SamplingOptions(temperature.value, topK.value, topP.value),
      )
    }
    val engine = llm ?: error("Text engine unavailable")
    _state.update { it.copy(llmContextTokens = engine.contextTokens) }

    val sent = SystemClock.elapsedRealtime()
    var firstToken = 0L
    val raw = StringBuilder()
    var stop = StopReason.COMPLETE
    var detail: String? = null

    appendMessage(
      chat.id,
      ChatMessage(
        Speaker.MODEL, "",
        accelerator = engine.accelerator,
        timestamp = System.currentTimeMillis(),
      ),
    )

    try {
      engine.send(prompt).collect { chunk ->
        if (chunk.isEmpty()) return@collect
        // The first token is when prefill - and on an NPU any just-in-time compilation -
        // has finished. It is usually most of a short reply's wall clock, so it is timed
        // separately rather than folded into a tokens-per-second figure that would then
        // describe neither phase.
        if (firstToken == 0L) firstToken = SystemClock.elapsedRealtime()
        raw.append(chunk)
        val (thinking, answer) = splitThinking(raw.toString())
        updateLastModelMessage(chat.id, answer, thinking)
      }
    } catch (e: CancellationException) {
      stop = StopReason.CANCELLED
      throw e
    } catch (e: Throwable) {
      stop = StopReason.ERROR
      detail = e.message ?: e.javaClass.simpleName
      Log.w(TAG, "generation failed", e)
    } finally {
      val done = SystemClock.elapsedRealtime()
      val (thinking, answer) = splitThinking(raw.toString())
      // A backend that owns its decode loop knows why it stopped; the guesses below are
      // only for the ones that just close the stream. An exception already set `stop`, so
      // this never overrides a real failure with the runtime's own tidier account of it.
      if (stop == StopReason.COMPLETE) {
        llm?.lastStop?.let { stop = it }
      }
      val counted = llm?.lastTokens?.takeIf { it > 0 }
      if (raw.isEmpty() && stop == StopReason.COMPLETE) {
        stop = StopReason.EMPTY
        detail = "The model produced no output. That usually means the context filled up - " +
          "start a new chat, or raise the window in Settings."
      }
      finishLastModelMessage(
        chat.id, answer, thinking,
        ReplyStats(
          ttftMillis = if (firstToken > 0) firstToken - sent else done - sent,
          decodeMillis = if (firstToken > 0) done - firstToken else 0,
          // Counted when the backend counts them, estimated otherwise - which is why
          // the label carries a tilde only in the second case.
          tokens = counted ?: (estimateTokens(answer) + estimateTokens(thinking.orEmpty())),
          stop = stop,
          detail = detail,
          tokensEstimated = counted == null,
        ),
        done - sent,
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
    _state.update { it.copy(generation = null) }
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
   * Changes the context window.
   *
   * Takes effect on the next model load, because `maxNumTokens` is an engine-construction
   * parameter: the KV cache is allocated once, for that size. Reloading here would also
   * discard the conversation the engine holds, which is a worse surprise than waiting.
   */
  fun setContextTokens(next: Int) {
    val clamped = next.coerceIn(512, 32768)
    contextTokens.value = clamped
    prefs().edit().putInt(KEY_CONTEXT, clamped).apply()
  }

  fun setTemperature(next: Double) {
    val clamped = next.coerceIn(0.0, 2.0)
    temperature.value = clamped
    prefs().edit().putFloat(KEY_TEMPERATURE, clamped.toFloat()).apply()
  }

  fun setTopK(next: Int) {
    val clamped = next.coerceIn(1, 200)
    topK.value = clamped
    prefs().edit().putInt(KEY_TOP_K, clamped).apply()
  }

  fun setTopP(next: Double) {
    val clamped = next.coerceIn(0.0, 1.0)
    topP.value = clamped
    prefs().edit().putFloat(KEY_TOP_P, clamped.toFloat()).apply()
  }

  fun setTextBackend(next: TextBackend) {
    textBackend.value = next
    prefs().edit().putString(KEY_BACKEND, next.name).apply()
    reloadEngine()
  }

  /** Drops the loaded engine so the next message picks up changed settings. */
  fun reloadEngine() {
    llm?.close()
    llm = null
    loadedLlmModelId = null
    _state.update { it.copy(llmContextTokens = 0) }
  }

  fun setHfToken(next: String) {
    val cleaned = next.trim()
    hfToken.value = cleaned
    prefs().edit().putString(KEY_HF_TOKEN, cleaned).apply()
    applyToken(cleaned)
  }

  /**
   * Hands the token to the downloader.
   *
   * Only the downloader: [browser] reads [hfToken] through a supplier instead, because
   * this runs from `init` and `browser` is a lazy declared two hundred lines below. That
   * shape — an initialiser reaching a property that has not been assigned yet — is exactly
   * what crashed v1.1.0 on launch, and it compiles without a warning.
   */
  private fun applyToken(token: String) {
    app.repository.accessToken = token.takeIf { it.isNotBlank() }
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



  companion object {
    private const val PREFS_NAME = "neuroforge"
    private const val KEY_POLICY = "accelerator_policy"
    private const val KEY_HF_TOKEN = "hugging_face_token"
    private const val KEY_CONTEXT = "context_tokens"
    private const val KEY_BACKEND = "text_backend"
    private const val KEY_TEMPERATURE = "temperature"
    private const val KEY_TOP_K = "top_k"
    private const val KEY_TOP_P = "top_p"
    private const val TAG = "AppViewModel"

    /** Job key for the whole-pipeline download; not a model id, so it cannot collide. */
    private const val IMAGE_PIPELINE_JOB = "pipeline:text-to-4k"

    fun factory(app: NeuroForgeApp) = viewModelFactory {
      initializer { AppViewModel(app) }
    }
  }
}
