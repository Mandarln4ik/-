package dev.neuroforge.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.ExpandMore
// Deliberately Photo rather than Image: `androidx.compose.foundation.Image` is the
// composable this file uses to draw a thumbnail, and importing an icon of the same name
// puts two different `Image` declarations in scope for no benefit.
import androidx.compose.material.icons.filled.Photo
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import android.graphics.BitmapFactory
import dev.neuroforge.core.Attachment
import dev.neuroforge.core.AttachmentKind
import dev.neuroforge.core.Chat
import dev.neuroforge.core.ChatKind
import dev.neuroforge.core.ChatMessage
import dev.neuroforge.core.OutputTarget
import dev.neuroforge.core.Speaker
import dev.neuroforge.core.StopReason
import dev.neuroforge.core.UpscaleStrategy
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Conversations, one per kind of thing you can generate.
 *
 * Kinds are separate rather than one chat that infers what you meant: a text reply, an
 * image and a piece of audio need different models and different controls, and a chat that
 * guesses gets it wrong exactly when the prompt is ambiguous.
 *
 * Image generation lives here too rather than on its own tab. A prompt, its settings and
 * its result are one thing; splitting them across two screens meant the picture appeared
 * somewhere other than where it was asked for, and the seed that produced it was on a
 * third.
 */
@Composable
fun ChatsScreen(vm: AppViewModel, state: UiState) {
  val active = state.activeChat
  if (active == null) {
    ChatList(vm, state)
  } else {
    ChatThread(vm, state, active)
  }
}

@Composable
private fun ChatList(vm: AppViewModel, state: UiState) {
  var kind by remember { mutableStateOf(ChatKind.TEXT) }
  val models = vm.availableModels(kind)

  LazyColumn(Modifier.fillMaxSize()) {
    item {
      SectionCard("New chat") {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          ChatKind.entries.forEach { k ->
            FilterChip(
              selected = kind == k,
              onClick = { kind = k },
              label = { Text(k.label) },
            )
          }
        }

        val reason = kind.unsupportedReason
        if (reason != null) {
          Text(
            reason,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 12.dp),
          )
        } else if (models.isEmpty()) {
          Text(
            "No ${kind.label.lowercase()} model downloaded yet. Get one from the Models tab.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 12.dp),
          )
        } else {
          Text(
            "Pick a model — it stays fixed for the whole conversation, so replies are " +
              "always attributable to one thing.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 12.dp, bottom = 4.dp),
          )
          models.forEach { spec ->
            OutlinedButton(
              onClick = { vm.createChat(kind, spec.id) },
              modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
            ) { Text(spec.displayName) }
          }
        }
      }
    }

    if (state.chats.isNotEmpty()) {
      item {
        SectionCard("Conversations") {
          // Most recently used first: the one being worked on is the one to hand.
          state.orderedChats.forEach { chat ->
            Row(
              Modifier.fillMaxWidth().padding(vertical = 4.dp),
              horizontalArrangement = Arrangement.SpaceBetween,
              verticalAlignment = Alignment.CenterVertically,
            ) {
              Column(
                Modifier.weight(1f).clickable { vm.openChat(chat.id) },
              ) {
                Text(chat.derivedTitle(), style = MaterialTheme.typography.bodyMedium)
                Text(
                  "${chat.kind.label} · ${chat.messages.size} messages",
                  style = MaterialTheme.typography.labelSmall,
                  color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
              }
              IconButton(onClick = { vm.deleteChat(chat.id) }) {
                Icon(
                  Icons.Filled.Delete,
                  contentDescription = "Delete ${chat.derivedTitle()}",
                  tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
              }
            }
          }
        }
      }
    }
  }
}

@Composable
private fun ChatThread(vm: AppViewModel, state: UiState, chat: Chat) {
  var draft by remember(chat.id) { mutableStateOf("") }
  var showSettings by remember(chat.id) { mutableStateOf(false) }

  Column(Modifier.fillMaxSize()) {
    Row(
      Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
      horizontalArrangement = Arrangement.SpaceBetween,
      verticalAlignment = Alignment.CenterVertically,
    ) {
      Column(Modifier.weight(1f)) {
        Text(chat.derivedTitle(), style = MaterialTheme.typography.titleSmall)
        Text(
          "${chat.kind.label} · ${chat.modelId}",
          style = MaterialTheme.typography.labelSmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
      }
      if (chat.kind == ChatKind.IMAGE) {
        IconButton(onClick = { showSettings = !showSettings }) {
          Icon(Icons.Filled.Tune, contentDescription = "Generation settings")
        }
      }
      OutlinedButton(onClick = { vm.openChat(null) }) { Text("Back") }
    }

    AnimatedVisibility(visible = showSettings) {
      ImageSettings(vm)
    }

    // Loading a text model is one blocking native call with no progress channel, so this
    // reports the stage and how long it has been going rather than a bar that would be
    // decoration. The first NPU load can include just-in-time compilation and take tens of
    // seconds, which is otherwise indistinguishable from a hang.
    state.llmLoad?.let { load ->
      Surface(
        color = MaterialTheme.colorScheme.secondaryContainer,
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
      ) {
        Column(Modifier.padding(12.dp)) {
          Text("${load.stage} — ${load.elapsedMillis / 1000}s", style = MaterialTheme.typography.bodySmall)
          LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 8.dp))
          load.tried.forEach { failure ->
            Text(
              failure,
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.onSecondaryContainer,
              modifier = Modifier.padding(top = 6.dp),
            )
          }
        }
      }
    }

    LazyColumn(Modifier.weight(1f).fillMaxWidth()) {
      items(chat.messages.size) { index -> Bubble(chat.messages[index]) }

      if (state.replying) {
        item {
          Column(Modifier.padding(16.dp)) {
            state.generation?.let { progress ->
              LinearProgressIndicator(
                progress = { progress.fraction },
                modifier = Modifier.fillMaxWidth(),
              )
              Text(
                progress.detail,
                style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.padding(top = 6.dp),
              )
            } ?: Text("Thinking…", style = MaterialTheme.typography.bodySmall)
            OutlinedButton(
              onClick = vm::cancelReply,
              modifier = Modifier.padding(top = 10.dp),
            ) { Text("Stop") }
          }
        }
      }
    }

    // How much of the window is left, so a reply that stops early is explicable before it
    // happens rather than after.
    state.contextBudget?.let { budget ->
      Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        LinearProgressIndicator(
          progress = { budget.fraction },
          modifier = Modifier.fillMaxWidth(),
          color = if (budget.tight) MaterialTheme.colorScheme.error
          else MaterialTheme.colorScheme.primary,
        )
        Text(
          budget.line() + if (budget.tight) " — start a new chat soon" else "",
          style = MaterialTheme.typography.labelSmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
          modifier = Modifier.padding(top = 4.dp),
        )
      }
    }

    if (chat.kind == ChatKind.TEXT && state.pending.isNotEmpty()) {
      PendingAttachments(state.pending, onRemove = vm::detach)
    }

    Row(
      Modifier.fillMaxWidth().padding(12.dp),
      horizontalArrangement = Arrangement.spacedBy(8.dp),
      verticalAlignment = Alignment.CenterVertically,
    ) {
      if (chat.kind == ChatKind.TEXT) {
        // Any MIME type, because the picker's type filter cannot express "text-ish": most
        // source files come back as octet-stream, and filtering on text/* would hide the
        // files this is most wanted for. What can actually be used is decided on the way
        // in, where the refusal can name the file.
        val picker = rememberLauncherForActivityResult(
          ActivityResultContracts.OpenDocument(),
        ) { uri -> uri?.let(vm::attach) }
        IconButton(
          onClick = { picker.launch(arrayOf("*/*")) },
          enabled = !state.replying,
        ) {
          Icon(Icons.Default.AttachFile, contentDescription = "Attach a file or photo")
        }
      }
      OutlinedTextField(
        value = draft,
        onValueChange = { draft = it },
        modifier = Modifier.weight(1f),
        enabled = !state.replying,
        placeholder = {
          Text(if (chat.kind == ChatKind.IMAGE) "Describe an image" else "Message")
        },
      )
      Button(
        onClick = { vm.send(draft); draft = "" },
        enabled = !state.replying && (draft.isNotBlank() || state.pending.isNotEmpty()),
      ) { Text("Send") }
    }
  }
}

/** The controls that used to be their own tab, next to the images they produce. */
@Composable
private fun ImageSettings(vm: AppViewModel) {
  val seed by vm.seed.collectAsStateWithLifecycle()
  val steps by vm.steps.collectAsStateWithLifecycle()
  val target by vm.target.collectAsStateWithLifecycle()
  val strategy by vm.strategy.collectAsStateWithLifecycle()

  SectionCard("Generation") {
    Text("Output", style = MaterialTheme.typography.labelSmall)
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
      OutputTarget.presets.take(3).forEach { preset ->
        FilterChip(
          selected = target == preset,
          onClick = { vm.setTarget(preset) },
          label = { Text(preset.label) },
        )
      }
    }

    Text("Upscale", style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(top = 10.dp))
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
      UpscaleStrategy.entries.forEach { s ->
        FilterChip(
          selected = strategy == s,
          onClick = { vm.setStrategy(s) },
          label = { Text(if (s == UpscaleStrategy.FAST) "Fast" else "Quality") },
        )
      }
    }

    Text(
      "Steps: $steps",
      style = MaterialTheme.typography.labelSmall,
      modifier = Modifier.padding(top = 10.dp),
    )
    Slider(
      value = steps.toFloat(),
      onValueChange = { vm.steps.value = it.toInt().coerceAtLeast(1) },
      valueRange = 1f..12f,
      steps = 10,
    )
    Text(
      "This model is step-distilled for 4. More steps change the image rather than " +
        "improving it.",
      style = MaterialTheme.typography.labelSmall,
      color = MaterialTheme.colorScheme.onSurfaceVariant,
    )

    Row(
      Modifier.fillMaxWidth().padding(top = 10.dp),
      horizontalArrangement = Arrangement.SpaceBetween,
      verticalAlignment = Alignment.CenterVertically,
    ) {
      Text("Seed $seed", style = MaterialTheme.typography.bodySmall)
      OutlinedButton(onClick = vm::randomizeSeed) { Text("Randomize") }
    }
  }
}

@Composable
private fun Bubble(message: ChatMessage) {
  val fromUser = message.speaker == Speaker.USER
  Row(
    Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp),
    horizontalArrangement = if (fromUser) Arrangement.End else Arrangement.Start,
  ) {
    Card(
      colors = CardDefaults.cardColors(
        containerColor = if (fromUser) MaterialTheme.colorScheme.primary
        else MaterialTheme.colorScheme.surfaceVariant,
      ),
      modifier = Modifier.fillMaxWidth(0.88f),
    ) {
      Column(Modifier.padding(12.dp)) {
        message.thinking?.let { ThinkingPanel(it) }

        message.imagePath?.let { path ->
          val bitmap = remember(path) {
            runCatching { BitmapFactory.decodeFile(path) }.getOrNull()
          }
          if (bitmap != null) {
            Image(
              bitmap = bitmap.asImageBitmap(),
              contentDescription = message.text,
              contentScale = ContentScale.Fit,
              modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(bitmap.width.toFloat() / bitmap.height)
                .clip(RoundedCornerShape(8.dp))
                .padding(bottom = 8.dp),
            )
          } else {
            Text(
              "Image missing from storage",
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.error,
            )
          }
        }

        if (message.text.isNotBlank()) {
          Text(
            message.text,
            style = MaterialTheme.typography.bodyMedium,
            color = if (fromUser) MaterialTheme.colorScheme.onPrimary
            else MaterialTheme.colorScheme.onSurface,
          )
        }

        // A text attachment is folded into the prompt, so the text above is the question
        // alone and nothing in the bubble would otherwise say which file it was about.
        if (message.attachments.isNotEmpty()) {
          SentAttachments(message.attachments)
        }

        // Which accelerator answered, what it cost, and why it stopped. Bound to a local
        // because a property from another module cannot be smart-cast after a null check.
        val stats = message.stats
        val accel = message.accelerator
        if (!fromUser && stats != null) {
          Text(
            stats.line(accel),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 6.dp),
          )
          // The detail only exists when something went wrong, and that is exactly when a
          // one-word status is not enough to act on.
          if (stats.stop == StopReason.ERROR || stats.stop == StopReason.EMPTY) {
            stats.detail?.let { why ->
              Text(
                why,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(top = 4.dp),
              )
            }
          }
        } else if (!fromUser && accel != null) {
          Text(
            accel.name + if (message.millis > 0) " · ${message.millis / 1000.0}s" else "",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 6.dp),
          )
        }
      }
    }
  }
}

/**
 * The model's reasoning, collapsed by default.
 *
 * Reasoning models emit more working than answer, so inline it buries the reply; dropped
 * entirely it takes away the one thing that explains a wrong result. Grey and behind an
 * arrow is the compromise: present, clearly not the answer, one tap away.
 */
@Composable
private fun ThinkingPanel(thinking: String) {
  var expanded by remember { mutableStateOf(false) }
  val rotation by animateFloatAsState(if (expanded) 180f else 0f, label = "thinking-arrow")

  Surface(
    color = MaterialTheme.colorScheme.surfaceVariant,
    shape = RoundedCornerShape(8.dp),
    modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
  ) {
    Column(Modifier.padding(horizontal = 10.dp, vertical = 6.dp)) {
      Row(
        Modifier.fillMaxWidth().clickable { expanded = !expanded },
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
      ) {
        Text(
          "Reasoning",
          style = MaterialTheme.typography.labelMedium,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Icon(
          Icons.Filled.ExpandMore,
          contentDescription = if (expanded) "Hide reasoning" else "Show reasoning",
          tint = MaterialTheme.colorScheme.onSurfaceVariant,
          modifier = Modifier.rotate(rotation),
        )
      }
      AnimatedVisibility(
        visible = expanded,
        enter = expandVertically() + fadeIn(),
        exit = shrinkVertically() + fadeOut(),
      ) {
        Text(
          thinking,
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
          modifier = Modifier.padding(top = 6.dp, bottom = 4.dp),
        )
      }
    }
  }
}

/**
 * What is staged for the next message, with a way to take it back.
 *
 * Shown above the input rather than inside it because a photo attached by mistake should
 * be obvious before Send, not after: an image costs a few hundred tokens of context and,
 * on a model that cannot see, the whole turn.
 */
@Composable
private fun PendingAttachments(
  pending: List<Attachment>,
  onRemove: (Attachment) -> Unit,
) {
  Row(
    Modifier
      .fillMaxWidth()
      .horizontalScroll(rememberScrollState())
      .padding(horizontal = 12.dp),
    horizontalArrangement = Arrangement.spacedBy(6.dp),
  ) {
    pending.forEach { attachment ->
      AssistChip(
        onClick = { onRemove(attachment) },
        label = { Text(attachment.fileName, maxLines = 1) },
        leadingIcon = { AttachmentIcon(attachment) },
        trailingIcon = {
          Icon(
            Icons.Default.Close,
            contentDescription = "Remove ${attachment.fileName}",
            modifier = Modifier.size(16.dp),
          )
        },
      )
    }
  }
}

/** The files a turn was sent with, under the bubble. */
@Composable
private fun SentAttachments(attachments: List<Attachment>) {
  Column(Modifier.padding(top = 6.dp)) {
    attachments.forEach { attachment ->
      Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
      ) {
        AttachmentIcon(attachment)
        Text(
          attachment.fileName,
          style = MaterialTheme.typography.labelSmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
          maxLines = 1,
        )
      }
      // A photo is worth showing; a text file was folded into the prompt and its name is
      // all there is to see.
      if (attachment.kind == AttachmentKind.IMAGE) {
        AsyncAttachmentImage(attachment)
      }
    }
  }
}

@Composable
private fun AttachmentIcon(attachment: Attachment) {
  Icon(
    if (attachment.kind == AttachmentKind.IMAGE) Icons.Default.Photo else Icons.Default.Description,
    contentDescription = null,
    modifier = Modifier.size(16.dp),
  )
}

/**
 * A thumbnail of an attached photo.
 *
 * Decoded at a bounded size with `inSampleSize` rather than loaded whole: a modern phone
 * camera produces a 12-megapixel JPEG, which is 48 MB as an ARGB bitmap and enough to
 * push a chat with a few photos in it straight into an OutOfMemoryError.
 */
@Composable
private fun AsyncAttachmentImage(attachment: Attachment) {
  val bitmap by produceState<android.graphics.Bitmap?>(null, attachment.localPath) {
    value = withContext(Dispatchers.IO) {
      runCatching {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(attachment.localPath, bounds)
        var sample = 1
        while (bounds.outWidth / sample > THUMBNAIL_PX || bounds.outHeight / sample > THUMBNAIL_PX) {
          sample *= 2
        }
        BitmapFactory.decodeFile(
          attachment.localPath,
          BitmapFactory.Options().apply { inSampleSize = sample },
        )
      }.getOrNull()
    }
  }
  bitmap?.let {
    Image(
      bitmap = it.asImageBitmap(),
      contentDescription = attachment.fileName,
      modifier = Modifier
        .padding(top = 4.dp)
        .heightIn(max = 180.dp)
        .clip(RoundedCornerShape(8.dp)),
    )
  }
}

private const val THUMBNAIL_PX = 512
