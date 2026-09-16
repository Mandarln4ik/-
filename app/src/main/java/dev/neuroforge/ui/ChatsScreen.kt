package dev.neuroforge.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import android.graphics.BitmapFactory
import dev.neuroforge.core.Chat
import dev.neuroforge.core.ChatKind
import dev.neuroforge.core.ChatMessage
import dev.neuroforge.core.Speaker
import java.io.File

/**
 * Conversations, one per kind of thing you can generate.
 *
 * Kinds are separate rather than one chat that infers what you meant: a text reply, an
 * image and a piece of audio need different models and different controls, and a chat that
 * guesses gets it wrong exactly when the prompt is ambiguous.
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
          state.chats.forEach { chat ->
            Row(
              Modifier.fillMaxWidth().padding(vertical = 4.dp),
              horizontalArrangement = Arrangement.SpaceBetween,
              verticalAlignment = Alignment.CenterVertically,
            ) {
              Column(Modifier.weight(1f)) {
                Text(chat.derivedTitle(), style = MaterialTheme.typography.bodyMedium)
                Text(
                  "${chat.kind.label} · ${chat.messages.size} messages",
                  style = MaterialTheme.typography.labelSmall,
                  color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
              }
              OutlinedButton(onClick = { vm.openChat(chat.id) }) { Text("Open") }
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
      OutlinedButton(onClick = { vm.openChat(null) }) { Text("Back") }
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

    Row(
      Modifier.fillMaxWidth().padding(12.dp),
      horizontalArrangement = Arrangement.spacedBy(8.dp),
      verticalAlignment = Alignment.CenterVertically,
    ) {
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
        enabled = !state.replying && draft.isNotBlank(),
      ) { Text("Send") }
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

        // Which accelerator answered, and how long it took: the two facts that explain a
        // slow reply without guessing.
        if (!fromUser && message.accelerator != null) {
          Text(
            buildString {
              append(message.accelerator.name)
              if (message.millis > 0) append(" · ${message.millis / 1000.0}s")
            },
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 6.dp),
          )
        }
      }
    }
  }
}
