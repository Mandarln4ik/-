package dev.neuroforge.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.neuroforge.core.ModelCatalog
import dev.neuroforge.core.formatBytes
import dev.neuroforge.models.ModelState

@Composable
fun ModelsScreen(vm: AppViewModel, state: UiState) {
  LazyColumn(Modifier.fillMaxSize()) {
    item {
      SectionCard("Weights") {
        Text(
          "Models download straight from Hugging Face to this device and stay here. " +
            "Nothing about a prompt or an image ever leaves the phone.",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
      }
    }

    items(ModelCatalog.all, key = { it.id }) { spec ->
      val files = state.modelStates[spec.id].orEmpty()
      val progress = state.downloads[spec.id]
      val ready = files.values.isNotEmpty() && files.values.all { it is ModelState.Ready }
      val needsConversion = files.values.any { it is ModelState.NeedsConversion }

      SectionCard(spec.displayName) {
        InfoRow("Role", spec.role.name)
        InfoRow("Size", formatBytes(spec.totalBytes))
        InfoRow("Targets", spec.accelerators.joinToString(" → "))
        InfoRow("Source", spec.files.first().originLabel())
        if (spec.notes.isNotEmpty()) {
          Text(
            spec.notes,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 6.dp),
          )
        }

        files.forEach { (name, fileState) ->
          val description = when (fileState) {
            is ModelState.Ready -> "ready (${formatBytes(fileState.bytes)})"
            is ModelState.Absent -> "not downloaded"
            is ModelState.Corrupt -> "problem: ${fileState.reason}"
            is ModelState.NeedsConversion -> "needs conversion — ${fileState.recipe}"
          }
          InfoRow(name, description)
        }

        if (progress != null) {
          Column(Modifier.padding(top = 10.dp)) {
            LinearProgressIndicator(
              progress = { progress.fraction },
              modifier = Modifier.fillMaxWidth(),
            )
            Text(
              "${formatBytes(progress.downloaded)} / ${formatBytes(progress.total)} " +
                "at ${formatBytes(progress.bytesPerSecond)}/s" +
                (if (progress.etaSeconds >= 0) " · ${progress.etaSeconds}s left" else ""),
              style = MaterialTheme.typography.labelSmall,
              modifier = Modifier.padding(top = 6.dp),
            )
          }
        }

        Row(
          Modifier.fillMaxWidth().padding(top = 12.dp),
          horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
          when {
            progress != null ->
              OutlinedButton(onClick = { vm.cancelDownload(spec) }) { Text("Cancel") }
            needsConversion ->
              // No download button for something a download cannot produce: the file has to
              // be converted on a desktop first, and offering the button would just fail.
              Text(
                "Convert it first — see docs/MODELS.md, then push the .tflite to this device.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary,
              )
            ready ->
              OutlinedButton(onClick = { vm.deleteModel(spec) }) { Text("Delete") }
            else ->
              Button(onClick = { vm.download(spec) }) {
                Text(
                  if (spec.files.size > 1) "Download all ${spec.files.size} files"
                  else "Download",
                )
              }
          }
        }
      }
    }

    item {
      SectionCard("Add a model from Hugging Face") {
        Text(
          "Every download that failed in this app failed because a file name in the " +
            "catalogue was a guess. Point at a repository and pick from what is really " +
            "there instead.",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        OutlinedTextField(
          value = state.browseRepo,
          onValueChange = vm::setBrowseRepo,
          modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
          singleLine = true,
          enabled = !state.browsing,
          label = { Text("owner/name") },
          placeholder = { Text("litert-community/Qwen3-0.6B") },
        )
        Button(
          onClick = vm::browse,
          enabled = !state.browsing && state.browseRepo.isNotBlank(),
          modifier = Modifier.padding(top = 10.dp),
        ) { Text(if (state.browsing) "Listing…" else "List files") }

        state.browseFiles.forEach { repoFile ->
          Row(
            Modifier.fillMaxWidth().padding(top = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
          ) {
            Text(
              repoFile.display,
              style = MaterialTheme.typography.bodySmall,
              modifier = Modifier.weight(1f),
            )
            if (repoFile.isModel) {
              OutlinedButton(
                onClick = { vm.downloadFromRepo(state.browseRepo.trim(), repoFile) },
              ) { Text("Get") }
            }
          }
        }
      }
    }

    item {
      SectionCard("Storage") {
        InfoRow("Used by models", formatBytes(state.diskUsage))
      }
    }
  }
}
