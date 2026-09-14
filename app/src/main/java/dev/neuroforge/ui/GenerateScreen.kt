package dev.neuroforge.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.item
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.neuroforge.core.OutputTarget
import dev.neuroforge.core.UpscaleStrategy
import dev.neuroforge.core.formatBytes

@Composable
fun GenerateScreen(vm: AppViewModel, state: UiState) {
  val prompt by vm.prompt.collectAsStateWithLifecycle()
  val seed by vm.seed.collectAsStateWithLifecycle()
  val steps by vm.steps.collectAsStateWithLifecycle()
  val target by vm.target.collectAsStateWithLifecycle()
  val strategy by vm.strategy.collectAsStateWithLifecycle()
  val busy = state.generation != null

  LazyColumn(Modifier.fillMaxSize()) {
    item {
      SectionCard("Prompt") {
        OutlinedTextField(
          value = prompt,
          onValueChange = { vm.prompt.value = it },
          modifier = Modifier.fillMaxWidth(),
          minLines = 3,
          enabled = !busy,
          label = { Text("What to draw") },
        )
      }
    }

    item {
      SectionCard("Output") {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          OutputTarget.presets.take(3).forEach { preset ->
            FilterChip(
              selected = target == preset,
              onClick = { vm.setTarget(preset) },
              enabled = !busy,
              label = { Text("${preset.width}²".takeIf { preset.width == preset.height } ?: preset.label) },
            )
          }
        }
        Row(
          Modifier.padding(top = 8.dp),
          horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
          OutputTarget.presets.drop(3).forEach { preset ->
            FilterChip(
              selected = target == preset,
              onClick = { vm.setTarget(preset) },
              enabled = !busy,
              label = { Text(preset.label) },
            )
          }
        }

        Row(
          Modifier.padding(top = 12.dp),
          horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
          UpscaleStrategy.entries.forEach { s ->
            FilterChip(
              selected = strategy == s,
              onClick = { vm.setStrategy(s) },
              enabled = !busy,
              label = { Text(if (s == UpscaleStrategy.FAST) "Fast" else "Quality") },
            )
          }
        }
      }
    }

    // The cost of the chosen route, before anything runs. A 4K job is minutes long; finding
    // out how many tiles it will take afterwards is not useful.
    state.plan?.let { plan ->
      item {
        SectionCard("Plan") {
          Text(
            plan.describe(),
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
          )
          InfoRow("Peak accumulator", formatBytes(plan.peakAccumulatorBytes))
          if (plan.cropsContent) {
            Text(
              "This target is not the generator's 1:1 aspect, so the final step crops. " +
                "A square target uses the whole image.",
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.primary,
              modifier = Modifier.padding(top = 6.dp),
            )
          }
        }
      }
    }

    item {
      SectionCard("Sampling") {
        InfoRow("Steps", steps.toString())
        Slider(
          value = steps.toFloat(),
          onValueChange = { vm.steps.value = it.toInt().coerceAtLeast(1) },
          valueRange = 1f..12f,
          steps = 10,
          enabled = !busy,
        )
        Text(
          "The model is step-distilled: 4 steps is what it was trained to produce. More " +
            "steps cost proportionally more time and change the image rather than refining it.",
          style = MaterialTheme.typography.labelSmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Row(
          Modifier.fillMaxWidth().padding(top = 12.dp),
          horizontalArrangement = Arrangement.SpaceBetween,
        ) {
          Text("Seed $seed", style = MaterialTheme.typography.bodySmall)
          OutlinedButton(onClick = vm::randomizeSeed, enabled = !busy) { Text("Randomize") }
        }
      }
    }

    item {
      Column(Modifier.padding(16.dp)) {
        if (busy) {
          val progress = state.generation!!
          LinearProgressIndicator(
            progress = { progress.fraction },
            modifier = Modifier.fillMaxWidth(),
          )
          Text(
            progress.detail,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(top = 8.dp),
          )
          Text(
            "elapsed ${progress.elapsedMillis / 1000}s",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
          )
          OutlinedButton(
            onClick = vm::cancelGeneration,
            modifier = Modifier.padding(top = 12.dp),
          ) { Text("Cancel") }
        } else {
          Button(onClick = vm::generate, modifier = Modifier.fillMaxWidth()) {
            Text("Generate ${target.label}")
          }
        }
      }
    }

    state.result?.let { result ->
      item {
        SectionCard("Result") {
          Image(
            bitmap = result.bitmap.asImageBitmap(),
            contentDescription = "Generated image",
            contentScale = ContentScale.Fit,
            modifier = Modifier
              .fillMaxWidth()
              .aspectRatio(result.bitmap.width.toFloat() / result.bitmap.height),
          )
          InfoRow("Size", "${result.bitmap.width}x${result.bitmap.height}")
          InfoRow("Total", "${result.totalMillis / 1000}s")
          result.timings.forEach { t ->
            InfoRow("  ${t.name}", "${t.millis / 1000.0}s on ${t.accelerator}")
          }
          Button(
            onClick = { vm.saveResult(result.bitmap) },
            modifier = Modifier.padding(top = 12.dp),
          ) { Text("Save PNG") }
        }
      }
    }
  }
}
