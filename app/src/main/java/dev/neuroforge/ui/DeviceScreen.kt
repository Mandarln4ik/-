package dev.neuroforge.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import dev.neuroforge.core.Accel

/**
 * What the hardware is, and — separately — what has actually been proven to run on it.
 *
 * The separation is the point. A static compatibility list can say "supported" on a device
 * where the graph you care about will not compile, and can say "unsupported" on silicon
 * newer than the list. Only the benchmark below settles it.
 */
@Composable
fun DeviceScreen(vm: AppViewModel, state: UiState) {
  val device = state.device

  LazyColumn(Modifier.fillMaxSize()) {
    item {
      SectionCard("Verdict") {
        Text(
          device?.verdict() ?: "Probing…",
          style = MaterialTheme.typography.bodyMedium,
          color = when {
            device == null -> MaterialTheme.colorScheme.onSurfaceVariant
            Accel.NPU in device.verifiedAccelerators -> MaterialTheme.colorScheme.secondary
            device.npuAvailable -> MaterialTheme.colorScheme.primary
            else -> MaterialTheme.colorScheme.onSurfaceVariant
          },
        )
      }
    }

    if (device != null) {
      item {
        SectionCard("Silicon") {
          InfoRow("Device", device.deviceModel)
          InfoRow("SoC", "${device.socManufacturer} ${device.socModel}", mono = true)
          InfoRow("Board platform", device.boardPlatform, mono = true)
          InfoRow("OS", device.androidRelease)
          InfoRow("ABIs", device.abis.joinToString(", "), mono = true)
        }
      }

      item {
        SectionCard("NPU path") {
          InfoRow("Vendor", device.npuVendor.name)
          InfoRow("LiteRT says supported", if (device.npuAvailable) "yes" else "no")
          InfoRow("Vendor runtime present", if (device.npuRuntimePresent) "yes" else "no")
          InfoRow(
            "APU device nodes",
            device.apuNodes.takeIf { it.isNotEmpty() }?.joinToString(", ") ?: "not visible",
            mono = true,
          )
          InfoRow(
            "Proven by benchmark",
            device.verifiedAccelerators.takeIf { it.isNotEmpty() }?.joinToString(", ")
              ?: "nothing yet — run the benchmark",
          )
          device.notes.forEach { note ->
            Text(
              "• $note",
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant,
              modifier = Modifier.padding(top = 6.dp),
            )
          }
        }
      }

      item {
        SectionCard("What reaches the APU here") {
          Text(
            "Three different answers, and only the honest split is useful:",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
          )
          InfoRow("×4 upscaler", "NPU — static shapes, integer convolutions")
          InfoRow(
            "Text model",
            // Not a claim about this device's silicon but about what is published for it:
            // litert-community ships graphs compiled per SoC, so the answer depends on
            // whether one exists for this chip.
            "NPU via LiteRT-LM, using the ${device.socModel} build if one is published",
          )
          InfoRow("Diffusion transformer", "CPU — blockwise int4, dynamic attention")
          Text(
            "The Models tab picks the bundle matching ${device.socModel} automatically when " +
              "a repository publishes per-accelerator builds. Nothing here claims the " +
              "generator runs on the APU, because no public diffusion model ships as an " +
              "NPU-compatible LiteRT graph yet.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 10.dp),
          )
        }
      }

      item {
        SectionCard("Measure it") {
          Text(
            "Runs the same graph on NPU, GPU and CPU and reports median latency. This is the " +
              "only claim on this screen that is evidence rather than inference. Any " +
              "downloaded model with a fixed input works — get one from the Models tab.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
          )
          Row(
            Modifier.fillMaxWidth().padding(top = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
          ) {
            Button(onClick = vm::runBenchmark, enabled = !state.benchmarkRunning) {
              Text(if (state.benchmarkRunning) "Running…" else "Run benchmark")
            }
            if (state.benchmarkRunning) {
              CircularProgressIndicator(Modifier.padding(top = 8.dp))
            }
          }

          state.benchmarkModel?.let { name ->
            Text(
              "Model: $name",
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant,
              modifier = Modifier.padding(top = 10.dp),
            )
          }

          state.benchmark?.let { report ->
            Text(
              report.table(),
              style = MaterialTheme.typography.bodySmall,
              fontFamily = FontFamily.Monospace,
              modifier = Modifier.padding(top = 14.dp),
            )
            report.npuSpeedupOverCpu()?.let { speedup ->
              Text(
                if (speedup >= 1.0) "The NPU is %.1fx faster than the CPU here.".format(speedup)
                else "The NPU is slower than the CPU on this graph (%.2fx) — for this stage, " +
                  "CPU is the right target.".format(speedup),
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(top = 8.dp),
              )
            }
          }
        }
      }

      item {
        Column(Modifier.padding(16.dp)) {
          Button(onClick = vm::refreshDevice) { Text("Re-probe") }
        }
      }
    }
  }
}
