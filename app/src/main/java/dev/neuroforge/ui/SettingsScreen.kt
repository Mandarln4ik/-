package dev.neuroforge.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.neuroforge.core.Accel
import dev.neuroforge.core.AcceleratorPolicy
import dev.neuroforge.core.ModelCatalog

/**
 * Where the accelerator choice lives.
 *
 * The interesting option is NPU-only. Every other mode falls back when a graph will not
 * compile, which is convenient and also the reason "runs on the NPU" is so often an empty
 * claim — you cannot tell a graph that reached the APU from one that quietly ran on the CPU.
 * Strict mode refuses to fall back, so the answer is unambiguous even when it is "no".
 */
@Composable
fun SettingsScreen(vm: AppViewModel, state: UiState) {
  val policy by vm.policy.collectAsStateWithLifecycle()

  LazyColumn(Modifier.fillMaxSize()) {
    item {
      SectionCard("Accelerator") {
        AcceleratorPolicy.entries.forEach { option ->
          androidx.compose.foundation.layout.Row(
            Modifier
              .fillMaxWidth()
              .padding(vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
          ) {
            RadioButton(
              selected = policy == option,
              onClick = { vm.setPolicy(option) },
            )
            Column(Modifier.padding(start = 8.dp)) {
              Text(option.label, style = MaterialTheme.typography.bodyMedium)
              Text(
                explain(option),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
              )
            }
          }
        }
      }
    }

    item {
      SectionCard("What can actually use the NPU") {
        Text(
          "An APU runs graphs with static shapes and integer convolutions. That is a " +
            "property of each model, not a setting — asking for the NPU on a graph that " +
            "cannot use it fails rather than getting faster.",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
          modifier = Modifier.padding(bottom = 10.dp),
        )
        ModelCatalog.all.forEach { spec ->
          InfoRow(
            spec.displayName,
            if (spec.npuCapable) "NPU possible" else "CPU/GPU only",
          )
        }
      }
    }

    item {
      SectionCard("Why the generator is not on the NPU") {
        Text(
          "The diffusion transformer carries blockwise int4 weights and dynamic attention. " +
            "An APU accelerates neither, so it runs on CPU with XNNPACK, where blockwise " +
            "int4 kernels exist at all. No public diffusion model ships as an " +
            "NPU-compatible LiteRT graph today; when one does, it drops into the same " +
            "catalogue and this line changes.",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
      }
    }

    state.device?.let { device ->
      item {
        SectionCard("This device") {
          InfoRow("SoC", "${device.socManufacturer} ${device.socModel}", mono = true)
          InfoRow("LiteRT says supported", if (device.npuAvailable) "yes" else "no")
          InfoRow("Vendor runtime present", if (device.npuRuntimePresent) "yes" else "no")
          InfoRow(
            "Proven by benchmark",
            device.verifiedAccelerators.takeIf { it.isNotEmpty() }?.joinToString(", ")
              ?: "nothing yet",
          )
          if (Accel.NPU in device.verifiedAccelerators) {
            Text(
              "The NPU has run a real graph here.",
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.secondary,
              modifier = Modifier.padding(top = 8.dp),
            )
          }
        }
      }
    }
  }
}

private fun explain(policy: AcceleratorPolicy): String = when (policy) {
  AcceleratorPolicy.AUTO ->
    "Try the NPU first, drop to GPU then CPU when a graph will not compile."
  AcceleratorPolicy.NPU_STRICT ->
    "Never fall back. A model the APU will not take fails with the reason, which is the " +
      "only way to be certain the NPU is doing the work."
  AcceleratorPolicy.GPU_FIRST ->
    "Skip the NPU entirely. Useful for comparing against the NPU result."
  AcceleratorPolicy.CPU_ONLY ->
    "The baseline everything else is measured against."
}
