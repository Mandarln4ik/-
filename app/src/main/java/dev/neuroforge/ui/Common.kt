package dev.neuroforge.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp

@Composable
fun SectionCard(title: String, content: @Composable () -> Unit) {
  Card(
    Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
  ) {
    Column(Modifier.padding(16.dp)) {
      Text(title, style = MaterialTheme.typography.titleSmall)
      Column(Modifier.padding(top = 10.dp)) { content() }
    }
  }
}

/** Label on the left, value on the right — the layout every spec row in this app uses. */
@Composable
fun InfoRow(label: String, value: String, mono: Boolean = false) {
  Row(
    Modifier.fillMaxWidth().padding(vertical = 3.dp),
    horizontalArrangement = Arrangement.SpaceBetween,
    verticalAlignment = Alignment.CenterVertically,
  ) {
    Text(
      label,
      style = MaterialTheme.typography.bodySmall,
      color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
    Text(
      value,
      style = MaterialTheme.typography.bodySmall,
      fontFamily = if (mono) FontFamily.Monospace else FontFamily.Default,
    )
  }
}
