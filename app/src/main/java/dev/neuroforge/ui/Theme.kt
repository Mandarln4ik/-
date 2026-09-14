package dev.neuroforge.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Accent = Color(0xFF7AA2FF)
private val AccentDark = Color(0xFF2B4B9E)

private val DarkScheme = darkColorScheme(
  primary = Accent,
  onPrimary = Color(0xFF06122E),
  secondary = Color(0xFF9BE8C8),
  background = Color(0xFF0B1020),
  surface = Color(0xFF121A2E),
  surfaceVariant = Color(0xFF1B2540),
  error = Color(0xFFFF8A80),
)

private val LightScheme = lightColorScheme(
  primary = AccentDark,
  secondary = Color(0xFF1C7A55),
  background = Color(0xFFF6F8FD),
  surface = Color(0xFFFFFFFF),
  surfaceVariant = Color(0xFFE6EBF6),
)

@Composable
fun NeuroForgeTheme(
  darkTheme: Boolean = isSystemInDarkTheme(),
  content: @Composable () -> Unit,
) {
  MaterialTheme(
    colorScheme = if (darkTheme) DarkScheme else LightScheme,
    content = content,
  )
}
