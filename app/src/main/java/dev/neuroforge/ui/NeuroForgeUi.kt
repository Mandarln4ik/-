package dev.neuroforge.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.Forum
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.lifecycle.compose.collectAsStateWithLifecycle

/**
 * Four tabs, not five.
 *
 * Image generation used to have its own, which split one activity across two screens: the
 * prompt was typed in one place and the picture appeared in another, with the seed that
 * produced it on a third. It now lives in the image chat, where the prompt, the settings
 * and the result are the same conversation.
 */
private enum class Tab(val label: String, val icon: ImageVector) {
  CHATS("Chats", Icons.Filled.Forum),
  DEVICE("Device", Icons.Filled.Memory),
  MODELS("Models", Icons.Filled.Download),
  SETTINGS("Settings", Icons.Filled.Tune),
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NeuroForgeUi(vm: AppViewModel) {
  val state by vm.state.collectAsStateWithLifecycle()
  var tab by remember { mutableStateOf(Tab.CHATS) }
  val snackbar = remember { SnackbarHostState() }

  // Errors are surfaced once and cleared, so a transient download failure does not stick to
  // the screen through the retry that fixes it.
  LaunchedEffect(state.error) {
    state.error?.let {
      snackbar.showSnackbar(it)
      vm.dismissError()
    }
  }

  Scaffold(
    modifier = Modifier.fillMaxSize(),
    topBar = {
      TopAppBar(title = {
        Column {
          Text("NeuroForge", style = MaterialTheme.typography.titleMedium)
          state.device?.let {
            Text(
              "${it.socManufacturer} ${it.socModel}",
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
          }
        }
      })
    },
    bottomBar = {
      NavigationBar {
        Tab.entries.forEach { t ->
          NavigationBarItem(
            selected = tab == t,
            onClick = { tab = t },
            icon = { Icon(t.icon, contentDescription = t.label) },
            label = { Text(t.label) },
          )
        }
      }
    },
    snackbarHost = { SnackbarHost(snackbar) },
  ) { padding ->
    Column(Modifier.padding(padding)) {
      when (tab) {
        Tab.CHATS -> ChatsScreen(vm, state)
        Tab.DEVICE -> DeviceScreen(vm, state)
        Tab.MODELS -> ModelsScreen(vm, state)
        Tab.SETTINGS -> SettingsScreen(vm, state)
      }
    }
  }
}
