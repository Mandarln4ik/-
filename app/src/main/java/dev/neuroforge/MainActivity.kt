package dev.neuroforge

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.neuroforge.ui.AppViewModel
import dev.neuroforge.ui.NeuroForgeUi
import dev.neuroforge.ui.NeuroForgeTheme

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
    setContent {
      NeuroForgeTheme {
        val vm: AppViewModel = viewModel(factory = AppViewModel.factory(application as NeuroForgeApp))
        NeuroForgeUi(vm)
      }
    }
  }
}
