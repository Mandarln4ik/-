# Запрос на aarch64-сборку компилятора LiteRT NPU

Черновик письма. Отправлять можно в два адреса — issue в
[`google-ai-edge/LiteRT`](https://github.com/google-ai-edge/LiteRT/issues) и в поддержку
MediaTek NeuroPilot. Первый адрес важнее: разрыв, судя по всему, именно там.

Все числа ниже — измеренные, не взятые из документации. Как проверить, написано в
[`NPU_COMPILATION.md`](NPU_COMPILATION.md).

---

**Тема:** aarch64 build of `libLiteRtCompilerPlugin_*` for on-device AOT compilation

Hello,

I maintain a small Android app that runs LiteRT models on the MediaTek APU of a
Dimensity 9500s (`Build.SOC_MODEL` reports `MT6991`). I would like to compile models
ahead-of-time on the phone itself, and I think the only thing missing is one shared library.
I want to check that reading with you before I go further.

**What I found.** In `ai-edge-litert` 2.2.0 the vendor compiler plugins are published for
x86_64 only:

| library in the wheel | `manylinux_2_27_x86_64` | `manylinux_2_27_aarch64` |
|---|---|---|
| `libLiteRtCompilerPlugin_MediaTek.so` | present | **absent** |
| `libLiteRtCompilerPlugin_Qualcomm.so` | present | absent |
| `libLiteRtCompilerPlugin_Samsung.so` | present | absent |
| `libLiteRtCompilerPlugin_google_tensor.so` | present | absent |
| `libLiteRtCompilerPlugin_IntelOpenvino.so` | present | absent |

The Python side is identical in both wheels — `ai_edge_litert/aot/vendors/mediatek/
mediatek_backend.py` is byte-for-byte the same — so what is missing is the native library it
loads, not the backend code. `ai-edge-litert-sdk-mediatek` 2.2.0 makes the same assumption
explicitly, in `setup.py`:

```python
IS_X86_ARCHITECTURE = platform.machine() in ('x86_64', 'i386', 'i686')
...
print('IGNORED: Currently LiteRT NPU AOT for MediaTek is only supported on'
      ' Linux x86 architecture.')
```

**Why I think this is packaging rather than a missing compiler.** That same `setup.py`
downloads the NeuroPilot SDK, and the SDK does ship an AArch64 build — just under a
different name:

| file | architecture | size |
|---|---|---|
| `neuro_pilot/v9_0_3/host/lib/libneuron_adapter.so` | x86-64 | 25.2 MB |
| `neuro_pilot/v9_0_3/usdk/lib64/libneuronusdk_adapter.so` | **AArch64** | 14.4 MB |

Their exported `Neuron*` symbols are **identical sets** — 104 symbols each, of which 30 are
`NeuronCompilation_*`, including `NeuronCompilation_create`, `NeuronCompilation_createWithOptions`
and `NeuronCompilation_finish`. I compared the two `nm -D` listings and `diff` reports no
difference at all. The AArch64 file is a normal stripped ELF, not a stub.

So the compilation API appears to be available on ARM already; what is not available is the
LiteRT plugin that drives it.

**What I am asking.**

1. To Google: could `libLiteRtCompilerPlugin_MediaTek.so` (and ideally the other vendor
   plugins) be built for `manylinux_2_27_aarch64` as well, or is there a reason the plugin
   itself cannot be cross-compiled that I am not seeing? If it is buildable from the LiteRT
   sources, a pointer to the right Bazel target would be enough — I am happy to build it
   myself and report back whether it works on the device.

2. To MediaTek: is `libneuronusdk_adapter.so` supported for compiling a model on-device, as
   opposed to only executing one that was compiled on a host? The symbol table says yes; I
   would rather hear it than infer it. If there are constraints — compilation only for the
   running SoC, a licence term, a required NeuroPilot version — those are exactly what I
   need to know before building anything on it.

**Why it matters for this device.** Without AOT, the first load of a `.litertlm` on the APU
includes a just-in-time compile that takes tens of seconds, and users cannot tell it from a
hang. With it, `litert-community`'s pre-compiled `..._mt6991.litertlm` bundles load quickly —
but that only helps for the models someone has already published for this exact chip.
Compiling on the phone would close that gap for every other model.

Happy to share the exact commands and the wheel hashes if that is useful.

Thanks,
