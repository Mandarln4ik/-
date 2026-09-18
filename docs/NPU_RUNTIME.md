# Рантайм NPU (NeuroPilot)

## Что нужно и почему этого нет в репозитории

LiteRT обращается к APU MediaTek через нативные библиотеки NeuroPilot —
`libneuron_adapter.so` и родственные. Они **не распространяются свободно**: их отдают
MediaTek и Google отдельно от Maven-артефакта, и положить их сюда нельзя.

Без них приложение собирается и работает — просто без NPU. `AcceleratorProbe` это
обнаруживает и пишет на экране Device прямым текстом, а каждая стадия уходит на GPU или CPU.
Молча работать медленнее, чем ожидалось, — худший из вариантов, поэтому он исключён.

## Куда класть

```
app/src/main/jniLibs/arm64-v8a/
    libneuron_adapter.so
    ...
```

Версия библиотек **обязана** совпадать с версией maven-артефакта LiteRT, здесь это
`com.google.ai.edge.litert:litert:2.1.6` (см. `gradle/libs.versions.toml`). Рассинхрон
версий проявляется как отказ компиляции графа, а не как внятная ошибка загрузки.

После добавления:

```bash
./gradlew :app:assembleDebug
```

`packaging { jniLibs { useLegacyPackaging = true } }` в `app/build.gradle.kts` уже
выставлен — нативный слой LiteRT делает `dlopen` и требует библиотеки распакованными
на диск, а не отображёнными прямо из APK.

## Где взять

**Сначала проверьте вкладку Device: возможно, брать не нужно.** На телефоне с Dimensity
рантайм NeuroPilot обычно уже лежит в vendor-образе, и `AcceleratorProbe` ищет его в
`/vendor/lib64`, `/system/lib64` и `/odm/lib64` — а не только в APK. Если он там есть,
класть ничего не надо.

Если нет — библиотека скачивается публично, и это не то, что здесь было написано раньше.
Пакет `ai-edge-litert-sdk-mediatek` при установке тянет NeuroPilot Express SDK (66 МБ) с
S3 самого MediaTek:

```bash
pip download --no-binary=:all: ai-edge-litert-sdk-mediatek   # URL внутри setup.py
```

Внутри архива, среди прочего:

| файл | архитектура | размер |
|---|---|---|
| `v8_0_10/usdk/lib64/libneuronusdk_adapter.mtk.so` | AArch64 | 13,7 МБ |
| `v9_0_3/usdk/lib64/libneuronusdk_adapter.so` | AArch64 | 14,4 МБ |

Первое имя — ровно одно из трёх, которые ищет `AcceleratorProbe`. То есть это именно тот
файл, который кладётся в `app/src/main/jniLibs/arm64-v8a/`.

**Но скачивать его должны вы, а не я.** В архиве лежит `LICENSE AGREEMENT.pdf` — «MediaTek
Confidential», лицензия NeuroPilot Express SDK, принимаемая самим фактом доступа к файлам.
Положить эти бинарники в публичный репозиторий нельзя; скачать их себе на свой телефон —
другое дело, и тогда лицензиат вы.

Остальные источники:

- Документация Google AI Edge: раздел NPU acceleration / MediaTek NeuroPilot
  (`ai.google.dev/edge/litert/next/mediatek`) — там же описан путь через AI Packs.
- Официальные сэмплы `google-ai-edge/litert-samples`, каталог
  `samples/litert/image_segmentation/kotlin_npu/` — есть два варианта: `android` (AOT,
  через Play AI Packs и dynamic features) и `android_jit` (компиляция графа на устройстве).

## AOT против JIT

Официальный сэмпл показывает оба пути, и разница существенная:

| | AOT (`android`) | JIT (`android_jit`) |
|---|---|---|
| Граф компилируется | заранее, под каждый SoC | на устройстве при первой загрузке |
| Доставка | Play AI Packs + dynamic feature на вендора | обычный APK |
| Первый запуск | быстрый | платит за компиляцию (секунды на крупном графе) |
| Установка | через `bundletool` с device groups | `adb install` |

NeuroForge использует **JIT**: он не требует ни публикации в Play, ни сборки app bundle,
ни таблицы device groups — а на Dimensity 9500s это ещё и принципиально, потому что в
таблице групп официального сэмпла такого SoC попросту нет (список заканчивается на
`Mediatek_MT6991_ANDROID_15`, то есть на Dimensity 9400).

Цена JIT — задержка первой компиляции. Бенчмарк показывает её отдельной колонкой `warmup`
и не смешивает со steady-state медианой.
