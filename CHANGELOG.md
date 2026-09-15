# NeuroForge v1.1.1

Исправление падения при запуске в v1.1.0.

---

## Что было сломано

```
java.lang.NullPointerException: Attempt to invoke interface method
'java.lang.Object kotlin.Lazy.getValue()' on a null object reference
    at dev.neuroforge.ui.AppViewModel.getPrefs(AppViewModel.kt:211)
    at dev.neuroforge.ui.AppViewModel.loadPolicy(AppViewModel.kt:205)
    at dev.neuroforge.ui.AppViewModel.<init>(AppViewModel.kt:68)
```

Приложение не открывалось вообще.

## Причина

Инициализаторы свойств в Kotlin выполняются **в порядке объявления**. Настройка ускорителя
читалась так:

```kotlin
val policy = MutableStateFlow(loadPolicy())   // строка 68
...
private val prefs by lazy { ... }             // строка 211
```

`loadPolicy()` обращается к `prefs`, но на момент вызова поле делегата ещё не присвоено —
отсюда NPE. Ошибка появилась вместе с самой настройкой в v1.1.0: поле оказалось ниже по
файлу, чем свойство, которое его использует.

## Исправление

Убрана не строка, а сама возможность такой ошибки. `prefs` больше не поле, а функция:

```kotlin
private fun prefs(): SharedPreferences =
  app.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
```

Порядок объявления перестал что-либо значить. Android кэширует экземпляры
`SharedPreferences`, так что повторные вызовы ничего не стоят.

Разбор сохранённого значения вынесен в ядро как чистая функция
`AcceleratorPolicy.fromStoredName()` и покрыт тестами — включая случай, когда в настройках
осталось имя режима, удалённого в более поздней версии. Этот код выполняется во время
конструирования, поэтому любой сбой в нём — это падение на старте, а не неверная настройка.

## Почему это не поймали

CI собирает APK, но не запускает его: падение в инициализаторе компилируется без единого
предупреждения. Тесты покрывают ядро, где Android нет, а `ui/` не покрыт ничем. Этот
конкретный класс ошибок теперь частично закрыт — логика, которая раньше жила в конструкторе
вью-модели, переехала в ядро под тесты, — но сам пробел остаётся: экраны проверяются только
компиляцией.

---

## Проверено

- **85 юнит-тестов** ядра, 0 падений.
- `:app:assembleDebug` и `:app:assembleRelease` собираются в CI.

## Артефакты

- `app-debug.apk` — подписан debug-ключом, ставится через `adb install` поверх.
- `app-release-unsigned.apk` — после R8, без подписи.

Всё содержимое v1.1.0 (рабочий бенчмарк, модели без ПК, настройки ускорителя) на месте —
см. [v1.1.0](https://github.com/Mandarln4ik/-/releases/tag/v1.1.0).
