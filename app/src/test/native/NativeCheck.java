import dev.neuroforge.runtime.LlamaCppNative;
import java.util.ArrayList;
import java.util.List;

/**
 * Runs the whole JNI surface against a real (if nonsensical) GGUF.
 *
 * Everything here is a run-time property that compiling proves nothing about: whether the
 * Kotlin object's external functions bind to the C++ symbols at all, whether a null pointer
 * is survivable, whether the chat template makes each turn a prefix extension of the last
 * (the multi-turn cache depends on it), and whether a long prompt decodes instead of
 * tripping a GGML_ASSERT. Two shipped bugs were found this way.
 */
public class NativeCheck {
  static int failures = 0;

  static void ok(String what, boolean cond, String detail) {
    System.out.printf("%-46s %s  %s%n", what, cond ? "ok  " : "FAIL", detail);
    if (!cond) failures++;
  }

  public static void main(String[] args) throws Exception {
    System.load(args[0]);
    LlamaCppNative n = LlamaCppNative.INSTANCE;
    n.init();

    // Every entry point has to tolerate a null pointer: a failed load leaves zeros behind,
    // and reaching native code with one must not be a segfault.
    ok("loadModel of a missing file returns 0", n.loadModel("/nonexistent.gguf") == 0L, "");
    ok("describeModel(0) is empty", n.describeModel(0L).isEmpty(), "");
    ok("contextSize(0) is 0", n.contextSize(0L) == 0, "");
    ok("newContext(0, ...) is 0", n.newContext(0L, 2048, 4) == 0L, "");
    ok("formatChat(0, ...) is null",
       n.formatChat(0L, new String[]{"user"}, new String[]{"hi"}, true) == null, "");
    ok("generate(0, 0, ...) reports ERROR",
       n.generate(0L, 0L, "hi", true, 8, 0.7f, 40, 0.9f, -1, t -> true)
         == LlamaCppNative.Stop.ERROR, "");
    n.clearMemory(0L);
    n.freeContext(0L);
    n.freeModel(0L);
    ok("the free functions survive a null pointer", true, "");

    long model = n.loadModel(args[1]);
    ok("loadModel", model != 0, "ptr=" + model);
    if (model == 0) System.exit(1);

    ok("describeModel", !n.describeModel(model).isEmpty(), n.describeModel(model));
    ok("trainedContextSize", n.trainedContextSize(model) == 256,
       String.valueOf(n.trainedContextSize(model)));

    // --- chat template -------------------------------------------------------
    String one = n.formatChat(model, new String[]{"system", "user"},
                              new String[]{"be nice", "hello"}, true);
    ok("formatChat applies the file's own template", one != null,
       one == null ? "null" : one.replace("\n", "\\n"));
    ok("  ...ends with the assistant opener",
       one != null && one.endsWith("<|im_start|>assistant\n"), "");

    // The multi-turn optimisation rests entirely on this being a prefix extension.
    String withoutAssistant = n.formatChat(model, new String[]{"system", "user"},
                                           new String[]{"be nice", "hello"}, false);
    String two = n.formatChat(model, new String[]{"system", "user", "assistant", "user"},
                              new String[]{"be nice", "hello", "hi there", "again"}, true);
    ok("a later turn extends the earlier transcript",
       two != null && withoutAssistant != null && two.startsWith(withoutAssistant),
       two == null ? "null" : "delta=" + two.substring(withoutAssistant.length())
                                            .replace("\n", "\\n"));

    // A long message forces the buffer-too-small retry inside formatChat.
    StringBuilder big = new StringBuilder();
    for (int i = 0; i < 5000; i++) big.append("word ");
    String huge = n.formatChat(model, new String[]{"user"}, new String[]{big.toString()}, true);
    ok("formatChat survives a message far past the buffer",
       huge != null && huge.length() > 25000, "len=" + (huge == null ? -1 : huge.length()));

    // --- generation ----------------------------------------------------------
    long ctx = n.newContext(model, 256, 2);
    ok("newContext", ctx != 0, "ptr=" + ctx);
    ok("contextSize", n.contextSize(ctx) == 256, String.valueOf(n.contextSize(ctx)));

    List<String> tokens = new ArrayList<>();
    int stop = n.generate(model, ctx, one, true, 16, 0.8f, 40, 0.9f, 7, t -> {
      tokens.add(t);
      return true;
    });
    ok("generate streams tokens", !tokens.isEmpty(), tokens.size() + " tokens, stop=" + stop);

    // Cancellation has to come back as CANCELLED, not as a truncated success.
    List<String> few = new ArrayList<>();
    int cancelled = n.generate(model, ctx, "a", false, 32, 0.8f, 40, 0.9f, 7, t -> {
      few.add(t);
      return few.size() < 3;
    });
    ok("a false return stops the loop",
       cancelled == LlamaCppNative.Stop.CANCELLED && few.size() == 3,
       "stop=" + cancelled + " after " + few.size());

    // Overrunning the window must report CONTEXT_FULL rather than looking like a finish.
    long small = n.newContext(model, 64, 2);
    int full = n.generate(model, small, one, true, 4096, 0.8f, 40, 0.9f, 7, t -> true);
    ok("running past the window reports CONTEXT_FULL",
       full == LlamaCppNative.Stop.CONTEXT_FULL, "stop=" + full);
    n.freeContext(small);

    // --- the KV cache actually advances -------------------------------------
    n.clearMemory(ctx);
    List<String> a = new ArrayList<>();
    n.generate(model, ctx, one, true, 4, 0.0f, 1, 1.0f, 7, t -> { a.add(t); return true; });
    List<String> b = new ArrayList<>();
    n.generate(model, ctx, "b", false, 4, 0.0f, 1, 1.0f, 7, t -> { b.add(t); return true; });
    n.clearMemory(ctx);
    List<String> c = new ArrayList<>();
    n.generate(model, ctx, one, true, 4, 0.0f, 1, 1.0f, 7, t -> { c.add(t); return true; });
    ok("clearMemory restores the first-turn result", a.equals(c), a + " vs " + c);
    ok("a continued turn differs from the first", !b.equals(a), b + " vs " + a);

    // The case that reaches production: n_batch is capped at 2048 however large the window
    // is, so any context above that can hold a prompt that will not fit in one batch. A
    // 4096-token window with a 3000-token paste is the shape of it, and before prefill was
    // chunked this was an abort(), not an error.
    long wide = n.newContext(model, 4096, 2);
    ok("a 4096 window really is 4096", n.contextSize(wide) == 4096,
       String.valueOf(n.contextSize(wide)));
    StringBuilder longPrompt = new StringBuilder();
    // Every "a " lands as four byte tokens in this toy vocabulary, so 700 of them is ~2800:
    // comfortably past n_batch (2048) and comfortably inside the 4096 window.
    for (int i = 0; i < 700; i++) longPrompt.append("a ");
    int[] got = {0};
    int wideStop = n.generate(model, wide, longPrompt.toString(), true, 4, 0.8f, 40, 0.9f, 7,
                              t -> { got[0]++; return true; });
    ok("a prompt longer than n_batch decodes instead of aborting",
       got[0] == 4 && wideStop == LlamaCppNative.Stop.LIMIT,
       got[0] + " tokens, stop=" + wideStop);
    n.freeContext(wide);

    n.freeContext(ctx);
    n.freeModel(model);
    System.out.println(failures == 0 ? "\nALL MODEL CHECKS PASSED" : "\n" + failures + " FAILURES");
    System.exit(failures);
  }
}
