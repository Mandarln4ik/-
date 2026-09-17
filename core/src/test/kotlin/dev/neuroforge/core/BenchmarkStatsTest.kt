package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The benchmark is this app's only evidence that the NPU does anything, so the way it can
 * lie matters as much as the way it computes.
 */
class BenchmarkStatsTest {

  private fun result(accel: Accel, samples: List<Double>) =
    BenchmarkResult("m", accel, samples, warmupMillis = 0.0)

  @Test
  fun `whole-millisecond samples are flagged as unresolvable`() {
    // The shipped bug: run() divided nanoseconds into a Long, so every sample was an
    // integer and NPU and CPU both reported 2.00 ms with a 1.00x speedup. The ratio was
    // arithmetically correct and told the user the opposite of the truth.
    val npu = result(Accel.NPU, List(20) { 2.0 })
    val cpu = result(Accel.CPU, List(20) { 2.0 })
    assertTrue(npu.quantised)
    assertTrue(cpu.quantised)

    val verdict = BenchmarkReport(listOf(npu, cpu)).interpretation()!!
    assertTrue(verdict.contains("whole millisecond"), verdict)
    // It must not present 1.00x as a finding.
    assertTrue(!verdict.contains("1.00x"), verdict)
  }

  @Test
  fun `fractional samples are not flagged`() {
    val r = result(Accel.NPU, listOf(1.62, 1.71, 1.58, 1.66))
    assertTrue(!r.quantised)
  }

  @Test
  fun `a gap smaller than the spread is reported as a tie`() {
    // Both around 2 ms, but noisy: the difference is not evidence of anything.
    val npu = result(Accel.NPU, listOf(2.0, 2.4, 1.6, 2.2, 1.8))
    val cpu = result(Accel.CPU, listOf(2.1, 2.5, 1.7, 2.3, 1.9))
    val verdict = BenchmarkReport(listOf(npu, cpu)).interpretation()!!
    assertTrue(verdict.contains("tied"), verdict)
  }

  @Test
  fun `a real speedup is reported as one`() {
    val npu = result(Accel.NPU, listOf(1.00, 1.01, 0.99, 1.00))
    val cpu = result(Accel.CPU, listOf(4.00, 4.02, 3.98, 4.00))
    val report = BenchmarkReport(listOf(npu, cpu))
    assertEquals(4.0, report.npuSpeedupOverCpu()!!, 0.05)
    assertTrue(report.interpretation()!!.contains("NPU is 4."), report.interpretation()!!)
  }

  @Test
  fun `the npu losing is stated plainly rather than softened`() {
    // A genuine finding, and the one the catalogue's CPU-only entries rest on.
    val npu = result(Accel.NPU, listOf(8.0, 8.1, 7.9, 8.0))
    val cpu = result(Accel.CPU, listOf(2.0, 2.1, 1.9, 2.0))
    val verdict = BenchmarkReport(listOf(npu, cpu)).interpretation()!!
    assertTrue(verdict.contains("CPU is 4."), verdict)
    assertTrue(verdict.contains("not one an APU accelerates"), verdict)
  }

  @Test
  fun `no interpretation without both sides`() {
    assertNull(BenchmarkReport(listOf(result(Accel.NPU, listOf(1.0, 1.1)))).interpretation())
    assertNull(
      BenchmarkReport(
        listOf(
          result(Accel.NPU, listOf(1.0, 1.1)),
          BenchmarkResult("m", Accel.CPU, emptyList(), 0.0, failed = "did not load"),
        )
      ).interpretation()
    )
  }

  @Test
  fun `warmup is separated rather than folded into the median`() {
    // The first NPU run includes graph compilation; averaging it in would describe neither
    // the first run nor the steady state.
    val r = summarize("m", Accel.NPU, listOf(900.0, 2.0, 2.1, 1.9), warmupCount = 1)
    assertEquals(900.0, r.warmupMillis, 1e-9)
    assertEquals(3, r.samples.size)
    assertTrue(r.median < 3.0, "median ${r.median} still carries the compile cost")
  }

  @Test
  fun `percentiles interpolate rather than snapping to the worst sample`() {
    val r = result(Accel.CPU, listOf(1.0, 2.0, 3.0, 4.0, 5.0))
    assertEquals(3.0, r.median, 1e-9)
    assertEquals(4.8, r.p95, 1e-9)
    assertEquals(1.0, r.best, 1e-9)
  }

  @Test
  fun `the table keeps three decimals so sub-millisecond gaps survive printing`() {
    val table = BenchmarkReport(
      listOf(result(Accel.NPU, listOf(1.618, 1.620)), result(Accel.CPU, listOf(2.414, 2.410)))
    ).table()
    assertTrue(table.contains("1.618") || table.contains("1.619"), table)
    assertTrue(table.contains("2.41"), table)
  }
}
