"""Unit tests for mcp-slim v1.6.0 SseChunkThrottle, LatencyHistogram, and mcp-slim bench."""

from __future__ import annotations

import json
import math
import unittest

from mcp_slim import (
    LatencyHistogram,
    SlimStats,
    SseChunkDecision,
    SseChunkThrottle,
    estimate_tokens,
)
from mcp_slim.cli import main, run_bench


class TestSseChunkThrottle(unittest.TestCase):
    """SseChunkThrottle (MS011): sliding exponential window burst budget."""

    def test_disabled_when_budget_zero(self) -> None:
        throttle = SseChunkThrottle(burst_budget_tokens=0)
        self.assertFalse(throttle.enabled)
        decision = throttle.evaluate("hello world")
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.throttled)
        self.assertEqual(decision.budget_tokens, 0)

    def test_first_chunk_always_passes_min_passthrough(self) -> None:
        # min_passthrough=1 means chunk 1 is never throttled even if huge
        throttle = SseChunkThrottle(burst_budget_tokens=1)
        big_text = "token " * 500  # ~500 tokens, well over budget of 1
        decision = throttle.evaluate(big_text)
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.throttled)

    def test_throttle_activates_after_budget_exceeded(self) -> None:
        throttle = SseChunkThrottle(burst_budget_tokens=10, decay=0.9, min_passthrough=0)
        # Feed chunks totalling well over 10 tokens
        decisions = []
        for _ in range(20):
            d = throttle.evaluate("hello world foo bar")  # ~4 tokens each
            decisions.append(d)
        # Eventually some chunks must be throttled
        throttled = [d for d in decisions if d.throttled]
        self.assertGreater(len(throttled), 0, "Expected some chunks to be throttled")

    def test_window_decays_allowing_recovery(self) -> None:
        # With high decay (0.01), window shrinks almost to zero each step
        throttle = SseChunkThrottle(burst_budget_tokens=5, decay=0.01, min_passthrough=0)
        # First chunk exceeds budget
        d1 = throttle.evaluate("hello world foo bar baz qux")  # ~6 tokens
        # Second chunk: window decayed to ~0.06, should pass
        d2 = throttle.evaluate("hi")  # 1 token; window ~ 0.06+1 < 5
        self.assertTrue(d2.allowed)

    def test_stats_budget_throttles_incremented(self) -> None:
        stats = SlimStats()
        throttle = SseChunkThrottle(burst_budget_tokens=1, decay=0.5, min_passthrough=0)
        for i in range(10):
            throttle.evaluate("token " * 5, stats=stats)
        self.assertGreater(stats.budget_throttles, 0)

    def test_snapshot_shape(self) -> None:
        throttle = SseChunkThrottle(burst_budget_tokens=100)
        throttle.evaluate("hello")
        snap = throttle.snapshot()
        self.assertIn("enabled", snap)
        self.assertIn("burst_budget_tokens", snap)
        self.assertIn("decay", snap)
        self.assertIn("window_tokens", snap)
        self.assertIn("chunks_seen", snap)
        self.assertIn("chunks_throttled", snap)
        self.assertIn("throttle_rate", snap)
        self.assertEqual(snap["chunks_seen"], 1)

    def test_reset_clears_window(self) -> None:
        throttle = SseChunkThrottle(burst_budget_tokens=100, decay=0.9)
        for _ in range(5):
            throttle.evaluate("some text with tokens")
        self.assertGreater(throttle.snapshot()["window_tokens"], 0)
        throttle.reset()
        snap = throttle.snapshot()
        self.assertEqual(snap["window_tokens"], 0.0)
        self.assertEqual(snap["chunks_seen"], 0)

    def test_invalid_decay_raises(self) -> None:
        with self.assertRaises(ValueError):
            SseChunkThrottle(burst_budget_tokens=100, decay=0.0)
        with self.assertRaises(ValueError):
            SseChunkThrottle(burst_budget_tokens=100, decay=1.0)
        with self.assertRaises(ValueError):
            SseChunkThrottle(burst_budget_tokens=100, decay=-0.5)

    def test_decision_fields_populated_on_throttle(self) -> None:
        throttle = SseChunkThrottle(burst_budget_tokens=1, decay=0.5, min_passthrough=0)
        # Force window over budget
        for _ in range(5):
            throttle.evaluate("token " * 3)
        # Find a throttled decision
        found_throttled = False
        for _ in range(10):
            d = throttle.evaluate("token " * 3)
            if d.throttled:
                found_throttled = True
                self.assertFalse(d.allowed)
                self.assertIn("MS011", d.drop_reason)
                self.assertIn("window_tokens", d.drop_reason)
                break
        self.assertTrue(found_throttled, "Expected at least one throttled decision")

    def test_window_accumulates_with_decay(self) -> None:
        throttle = SseChunkThrottle(burst_budget_tokens=10000, decay=0.9, min_passthrough=0)
        # estimate_tokens("a b c d e f g h i j") => ~5 tokens
        # Geometric series steady-state: chunk_tokens / (1 - decay) = 5 / 0.1 = 50
        for _ in range(200):
            throttle.evaluate("a b c d e f g h i j")
        snap = throttle.snapshot()
        # Window should converge near 50 (5/(1-0.9))
        chunk_tokens = estimate_tokens("a b c d e f g h i j")
        expected_steady_state = chunk_tokens / (1 - 0.9)
        self.assertAlmostEqual(snap["window_tokens"], expected_steady_state, delta=5.0)


class TestLatencyHistogram(unittest.TestCase):
    """LatencyHistogram (MS012): JSON-RPC latency collection and Prometheus export."""

    def test_basic_record_and_stats(self) -> None:
        h = LatencyHistogram()
        h.record("tools/call", 10.0)
        h.record("tools/call", 20.0)
        h.record("tools/call", 30.0)
        s = h.stats("tools/call")
        self.assertEqual(s["count"], 3)
        self.assertAlmostEqual(s["mean_ms"], 20.0, places=2)
        self.assertAlmostEqual(s["min_ms"], 10.0, places=2)
        self.assertAlmostEqual(s["max_ms"], 30.0, places=2)

    def test_percentiles_single_sample(self) -> None:
        h = LatencyHistogram()
        h.record("tools/list", 42.5)
        s = h.stats("tools/list")
        self.assertEqual(s["count"], 1)
        self.assertAlmostEqual(s["p50_ms"], 42.5, places=2)
        self.assertAlmostEqual(s["p99_ms"], 42.5, places=2)

    def test_percentiles_ordered(self) -> None:
        h = LatencyHistogram()
        for ms in [1.0, 2.0, 3.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0, 10000.0]:
            h.record("method", ms)
        s = h.stats("method")
        self.assertLessEqual(s["p50_ms"], s["p90_ms"])
        self.assertLessEqual(s["p90_ms"], s["p95_ms"])
        self.assertLessEqual(s["p95_ms"], s["p99_ms"])

    def test_method_allowlist_filters(self) -> None:
        h = LatencyHistogram(method_allowlist=["tools/call"])
        h.record("tools/call", 10.0)
        h.record("tools/list", 5.0)  # should be filtered
        self.assertIn("tools/call", h.methods())
        self.assertNotIn("tools/list", h.methods())

    def test_empty_method_returns_zeros(self) -> None:
        h = LatencyHistogram()
        s = h.stats("nonexistent")
        self.assertEqual(s["count"], 0)
        self.assertEqual(s["mean_ms"], 0.0)

    def test_all_stats_returns_all_methods(self) -> None:
        h = LatencyHistogram()
        h.record("a", 1.0)
        h.record("b", 2.0)
        h.record("c", 3.0)
        all_s = h.all_stats()
        self.assertSetEqual(set(all_s.keys()), {"a", "b", "c"})

    def test_total_observations(self) -> None:
        h = LatencyHistogram()
        for i in range(7):
            h.record("tools/call", float(i))
        for i in range(3):
            h.record("tools/list", float(i))
        self.assertEqual(h.total_observations(), 10)

    def test_negative_latency_clamped_to_zero(self) -> None:
        h = LatencyHistogram()
        h.record("m", -5.0)
        s = h.stats("m")
        self.assertEqual(s["min_ms"], 0.0)

    def test_prometheus_text_format_structure(self) -> None:
        h = LatencyHistogram()
        h.record("tools/call", 10.0)
        h.record("tools/call", 50.0)
        text = h.prometheus_text()
        self.assertIn("# HELP", text)
        self.assertIn("# TYPE", text)
        self.assertIn("mcp_slim_jsonrpc_latency_ms_bucket", text)
        self.assertIn('le="+Inf"', text)
        self.assertIn("mcp_slim_jsonrpc_latency_ms_count", text)
        self.assertIn("mcp_slim_jsonrpc_latency_ms_sum", text)
        self.assertIn("mcp_slim_jsonrpc_latency_percentile_ms", text)
        self.assertIn('quantile="0.5"', text)
        self.assertIn('quantile="0.99"', text)

    def test_prometheus_text_custom_prefix(self) -> None:
        h = LatencyHistogram()
        h.record("tools/call", 100.0)
        text = h.prometheus_text(prefix="myapp_rpc")
        self.assertIn("myapp_rpc_latency_ms_bucket", text)

    def test_prometheus_bucket_counts_non_decreasing(self) -> None:
        h = LatencyHistogram()
        for ms in [1.0, 5.0, 25.0, 100.0, 1000.0]:
            h.record("tools/call", ms)
        text = h.prometheus_text()
        # Extract bucket lines for tools/call
        bucket_lines = [
            line for line in text.splitlines()
            if "mcp_slim_jsonrpc_latency_ms_bucket" in line and 'method="tools/call"' in line
        ]
        # Parse counts and verify non-decreasing
        counts = []
        for line in bucket_lines:
            count_str = line.rsplit(" ", 1)[-1]
            counts.append(int(count_str))
        for i in range(1, len(counts)):
            self.assertGreaterEqual(counts[i], counts[i - 1], "Bucket counts must be non-decreasing")

    def test_reset_clears_all_data(self) -> None:
        h = LatencyHistogram()
        h.record("tools/call", 10.0)
        h.reset()
        self.assertEqual(h.methods(), [])
        self.assertEqual(h.total_observations(), 0)

    def test_snapshot_shape(self) -> None:
        h = LatencyHistogram()
        h.record("tools/call", 10.0)
        snap = h.snapshot()
        self.assertIn("buckets_ms", snap)
        self.assertIn("methods", snap)
        self.assertIn("total_observations", snap)
        self.assertEqual(snap["total_observations"], 1)

    def test_invalid_buckets_raise(self) -> None:
        with self.assertRaises(ValueError):
            LatencyHistogram(buckets=[100.0, 50.0, 10.0])  # not sorted

    def test_custom_buckets_respected(self) -> None:
        h = LatencyHistogram(buckets=[10.0, 100.0, 1000.0])
        h.record("m", 5.0)
        h.record("m", 50.0)
        h.record("m", 500.0)
        snap = h.snapshot()
        self.assertEqual(snap["buckets_ms"], [10.0, 100.0, 1000.0])


class TestBenchCli(unittest.TestCase):
    """mcp-slim bench subcommand integration tests."""

    def test_bench_subcommand_runs_successfully(self) -> None:
        rc = main(["bench", "--turns", "3"])
        self.assertEqual(rc, 0)

    def test_bench_json_output_valid(self, capsys=None) -> None:
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            rc = main(["bench", "--turns", "5", "--json"])
        finally:
            sys.stdout = old_stdout
        self.assertEqual(rc, 0)
        output = captured.getvalue()
        data = json.loads(output)
        self.assertEqual(data["turns"], 5)
        self.assertIn("total_tokens_before", data)
        self.assertIn("total_tokens_after", data)
        self.assertIn("tokens_saved", data)
        self.assertGreater(data["tokens_saved"], 0)
        self.assertIn("latency_ms", data)
        self.assertIn("tools/list", data["latency_ms"])
        self.assertIn("tools/call", data["latency_ms"])

    def test_bench_json_latency_counts_match_turns(self) -> None:
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["bench", "--turns", "7", "--json"])
        finally:
            sys.stdout = old_stdout
        data = json.loads(captured.getvalue())
        self.assertEqual(data["latency_ms"]["tools/list"]["count"], 7)
        self.assertEqual(data["latency_ms"]["tools/call"]["count"], 7)

    def test_bench_with_sse_burst_budget_json(self) -> None:
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["bench", "--turns", "5", "--sse-burst-budget", "50", "--json"])
        finally:
            sys.stdout = old_stdout
        data = json.loads(captured.getvalue())
        self.assertIsNotNone(data["sse_throttle"])
        snap = data["sse_throttle"]
        self.assertTrue(snap["enabled"])
        self.assertEqual(snap["burst_budget_tokens"], 50)
        self.assertGreaterEqual(snap["chunks_seen"], 1)

    def test_bench_no_sse_budget_produces_null_throttle(self) -> None:
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["bench", "--turns", "2", "--json"])
        finally:
            sys.stdout = old_stdout
        data = json.loads(captured.getvalue())
        self.assertIsNone(data["sse_throttle"])

    def test_bench_reduction_pct_positive(self) -> None:
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["bench", "--turns", "1", "--json"])
        finally:
            sys.stdout = old_stdout
        data = json.loads(captured.getvalue())
        self.assertGreater(data["reduction_pct"], 0.0)

    def test_run_bench_function_direct(self) -> None:
        rc = run_bench(turns=4)
        self.assertEqual(rc, 0)

    def test_run_bench_json_direct(self) -> None:
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            rc = run_bench(turns=2, json_output=True)
        finally:
            sys.stdout = old_stdout
        self.assertEqual(rc, 0)
        data = json.loads(captured.getvalue())
        self.assertEqual(data["turns"], 2)


if __name__ == "__main__":
    unittest.main()
