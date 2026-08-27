"""[P415] The DeBERTa L2 load must never block engine startup.

A stalled HF-CDN download of the DeBERTa base encoder previously hung the
runner __init__ for ~14 min (2026-08-27) with only venue-resting stops
protecting the book. This L2 is low-value/near-zero-weighted (P228/P296), so a
stalled or failed load must degrade to the F&G heuristic and let startup
proceed. These pin: startup returns within the timeout even when the load
hangs, a hang leaves the engine degraded (is_ready False), a fast load still
succeeds, a late-finishing download harmlessly recovers, and the HF call is
dispatched off the init thread (a future inline from_pretrained in __init__
would re-introduce the hang)."""
import inspect
import time

from agents import sentiment_deberta as sd


def test_a_stalled_load_does_not_block_startup(monkeypatch, tmp_path):
    f = tmp_path / "m.pt"
    f.write_text("x")
    monkeypatch.setattr(sd.DeBERTaSentimentEngine, "LOAD_TIMEOUT_SEC", 0.3)

    def _hang(self, _path, device="auto"):
        time.sleep(3.0)  # simulate a stalled HF download

    monkeypatch.setattr(sd.DeBERTaSentimentEngine, "_load_model", _hang)
    t0 = time.time()
    eng = sd.DeBERTaSentimentEngine(model_path=str(f))
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"startup blocked on a stalled load ({elapsed:.1f}s)"
    assert eng.is_ready is False  # degraded to the F&G heuristic


def test_a_fast_load_still_succeeds(monkeypatch, tmp_path):
    f = tmp_path / "m.pt"
    f.write_text("x")

    def _ok(self, _path, device="auto"):
        self.model = object()
        self._loaded = True

    monkeypatch.setattr(sd.DeBERTaSentimentEngine, "_load_model", _ok)
    eng = sd.DeBERTaSentimentEngine(model_path=str(f))
    assert eng.is_ready is True


def test_a_late_finishing_load_recovers_the_model(monkeypatch, tmp_path):
    """A download that finishes AFTER the timeout harmlessly recovers the model
    (benign late write) — degraded meanwhile, not broken."""
    f = tmp_path / "m.pt"
    f.write_text("x")
    monkeypatch.setattr(sd.DeBERTaSentimentEngine, "LOAD_TIMEOUT_SEC", 0.2)

    def _slow_ok(self, _path, device="auto"):
        time.sleep(0.6)
        self.model = object()
        self._loaded = True

    monkeypatch.setattr(sd.DeBERTaSentimentEngine, "_load_model", _slow_ok)
    eng = sd.DeBERTaSentimentEngine(model_path=str(f))
    assert eng.is_ready is False  # degraded right after startup
    time.sleep(0.8)
    assert eng.is_ready is True   # recovered once the late load finished


def test_missing_model_file_is_a_clean_degrade(tmp_path):
    eng = sd.DeBERTaSentimentEngine(model_path=str(tmp_path / "nope.pt"))
    assert eng.is_ready is False


def test_load_is_dispatched_off_the_init_thread():
    src = inspect.getsource(sd.DeBERTaSentimentEngine.__init__)
    assert "threading.Thread" in src, "load must run in a thread"
    assert "_load_model" in src
    assert "join(self.LOAD_TIMEOUT_SEC)" in src.replace(" ", "")
    # the blocking HF call must NOT be inline in __init__ (scan non-comment
    # lines — the fix comment itself names from_pretrained, P177)
    code_lines = [l for l in src.splitlines() if not l.strip().startswith("#")]
    assert not any("from_pretrained" in l for l in code_lines), (
        "HF call must not block __init__")
