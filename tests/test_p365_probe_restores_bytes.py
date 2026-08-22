"""[P365] The falsification harness's "restored byte-identically" check could
not observe the property it named.

Found by answering "any remaining tasks" honestly: two files showed as
modified in the shared working tree with an EMPTY content diff — pure
line-ending residue, left by my own probe runs.

    read:     io.open(target, encoding="utf-8").read()   # CRLF -> LF
    restore:  io.open(..., newline="").write(original)   # writes LF
    verify:   re-read in text mode and compare TEXT

The verification decoded both sides, so it compared text to text and passed
while the FILE on disk had changed from CRLF to LF. **A check that cannot
fail (P174) — inside the harness built to catch exactly that class.**

The cost was not cosmetic. In a shared working tree it left every probed file
showing as modified, i.e. clutter indistinguishable from another session's
legitimate work (P344's rule about measurement residue) — and "whose change
is this?" is precisely the question that produced three partial commits in
one day (P352b, P357b, P357d).

Fixed by keeping the RAW BYTES for restore and verifying against bytes. The
decoded text is still what anchors are matched against, because probe `old`
strings are written with bare LF.
"""

import io
import pathlib

import pytest

import tools.falsify as fz
from tools.falsify import Probe, run_probe

CRLF = b"line one\r\nMARKER = 1\r\nline three\r\n"


def _rc(code):
    return type("R", (), {"returncode": code,
                          "stdout": "1 failed" if code else "ok",
                          "stderr": ""})()


def _stateful(path, marker):
    """GREEN until the mutation lands, RED after — the shape a real guard
    has. A fake that is red from the start trips the harness's own
    ALREADY-RED refusal, which is correct behaviour and would make these
    tests measure that clause instead of the one they name (P357's lesson)."""
    def _run(_t):
        return _rc(1) if marker in path.read_text(encoding="utf-8") else _rc(0)
    return _run


def _probe(tmp_path, monkeypatch, red=True):
    monkeypatch.setattr(fz, "REPO", tmp_path)
    (tmp_path / "s.py").write_bytes(CRLF)
    monkeypatch.setattr(fz, "_pytest",
                        _stateful(tmp_path / "s.py", "MARKER = 2") if red
                        else (lambda t: _rc(0)))
    return Probe(name="t", path="s.py", old="MARKER = 1", new="MARKER = 2",
                 expect_red=["missing.py"])


def test_a_CRLF_file_is_restored_byte_for_byte(tmp_path, monkeypatch):
    """The defect: content came back identical and the BYTES did not."""
    p = _probe(tmp_path, monkeypatch)
    assert run_probe(p) is True, p.detail
    assert (tmp_path / "s.py").read_bytes() == CRLF, (
        "the probe rewrote line endings — in a shared tree that leaves the "
        "file showing as modified, indistinguishable from someone else's work"
    )


def test_it_restores_bytes_even_when_the_probe_FAILS(tmp_path, monkeypatch):
    """A vacuous probe still has to leave the tree exactly as it found it —
    the restore is in a `finally` and the failure path is the common one."""
    p = _probe(tmp_path, monkeypatch, red=False)
    assert run_probe(p) is False
    assert p.result == "VACUOUS"
    assert (tmp_path / "s.py").read_bytes() == CRLF


def test_an_LF_file_is_untouched_too(tmp_path, monkeypatch):
    """The common case must not regress while fixing the CRLF one."""
    monkeypatch.setattr(fz, "REPO", tmp_path)
    lf = b"a\nMARKER = 1\nb\n"
    (tmp_path / "s.py").write_bytes(lf)
    monkeypatch.setattr(fz, "_pytest",
                        _stateful(tmp_path / "s.py", "MARKER = 2"))
    p = Probe(name="t", path="s.py", old="MARKER = 1", new="MARKER = 2",
              expect_red=["missing.py"])
    assert run_probe(p) is True, p.detail
    assert (tmp_path / "s.py").read_bytes() == lf


def test_anchors_still_match_across_a_CRLF_file(tmp_path, monkeypatch):
    """Why the decoded text is still read: probe `old` strings are written
    with bare LF, so matching against raw CRLF bytes would fail for any
    multi-line anchor. Keeping both is the point of the fix."""
    monkeypatch.setattr(fz, "REPO", tmp_path)
    (tmp_path / "s.py").write_bytes(b"def f():\r\n    return 1\r\n")
    seen = {}

    def _fake(_t):
        seen["src"] = (tmp_path / "s.py").read_text(encoding="utf-8")
        return _rc(1) if "return 2" in seen["src"] else _rc(0)

    monkeypatch.setattr(fz, "_pytest", _fake)
    p = Probe(name="t", path="s.py",
              old="def f():\n    return 1",      # bare LF, as probes are written
              new="def f():\n    return 2",
              expect_red=["missing.py"])
    assert run_probe(p) is True, (
        f"a multi-line anchor failed against a CRLF file: {p.detail}"
    )
    assert "return 2" in seen["src"], "the mutation never reached the file"


def test_the_verification_compares_BYTES_not_decoded_text():
    """The structural half. The old check re-decoded both sides, which is why
    it could not see the difference it was named for."""
    src = pathlib.Path(fz.__file__).read_text(encoding="utf-8")
    i = src.index("did NOT restore byte-identically")
    window = src[max(0, i - 700):i]
    assert "read_bytes()" in window, (
        "the restore verification no longer compares bytes — it cannot "
        "observe the property it claims (P174)"
    )
    assert 'io.open(target, encoding="utf-8").read()' not in window, (
        "the verification re-decodes, so CRLF->LF passes silently again"
    )


def test_a_genuinely_unrestored_file_is_still_caught(tmp_path, monkeypatch):
    """Anti-vacuity (P174): the check must still fire when the tree really is
    left carrying the defect."""
    monkeypatch.setattr(fz, "REPO", tmp_path)
    (tmp_path / "s.py").write_bytes(CRLF)
    monkeypatch.setattr(fz, "_pytest",
                        _stateful(tmp_path / "s.py", "MARKER = 2"))

    real = pathlib.Path.write_bytes

    def _sabotage(self, data):
        return real(self, b"NOT RESTORED\r\n")

    monkeypatch.setattr(pathlib.Path, "write_bytes", _sabotage)
    p = Probe(name="t", path="s.py", old="MARKER = 1", new="MARKER = 2",
              expect_red=["missing.py"])
    run_probe(p)
    monkeypatch.undo()
    assert p.result == "NOT_RESTORED", (
        f"a file left unrestored was not reported (got {p.result})"
    )
    assert io  # keep the import meaningful for readers
