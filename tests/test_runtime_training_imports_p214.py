"""[P214] Every `training.*` module a RUNTIME module imports must be in the image.

`Dockerfile.engine` copies whole runtime packages but only an ALLOWLIST of
`training/` files, because most of `training/` has no business in a live trading
container. That allowlist and the runtime import sites are two files that have to
agree, and nothing checked that they did. Two live consequences, both found by
reading the engine log rather than by any test:

  * `training.scripts.wavelet_denoise` — imported EVERY TICK by
    `data_mgmt/market_data_pipeline.py:855` to produce 5 of the 122 model
    features. Never copied, so the import raised ModuleNotFoundError, the
    `except` fell back to RAW values, and serving disagreed with training on
    rsi_14/macd_12_26/bb_width_20/atr_14/vol_ratio_s for as long as this image
    has existed. (Blast radius is bounded: the per-asset GMMs take 12 features,
    none denoised, so REGIME classification is unaffected. The 5 land in the DRL
    observation vector — SHADOW, so no live orders — but they do confound the
    live IC used to judge whether the DRL should ever be re-promoted.)

  * `training.model_alpha.sequence_alpha_model` — the class definitions that
    unpickle `models/model_alpha/{ASSET}/sequence_alpha_v1_best.pt`. The
    checkpoints ARE in the volume; the classes were not. Live logged
    `No module named 'training.model_alpha'` for BTC and SOL, and model_alpha —
    an ADVISE agent in the authority matrix — emitted `+0.00/0.00` for two of
    three assets while HEALTH_S7 reported "model_alpha loaded".

Both share the shape this file exists to prevent: a swallowed ImportError whose
fallback is *plausible output*. The failure does not look like a failure — it
looks like a feature with no signal.

Same family as P192 (`.dockerignore` silently removing a COPYed script) and
P165 (`core.canonical_imports` unimportable for the life of the repo, swallowed
by an `except ImportError` five lines below).
"""

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_DOCKERFILE = _REPO / "Dockerfile.engine"

# Packages copied wholesale into the image — i.e. code that runs in production.
_RUNTIME_DIRS = [
    "core", "agents", "data_mgmt", "integration", "drl", "execution",
    "exchange", "signals", "defense", "risk", "strategies", "engine",
    "portfolio", "liquidity", "market", "orchestration", "infra",
]

_IMPORT_RE = re.compile(r"(?:^|\n)\s*(?:from|import)\s+(training\.[\w.]+)")

# Modules a runtime file imports that are DELIBERATELY not in the image. Each
# entry is a decision with a reason, not a suppression — and the entry itself is
# checked below (a stale exemption naming a now-shipped module fails), so this
# cannot quietly become a place where new breakage is parked.
#
# The bar for adding one: shipping the module would ENABLE a code path that has
# never executed in production. That is a live behaviour change and belongs to
# the operator, not to a dependency fix (P141/P177). Restoring a path that is
# merely degraded — the wavelet features, the sequence-alpha checkpoints — does
# NOT meet that bar and must be fixed instead.
_INTENTIONALLY_NOT_SHIPPED = {
    "training.regime.regime_classifier": (
        "orchestration/strategic_coordinator.py:200 — EnsembleRegimeClassifier "
        "has NEVER loaded in production (live logs 'Ensemble Regime Classifier "
        "not available: No module named training.regime'), so its consumer at "
        ":595 has never run. It is a SHORT filter and only tightens (reduces "
        "max_short_exposure, blocks new shorts in bullish regimes), but "
        "switching on a never-executed decision path on a live account is an "
        "operator call. The module is stdlib-only, so shipping it is a one-line "
        "change whenever that call is made."
    ),
}


def _copied_training_paths() -> set:
    """Every `training/...` token appearing in a COPY line."""
    df = _DOCKERFILE.read_text(encoding="utf-8")
    out = set()
    for line in df.splitlines():
        if not line.startswith("COPY"):
            continue
        for tok in line.split():
            if tok.startswith("training/"):
                out.add(tok)
    return out


def _runtime_training_imports() -> dict:
    """{dotted_module: [importer paths]} across runtime code."""
    found: dict = {}
    files = [_REPO / "main.py"]
    for d in _RUNTIME_DIRS:
        p = _REPO / d
        if p.is_dir():
            files.extend(p.rglob("*.py"))
    for f in files:
        try:
            src = f.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        for m in _IMPORT_RE.finditer(src):
            found.setdefault(m.group(1), []).append(
                str(f.relative_to(_REPO)).replace("\\", "/"))
    return found


def _is_shipped(dotted: str, copied: set) -> bool:
    """A module ships if its file is copied, or any ancestor package dir is."""
    rel = dotted.replace(".", "/")
    if f"{rel}.py" in copied or rel in copied:
        return True
    # a directory copied wholesale, e.g. `training/exit_drl`
    return any(rel.startswith(c.rstrip("/") + "/") for c in copied)


class TestEveryRuntimeTrainingImportShips:

    def test_the_scanner_finds_the_known_import_sites(self):
        """If this returns nothing the tests below pass vacuously — the exact
        defect class this file is about (P174)."""
        found = _runtime_training_imports()
        assert len(found) >= 4, f"scanner found too little: {sorted(found)}"
        assert "training.scripts.wavelet_denoise" in found

    @pytest.mark.parametrize("dotted", sorted(_runtime_training_imports()))
    def test_module_is_in_the_image(self, dotted):
        copied = _copied_training_paths()
        importers = _runtime_training_imports()[dotted]
        if dotted in _INTENTIONALLY_NOT_SHIPPED:
            pytest.skip(f"deliberate: {_INTENTIONALLY_NOT_SHIPPED[dotted]}")
        assert _is_shipped(dotted, copied), (
            f"{dotted} is imported by RUNTIME code ({importers[0]}) but is not "
            f"copied into the engine image. The import will raise "
            f"ModuleNotFoundError in production; if it sits under a try/except "
            f"the fallback will look like ordinary output, not a failure.\n"
            f"FIX: add it to the training allowlist in Dockerfile.engine.\n"
            f"Currently copied: {sorted(copied)}"
        )

    def test_the_two_regressions_specifically(self):
        """Named explicitly so the intent survives even if the scanner is
        later narrowed."""
        copied = _copied_training_paths()
        for dotted in ("training.scripts.wavelet_denoise",
                       "training.model_alpha.sequence_alpha_model"):
            assert _is_shipped(dotted, copied), dotted

    def test_every_exemption_is_still_real(self):
        """A stale exemption is worse than none — it would silently cover a
        module that later got shipped, or one nobody imports any more, and the
        next real breakage would be parked here without anyone noticing."""
        copied = _copied_training_paths()
        found = _runtime_training_imports()
        for dotted, reason in _INTENTIONALLY_NOT_SHIPPED.items():
            assert dotted in found, (
                f"{dotted} is exempted but no runtime module imports it — "
                f"delete the exemption"
            )
            assert not _is_shipped(dotted, copied), (
                f"{dotted} IS now in the image — delete the exemption so the "
                f"gate covers it"
            )
            assert len(reason) > 80, "an exemption must record WHY, not just that"

    def test_copied_training_files_actually_exist(self):
        """A COPY naming a missing file fails the BUILD (P192's symptom)."""
        for tok in _copied_training_paths():
            assert (_REPO / tok).exists(), f"Dockerfile COPYs missing {tok}"

    def test_packages_carry_their_init(self):
        """Copying `training/x/y.py` without `training/x/__init__.py` leaves
        `training.x` unimportable — the same ModuleNotFoundError by another
        route."""
        copied = _copied_training_paths()
        pkgs = {tok.rsplit("/", 1)[0] for tok in copied if tok.endswith(".py")}
        for pkg in pkgs:
            if pkg == "training":
                continue
            assert f"{pkg}/__init__.py" in copied, (
                f"{pkg} files are copied but {pkg}/__init__.py is not — the "
                f"package cannot be imported"
            )

    def test_nothing_shipped_is_dockerignored(self):
        """P192: naming a file in the COPY is not putting it in the context."""
        di = [ln.strip() for ln in
              (_REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
              if ln.strip() and not ln.strip().startswith("#")]
        excluded = [ln for ln in di if not ln.startswith("!")]
        negated = {ln[1:] for ln in di if ln.startswith("!")}
        for tok in _copied_training_paths():
            if tok in negated:
                continue
            for pat in excluded:
                base = pat.rstrip("/")
                if tok == base or tok.startswith(base + "/"):
                    pytest.fail(
                        f"{tok} is COPYed but excluded by .dockerignore rule "
                        f"'{pat}' with no negation — the build will fail or the "
                        f"file will be silently absent"
                    )


class TestTheLibraryShipsWithTheModule:

    def test_pywavelets_is_a_runtime_dependency(self):
        """Shipping wavelet_denoise.py without PyWavelets just moves the
        failure from ModuleNotFoundError on `training.scripts` to
        ModuleNotFoundError on `pywt` — same silent RAW fallback."""
        req = (_REPO / "requirements-runtime.txt").read_text(encoding="utf-8")
        assert re.search(r"(?im)^\s*pywavelets\b", req), (
            "training/scripts/wavelet_denoise.py imports pywt"
        )

    def test_the_module_still_imports_pywt(self):
        """If it stops needing pywt, the requirement above is dead weight."""
        src = (_REPO / "training" / "scripts" / "wavelet_denoise.py").read_text(
            encoding="utf-8", errors="replace")
        assert re.search(r"(?m)^\s*import pywt", src)


class TestTheFallbackStaysVisible:

    def test_the_wavelet_fallback_logs_a_warning(self):
        """Belt and braces: if the module goes missing again, the degradation
        must not be silent. A DEBUG here would be P160 exactly."""
        src = (_REPO / "data_mgmt" / "market_data_pipeline.py").read_text(
            encoding="utf-8", errors="replace")
        i = src.index("from training.scripts.wavelet_denoise import")
        w = src[i:i + 1200]
        assert "logger.warning" in w
        assert "using raw" in w
