from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_main_hot_path_uses_timezone_aware_now_instead_of_utcnow():
    text = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    assert "utcnow(" not in text


def test_selected_runtime_hot_paths_use_timezone_aware_now():
    for rel_path in (
        "execution/loop_controller.py",
        "risk/global_exposure_cap.py",
        "risk/volatility_targeting.py",
        "risk/stop_loss_authority.py",
    ):
        text = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
        assert "utcnow(" not in text, f"{rel_path} still uses utcnow()"
