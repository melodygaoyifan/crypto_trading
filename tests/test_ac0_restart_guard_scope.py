from main import HMATSProductionRunner


def _make_runner():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._ac0_post_restart_tick = True
    runner._ac0_requires_entry_block = True
    runner._ac0_restored_assets = {"SOL"}
    runner._ac4_extended_cooldown = 0
    return runner


def test_ac0_blocks_only_restored_asset_entries():
    runner = _make_runner()

    assert runner._get_ac0_entry_block_reason("SOL", True) == (
        "post-restart cooldown -restored positions still reconciling"
    )
    assert runner._get_ac0_entry_block_reason("ETH", True) == ""


def test_ac0_crash_loop_extension_still_blocks_all_assets():
    runner = _make_runner()
    runner._ac4_extended_cooldown = 2

    assert runner._get_ac0_entry_block_reason("SOL", True) == (
        "post-restart cooldown -crash-loop protection"
    )
    assert runner._get_ac0_entry_block_reason("ETH", True) == (
        "post-restart cooldown -crash-loop protection"
    )


def test_ac0_ignores_non_entry_requests():
    runner = _make_runner()

    assert runner._get_ac0_entry_block_reason("SOL", False) == ""
