from pathlib import Path


def test_kraken_quant_agent_header_matches_live_quant_identity():
    agent_path = Path(__file__).parent.parent / "agents" / "kraken_quant_agent.py"
    content = agent_path.read_text(encoding="utf-8")

    assert "Primary live quant path: main.py Best-of-N runtime + execution gating." in content
    assert "Primary quant path: agents/quant_agent.py" not in content
