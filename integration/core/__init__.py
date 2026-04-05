"""integration.core - legacy subpackage."""

VERSION = "5.1.0-CONSOLIDATED"

# [T13] compat.py dead code - all wrappers have broken import paths, 0 consumers
AgentSignal = None
MetaIntegration = None

__all__ = [
    "VERSION",
    "AgentSignal",
    "MetaIntegration",
]