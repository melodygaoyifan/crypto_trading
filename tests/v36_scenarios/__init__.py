"""
V36 Scenario Tests
==================

Scenario-based testing and validation for HMATS trading logic.
Migrated from integration/core/ to ensure trading logic is properly validated.

Modules:
    scenario_walkthroughs.py - Core scenario testing (813 lines)
    scenario_walkthroughs_unified.py - Unified v3.2 + v3.6 testing (874 lines)
    test_production_patches.py - Production reliability tests (500 lines)
    test_v4_modules.py - v4 module verification tests (479 lines)

Total: 2,666 lines of test code
"""

# Explicit imports for type checking and IDE support
# Use explicit module imports instead of star imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type stubs for IDE support
    from . import scenario_walkthroughs
    from . import scenario_walkthroughs_unified
    from . import test_production_patches
    from . import test_v4_modules

__all__ = [
    "scenario_walkthroughs",
    "scenario_walkthroughs_unified",
    "test_production_patches",
    "test_v4_modules",
]
