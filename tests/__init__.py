"""
Hybridyzer Verifier Layer - Test Suite

=============================================================================
CURSOR AUTO-RUN SAFETY
=============================================================================

This test suite is designed to be SAFE for Cursor's automatic test execution.

SAFE TESTS (auto-run in Cursor):
- test_ev_consistency.py       - EV math validation
- test_no_lookahead.py         - Lookahead bias detection
- test_future_leak.py          - Data leakage detection  
- test_time_alignment.py       - Timestamp/index alignment
- test_backtest_structures.py  - Schema validation
- test_tools_init.py           - Tools initialization smoke tests

MANUAL-ONLY (require explicit invocation):
- Any test marked with @pytest.mark.manual
- Any test marked with @pytest.mark.slow
- Any test marked with @pytest.mark.gpu

SCRIPTS THAT ARE MANUAL-ONLY (never run via pytest):
- python backtest.py
- python train.py
- python tools/audit_backtest.py (standalone)
- python tools/walk_forward.py (standalone)
- python tools/profile_data.py (standalone)

Usage:
    # Run all safe tests (Cursor auto-run)
    pytest tests/ -v
    
    # Run specific test module
    pytest tests/test_ev_consistency.py -v
    
    # Run manual tests explicitly
    pytest tests/ -m manual -v
    
    # Run with coverage
    pytest tests/ --cov=core --cov=tools --cov-report=html
"""

from pathlib import Path


# Package version
__version__ = "0.1.0"

# Test module names for discovery
TEST_MODULES = [
    "test_ev_consistency",      # EV math - SAFE
    "test_no_lookahead",        # Lookahead prevention - SAFE
    "test_future_leak",         # Leakage detection - SAFE
    "test_time_alignment",      # Index alignment - SAFE
    "test_backtest_structures", # Schema validation - SAFE
    "test_tools_init",          # Tools smoke tests - SAFE
]

# Manual-only test markers
MANUAL_MARKERS = ["manual", "slow", "gpu", "integration"]


def get_test_dir() -> Path:
    """Return the path to the tests directory."""
    return Path(__file__).parent


def list_test_modules() -> list[str]:
    """List all available test modules."""
    return TEST_MODULES


def list_safe_tests() -> list[str]:
    """List tests safe for Cursor auto-run."""
    return TEST_MODULES


def list_manual_tests() -> list[str]:
    """List tests that require manual invocation."""
    return ["Any test marked with @pytest.mark.manual"]


__all__ = [
    "__version__",
    "TEST_MODULES",
    "MANUAL_MARKERS",
    "get_test_dir",
    "list_test_modules",
    "list_safe_tests",
    "list_manual_tests",
]
