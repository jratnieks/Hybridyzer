# core/profiles.py
"""
Configuration profiles for FinalSignalGenerator.
Defines named profiles with regime policies, thresholds, and other settings.
"""

from typing import Dict, Optional, Any


# Define available profiles
PROFILES: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # TREND_UP_ONLY: Conservative long-only in uptrends
    # ==========================================================================
    "trend_up_only": {
        "disable_regime": {
            "trend_up": False,
            "trend_down": True,
            "chop": True,
        },
        "regime_policy": {
            "trend_up": {"long": True, "short": False},
            "trend_down": {"long": False, "short": False},
            "chop": {"long": False, "short": False},
        },
        "probability_threshold": None,  # Use default or CLI override
        "disable_shorts": True,
    },
    
    # ==========================================================================
    # SHORT_ONLY: Only short trades in downtrends
    # ==========================================================================
    "short_only": {
        "disable_regime": {
            "trend_up": True,       # Skip uptrends
            "trend_down": False,    # Trade in downtrends
            "chop": True,           # Skip chop
        },
        "regime_policy": {
            "trend_up": {"long": False, "short": False},
            "trend_down": {"long": False, "short": True},  # Shorts only
            "chop": {"long": False, "short": False},
        },
        "probability_threshold": None,  # Use default or CLI override
        "disable_shorts": False,  # Must be False to allow shorts
    },
    
    # ==========================================================================
    # FULL: Trade both directions in trending regimes
    # ==========================================================================
    "full": {
        "disable_regime": {
            "trend_up": False,      # Trade in uptrends
            "trend_down": False,    # Trade in downtrends
            "chop": True,           # Skip chop (too noisy)
        },
        "regime_policy": {
            "trend_up": {"long": True, "short": False},     # Longs in uptrend
            "trend_down": {"long": False, "short": True},   # Shorts in downtrend
            "chop": {"long": False, "short": False},        # No trades in chop
        },
        "probability_threshold": None,  # Use default or CLI override
        "disable_shorts": False,  # Allow shorts
    },
    
    # ==========================================================================
    # FULL_AGGRESSIVE: Trade both directions in ALL regimes (including chop)
    # ==========================================================================
    "full_aggressive": {
        "disable_regime": {
            "trend_up": False,
            "trend_down": False,
            "chop": False,          # Trade even in chop
        },
        "regime_policy": {
            "trend_up": {"long": True, "short": False},
            "trend_down": {"long": False, "short": True},
            "chop": {"long": True, "short": True},  # Both directions in chop
        },
        "probability_threshold": None,
        "disable_shorts": False,
    },
}


def get_profile(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a profile by name.
    
    Args:
        name: Profile name
        
    Returns:
        Profile dictionary or None if not found
    """
    return PROFILES.get(name)


def list_profiles() -> list:
    """Return list of available profile names."""
    return list(PROFILES.keys())

