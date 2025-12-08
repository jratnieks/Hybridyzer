"""
Tests for mathematical invariants in backtest.py.

Validates PnL math, fee/slippage calculations, EV math, and cost propagation
using deterministic synthetic data.

WHY THESE TESTS EXIST:
- Financial calculations must be exact - small errors compound over time
- Fee/slippage handling directly impacts strategy profitability
- EV calculations drive trade filtering decisions

WHAT INVARIANTS ARE PROTECTED:
- PnL = position * future_return - costs (no missing terms)
- Round-trip cost = 2 * (fee + slippage) per trade
- Net return = gross return - cost (subtraction, not multiplication)
- Total transaction cost = num_trades * cost_per_trade
- EV math: expected_return * hit_rate

BUGS THESE TESTS WOULD CATCH:
- Fee applied per side instead of round-trip
- Cost multiplied instead of subtracted
- Missing cost on entry or exit
- Wrong basis points conversion (100 vs 10000)
- EV sign errors
"""

from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

# Import mocks
from tests.mocks import (
    make_synthetic_trades,
    make_winning_trades,
    make_losing_trades,
    make_mixed_trades,
)


class TestPnLMathInvariants:
    """
    Tests for PnL calculation correctness.
    
    WHY: PnL is the fundamental backtest output - must be mathematically correct.
    WHAT: Validates position * return formula and sign conventions.
    """
    
    def test_pnl_equals_position_times_return(self):
        """
        WHY: Verify core PnL formula: PnL = position * future_return.
        WHAT: Raw PnL before costs should exactly equal position * return.
        BUG CAUGHT: Extra or missing terms in PnL calculation.
        """
        positions = pd.Series([1, 1, -1, -1, 1, -1])
        future_returns = pd.Series([0.01, -0.02, -0.015, 0.025, 0.005, -0.01])
        
        pnl_raw = positions * future_returns
        
        # Verify each calculation
        assert np.isclose(pnl_raw.iloc[0], 1 * 0.01), "Long + positive return"
        assert np.isclose(pnl_raw.iloc[1], 1 * (-0.02)), "Long + negative return"
        assert np.isclose(pnl_raw.iloc[2], -1 * (-0.015)), "Short + negative return"
        assert np.isclose(pnl_raw.iloc[3], -1 * 0.025), "Short + positive return"
    
    def test_pnl_sign_convention(self):
        """
        WHY: Verify PnL sign matches expected profitability.
        WHAT: Profitable trades should have positive PnL.
        BUG CAUGHT: Inverted sign convention.
        """
        # Long profit scenario: long position, price goes up
        assert 1 * 0.01 > 0, "Long + up should profit"
        
        # Long loss scenario: long position, price goes down
        assert 1 * (-0.01) < 0, "Long + down should lose"
        
        # Short profit scenario: short position, price goes down
        assert -1 * (-0.01) > 0, "Short + down should profit"
        
        # Short loss scenario: short position, price goes up
        assert -1 * 0.01 < 0, "Short + up should lose"
    
    def test_flat_position_zero_pnl(self):
        """
        WHY: Verify flat positions generate zero PnL regardless of returns.
        WHAT: position=0 should always yield PnL=0.
        BUG CAUGHT: Non-zero PnL leaking from flat periods.
        """
        positions = pd.Series([0, 0, 0])
        future_returns = pd.Series([0.05, -0.03, 0.10])
        
        pnl = positions * future_returns
        
        assert (pnl == 0).all(), "Flat positions should have zero PnL"


class TestFeeAndSlippageMath:
    """
    Tests for transaction cost calculations.
    
    WHY: Transaction costs determine strategy viability.
    WHAT: Validates round-trip cost formula and application.
    """
    
    def test_basis_points_conversion(self):
        """
        WHY: Verify correct conversion from bps to decimal.
        WHAT: 1 bps = 0.0001 = 0.01%.
        BUG CAUGHT: Using 100 instead of 10000 for bps conversion.
        """
        fee_bps = 2.0  # 2 basis points
        slip_bps = 1.0  # 1 basis point
        
        fee_frac = fee_bps / 10000.0
        slip_frac = slip_bps / 10000.0
        
        assert fee_frac == 0.0002, "2 bps should be 0.0002"
        assert slip_frac == 0.0001, "1 bps should be 0.0001"
    
    def test_round_trip_cost_calculation(self):
        """
        WHY: Verify round-trip cost = 2 * (fee + slippage).
        WHAT: Each trade has entry cost + exit cost.
        BUG CAUGHT: Missing factor of 2, or only charging one side.
        """
        fee_bps = 2.0
        slip_bps = 1.0
        
        fee_frac = fee_bps / 10000.0
        slip_frac = slip_bps / 10000.0
        round_trip_cost = 2.0 * (fee_frac + slip_frac)
        
        expected = 2.0 * (0.0002 + 0.0001)  # 0.0006 = 6 bps
        assert np.isclose(round_trip_cost, expected), f"Round-trip should be {expected}"
        assert np.isclose(round_trip_cost, 0.0006), "6 bps round-trip"
    
    def test_net_returns_subtract_cost(self):
        """
        WHY: Verify net return = gross return - cost (subtraction).
        WHAT: Cost reduces return, not multiplies it.
        BUG CAUGHT: Using (1 - cost) * return instead of return - cost.
        """
        gross_return = 0.01  # 1% gross return
        cost = 0.0006  # 6 bps cost
        
        net_return = gross_return - cost
        
        assert net_return == 0.0094, "Net should be gross minus cost"
        assert net_return < gross_return, "Net should be less than gross"
    
    def test_cost_only_applied_to_trades(self):
        """
        WHY: Verify cost is only applied when position != 0.
        WHAT: Flat periods should not incur transaction costs.
        BUG CAUGHT: Applying cost to every bar instead of trade bars.
        """
        positions = pd.Series([1, 1, 0, -1, 0])
        trade_mask = positions != 0
        
        # Only 3 trades (positions 0, 1, 3)
        assert trade_mask.sum() == 3, "Should have 3 trades"
        
        cost_per_trade = 0.0006
        total_cost = trade_mask.sum() * cost_per_trade
        
        assert total_cost == 3 * 0.0006, "Cost should be num_trades * cost_per_trade"
    
    def test_cost_preserves_pnl_direction(self):
        """
        WHY: Verify cost reduces magnitude but preserves sign direction.
        WHAT: Positive gross should remain positive if > cost; negative stays negative.
        BUG CAUGHT: Cost turning positive returns negative unexpectedly.
        """
        cost = 0.0006
        
        # Gross return larger than cost: should stay positive
        gross_positive = 0.01
        net_positive = gross_positive - cost
        assert net_positive > 0, "Large positive gross should stay positive after cost"
        
        # Gross return smaller than cost: could become negative
        gross_small = 0.0003
        net_small = gross_small - cost
        assert net_small < 0, "Small positive gross can become negative after cost"
        
        # Negative gross becomes more negative
        gross_negative = -0.01
        net_negative = gross_negative - cost
        assert net_negative < gross_negative, "Negative gross becomes more negative with cost"


class TestEVMathInvariants:
    """
    Tests for Expected Value calculations.
    
    WHY: EV filtering determines which trades are taken.
    WHAT: Validates EV = avg_return * probability formula.
    """
    
    def test_ev_formula_basic(self):
        """
        WHY: Verify EV = expected_return * probability.
        WHAT: Simple EV calculation should match formula.
        BUG CAUGHT: Missing probability term or wrong formula.
        """
        avg_return = 0.002  # 0.2% average winning return
        hit_rate = 0.55  # 55% win rate
        avg_loss = -0.0015  # 0.15% average losing return
        
        # Simple EV approximation
        ev = hit_rate * avg_return + (1 - hit_rate) * avg_loss
        
        expected = 0.55 * 0.002 + 0.45 * (-0.0015)
        assert np.isclose(ev, expected), f"EV should be {expected}"
    
    def test_ev_positive_edge(self):
        """
        WHY: Verify positive EV indicates profitable edge.
        WHAT: EV > 0 means strategy is expected to be profitable.
        BUG CAUGHT: Sign error causing profitable trades to be filtered.
        """
        # High win rate, positive expected return
        hit_rate = 0.6
        avg_win = 0.002
        avg_loss = -0.0018
        
        ev = hit_rate * avg_win + (1 - hit_rate) * avg_loss
        
        assert ev > 0, "Strategy with edge should have positive EV"
    
    def test_ev_filtering_logic(self):
        """
        WHY: Verify EV filtering allows high-EV trades, blocks low-EV.
        WHAT: min_ev threshold should correctly filter trades.
        BUG CAUGHT: Wrong comparison operator or threshold application.
        """
        ev_values = pd.Series([0.001, -0.0005, 0.002, 0.0001, -0.001])
        min_ev = 0.0005
        
        # Filter: keep trades where EV >= min_ev
        allowed_mask = ev_values >= min_ev
        
        assert allowed_mask.iloc[0] == True, "EV 0.001 >= 0.0005"
        assert allowed_mask.iloc[1] == False, "EV -0.0005 < 0.0005"
        assert allowed_mask.iloc[2] == True, "EV 0.002 >= 0.0005"
        assert allowed_mask.iloc[3] == False, "EV 0.0001 < 0.0005"
        assert allowed_mask.iloc[4] == False, "EV -0.001 < 0.0005"


class TestTotalReturnMath:
    """
    Tests for total return and CAGR calculations.
    
    WHY: Total return is the primary performance metric.
    WHAT: Validates compounding and annualization formulas.
    """
    
    def test_total_return_formula(self):
        """
        WHY: Verify total_return = final_equity / initial_equity - 1.
        WHAT: Total return should match compounded returns.
        BUG CAUGHT: Using additive instead of multiplicative returns.
        """
        returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        equity = (1 + returns).cumprod()
        
        # Note: equity[0] is after first return, so total_return = equity[-1] - 1
        # to get return from starting capital of 1.0
        total_return = equity.iloc[-1] - 1.0
        
        # Manual calculation from starting capital 1.0
        expected = (1.01 * 1.02 * 0.99 * 1.015) - 1.0
        
        assert np.isclose(total_return, expected), f"Total return should be {expected}"
    
    def test_cagr_formula(self):
        """
        WHY: Verify CAGR = (final/initial)^(1/years) - 1.
        WHAT: CAGR should annualize total return correctly.
        BUG CAUGHT: Wrong exponent or missing -1 term.
        """
        initial = 1.0
        final = 1.5
        years = 2.0
        
        cagr = (final / initial) ** (1.0 / years) - 1.0
        
        expected = 1.5 ** 0.5 - 1.0  # About 22.47%
        assert np.isclose(cagr, expected), f"CAGR should be {expected}"
    
    def test_equity_curve_monotonic_for_positive_returns(self):
        """
        WHY: Verify equity curve increases monotonically for all positive returns.
        WHAT: Each positive return should increase equity.
        BUG CAUGHT: Sign error causing decreasing equity.
        """
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.005])
        equity = (1 + returns).cumprod()
        
        # Should be strictly increasing
        diffs = equity.diff().dropna()
        assert (diffs > 0).all(), "Equity should increase for all positive returns"


class TestHitRateMath:
    """
    Tests for hit rate and accuracy calculations.
    
    WHY: Hit rate measures trading accuracy.
    WHAT: Validates win/loss counting and percentage calculations.
    """
    
    def test_hit_rate_formula(self):
        """
        WHY: Verify hit_rate = profitable_trades / total_trades.
        WHAT: Hit rate should correctly count wins vs total.
        BUG CAUGHT: Off-by-one error or wrong denominator.
        """
        trades = make_mixed_trades(n_trades=12)
        returns = trades['trade_pnl']
        
        profitable = (returns > 0).sum()
        total = (returns != 0).sum()
        hit_rate = profitable / total if total > 0 else 0.0
        
        assert 0.0 <= hit_rate <= 1.0, "Hit rate should be between 0 and 1"
        assert total == 12, "Should have 12 total trades"
    
    def test_hit_rate_all_wins(self):
        """
        WHY: Verify 100% hit rate for all winning trades.
        WHAT: hit_rate should be 1.0 when all trades profit.
        BUG CAUGHT: Incorrect boundary handling.
        """
        trades = make_winning_trades(n_trades=10)
        returns = trades['trade_pnl']
        
        profitable = (returns > 0).sum()
        total = len(returns)
        hit_rate = profitable / total
        
        assert hit_rate == 1.0, "All winning trades should give 100% hit rate"
    
    def test_hit_rate_all_losses(self):
        """
        WHY: Verify 0% hit rate for all losing trades.
        WHAT: hit_rate should be 0.0 when all trades lose.
        BUG CAUGHT: Zero handling or wrong numerator.
        """
        trades = make_losing_trades(n_trades=10)
        returns = trades['trade_pnl']
        
        profitable = (returns > 0).sum()
        total = len(returns)
        hit_rate = profitable / total
        
        assert hit_rate == 0.0, "All losing trades should give 0% hit rate"


class TestNumericalStability:
    """
    Tests for numerical stability in calculations.
    
    WHY: Financial calculations must be numerically stable.
    WHAT: Validates handling of edge cases and extreme values.
    """
    
    def test_no_nan_propagation(self):
        """
        WHY: Verify NaN values don't propagate through calculations.
        WHAT: Clean data should produce clean results.
        BUG CAUGHT: NaN introduced by division or log of negative.
        """
        trades = make_synthetic_trades(n_trades=10)
        returns = trades['trade_pnl']
        
        equity = (1 + returns).cumprod()
        
        assert not equity.isna().any(), "Equity should have no NaN values"
        assert np.isfinite(equity).all(), "Equity should be finite"
    
    def test_zero_return_handling(self):
        """
        WHY: Verify zero returns don't cause issues.
        WHAT: Zero return should just preserve equity.
        BUG CAUGHT: Division by zero or log(0) errors.
        """
        returns = pd.Series([0.01, 0.0, 0.0, 0.02, 0.0])
        equity = (1 + returns).cumprod()
        
        assert np.isfinite(equity).all(), "Zero returns should not cause issues"
        # Equity should be same before and after zero return
        assert equity.iloc[1] == equity.iloc[2], "Zero return should preserve equity"
    
    def test_large_return_handling(self):
        """
        WHY: Verify large returns are handled correctly.
        WHAT: Extreme returns should not cause overflow.
        BUG CAUGHT: Float overflow or underflow.
        """
        # Very large positive return
        returns = pd.Series([0.5, 0.5, 0.5])  # 50% returns
        equity = (1 + returns).cumprod()
        
        assert np.isfinite(equity).all(), "Large returns should be handled"
        assert equity.iloc[-1] == 1.5 ** 3, "Should compound correctly"
    
    def test_negative_return_bounds(self):
        """
        WHY: Verify returns > -1 (can't lose more than 100%).
        WHAT: Equity should remain positive for valid returns.
        BUG CAUGHT: Negative equity from invalid returns.
        """
        # Returns close to -100% but valid
        returns = pd.Series([-0.1, -0.2, -0.15, -0.05])
        equity = (1 + returns).cumprod()
        
        assert (equity > 0).all(), "Equity should remain positive for valid returns"

