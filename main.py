# main.py
"""
Main entry point for Hybridyzer trading system.
"""

import pandas as pd
from modules.superma import SuperMA4hr
from modules.trendmagic import TrendMagicV2
from modules.pvt_eliminator import PVTEliminator
from modules.pivots_rsi import PivotRSIContext
from modules.linreg_channel import LinRegChannelContext
from modules.ohlcv_context import OHLCVContext
from core.hybrid_engine import HybridEngine
from data.btc_data_loader import BTCDataLoader


def main():
    """
    Main execution function.
    """
    # Initialize data loader
    data_loader = BTCDataLoader()
    
    # Load data from available CSV file
    try:
        df = data_loader.load_data()
        print(f"Loaded {len(df)} rows of data")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize signal modules (produce signals)
    signal_modules = [
        SuperMA4hr(),
        TrendMagicV2(),
        PVTEliminator(),
    ]
    
    # Initialize context modules (produce features only, no signals)
    context_modules = [
        PivotRSIContext(),
        LinRegChannelContext(),
        OHLCVContext(),
    ]
    
    # Initialize hybrid engine
    engine = HybridEngine(
        signal_modules=signal_modules,
        context_modules=context_modules
    )
    
    # Process data through engine
    results = engine.process(df)
    
    # TODO: Output results, save to file, or visualize
    print("Processing complete.")
    print(f"Results shape: {results.shape}")
    print(f"Results columns: {results.columns.tolist()}")


if __name__ == "__main__":
    main()
