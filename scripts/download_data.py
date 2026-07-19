"""
Ingestion script to download, validate, and cache historical stock data for market registries.
Saves datasets in CSV format under data/stocks/ and generates download_report.csv.
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.data import DataLoader, DataStorage
from ai_engine.data.tickers import load_registry
from ai_engine.utils.config import settings
from ai_engine.utils.logging import logger

def format_bytes(size_bytes: int) -> str:
    """Format bytes into a human-readable string (KB, MB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def run_download_pipeline():
    parser = argparse.ArgumentParser(description="Multi-Market Stock Index Downloader & Preprocessing Ingestion Pipeline")
    parser.add_argument(
        "--index",
        type=str,
        default="nifty50",
        help="Target index name (nifty50, nifty100, nifty200, nifty500, sp500, nasdaq100, dowjones30) or path to custom JSON file."
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date for historical stock prices (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date for historical stock prices (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1d", "1h", "30m", "15m", "5m", "1m"],
        help="yFinance frequency interval settings."
    )
    
    args = parser.parse_args()
    
    logger.info("Initializing multi-index data ingestion pipeline...")
    logger.info(f"Configuration: Index={args.index} | Range={args.start} to {args.end} | Interval={args.interval}")
    
    # 1. Load constituents registry
    try:
        tickers = load_registry(args.index)
        logger.info(f"Loaded registry. Constituent count: {len(tickers)}")
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")
        sys.exit(1)
        
    # Force storage format to CSV for raw data
    storage = DataStorage(file_format="csv")
    loader = DataLoader(storage=storage, index_name=args.index)
    
    successful = []
    failed = {}
    report_rows = []
    total_rows = 0

    # Download constituent loop
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Querying {ticker}...", end="\r")
        try:
            dataset = loader.get_ticker_data(
                ticker=ticker,
                start_date=args.start,
                end_date=args.end,
                force_download=True,
                interval=args.interval
            )
            
            # Count missing values
            df = dataset.df
            missing_cells = df.isnull().sum().sum()
            row_count = len(df)
            
            successful.append(ticker)
            total_rows += row_count
            
            report_rows.append({
                "Ticker": ticker,
                "Downloaded Rows": row_count,
                "Missing Values": missing_cells,
                "Status": "SUCCESS"
            })
        except Exception as e:
            failed[ticker] = str(e)
            logger.error(f"Failed to ingest symbol '{ticker}': {e}")
            report_rows.append({
                "Ticker": ticker,
                "Downloaded Rows": 0,
                "Missing Values": 0,
                "Status": f"FAILED ({str(e)})"
            })
            
    print("\nData loading complete. Writing download reports...")
    
    # Write download_report.csv
    report_df = pd.DataFrame(report_rows)
    report_csv_path = settings.BASE_PATH / "download_report.csv"
    report_df.to_csv(report_csv_path, index=False)
    logger.info(f"Saved tabular download report to: {report_csv_path.absolute()}")
    
    # Compute folder size of storage directory
    raw_dir = storage.raw_dir
    total_size_bytes = sum(f.stat().st_size for f in raw_dir.glob("*.csv") if f.is_file())
    formatted_size = format_bytes(total_size_bytes)
    
    # Write detailed txt execution report
    summary_report = f"""
=========================================================================
                    DATA INGESTION PIPELINE SUMMARY
=========================================================================
Execution Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Index Target        : {args.index}
Date Range Query    : {args.start} to {args.end} (Interval: {args.interval})
Storage Directory   : {raw_dir.absolute()}

Ingestion Metrics:
-----------------
Total Tickers Checked : {len(tickers)}
Successful Downloads  : {len(successful)} / {len(tickers)} ({len(successful)/len(tickers)*100:.1f}%)
Failed Downloads      : {len(failed)} / {len(tickers)}
Total Records Saved   : {total_rows:,} rows
Raw Cache Folder Size : {formatted_size}
Report Location       : {report_csv_path}
=========================================================================
"""
    print(summary_report)
    
    # Write summary txt report to docs folder
    summary_txt_path = settings.BASE_PATH / "docs" / "ingestion_report.txt"
    try:
        summary_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write(summary_report)
        logger.info(f"Saved summary execution report to: {summary_txt_path.absolute()}")
    except Exception as e:
        logger.error(f"Failed to save text summary report: {e}")

if __name__ == "__main__":
    run_download_pipeline()
