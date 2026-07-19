"""
Ingestion script to download, validate, and cache historical stock data for all NIFTY 50 tickers.
Saves datasets in CSV format under data/raw/ and generates a summary execution report.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.data import DataLoader, DataStorage, NIFTY_50_TICKERS
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
    logger.info("Initializing historical NIFTY 50 data download pipeline...")
    
    # 1. Define bounds
    start_date = "2010-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Target date range: {start_date} to {end_date}")
    
    # Force storage format to CSV for raw data
    storage = DataStorage(file_format="csv")
    loader = DataLoader(storage=storage)
    
    successful = []
    failed = {}
    total_rows = 0

    logger.info(f"Preparing download for {len(NIFTY_50_TICKERS)} tickers...")
    
    # Download sequential loop to capture granular metrics for the summary report
    for i, ticker in enumerate(NIFTY_50_TICKERS, start=1):
        print(f"[{i}/{len(NIFTY_50_TICKERS)}] Processing {ticker}...", end="\r")
        try:
            # get_ticker_data handles cache hits internally
            dataset = loader.get_ticker_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                force_download=False
            )
            successful.append(ticker)
            total_rows += len(dataset)
        except Exception as e:
            failed[ticker] = str(e)
            logger.error(f"Failed to process {ticker}: {e}")
            
    print("\nProcessing complete. Gathering storage statistics...")
    
    # Compute folder size using Storage directory settings
    raw_dir = storage.raw_dir
    total_size_bytes = sum(f.stat().st_size for f in raw_dir.glob("*") if f.is_file())
    formatted_size = format_bytes(total_size_bytes)
    
    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    summary_report = f"""
=========================================================================
                    DATA INGESTION PIPELINE REPORT
=========================================================================
Execution Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Date Range Requested: {start_date} to {end_date}
Storage Directory   : {raw_dir.absolute()}
File Storage Format : CSV (.csv + .metadata.json)

Ingestion Metrics:
-----------------
Total Tickers      : {len(NIFTY_50_TICKERS)}
Successful Downloads: {len(successful)} / {len(NIFTY_50_TICKERS)} ({len(successful)/len(NIFTY_50_TICKERS)*100:.1f}%)
Failed Downloads    : {len(failed)} / {len(NIFTY_50_TICKERS)}
Total Records Saved : {total_rows:,} rows
Raw Cache Directory Size: {formatted_size}

Successful Tickers  : {', '.join(successful[:10])}... (and {max(0, len(successful)-10)} more)
"""
    
    if failed:
        summary_report += "\nFailed Tickers & Root Cause:\n"
        for ticker, error in failed.items():
            summary_report += f"- {ticker}: {error}\n"
            
    summary_report += "========================================================================="
    
    # Output report to console
    print(summary_report)
    
    # Write report to docs folder for tracking
    report_path = settings.BASE_PATH / "docs" / "ingestion_report.txt"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(summary_report)
        logger.info(f"Pipeline report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save pipeline report: {e}")

if __name__ == "__main__":
    run_download_pipeline()
