"""
Preprocessing Runner Script

Goal: Script to run the data preprocessing pipeline.
This module provides a command-line interface for running preprocessing.

Usage:
    # Activate virtual environment first
    source venv/bin/activate
    
    # Basic usage
    python scripts/run_preprocessing.py
    
    # With custom paths
    python scripts/run_preprocessing.py --data-dir data/ml-latest-small --output-dir data/processed
    
    # With verbose logging
    python scripts/run_preprocessing.py --verbose
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.preprocessing import MovieDataPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Run movie data preprocessing pipeline')
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/ml-latest-small',
        help='Path to raw data directory (default: data/ml-latest-small)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Path to output directory (default: data/processed)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    # Check for required files
    required_files = ['movies.csv', 'ratings.csv', 'tags.csv', 'links.csv']
    missing_files = []
    
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        logger.error(f"Please ensure all MovieLens files are in: {data_dir}")
        sys.exit(1)
    
    try:
        logger.info("Starting movie data preprocessing pipeline...")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Initialize preprocessor
        preprocessor = MovieDataPreprocessor(
            data_dir=str(data_dir),
            output_dir=args.output_dir
        )
        
        # Run preprocessing
        preprocessor.process_data()
        
        logger.info("✅ Preprocessing completed successfully!")
        logger.info(f"Processed data saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# the main() function is only called when run directly.
# if the file is imported, the main() function is not called.
# this is a good practice to avoid running the main() function when the file is imported.
if __name__ == "__main__":
    main()
