#!/usr/bin/env python3
"""
Batch inference script for the Cascade Model Engine.
Loads a processed CSV, applies the cascade stack, and outputs both binary and probability predictions.
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
import pandas as pd
import numpy as np
from utils.seq_windows import make_windows
from app.cascade_engine import CascadeModelEngine

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference using CascadeModelEngine"
    )
    parser.add_argument(
        "--input", type=str,
        default="data/processed_dataset_labeled.csv",
        help="Path to processed input CSV"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/final_predictions.csv",
        help="Path to save predictions CSV"
    )
    parser.add_argument(
        "--threshold", type=float,
        default=None,
        help="Optional threshold for binary decision (overrides engine default)"
    )
    args = parser.parse_args()

    # Load and prepare data
    df = pd.read_csv(args.input)
    # Extract features and optionally labels
    X = df.drop(columns=["label", "Date"], errors="ignore").values
    y = df["label"].values if "label" in df.columns else None

    # Create rolling windows for RNN input
    X_seq, _ = make_windows(X, y if y is not None else np.zeros(len(X)), n_steps=5)

    # Initialize the cascade engine
    engine = CascadeModelEngine()

    # Get base predictions and probabilities
    preds, probs = engine.predict(X_seq)

    # Override threshold if provided
    if args.threshold is not None:
        preds = (probs >= args.threshold).astype(int)

    # Save results
    out_df = pd.DataFrame({
        "pred": preds,
        "prob": probs
    })
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(preds)} predictions to {args.output}")

if __name__ == "__main__":
    main()
