#!/usr/bin/env python3
"""
Clean training data by removing extreme outliers
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('training_data_augmented.csv')
print(f"Original samples: {len(df)}")

# Remove extreme outliers (values that would cause numerical issues)
# Use more aggressive filtering to remove unstable system states
threshold_def = min(df['kappa_def_95'].quantile(0.95), 1000)  # Keep 95% or cap at 1000
threshold_att = min(df['kappa_att_95'].quantile(0.95), 100)   # Keep 95% or cap at 100

print(f"\nThresholds (99th percentile):")
print(f"  kappa_def_95: {threshold_def:.2f}")
print(f"  kappa_att_95: {threshold_att:.2f}")

# Filter
df_clean = df[
    (df['kappa_def_95'] <= threshold_def) & 
    (df['kappa_att_95'] <= threshold_att)
].copy()

print(f"\nAfter filtering: {len(df_clean)} samples")
print(f"Removed: {len(df) - len(df_clean)} samples ({100*(len(df)-len(df_clean))/len(df):.1f}%)")

# Save cleaned data
df_clean.to_csv('training_data_clean.csv', index=False)
print(f"\nSaved to training_data_clean.csv")

# Show new statistics
print("\nCleaned kappa_def_95 statistics:")
print(df_clean['kappa_def_95'].describe())

print("\nCleaned kappa_att_95 statistics:")
print(df_clean['kappa_att_95'].describe())
