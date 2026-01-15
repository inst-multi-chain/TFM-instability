#!/usr/bin/env python3
"""
Generate Boundary Training Data for epsilon=16, lambda=16 (All-Max scenario)

This script generates training samples specifically for the worst-case scenario
where all elasticity parameters are at maximum (16.0), comparable to EIP-1559.
This addresses the OOD (Out-of-Distribution) problem where the model has never
seen this extreme corner case during training.

Rationale:
- Training data has epsilon/lambda randomly distributed in [0, 16]
- But "all elements = 16" is a corner point in high-dimensional space
- Model needs explicit training on this boundary case for robust prediction
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import subprocess
import shutil
from pathlib import Path

# Configuration
NUM_SAMPLES = 500  # Number of boundary samples to generate
EPSILON_FIXED = 16.0  # Fixed epsilon value (maximum)
LAMBDA_FIXED = 16.0   # Fixed lambda value (maximum)

# Simulation parameters
NUM_SHARDS = 7
G_MAX = 2_000_000
TARGET_DEMAND = G_MAX / 2

# Delta range
DELTA_MIN = 0.05
DELTA_MAX = 0.5

# Alpha range (cross-shard ratio)
ALPHA_MIN = 0.05
ALPHA_MAX = 0.95

# d_max possibilities
D_MAX_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]

# Paths
SIMULATION_DIR = Path('../simulations/time-consuming-simulations/boundary-training')
CONFIG_TEMPLATE = 'config_template.yml'
OUTPUT_CSV = 'boundary_training_data.csv'


def create_simulation_dir():
    """Create simulation directory"""
    SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created simulation directory: {SIMULATION_DIR}")


def generate_random_parameters():
    """Generate random parameters for one sample"""
    # Random delta
    delta = np.random.uniform(DELTA_MIN, DELTA_MAX)
    
    # Random alpha (total cross-shard ratio)
    alpha_total = np.random.uniform(ALPHA_MIN, ALPHA_MAX)
    
    # Split into inbound/outbound (random split)
    split_ratio = np.random.uniform(0.2, 0.8)
    alpha_inbound = alpha_total * split_ratio
    alpha_outbound = alpha_total * (1 - split_ratio)
    
    # Random d_max
    d_max = np.random.choice(D_MAX_VALUES)
    
    return {
        'delta': delta,
        'alpha_total': alpha_total,
        'alpha_inbound': alpha_inbound,
        'alpha_outbound': alpha_outbound,
        'd_max': d_max
    }


def build_demand_matrix(alpha_inbound, alpha_outbound):
    """Build demand matrix from alpha ratios"""
    matrix = np.zeros((NUM_SHARDS, NUM_SHARDS))
    T = TARGET_DEMAND
    
    # Shard 0
    shard0_local = 1.0 - alpha_inbound - alpha_outbound
    matrix[0, 0] = shard0_local * T
    
    # Shard 0 outbound
    outbound_per_shard = (alpha_outbound * T) / (NUM_SHARDS - 1)
    for j in range(1, NUM_SHARDS):
        matrix[0, j] = outbound_per_shard
    
    # Shard 0 inbound
    inbound_per_shard = (alpha_inbound * T) / (NUM_SHARDS - 1)
    for i in range(1, NUM_SHARDS):
        matrix[i, 0] = inbound_per_shard
    
    # Other shards
    other_cross_ratio = alpha_inbound + alpha_outbound
    other_local_ratio = 1.0 - other_cross_ratio
    
    for i in range(1, NUM_SHARDS):
        matrix[i, i] = other_local_ratio * T
    
    symmetric_traffic = (other_cross_ratio * T - inbound_per_shard - outbound_per_shard) / 10
    
    for i in range(1, NUM_SHARDS):
        for j in range(1, NUM_SHARDS):
            if i != j:
                matrix[i, j] = symmetric_traffic
    
    return matrix


def create_config(params, sample_id):
    """Create config.yml for simulation"""
    demand_matrix = build_demand_matrix(params['alpha_inbound'], params['alpha_outbound'])
    
    # All-16 epsilon matrix
    epsilon_matrix = [[EPSILON_FIXED] * NUM_SHARDS for _ in range(NUM_SHARDS)]
    
    # All-16 lambda matrix
    lambda_matrix = [[LAMBDA_FIXED] * NUM_SHARDS for _ in range(NUM_SHARDS)]
    
    # Delay weights (spike distribution)
    d_max = params['d_max']
    delay_weights = [0] * (d_max - 1) + [1]
    
    config = {
        'simulation': {
            'total_steps': 5000,
            'l_target': 0.5,
            'delta': float(params['delta']),
            'perturbation': {'enabled': False},
            'demand_shock': {
                'enabled': True,
                'start_step': 100,
                'end_step': 101,
                'multiplier': 15,
                'target_shards': [0]
            }
        },
        'network': {'num_shards': NUM_SHARDS},
        'shards': [{'id': i, 'g_max': G_MAX} for i in range(NUM_SHARDS)],
        'demand': {
            'base_demand_matrix': [[float(v) for v in row] for row in demand_matrix],
            'epsilon_matrix': epsilon_matrix,
            'lambda_matrix': lambda_matrix
        },
        'delay': {
            'max_delay': d_max,
            'weights': delay_weights
        }
    }
    
    config_path = SIMULATION_DIR / f'config_{sample_id}.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def run_simulation(config_path, sample_id):
    """Run Go simulation and parse results"""
    # TODO: Implement actual simulation run
    # For now, return dummy data
    # You'll need to replace this with actual simulation execution
    
    # Placeholder: generate reasonable kappa values
    # In reality, with epsilon=16, kappa values should be quite high
    kappa_def_95 = np.random.uniform(5, 50)  # High defense amplification
    kappa_att_95 = np.random.uniform(3, 40)  # High attack amplification
    
    return kappa_def_95, kappa_att_95


def generate_boundary_samples():
    """Generate all boundary samples"""
    print(f"\n{'='*80}")
    print(f"GENERATING {NUM_SAMPLES} BOUNDARY TRAINING SAMPLES")
    print(f"Epsilon matrix: ALL elements = {EPSILON_FIXED}")
    print(f"Lambda matrix: ALL elements = {LAMBDA_FIXED}")
    print(f"{'='*80}\n")
    
    create_simulation_dir()
    
    samples = []
    
    for i in range(NUM_SAMPLES):
        # Generate random parameters
        params = generate_random_parameters()
        
        print(f"Sample {i+1}/{NUM_SAMPLES}: δ={params['delta']:.3f}, "
              f"α_in={params['alpha_inbound']:.3f}, α_out={params['alpha_outbound']:.3f}, "
              f"d_max={params['d_max']}")
        
        # Create config
        config_path = create_config(params, i)
        
        # Run simulation (TODO: implement actual simulation)
        kappa_def, kappa_att = run_simulation(config_path, i)
        
        # Prepare epsilon/lambda matrix columns
        epsilon_flat = [EPSILON_FIXED] * (NUM_SHARDS * NUM_SHARDS)
        lambda_flat = [LAMBDA_FIXED] * (NUM_SHARDS * NUM_SHARDS)
        
        # Build sample row
        sample = {
            'delta': params['delta'],
            'alpha_total': params['alpha_total'],
            'alpha_inbound': params['alpha_inbound'],
            'alpha_outbound': params['alpha_outbound'],
            'd_max': params['d_max'],
            'avg_epsilon': EPSILON_FIXED,
            'avg_lambda': LAMBDA_FIXED,
            'kappa_def_95': kappa_def,
            'kappa_att_95': kappa_att
        }
        
        # Add epsilon matrix columns
        for idx, val in enumerate(epsilon_flat):
            row = idx // NUM_SHARDS
            col = idx % NUM_SHARDS
            sample[f'epsilon_{row}_{col}'] = val
        
        # Add lambda matrix columns
        for idx, val in enumerate(lambda_flat):
            row = idx // NUM_SHARDS
            col = idx % NUM_SHARDS
            sample[f'lambda_{row}_{col}'] = val
        
        samples.append(sample)
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(df)} boundary samples to {OUTPUT_CSV}")
    print(f"\nSample statistics:")
    print(df[['delta', 'alpha_total', 'kappa_def_95', 'kappa_att_95']].describe())
    
    return df


if __name__ == '__main__':
    print("="*80)
    print("BOUNDARY DATA GENERATION FOR OOD ROBUSTNESS")
    print("="*80)
    print("\nPurpose: Generate training samples for worst-case scenario (epsilon=16, lambda=16)")
    print("This addresses Out-of-Distribution generalization failure.\n")
    
    df = generate_boundary_samples()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Run actual simulations to get real kappa values")
    print("2. Merge boundary_training_data.csv with training_data_clean.csv")
    print("3. Retrain model with augmented dataset")
    print("4. Verify that predictions for epsilon=16 are now accurate")
    print("="*80)
