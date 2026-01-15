#!/usr/bin/env python3
"""
Predict Max Delta using Neural Network
Combines configuration ratios from critical_delta_experiment.py with config.yml elasticity data

Experiment Setup:
- 6 configurations from critical_delta_experiment.py (inbound/outbound ratios)
- Epsilon/Lambda matrices and d_max from config.yml
- Build full demand matrix for each config (like critical_delta_experiment does)
- Iterate delta values, use NN to predict kappa, calculate Gi/Ri
- Find first delta where |1 - Gi| + Ri >= 1 (stability boundary)
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Add nn-model to path
sys.path.insert(0, '../../../nn-model')

from predict_with_physics_features import KappaPredictorWithPhysics

# Constants
NUM_SHARDS = 7
G_MAX = 2_000_000
TARGET_TOTAL_DEMAND = G_MAX / 2

DELTA_START = 0.001
DELTA_STEP = 0.001
MAX_DELTA = 1.0


def load_config() -> dict:
    """Load configuration from config.yml"""
    try:
        with open('config.yml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load config.yml: {e}")
        return None


def build_demand_matrix_from_shard0(shard0_inbound: float, shard0_outbound: float) -> np.ndarray:
    """
    Build demand matrix ensuring equilibrium (copied from critical_delta_experiment.py)
    
    Args:
        shard0_inbound: shard 0 inbound ratio (relative to T)
        shard0_outbound: shard 0 outbound ratio (relative to T)
    
    Returns:
        7x7 demand matrix (absolute values)
    """
    N = NUM_SHARDS
    T = TARGET_TOTAL_DEMAND
    
    matrix = np.zeros((N, N))
    
    # Shard 0
    shard0_local = 1.0 - shard0_inbound - shard0_outbound
    assert shard0_local >= 0, f"Shard 0 local negative: {shard0_local}"
    
    matrix[0, 0] = shard0_local * T
    
    # Shard 0 outbound: evenly distribute to other shards
    outbound_per_shard = (shard0_outbound * T) / (N - 1)
    for j in range(1, N):
        matrix[0, j] = outbound_per_shard
    
    # Shard 0 inbound: comes from other shards
    inbound_per_shard = (shard0_inbound * T) / (N - 1)
    for i in range(1, N):
        matrix[i, 0] = inbound_per_shard
    
    # Other shards (1-6): make them symmetric among themselves
    other_cross_ratio = shard0_inbound + shard0_outbound
    other_local_ratio = 1.0 - other_cross_ratio
    
    for i in range(1, N):
        matrix[i, i] = other_local_ratio * T
    
    # Symmetric traffic value
    symmetric_traffic = (other_cross_ratio * T - inbound_per_shard - outbound_per_shard) / 10
    
    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                matrix[i, j] = symmetric_traffic
    
    return matrix


def compute_gi_ri(delta: float, demand_matrix: np.ndarray, epsilon_matrix: np.ndarray,
                  lambda_matrix: np.ndarray, kappa_def: float, kappa_att: float) -> Tuple[float, float, dict]:
    """
    Compute Gi and Ri for shard 0 using phase map formulation
    
    Formula (from Figure7/prediction.py):
        defense_total = Lambda_0_local + kappa_def * Lambda_0_self_delay
        attack_total = kappa_att * Lambda_j0_delay
        Gi = delta * defense_total
        Ri = attack_total / defense_total
    
    Returns:
        (Gi, Ri, details_dict)
    """
    T = TARGET_TOTAL_DEMAND
    
    # Calculate alpha ratios for shard 0
    alpha_0_local = demand_matrix[0, 0] / T
    alpha_j_to_0_in = sum(demand_matrix[i, 0] for i in range(1, NUM_SHARDS)) / T  # Total inbound
    alpha_0_to_j_out = sum(demand_matrix[0, j] for j in range(1, NUM_SHARDS)) / T  # Total outbound
    
    # Average elasticities (use first element as they're all the same)
    lambda_00 = lambda_matrix[0, 0]  # Local elasticity
    epsilon_j0 = np.mean([epsilon_matrix[i, 0] for i in range(1, NUM_SHARDS)])  # Inbound elasticity
    lambda_j0 = np.mean([lambda_matrix[i, 0] for i in range(1, NUM_SHARDS)])  # Delayed attack elasticity
    
    # Calculate forces
    Lambda_0_local = lambda_00 * alpha_0_local
    Lambda_0_self_delay = epsilon_j0 * alpha_j_to_0_in  # Self-delayed demand from inbound
    Lambda_j0_delay = lambda_j0 * alpha_j_to_0_in  # Attack from delayed inbound
    
    # Calculate Gi and Ri
    defense_total = Lambda_0_local + kappa_def * Lambda_0_self_delay
    attack_total = kappa_att * Lambda_j0_delay
    
    Gi = delta * defense_total
    Ri = attack_total / defense_total if defense_total > 0 else float('inf')
    
    details = {
        'alpha_0_local': alpha_0_local,
        'alpha_j_to_0_in': alpha_j_to_0_in,
        'alpha_0_to_j_out': alpha_0_to_j_out,
        'Lambda_0_local': Lambda_0_local,
        'Lambda_0_self_delay': Lambda_0_self_delay,
        'Lambda_j0_delay': Lambda_j0_delay,
        'defense_total': defense_total,
        'attack_total': attack_total,
        'kappa_def': kappa_def,
        'kappa_att': kappa_att
    }
    
    return Gi, Ri, details


def check_stability(Gi: float, Ri: float) -> bool:
    """
    Check stability using phase map condition: |1 - Gi| + Ri < 1
    
    Returns:
        True if stable, False if unstable
    """
    return (Gi * (1 + Ri) < 2) and (Ri < 1)


def find_critical_delta_for_config(config_name: str, alpha_inbound: float, 
                                   alpha_outbound: float, config_data: dict, d_max: int = 5) -> Tuple[float, List[Dict]]:
    """
    Find critical delta for given configuration
    
    Args:
        config_name: name of configuration
        alpha_inbound: inbound ratio from critical_delta_experiment.py
        alpha_outbound: outbound ratio from critical_delta_experiment.py
        config_data: config.yml data (for epsilon/lambda)
        d_max: max delay (5 for Polkadot, 9 for Cosmos)
    
    Returns:
        (critical_delta, results_list)
    """
    # Build demand matrix from ratios
    demand_matrix = build_demand_matrix_from_shard0(alpha_inbound, alpha_outbound)
    
    # Extract data from config.yml
    epsilon_matrix = np.array(config_data['demand']['epsilon_matrix'], dtype=float)
    lambda_matrix = np.array(config_data['demand']['lambda_matrix'], dtype=float)
    
    # Calculate alpha values for NN input
    T = TARGET_TOTAL_DEMAND
    alpha_total = alpha_inbound + alpha_outbound
    
    # Load predictor
    predictor = KappaPredictorWithPhysics(model_dir='/Users/epsilon/code/go/shard-TFM-instability/sharded-system/nn-model/models_weighted_v2')
    
    results = []
    critical_delta = None
    
    delta = DELTA_START
    while delta <= MAX_DELTA:
        # Predict kappa values using NN
        try:
            kappa_def, kappa_att = predictor.predict(
                delta=delta,
                alpha_total=alpha_total,
                alpha_inbound=alpha_inbound,
                alpha_outbound=alpha_outbound,
                d_max=d_max,
                epsilon_matrix=epsilon_matrix,
                lambda_matrix=lambda_matrix
            )
        except Exception as e:
            print(f"⚠️  Prediction error at delta={delta}: {e}")
            break
        
        # Calculate Gi and Ri using demand matrix
        Gi, Ri, details = compute_gi_ri(delta, demand_matrix, epsilon_matrix, 
                                        lambda_matrix, kappa_def, kappa_att)
        
        # Check stability using phase map condition
        stable = check_stability(Gi, Ri)
        
        result = {
            'config': config_name,
            'delta': round(delta, 4),
            'alpha_inbound': round(alpha_inbound, 4),
            'alpha_outbound': round(alpha_outbound, 4),
            'kappa_def': round(kappa_def, 4),
            'kappa_att': round(kappa_att, 4),
            'Gi': round(Gi, 4),
            'Ri': round(Ri, 4),
            'stable': stable
        }
        results.append(result)
        
        print(f"  δ={delta:.4f} | κ_def={kappa_def:.4f} κ_att={kappa_att:.4f} | "
              f"Gi={Gi:.4f} Ri={Ri:.4f} | stable={stable}")
        
        # Check if this is the first unstable delta
        if not stable and critical_delta is None:
            critical_delta = delta
            print(f"  ⚠️  CRITICAL DELTA FOUND: {critical_delta:.4f}")
            break
        
        delta += DELTA_STEP
    
    if critical_delta is None:
        critical_delta = MAX_DELTA
        print(f"  ❌ No critical delta found up to {MAX_DELTA}")
    
    return critical_delta, results


def main():
    """
    Run prediction for 6 configurations using ratios from critical_delta_experiment.py
    and elasticity data from config.yml
    """
    print("=" * 80)
    print("PREDICT MAX DELTA USING NEURAL NETWORK")
    print("=" * 80)
    
    # Load config.yml for epsilon/lambda/d_max
    config = load_config()
    if config is None:
        return
    
    print(f"\n✅ Loaded config.yml")
    print(f"  Epsilon values: {config['demand']['epsilon_matrix'][0][0]}")
    print(f"  Lambda values: {config['demand']['lambda_matrix'][0][0]}")
    print(f"  d_max: {config['delay']['max_delay']}")
    
    # 6 configurations from critical-delta-experiment
    # Format: (name, alpha_inbound, alpha_outbound)
    polkadot_configs = [
        ('Asset Hub', 0.134, 0.14),
        ('Bifrost', 0.06, 0.126),
        ('Hydration', 0.133, 0.083),
        ('Moonbeam', 0.255, 0.178),
    ]
    
    cosmos_configs = [
        ('Cosmos Hub', 0.171, 0.074),
        ('Osmosis', 0.071, 0.048),
    ]
    
    configs = polkadot_configs + cosmos_configs
    
    all_results = []
    critical_deltas = {}
    
    for config_name, alpha_inbound, alpha_outbound in configs:
        print(f"\n{'=' * 80}")
        print(f"Config: {config_name}")
        print(f"  α_inbound: {alpha_inbound:.3f}, α_outbound: {alpha_outbound:.3f}")
        print(f"{'=' * 80}")
        
        critical_delta, results = find_critical_delta_for_config(
            config_name, alpha_inbound, alpha_outbound, config
        )
        
        critical_deltas[config_name] = critical_delta
        all_results.extend(results)
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('nn_predicted_critical_deltas.csv', index=False)
    print(f"\n✅ Results saved to nn_predicted_critical_deltas.csv")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY: Critical Delta for Each Configuration")
    print(f"{'=' * 80}")
    for config_name, critical_delta in critical_deltas.items():
        print(f"{config_name:30s}: δ_crit = {critical_delta:.4f}")
    
    # Overall critical delta (minimum across all configs)
    min_critical_delta = min(critical_deltas.values())
    print(f"\n{'=' * 80}")
    print(f"OVERALL CRITICAL DELTA (Most Restrictive): {min_critical_delta:.4f}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
