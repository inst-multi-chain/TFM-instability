#!/usr/bin/env python3
"""
Delta Max Validation Experiment

Compare empirical δ_max (from simulation) vs calculated δ_max (from NN prediction)
for randomly generated configurations.

This validates the NN model's ability to predict stability boundaries
for general (not just specific) configurations.

Output: CSV with columns [config_id, empirical_delta_max, calculated_delta_max, ...]
        and a scatter plot showing correlation.
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import subprocess
import shutil
import time
import glob
from typing import Dict, List, Tuple, Optional

# Add paths
sys.path.append('../../../nn-model')
sys.path.insert(0, '../critical-delta-experiment')

from predict_with_physics_features import KappaPredictorWithPhysics

# ===================== CONSTANTS =====================

NUM_SHARDS = 7
G_MAX = 2_000_000
TARGET_T = G_MAX / 2
L_TARGET = 0.5

# Delta search parameters
DELTA_START = 0.0001
DELTA_STEP = 0.0001  # Finer step for more precise search
MAX_DELTA = 1.0

# Random config parameters
EPSILON_MIN = 0.0
EPSILON_MAX = 16.0
ALPHA_MIN = 0.01
ALPHA_MAX = 0.99
D_MAX_MIN = 3
D_MAX_MAX = 10

# Simulation parameters
TOTAL_STEPS = 2000


# ===================== CONFIG GENERATION (from data_generator.py) =====================

def build_7shard_self_consistent_matrix(alpha: float) -> Tuple[List[List[float]], float, float]:
    """Build 7x7 self-consistent demand matrix with inbound 30-40%"""
    alpha = max(0.0, min(1.0, float(alpha)))
    
    TARGET_T_INT = int(TARGET_T)
    matrix = [[0 for _ in range(NUM_SHARDS)] for _ in range(NUM_SHARDS)]
    
    # Inbound ratio: 30-40%, rest is outbound
    inbound_ratio = np.random.uniform(0.2, 0.99)
    cross_total_0 = int(alpha * TARGET_T_INT)
    
    inbound_0 = int(inbound_ratio * cross_total_0)
    in_base = inbound_0 // (NUM_SHARDS - 1)
    in_remainder = inbound_0 % (NUM_SHARDS - 1)
    
    for i in range(1, NUM_SHARDS):
        matrix[i][0] = in_base + (1 if i - 1 < in_remainder else 0)
    
    outbound_0 = cross_total_0 - inbound_0
    out_base = outbound_0 // (NUM_SHARDS - 1)
    out_remainder = outbound_0 % (NUM_SHARDS - 1)
    
    for j in range(1, NUM_SHARDS):
        matrix[0][j] = out_base + (1 if j - 1 < out_remainder else 0)
    
    matrix[0][0] = TARGET_T_INT - sum(matrix[0][j] for j in range(1, NUM_SHARDS)) - sum(matrix[i][0] for i in range(1, NUM_SHARDS))
    
    diagonal_value = matrix[0][0]
    for i in range(1, NUM_SHARDS):
        matrix[i][i] = diagonal_value
    
    sample_shard = 1
    already_allocated = matrix[sample_shard][sample_shard] + matrix[sample_shard][0] + matrix[0][sample_shard]
    remaining = TARGET_T_INT - already_allocated
    
    num_peers = NUM_SHARDS - 2
    half_remaining = remaining // 2
    flow_per_peer = half_remaining // num_peers
    
    for i in range(1, NUM_SHARDS):
        for j in range(i + 1, NUM_SHARDS):
            matrix[i][j] = flow_per_peer
            matrix[j][i] = flow_per_peer
    
    for i in range(1, NUM_SHARDS):
        outbound = sum(matrix[i][j] for j in range(NUM_SHARDS) if j != i)
        inbound = sum(matrix[j][i] for j in range(NUM_SHARDS) if j != i)
        matrix[i][i] = TARGET_T_INT - outbound - inbound
    
    outbound_0 = sum(matrix[0][j] for j in range(NUM_SHARDS) if j != 0)
    inbound_0 = sum(matrix[j][0] for j in range(NUM_SHARDS) if j != 0)
    matrix[0][0] = TARGET_T_INT - outbound_0 - inbound_0
    
    matrix_float = [[float(matrix[i][j]) for j in range(NUM_SHARDS)] for i in range(NUM_SHARDS)]
    
    actual_outbound_0 = sum(matrix[0][j] for j in range(1, NUM_SHARDS))
    actual_inbound_0 = sum(matrix[i][0] for i in range(1, NUM_SHARDS))
    
    alpha_outbound = float(actual_outbound_0) / TARGET_T_INT
    alpha_inbound = float(actual_inbound_0) / TARGET_T_INT
    
    return matrix_float, alpha_inbound, alpha_outbound


def verify_equilibrium(matrix: List[List[float]], tolerance: float = 1.0) -> bool:
    """Verify demand matrix equilibrium"""
    for i in range(NUM_SHARDS):
        local = matrix[i][i]
        outbound = sum(matrix[i][j] for j in range(NUM_SHARDS) if j != i)
        inbound = sum(matrix[j][i] for j in range(NUM_SHARDS) if j != i)
        total = local + outbound + inbound
        
        if abs(total - TARGET_T) > tolerance or local < 0:
            return False
    return True


def generate_random_config() -> Optional[Dict]:
    """Generate random configuration for delta_max validation"""
    max_attempts = 100
    
    for attempt in range(max_attempts):
        try:
            alpha_total = np.random.uniform(ALPHA_MIN, ALPHA_MAX)
            demand_matrix, alpha_inbound, alpha_outbound = build_7shard_self_consistent_matrix(alpha_total)
            
            if not verify_equilibrium(demand_matrix):
                continue
            
            # Fixed elasticity matrices: all 16 to avoid OOD
            epsilon_matrix = np.random.uniform(EPSILON_MIN, EPSILON_MAX, (NUM_SHARDS, NUM_SHARDS))
            lambda_matrix = np.random.uniform(EPSILON_MIN, EPSILON_MAX, (NUM_SHARDS, NUM_SHARDS))
            np.fill_diagonal(epsilon_matrix, 0.0)
            
            # Random d_max
            d_max = np.random.randint(D_MAX_MIN, D_MAX_MAX + 1)
            delay_weights = [0.0] * d_max
            delay_weights[-1] = 1.0  # Spike distribution
            
            return {
                'alpha_total': float(alpha_total),
                'alpha_inbound': float(alpha_inbound),
                'alpha_outbound': float(alpha_outbound),
                'demand_matrix': demand_matrix,
                'epsilon_matrix': epsilon_matrix.tolist(),
                'lambda_matrix': lambda_matrix.tolist(),
                'd_max': int(d_max),
                'delay_weights': delay_weights
            }
        except:
            continue
    
    return None


# ===================== SIMULATION (from critical_delta_experiment.py) =====================

def create_config_yml(config: Dict, delta: float, output_path: str = 'config.yml'):
    """Create config.yml for simulation"""
    shards = [{'id': i, 'g_max': G_MAX} for i in range(NUM_SHARDS)]
    
    yml_config = {
        'simulation': {
            'total_steps': TOTAL_STEPS,
            'l_target': L_TARGET,
            'delta': delta,
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
        'shards': shards,
        'demand': {
            'base_demand_matrix': config['demand_matrix'],
            'epsilon_matrix': config['epsilon_matrix'],
            'lambda_matrix': config['lambda_matrix']
        },
        'delay': {
            'max_delay': config['d_max'],
            'weights': config['delay_weights']
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(yml_config, f, default_flow_style=False)


def run_simulation() -> Tuple[bool, float]:
    """Run Go simulation and check convergence"""
    env = os.environ.copy()
    env['DISABLE_LOG'] = 'true'
    
    try:
        shutil.copy('config.yml', '../../../config.yml')
        
        result = subprocess.run(
            ['go', 'run', 'main.go'],
            cwd='../../..',
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        if result.returncode != 0:
            return False, 0.5
        
        # Move and parse log
        log_files = glob.glob('../../../enhanced_simulation_analysis_*.log')
        if not log_files:
            return False, 0.5
        
        latest_log = max(log_files, key=os.path.getctime)
        shutil.move(latest_log, './temp_log.log')
        
        converged, final_load = parse_log('./temp_log.log')
        
        try:
            os.remove('./temp_log.log')
        except:
            pass
        
        return converged, final_load
        
    except subprocess.TimeoutExpired:
        return False, 0.5
    except Exception as e:
        return False, 0.5


def parse_log(log_path: str) -> Tuple[bool, float]:
    """Parse log file to check convergence"""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        loads = []
        in_csv = False
        
        for line in lines:
            line = line.strip()
            if ',' in line and not line.startswith('│'):
                parts = line.split(',')
                if len(parts) >= 8:
                    try:
                        step = int(parts[0])
                        in_csv = True
                        num_data_cols = len(parts) - 1
                        num_shards = num_data_cols // 2
                        load0_index = 1 + num_shards
                        load0 = float(parts[load0_index])
                        loads.append(load0)
                    except:
                        continue
        
        if len(loads) < 100:
            return False, 0.5
        
        final_load = loads[-1]
        converged = abs(final_load - 0.5) < 0.01
        
        return converged, final_load
        
    except Exception as e:
        return False, 0.5


def find_empirical_delta_max(config: Dict) -> float:
    """Find empirical delta_max through simulation (binary search for speed)"""
    low = DELTA_START
    high = MAX_DELTA
    last_converged = low
    
    # First, check if even the minimum delta converges
    create_config_yml(config, low)
    converged, _ = run_simulation()
    if not converged:
        return low
    
    # Binary search for critical delta
    while high - low > 0.005:  # 0.5% precision
        mid = (low + high) / 2
        create_config_yml(config, mid)
        converged, _ = run_simulation()
        
        if converged:
            low = mid
            last_converged = mid
        else:
            high = mid
    
    return last_converged


# ===================== NN PREDICTION (from predict_max_delta_nn.py) =====================

def compute_gi_ri(delta: float, demand_matrix: np.ndarray, epsilon_matrix: np.ndarray,
                  lambda_matrix: np.ndarray, kappa_def: float, kappa_att: float) -> Tuple[float, float]:
    """Compute Gi and Ri for stability check"""
    T = TARGET_T
    
    alpha_0_local = demand_matrix[0, 0] / T
    alpha_j_to_0_in = sum(demand_matrix[i, 0] for i in range(1, NUM_SHARDS)) / T
    
    lambda_00 = lambda_matrix[0, 0]
    epsilon_j0 = np.mean([epsilon_matrix[i, 0] for i in range(1, NUM_SHARDS)])
    lambda_j0 = np.mean([lambda_matrix[i, 0] for i in range(1, NUM_SHARDS)])
    
    Lambda_0_local = lambda_00 * alpha_0_local
    Lambda_0_self_delay = epsilon_j0 * alpha_j_to_0_in
    Lambda_j0_delay = lambda_j0 * alpha_j_to_0_in
    
    defense_total = Lambda_0_local + kappa_def * Lambda_0_self_delay
    attack_total = kappa_att * Lambda_j0_delay
    
    Gi = delta * defense_total
    Ri = attack_total / defense_total if defense_total > 0 else float('inf')
    
    return Gi, Ri


def check_stability(Gi: float, Ri: float) -> bool:
    """Check stability using phase map condition"""
    return (Gi * (1 + Ri) < 2) and (Ri < 1)


def find_calculated_delta_max(config: Dict, predictor: KappaPredictorWithPhysics) -> float:
    """Find calculated delta_max using NN prediction"""
    demand_matrix = np.array(config['demand_matrix'])
    epsilon_matrix = np.array(config['epsilon_matrix'])
    lambda_matrix = np.array(config['lambda_matrix'])
    
    alpha_total = config['alpha_total']
    alpha_inbound = config['alpha_inbound']
    alpha_outbound = config['alpha_outbound']
    d_max = config['d_max']
    
    last_stable_delta = DELTA_START
    delta = DELTA_START
    
    while delta <= MAX_DELTA:
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
            
            Gi, Ri = compute_gi_ri(delta, demand_matrix, epsilon_matrix, 
                                   lambda_matrix, kappa_def, kappa_att)
            
            stable = check_stability(Gi, Ri)
            
            if stable:
                last_stable_delta = delta
            else:
                break
                
        except Exception as e:
            break
        
        delta += DELTA_STEP
    
    return 0.5*last_stable_delta


# ===================== MAIN EXPERIMENT =====================

def run_experiment(num_samples: int = 1000, output_csv: str = 'delta_max_validation.csv',
                   checkpoint_interval: int = 50):
    """Run delta_max validation experiment"""
    print("=" * 80)
    print("DELTA MAX VALIDATION EXPERIMENT")
    print("=" * 80)
    print(f"Samples: {num_samples}")
    print(f"Output: {output_csv}")
    print("=" * 80)
    
    # Load NN predictor
    print("\n📦 Loading NN predictor...")
    predictor = KappaPredictorWithPhysics(
        model_dir='/Users/epsilon/code/go/shard-TFM-instability/sharded-system/nn-model/models_weighted_v2'
    )
    print("✅ NN predictor loaded")
    
    results = []
    failed_count = 0
    start_time = time.time()
    
    for i in range(num_samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'='*60}")
        
        # Generate random config
        config = generate_random_config()
        if config is None:
            print("❌ Failed to generate config")
            failed_count += 1
            continue
        
        print(f"✅ Config: α_total={config['alpha_total']:.3f}, "
              f"α_in={config['alpha_inbound']:.3f}, α_out={config['alpha_outbound']:.3f}, "
              f"d_max={config['d_max']}")
        
        # Find empirical delta_max (simulation)
        print("🔬 Finding empirical δ_max (simulation)...")
        t1 = time.time()
        empirical_delta_max = find_empirical_delta_max(config)
        t_empirical = time.time() - t1
        print(f"✅ Empirical δ_max = {empirical_delta_max:.4f} ({t_empirical:.1f}s)")
        
        # Find calculated delta_max (NN)
        print("🧠 Finding calculated δ_max (NN)...")
        t2 = time.time()
        calculated_delta_max = find_calculated_delta_max(config, predictor)
        t_calculated = time.time() - t2
        print(f"✅ Calculated δ_max = {calculated_delta_max:.4f} ({t_calculated:.1f}s)")
        
        # Compute error
        error = calculated_delta_max - empirical_delta_max
        rel_error = error / empirical_delta_max if empirical_delta_max > 0 else 0
        
        print(f"📊 Error: {error:.4f} ({rel_error*100:.1f}%)")
        
        # Store result
        result = {
            'config_id': i,
            'alpha_total': config['alpha_total'],
            'alpha_inbound': config['alpha_inbound'],
            'alpha_outbound': config['alpha_outbound'],
            'd_max': config['d_max'],
            'avg_epsilon': np.mean(config['epsilon_matrix']),
            'avg_lambda': np.mean(config['lambda_matrix']),
            'empirical_delta_max': empirical_delta_max,
            'calculated_delta_max': calculated_delta_max,
            'error': error,
            'rel_error': rel_error
        }
        results.append(result)
        
        # Checkpoint
        if len(results) % checkpoint_interval == 0:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / len(results)
            eta = avg_time * (num_samples - len(results))
            
            print(f"\n💾 Checkpoint: {len(results)}/{num_samples} saved")
            print(f"   Avg time: {avg_time:.1f}s/sample | ETA: {eta/60:.1f}min")
    
    # Final save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"✅ Completed: {len(results)} samples")
    print(f"❌ Failed: {failed_count}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"💾 Output: {output_csv}")
    
    if len(results) > 0:
        print(f"\n📊 Statistics:")
        print(f"   Mean error: {df['error'].mean():.4f}")
        print(f"   Std error: {df['error'].std():.4f}")
        print(f"   Mean |rel_error|: {df['rel_error'].abs().mean()*100:.1f}%")
        print(f"   Correlation: {df['empirical_delta_max'].corr(df['calculated_delta_max']):.4f}")
    
    print("=" * 80)
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Delta Max Validation Experiment')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples (default: 1000)')
    parser.add_argument('--output', type=str, default='delta_max_validation.csv',
                       help='Output CSV file')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Checkpoint interval')
    
    args = parser.parse_args()
    
    run_experiment(
        num_samples=args.num_samples,
        output_csv=args.output,
        checkpoint_interval=args.checkpoint_interval
    )
