#!/usr/bin/env python3
"""
Critical Delta Experiment
Test critical delta values for different configurations (convergence boundary)

Experiment Setup:
- 7 shards (shard 0-6)
- All shards epsilon = 16
- Delta starts from 0.01, step 0.001, increasing until divergence
"""

import pandas as pd
import yaml
import subprocess
import os
import shutil
import numpy as np
from typing import Dict, List, Tuple

# Constants
NUM_SHARDS = 7
FIXED_EPSILON = 16.0
G_MAX = 2_000_000
TARGET_TOTAL_DEMAND = G_MAX / 2

DELTA_START = 0.001
DELTA_STEP = 0.001
MAX_DELTA = 1.0

# Delay configurations
POLKADOT_MAX_DELAY = 5
COSMOS_MAX_DELAY = 9

POLKADOT_SPIKE_WEIGHTS = [0, 0, 0, 0, 1]  # 5 weights for max_delay=5
COSMOS_SPIKE_WEIGHTS = [0, 0, 0, 0, 0, 0, 0, 0, 1]  # 9 weights for max_delay=9


def build_demand_matrix_from_shard0(shard0_inbound: float, shard0_outbound: float, mode: str = 'polkadot') -> np.ndarray:
    """
    Build demand matrix ensuring equilibrium: each shard local + outbound + inbound = 1.0 * T
    
    Matrix M[i][j]: demand from shard i to shard j
    - Row i sum (excluding diagonal) = total outbound from i
    - Column i sum (excluding diagonal) = total inbound to i
    - M[i][i] = local demand of i
    
    Args:
        shard0_inbound: shard 0 inbound ratio (relative to T)
        shard0_outbound: shard 0 outbound ratio (relative to T)
        mode: 'polkadot' or 'cosmos'
    
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
    # Key insight: shards 1-6 form a symmetric subgraph
    
    # Each shard i (i>0) has:
    # - local demand: same ratio as shard 0
    # - traffic to/from shard 0: already set (asymmetric)
    # - traffic among shards 1-6: should be symmetric
    
    other_cross_ratio = shard0_inbound + shard0_outbound
    other_local_ratio = 1.0 - other_cross_ratio
    
    for i in range(1, N):
        matrix[i, i] = other_local_ratio * T
    
    # For shards 1-6, we need to find the symmetric traffic value x such that:
    # For each shard i (i>0):
    #   local + (out_to_0 + 5*x) + (in_from_0 + 5*x) = T
    #   other_local_ratio*T + (inbound_per_shard + 5*x) + (outbound_per_shard + 5*x) = T
    #   5*x + 5*x = T - other_local_ratio*T - inbound_per_shard - outbound_per_shard
    #   10*x = T * (1 - other_local_ratio) - inbound_per_shard - outbound_per_shard
    #   10*x = T * other_cross_ratio - inbound_per_shard - outbound_per_shard
    
    symmetric_traffic = (other_cross_ratio * T - inbound_per_shard - outbound_per_shard) / 10
    
    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                matrix[i, j] = symmetric_traffic
    
    # Verify equilibrium
    for i in range(N):
        local = matrix[i, i]
        outbound = sum(matrix[i, j] for j in range(N) if j != i)
        inbound = sum(matrix[j, i] for j in range(N) if j != i)
        total = local + outbound + inbound
        if abs(total - T) > 1:
            print(f"⚠️  Shard {i} not balanced: local={local:.0f}, out={outbound:.0f}, in={inbound:.0f}, total={total:.0f}")
    
    return matrix


def demand_matrix_to_yaml(matrix: np.ndarray) -> list:
    """Convert numpy matrix to YAML format list"""
    return [[float(val) for val in row] for row in matrix]


def scale_demand_matrix(matrix: np.ndarray, epsilon: float) -> np.ndarray:
    """Scale demand matrix by epsilon (matrix is already in absolute units)"""
    return matrix * epsilon


def create_config_for_delta(delta: float, demand_matrix: np.ndarray, config_template: dict, mode: str = 'polkadot') -> dict:
    """
    Create configuration for specific delta value
    
    Args:
        delta: basefee update ratio
        demand_matrix: demand matrix (already in absolute units, TARGET_TOTAL_DEMAND per shard)
        config_template: configuration template
        mode: 'polkadot' or 'cosmos' to set appropriate delay configuration
    
    Returns:
        complete configuration dictionary
    """
    import copy
    config = copy.deepcopy(config_template)
    
    config['simulation']['delta'] = float(delta)
    
    # Matrix is already in absolute units, just convert to YAML
    config['demand']['base_demand_matrix'] = demand_matrix_to_yaml(demand_matrix)
    
    config['network']['num_shards'] = NUM_SHARDS
    
    # Set delay configuration based on mode
    if mode.lower() == 'cosmos':
        config['delay'] = {'max_delay': COSMOS_MAX_DELAY, 'weights': COSMOS_SPIKE_WEIGHTS}
        config['demand']['alpha_j_history'] = COSMOS_SPIKE_WEIGHTS
    else:  # polkadot
        config['delay'] = {'max_delay': POLKADOT_MAX_DELAY, 'weights': POLKADOT_SPIKE_WEIGHTS}
        config['demand']['alpha_j_history'] = POLKADOT_SPIKE_WEIGHTS
    
    return config


def run_simulation(config: dict) -> Tuple[bool, float]:
    """
    Run simulation and check convergence
    
    Returns:
        (converged, final_load)
    """
    # Save config to current directory
    config_path = os.path.join(os.getcwd(), 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Disable log generation
    env = os.environ.copy()
    env['DISABLE_LOG'] = 'true'
    
    try:
        # Copy config to sharded-system directory
        import shutil
        sharded_config = '../../../config.yml'
        shutil.copy(config_path, sharded_config)
        
        result = subprocess.run(
            ['go', 'run', 'main.go'],
            cwd='../../..',
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        if result.returncode != 0:
            print(f"⚠️  Simulation failed with code {result.returncode}")
            print(f"stderr: {result.stderr[:200]}")
            return False, 0.5
        
        # Move log files to current directory
        import glob
        sharded_logs = glob.glob('../../../enhanced_simulation_analysis_*.log')
        for log in sharded_logs:
            basename = os.path.basename(log)
            dest = os.path.join(os.getcwd(), basename)
            shutil.move(log, dest)
        
        converged, final_load = parse_latest_log()
        return converged, final_load
        
    except subprocess.TimeoutExpired:
        print("⚠️  Simulation timeout")
        return False, 0.5
    except Exception as e:
        print(f"⚠️  Simulation error: {e}")
        return False, 0.5


def parse_latest_log() -> Tuple[bool, float]:
    """
    Parse latest log file to check convergence (final load close to 0.5)
    
    Returns:
        (converged, final_load)
    """
    import glob
    
    log_files = glob.glob('enhanced_simulation_analysis_*.log')
    if not log_files:
        print(f"⚠️  No log files found in {os.getcwd()}")
        return False, 0.5
    
    # Get the latest log file
    latest_log = max(log_files, key=os.path.getctime)
    
    try:
        # Read the log file and extract shard 0 load from CSV data
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        
        # Parse CSV data at the end of the log
        # CSV format: step,basefee0,basefee1,basefee2,load0,load1,load2,...
        # We want load0 (4th column, index 3 if counting from 0, or index 4 in 1-based)
        loads = []
        in_csv = False
        
        for line in lines:
            line = line.strip()
            
            # Skip until we find CSV data (lines with comma-separated numbers)
            if not in_csv:
                # Check if this looks like CSV data (step number followed by floats)
                if ',' in line and not line.startswith('│'):
                    parts = line.split(',')
                    if len(parts) > 1:
                        try:
                            int(parts[0])  # First column should be step number
                            in_csv = True
                        except:
                            continue
            
            if in_csv and ',' in line:
                parts = line.split(',')
                if len(parts) >= 4:  # Need at least step + 3 values
                    try:
                        # Load for shard 0 is typically the 4th column (index 3)
                        # But CSV has: step, basefee0, basefee1, basefee2, load0, load1, load2, ...
                        # So load0 is at index 4 (assuming 3 basefees first)
                        # Let's find it: if we have N shards, format is step + N basefees + N loads
                        # For 7 shards: step, bf0-bf6 (7 cols), load0-load6 (7 cols) = 15 cols
                        # But we only see 6 data cols, so maybe 3 shards output
                        # Let's assume load0 is at index (1 + num_shards)
                        num_data_cols = len(parts) - 1  # Exclude step column
                        num_shards = num_data_cols // 2  # Half are basefees, half are loads
                        load0_index = 1 + num_shards  # Skip step + all basefees
                        
                        load0 = float(parts[load0_index])
                        loads.append(load0)
                    except:
                        continue
        if not loads:
            print("⚠️  Log parse produced no load samples")
            return False, 0.5
        
        final_load = loads[-1]
        converged = abs(final_load - 0.5) < 0.01
        
        return converged, final_load
        
    except Exception as e:
        print(f"⚠️  Error parsing log: {e}")
        return False, 0.5


def find_critical_delta(demand_matrix: np.ndarray, config_template: dict, 
                       config_name: str, mode: str = 'polkadot') -> Tuple[float, List[Tuple[float, bool, float]]]:
    """
    Find first delta value that causes divergence
    Start from DELTA_START and increment by DELTA_STEP until divergence is found
    
    Returns:
        (first_diverged_delta, history) where history is [(delta, converged, final_load), ...]
    """
    print(f"\n🔍 Finding critical delta for: {config_name}")
    
    history = []
    last_converged_delta = None
    first_diverged_delta = None
    
    delta = DELTA_START
    
    while delta <= MAX_DELTA:
        config = create_config_for_delta(delta, demand_matrix, config_template, mode)
        converged, final_load = run_simulation(config)
        history.append((delta, converged, final_load))
        
        if not converged:
            first_diverged_delta = delta
            print(f"  ✅ First divergence at δ = {delta:.3f} (load={final_load:.4f})")
            break
        
        # Only print every 10th converged result to reduce output
        if len(history) % 10 == 0:
            print(f"  δ={delta:.3f}: ✓ (tested {len(history)} values so far)")
        
        last_converged_delta = delta
        delta += DELTA_STEP
    
    if first_diverged_delta is None:
        print(f"  ⚠️  No divergence found up to δ={MAX_DELTA}")
        return MAX_DELTA, history
    
    return first_diverged_delta, history


def load_config_template() -> dict:
    """Load configuration template"""
    try:
        with open('config.yml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ config.yml not found")
        return None


def backup_config():
    """Backup original configuration"""
    try:
        shutil.copy('config.yml', 'config_backup.yml')
        print("✅ Created config backup")
        return True
    except Exception as e:
        print(f"❌ Failed to backup config: {e}")
        return False


def restore_config():
    """Restore original configuration"""
    try:
        shutil.copy('config_backup.yml', 'config.yml')
        print("✅ Restored original config")
        return True
    except Exception as e:
        print(f"❌ Failed to restore config: {e}")
        return False


def save_results(results: List[Dict], filename: str):
    """Save experiment results to CSV"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")


def save_individual_results(history: List[Tuple[float, bool, float]], config_name: str, filename: str):
    """Save detailed results for individual config to CSV"""
    try:
        data = {
            'delta': [h[0] for h in history],
            'converged': [h[1] for h in history],
            'final_load': [h[2] for h in history]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"  💾 Saved {len(history)} test results to: {filename}")
    except Exception as e:
        print(f"  ❌ Failed to save individual results: {e}")


def main():
    """Main function"""
    print("="*80)
    print("Critical Delta Experiment")
    print("="*80)
    
    if not backup_config():
        return
    
    config_template = load_config_template()
    if config_template is None:
        restore_config()
        return
    
    results = []
    
    print("\n" + "="*80)
    print("Configuration Input")
    print("="*80)
    
    # ========================================================================
    # USER INPUT - Only provide shard 0 data
    # ========================================================================
    
    polkadot_configs = [
        {'name': 'Asset Hub', 'shard0_inbound': 0.134, 'shard0_outbound': 0.14},
        {'name': 'Bifrost', 'shard0_inbound': 0.06, 'shard0_outbound': 0.126},
        {'name': 'Hydration', 'shard0_inbound': 0.133, 'shard0_outbound': 0.083},
        {'name': 'Moonbeam', 'shard0_inbound': 0.255, 'shard0_outbound': 0.178},
    ]
    
    cosmos_configs = [
        {'name': 'Cosmos Hub', 'shard0_inbound': 0.171, 'shard0_outbound': 0.074},
        {'name': 'Osmosis', 'shard0_inbound': 0.071, 'shard0_outbound': 0.048},
    ]
    
    # ========================================================================
    
    print(f"\nLoaded {len(polkadot_configs)} Polkadot configs")
    print(f"Loaded {len(cosmos_configs)} Cosmos configs (Polkadot-style, asymmetric)")
    
    for idx, config_data in enumerate(polkadot_configs, 1):
        print(f"\n{'='*80}")
        print(f"Testing Polkadot Config {idx}/{len(polkadot_configs)}: {config_data['name']}")
        print(f"{'='*80}")
        
        try:
            shard0_total = config_data['shard0_inbound'] + config_data['shard0_outbound']
            print(f"Shard 0: inbound={config_data['shard0_inbound']:.3f}, "
                  f"outbound={config_data['shard0_outbound']:.3f}, "
                  f"total={shard0_total:.3f}")
            
            if shard0_total > 0.5:
                print(f"⚠️  WARNING: Shard 0 total ratio ({shard0_total:.3f}) > 0.5, local ratio will be negative!")
                continue
            
            demand_matrix = build_demand_matrix_from_shard0(
                config_data['shard0_inbound'],
                config_data['shard0_outbound'],
                mode='polkadot'
            )
            
            critical_delta, history = find_critical_delta(
                demand_matrix, 
                config_template, 
                config_data['name'],
                mode='polkadot'
            )
            
            # Save individual results for this config
            config_filename = config_data['name'].replace(' ', '_').lower()
            save_individual_results(history, config_data['name'], f'results_{config_filename}.csv')
            
            results.append({
                'config_name': config_data['name'],
                'mode': 'Polkadot',
                'critical_delta': critical_delta,
                'shard0_inbound': config_data['shard0_inbound'],
                'shard0_outbound': config_data['shard0_outbound'],
                'shard0_total': shard0_total,
                'num_tests': len(history),
            })
            
        except Exception as e:
            print(f"❌ Error testing {config_data['name']}: {e}")
            continue
    
    for idx, config_data in enumerate(cosmos_configs, 1):
        print(f"\n{'='*80}")
        print(f"Testing Cosmos Config {idx}/{len(cosmos_configs)} (Polkadot-style): {config_data['name']}")
        print(f"{'='*80}")
        
        try:
            shard0_total = config_data['shard0_inbound'] + config_data['shard0_outbound']
            print(f"Shard 0: inbound={config_data['shard0_inbound']:.3f}, "
                  f"outbound={config_data['shard0_outbound']:.3f}, "
                  f"total={shard0_total:.3f}")
            
            if shard0_total > 0.5:
                print(f"⚠️  WARNING: Shard 0 total ratio ({shard0_total:.3f}) > 0.5, local ratio will be negative!")
                continue
            
            demand_matrix = build_demand_matrix_from_shard0(
                config_data['shard0_inbound'],
                config_data['shard0_outbound'],
                mode='cosmos'
            )
            
            critical_delta, history = find_critical_delta(
                demand_matrix,
                config_template,
                config_data['name'],
                mode='cosmos'
            )
            
            # Save individual results for this config
            config_filename = config_data['name'].replace(' ', '_').lower()
            save_individual_results(history, config_data['name'], f'results_{config_filename}.csv')
            
            results.append({
                'config_name': config_data['name'],
                'mode': 'Cosmos (Polkadot-style)',
                'critical_delta': critical_delta,
                'shard0_inbound': config_data['shard0_inbound'],
                'shard0_outbound': config_data['shard0_outbound'],
                'shard0_total': shard0_total,
                'num_tests': len(history),
            })
            
        except Exception as e:
            print(f"❌ Error testing {config_data['name']}: {e}")
            continue
    
    if results:
        save_results(results, 'critical_delta_results.csv')
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        for r in results:
            print(f"{r['config_name']:30s} ({r['mode']:8s}): δ_critical = {r['critical_delta']:.4f}")
    else:
        print("\n⚠️  No results to save")
    
    restore_config()


if __name__ == '__main__':
    main()
