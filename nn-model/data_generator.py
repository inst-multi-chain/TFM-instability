#!/usr/bin/env python3
"""
Neural Network Training Data Generator

Requirements:
1. 7 shards with uniform cross-shard ratio
2. Shard 0 anchor: inbound + outbound = alpha_total (randomly split)
3. All elasticities random [0, 16]
4. Spike latency, d_max random [3, 10]
5. Verify equilibrium for all configs
"""

import numpy as np
import yaml
import subprocess
import os
import glob
import time
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ===================== CONSTANTS =====================

NUM_SHARDS = 7
G_MAX = 2000000
L_TARGET = 0.5
TARGET_T = G_MAX * L_TARGET
EQUIL_P = 1.0

TOTAL_STEPS = 2000
START_STEP = 100  
END_STEP = 1999

EPSILON_MIN = 0.0
EPSILON_MAX = 16.0
ALPHA_MIN = 0.01
ALPHA_MAX = 0.99
DELTA_MIN = 0.01
DELTA_MAX = 0.50
D_MAX_MIN = 3
D_MAX_MAX = 10

# ===================== DEMAND MATRIX =====================

def build_7shard_self_consistent_matrix(alpha: float) -> Tuple[List[List[float]], float, float]:
    """
    Build 7x7 self-consistent demand matrix (based on Figure4)
    
    All values are integers to avoid floating-point accumulation errors.
    
    Returns:
        (demand_matrix, alpha_inbound, alpha_outbound)
    
    Equilibrium constraint for each shard i:
        matrix[i,i] + sum(matrix[i,j] for j≠i) + sum(matrix[j,i] for j≠i) = TARGET_T
        = local_i + outbound_from_i + inbound_to_i = TARGET_T
    
    Strategy (similar to Figure4 3-shard case):
    - Shard 0 (anchor): alpha = (outbound_0 + inbound_0) / TARGET_T
      * Randomly split: outbound_0 and inbound_0
      * Local: matrix[0,0] = TARGET_T - outbound_0 - inbound_0
    - Other shards (1-6): symmetric construction
      * All diagonal elements equal to matrix[0,0]
      * Cross-flows distributed to maintain equilibrium
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    
    TARGET_T_INT = int(TARGET_T)
    matrix = [[0 for _ in range(NUM_SHARDS)] for _ in range(NUM_SHARDS)]
    
    # Random split for shard 0
    outbound_ratio = np.random.uniform(0, 1)
    
    # Total cross-shard demand for shard 0 (integer)
    cross_total_0 = int(alpha * TARGET_T_INT)
    
    # Outbound from shard 0 (distributed evenly to other shards as integers)
    outbound_0 = int(outbound_ratio * cross_total_0)
    out_base = outbound_0 // (NUM_SHARDS - 1)
    out_remainder = outbound_0 % (NUM_SHARDS - 1)
    
    for j in range(1, NUM_SHARDS):
        matrix[0][j] = out_base + (1 if j - 1 < out_remainder else 0)
    
    # Inbound to shard 0 (received evenly from other shards as integers)
    inbound_0 = cross_total_0 - outbound_0
    in_base = inbound_0 // (NUM_SHARDS - 1)
    in_remainder = inbound_0 % (NUM_SHARDS - 1)
    
    for i in range(1, NUM_SHARDS):
        matrix[i][0] = in_base + (1 if i - 1 < in_remainder else 0)
    
    # Local demand for shard 0
    matrix[0][0] = TARGET_T_INT - sum(matrix[0][j] for j in range(1, NUM_SHARDS)) - sum(matrix[i][0] for i in range(1, NUM_SHARDS))
    
    # All other diagonal elements equal to matrix[0,0]
    diagonal_value = matrix[0][0]
    for i in range(1, NUM_SHARDS):
        matrix[i][i] = diagonal_value
    
    # For shards 1-6: Use symmetric matrix, discard remainders, absorb error in local
    # All shards 1-6 have same remaining budget
    sample_shard = 1
    already_allocated = matrix[sample_shard][sample_shard] + matrix[sample_shard][0] + matrix[0][sample_shard]
    remaining = TARGET_T_INT - already_allocated
    
    # For symmetric matrix: outbound = inbound
    # So: outbound + inbound = remaining
    # outbound = inbound = remaining / 2
    
    num_peers = NUM_SHARDS - 2  # 5 peers for each shard in {1..6}
    
    # Divide by 2 first, DISCARD remainder
    half_remaining = remaining // 2
    
    # Divide among peers, DISCARD remainder  
    flow_per_peer = half_remaining // num_peers
    
    # Fill symmetric flows between shards 1-6
    for i in range(1, NUM_SHARDS):
        for j in range(i + 1, NUM_SHARDS):
            matrix[i][j] = flow_per_peer
            matrix[j][i] = flow_per_peer
    
    # Recalculate local for each shard to absorb all rounding errors
    for i in range(1, NUM_SHARDS):
        outbound = sum(matrix[i][j] for j in range(NUM_SHARDS) if j != i)
        inbound = sum(matrix[j][i] for j in range(NUM_SHARDS) if j != i)
        # Adjust local to make total = TARGET_T
        matrix[i][i] = TARGET_T_INT - outbound - inbound
    
    # Also recalculate shard 0's local (in case flows to/from it changed)
    outbound_0 = sum(matrix[0][j] for j in range(NUM_SHARDS) if j != 0)
    inbound_0 = sum(matrix[j][0] for j in range(NUM_SHARDS) if j != 0)
    matrix[0][0] = TARGET_T_INT - outbound_0 - inbound_0
    
    # Convert to float
    matrix_float = [[float(matrix[i][j]) for j in range(NUM_SHARDS)] for i in range(NUM_SHARDS)]
    
    # Calculate actual alpha_inbound and alpha_outbound as fractions
    actual_outbound_0 = sum(matrix[0][j] for j in range(1, NUM_SHARDS))
    actual_inbound_0 = sum(matrix[i][0] for i in range(1, NUM_SHARDS))
    
    alpha_outbound = float(actual_outbound_0) / TARGET_T_INT
    alpha_inbound = float(actual_inbound_0) / TARGET_T_INT
    
    return matrix_float, alpha_inbound, alpha_outbound


def verify_equilibrium_7shard(matrix: List[List[float]], tolerance: float = 1.0) -> Tuple[bool, str]:
    """Verify 7-shard demand matrix satisfies equilibrium"""
    for i in range(NUM_SHARDS):
        local = matrix[i][i]
        outbound = sum(matrix[i][j] for j in range(NUM_SHARDS) if j != i)
        inbound = sum(matrix[j][i] for j in range(NUM_SHARDS) if j != i)
        total = local + outbound + inbound
        
        if abs(total - TARGET_T) > tolerance:
            return False, f"Shard {i}: total={total:.0f}, expected={TARGET_T:.0f}"
        
        if local < 0:
            return False, f"Shard {i} has negative local demand"
    
    return True, "OK"


# ===================== CONFIG GENERATION =====================

def generate_random_config() -> Optional[Dict]:
    """Generate random configuration ensuring equilibrium"""
    max_attempts = 100
    
    for attempt in range(max_attempts):
        try:
            delta = np.random.uniform(DELTA_MIN, DELTA_MAX)
            alpha_total = np.random.uniform(ALPHA_MIN, ALPHA_MAX)
            
            demand_matrix, alpha_inbound, alpha_outbound = build_7shard_self_consistent_matrix(alpha_total)
            
            is_valid, msg = verify_equilibrium_7shard(demand_matrix)
            if not is_valid:
                continue
            
            epsilon_matrix = np.random.uniform(EPSILON_MIN, EPSILON_MAX, (NUM_SHARDS, NUM_SHARDS))
            lambda_matrix = np.random.uniform(EPSILON_MIN, EPSILON_MAX, (NUM_SHARDS, NUM_SHARDS))
            np.fill_diagonal(epsilon_matrix, 0.0)
            
            d_max = np.random.randint(D_MAX_MIN, D_MAX_MAX + 1)
            delay_weights = [0.0] * d_max
            delay_weights[-1] = 1.0
            
            return {
                'delta': float(delta),
                'alpha_total': float(alpha_total),
                'alpha_inbound': float(alpha_inbound),
                'alpha_outbound': float(alpha_outbound),
                'demand_matrix': [[float(x) for x in row] for row in demand_matrix],
                'epsilon_matrix': epsilon_matrix.tolist(),
                'lambda_matrix': lambda_matrix.tolist(),
                'd_max': int(d_max),
                'delay_weights': delay_weights
            }
        except Exception as e:
            continue
    
    return None


def generate_boundary_config() -> Optional[Dict]:
    """
    Generate boundary configuration with epsilon=16, lambda=16 (All-Max scenario)
    This addresses OOD problem for worst-case analysis comparable to EIP-1559.
    """
    max_attempts = 100
    
    for attempt in range(max_attempts):
        try:
            delta = np.random.uniform(DELTA_MIN, DELTA_MAX)
            alpha_total = np.random.uniform(ALPHA_MIN, ALPHA_MAX)
            
            demand_matrix, alpha_inbound, alpha_outbound = build_7shard_self_consistent_matrix(alpha_total)
            
            is_valid, msg = verify_equilibrium_7shard(demand_matrix)
            if not is_valid:
                continue
            
            # FIXED: All epsilon = 16.0
            epsilon_matrix = np.full((NUM_SHARDS, NUM_SHARDS), 16.0)
            np.fill_diagonal(epsilon_matrix, 0.0)
            
            # FIXED: All lambda = 16.0
            lambda_matrix = np.full((NUM_SHARDS, NUM_SHARDS), 16.0)
            
            d_max = np.random.randint(D_MAX_MIN, D_MAX_MAX + 1)
            delay_weights = [0.0] * d_max
            delay_weights[-1] = 1.0
            
            return {
                'delta': float(delta),
                'alpha_total': float(alpha_total),
                'alpha_inbound': float(alpha_inbound),
                'alpha_outbound': float(alpha_outbound),
                'demand_matrix': [[float(x) for x in row] for row in demand_matrix],
                'epsilon_matrix': epsilon_matrix.tolist(),
                'lambda_matrix': lambda_matrix.tolist(),
                'd_max': int(d_max),
                'delay_weights': delay_weights
            }
        except Exception as e:
            continue
    
    return None


def create_config_yml(config: Dict, output_path: str = 'config.yml'):
    """Create config.yml file"""
    shards = [{'id': i, 'g_max': G_MAX} for i in range(NUM_SHARDS)]
    
    yml_config = {
        'simulation': {
            'total_steps': TOTAL_STEPS,
            'l_target': L_TARGET,
            'delta': config['delta'],
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
        yaml.dump(yml_config, f, default_flow_style=False, sort_keys=False)


# ===================== SIMULATION =====================

def run_simulation(config_dir: str, timeout: int = 120) -> bool:
    """Run Go simulation"""
    try:
        result = subprocess.run(
            ['go', 'run', 'main.go'],
            cwd=config_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except:
        return False


def find_latest_log(log_dir: str) -> Optional[str]:
    """Find latest log file"""
    pattern = os.path.join(log_dir, "enhanced_simulation_analysis_*.log")
    logs = glob.glob(pattern)
    return max(logs, key=os.path.getctime) if logs else None


def parse_simulation_log(log_path: str) -> Optional[Dict[int, pd.DataFrame]]:
    """
    Parse simulation log from enhanced table format
    
    Extract Fee and Load for all NUM_SHARDS shards from formatted table output
    """
    try:
        data = {i: [] for i in range(NUM_SHARDS)}
        current_step = None
        current_row_shards = []
        
        with open(log_path, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                
                # Skip empty lines or non-table lines
                if not line_stripped or not line_stripped.startswith('│'):
                    continue
                
                # Skip separator and header lines
                if any(char in line_stripped for char in ['├', '┼', '┬', '╠', '═', 'STEP', 'SHARD', 'Fee/Load/Ratio']):
                    continue
                
                # Split by │ and clean
                parts = [p.strip() for p in line_stripped.split('│') if p.strip()]
                
                if len(parts) < 2:
                    continue
                
                # Check if first part is a step number
                try:
                    step = int(parts[0])
                    # New step encountered
                    if current_step is not None and current_row_shards:
                        # Save accumulated shard data for previous step
                        for shard_idx, (fee, load) in enumerate(current_row_shards):
                            if shard_idx < NUM_SHARDS:
                                data[shard_idx].append({'Step': current_step, 'Fee': fee, 'Load': load})
                    
                    current_step = step
                    current_row_shards = []
                    parts = parts[1:]  # Remove step from parts
                except:
                    # Not a step number, continue with current step
                    if current_step is None:
                        continue
                
                # Extract fee/load from remaining parts
                for part in parts:
                    if '/' in part and '%' in part:
                        try:
                            # Extract just the fee/load/ratio part (first sub-column)
                            fee_load_str = part.split('│')[0].strip()
                            values = fee_load_str.split('/')
                            
                            if len(values) >= 2:
                                fee = float(values[0].strip())
                                load_str = values[1].strip()
                                
                                # Handle "1000k" format
                                if 'k' in load_str.lower():
                                    load = float(load_str.lower().replace('k', '').strip()) * 1000
                                else:
                                    load = float(load_str)
                                
                                current_row_shards.append((fee, load))
                        except:
                            continue
        
        # Don't forget last step
        if current_step is not None and current_row_shards:
            for shard_idx, (fee, load) in enumerate(current_row_shards):
                if shard_idx < NUM_SHARDS:
                    data[shard_idx].append({'Step': current_step, 'Fee': fee, 'Load': load})
        
        # Convert to DataFrames
        result = {}
        for sid in range(NUM_SHARDS):
            if data[sid]:
                result[sid] = pd.DataFrame(data[sid])
        
        return result if result else None
        
    except Exception as e:
        print(f"Parse error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ===================== KAPPA CALCULATION (based on Figure3/Figure5) =====================

def calculate_kappa_def_timeseries(delta_p: np.ndarray, weights: List[float], 
                                   start_idx: int, end_idx: int) -> List[float]:
    """
    Calculate κ_def(t) (based on Figure3 calc_kappa_timeseries)
    
    κ_def(t) = Σ_d w_d * |ΔP_i(t-d)| / |ΔP_i(t)| * 1(same-sign)
    """
    dmax = len(weights)
    end_idx = min(end_idx, len(delta_p) - 1)
    
    if start_idx + dmax > end_idx:
        return []
    
    ks = []
    eps = 1e-12
    
    for t in range(start_idx + dmax, end_idx + 1):
        cur = delta_p[t]
        if not np.isfinite(cur) or abs(cur) < eps:
            continue
        
        s_cur = np.sign(cur)
        k = 0.0
        
        for d in range(1, dmax + 1):
            prev = delta_p[t - d]
            if not np.isfinite(prev) or prev == 0.0:
                continue
            
            if np.sign(prev) == s_cur:  # same-sign mask
                k += weights[d - 1] * (abs(prev) / abs(cur))
        
        ks.append(k)
    
    return ks


def calculate_kappa_att_timeseries(delta_p_i: np.ndarray, delta_p_j: np.ndarray,
                                   weights: List[float], start_idx: int, end_idx: int) -> List[float]:
    """
    Calculate κ_att(t) (based on Figure8)
    
    κ_att_{j→i}(t) = Σ_d w_d * |ΔP_j(t-d)| / |ΔP_j(t)| * 1(opposite-sign)
    """
    dmax = len(weights)
    end_idx = min(end_idx, len(delta_p_i) - 1, len(delta_p_j) - 1)
    
    if start_idx + dmax > end_idx:
        return []
    
    ks = []
    eps = 1e-12
    
    for t in range(start_idx + dmax, end_idx + 1):
        cur_j = delta_p_j[t]
        cur_i = delta_p_i[t]
        
        if not np.isfinite(cur_j) or abs(cur_j) < eps:
            continue
        
        s_i = np.sign(cur_i)
        k = 0.0
        
        for d in range(1, dmax + 1):
            prev_j = delta_p_j[t - d]
            if not np.isfinite(prev_j) or prev_j == 0.0:
                continue
            
            if np.sign(prev_j) != s_i:  # opposite-sign mask
                k += weights[d - 1] * (abs(prev_j) / abs(cur_j))
        
        ks.append(k)
    
    return ks


def calculate_95th_percentile_kappas(shard_data: Dict[int, pd.DataFrame], 
                                     config: Dict) -> Dict[str, float]:
    """
    Calculate 95th percentile of κ_def and κ_att (based on Figure5)
    """
    if 0 not in shard_data:
        return {'kappa_def_95': np.nan, 'kappa_att_95': np.nan}
    
    df0 = shard_data[0]
    delta_p_0 = (df0['Fee'].values - EQUIL_P)
    weights = config['delay_weights']
    
    kappa_def_series = calculate_kappa_def_timeseries(delta_p_0, weights, START_STEP, END_STEP)
    
    # κ_att: average all source shards attacking shard 0
    all_kappa_att = []
    for source_shard in range(1, NUM_SHARDS):
        if source_shard not in shard_data:
            continue
        
        df_j = shard_data[source_shard]
        delta_p_j = (df_j['Fee'].values - EQUIL_P)
        kappa_att_series = calculate_kappa_att_timeseries(delta_p_0, delta_p_j, weights, START_STEP, END_STEP)
        
        if kappa_att_series:
            all_kappa_att.append(kappa_att_series)
    
    # Calculate 95th percentile
    kappa_def_valid = [k for k in kappa_def_series if np.isfinite(k)]
    
    if all_kappa_att:
        # Flatten all kappa_att values from different source shards
        all_kappa_values = []
        for kappa_series in all_kappa_att:
            all_kappa_values.extend([k for k in kappa_series if np.isfinite(k)])
        kappa_att_valid = all_kappa_values
    else:
        kappa_att_valid = []
    
    kappa_def_95 = np.percentile(kappa_def_valid, 95) if kappa_def_valid else 0.0
    kappa_att_95 = np.percentile(kappa_att_valid, 95) if kappa_att_valid else 0.0
    
    return {
        'kappa_def_95': float(kappa_def_95),
        'kappa_att_95': float(kappa_att_95)
    }


# ===================== MAIN FUNCTION =====================

def generate_single_training_sample(sample_id: int, sharded_system_dir: str) -> Optional[Dict]:
    """
    Generate single training sample
    
    Args:
        sample_id: Sample ID
        sharded_system_dir: Absolute path to sharded-system directory
        
    Returns:
        Training sample dictionary
    """
    print(f"\n{'='*60}")
    print(f"Sample {sample_id}")
    print(f"{'='*60}")
    
    try:
        config = generate_random_config()
        if config is None:
            print("❌ Failed to generate config")
            return None
        
        print(f"✅ Config: δ={config['delta']:.3f}, α={config['alpha_total']:.3f}, d_max={config['d_max']}")
        
        # Verify equilibrium
        is_valid, msg = verify_equilibrium_7shard(config['demand_matrix'])
        if not is_valid:
            print(f"❌ Equilibrium verification failed: {msg}")
            return None
        print(f"✅ Equilibrium verified: {msg}")
        
        config_path = os.path.join(sharded_system_dir, 'config.yml')
        create_config_yml(config, config_path)
        
        print("🚀 Running simulation...")
        if not run_simulation(sharded_system_dir):
            print("❌ Simulation failed")
            return None
        print("✅ Simulation OK")
        
        # 解析log
        log_path = find_latest_log(sharded_system_dir)
        if not log_path:
            print("❌ No log found")
            return None
        
        shard_data = parse_simulation_log(log_path)
        if not shard_data:
            print("❌ Failed to parse log")
            return None
        print(f"✅ Parsed {len(shard_data)} shards")
        
        # Delete log immediately after parsing
        try:
            os.remove(log_path)
        except:
            pass
        
        kappas = calculate_95th_percentile_kappas(shard_data, config)
        print(f"✅ κ_def={kappas['kappa_def_95']:.4f}, κ_att={kappas['kappa_att_95']:.4f}")
        
        return _build_sample_dict(config, kappas)
    
    except Exception as e:
        print(f"❌ Sample generation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_boundary_training_sample(sample_id: int, sharded_system_dir: str) -> Optional[Dict]:
    """
    Generate boundary training sample with epsilon=16, lambda=16 (All-Max scenario)
    
    Args:
        sample_id: Sample ID
        sharded_system_dir: Absolute path to sharded-system directory
        
    Returns:
        Training sample dictionary
    """
    print(f"\n{'='*60}")
    print(f"Sample {sample_id} - BOUNDARY (ε=16, λ=16)")
    print(f"{'='*60}")
    
    try:
        config = generate_boundary_config()
        if config is None:
            print("❌ Failed to generate boundary config")
            return None
        
        print(f"✅ Config: δ={config['delta']:.3f}, α={config['alpha_total']:.3f}, d_max={config['d_max']}, ε=16.0, λ=16.0")
        
        # Verify equilibrium
        is_valid, msg = verify_equilibrium_7shard(config['demand_matrix'])
        if not is_valid:
            print(f"❌ Equilibrium verification failed: {msg}")
            return None
        print(f"✅ Equilibrium verified: {msg}")
        
        config_path = os.path.join(sharded_system_dir, 'config.yml')
        create_config_yml(config, config_path)
        
        print("🚀 Running simulation...")
        if not run_simulation(sharded_system_dir):
            print("❌ Simulation failed")
            return None
        print("✅ Simulation OK")
        
        # 解析log
        log_path = find_latest_log(sharded_system_dir)
        if not log_path:
            print("❌ No log found")
            return None
        
        shard_data = parse_simulation_log(log_path)
        if not shard_data:
            print("❌ Failed to parse log")
            return None
        print(f"✅ Parsed {len(shard_data)} shards")
        
        # Delete log immediately after parsing
        try:
            os.remove(log_path)
        except:
            pass
        
        kappas = calculate_95th_percentile_kappas(shard_data, config)
        print(f"✅ κ_def={kappas['kappa_def_95']:.4f}, κ_att={kappas['kappa_att_95']:.4f}")
        
        return _build_sample_dict(config, kappas)
    
    except Exception as e:
        print(f"❌ Sample generation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _build_sample_dict(config: Dict, kappas: Dict) -> Dict:
    """Helper function to build sample dictionary from config and kappas"""
    # Flatten epsilon and lambda matrices for CSV storage
    epsilon_flat = {}
    lambda_flat = {}
    for i in range(NUM_SHARDS):
        for j in range(NUM_SHARDS):
            epsilon_flat[f'epsilon_{i}_{j}'] = config['epsilon_matrix'][i][j]
            lambda_flat[f'lambda_{i}_{j}'] = config['lambda_matrix'][i][j]
    
    result = {
        'delta': config['delta'],
        'alpha_total': config['alpha_total'],
        'alpha_inbound': config['alpha_inbound'],
        'alpha_outbound': config['alpha_outbound'],
        'd_max': config['d_max'],
        'avg_epsilon': float(np.mean(config['epsilon_matrix'])),
        'avg_lambda': float(np.mean(config['lambda_matrix'])),
        'kappa_def_95': kappas['kappa_def_95'],
        'kappa_att_95': kappas['kappa_att_95']
    }
    
    # Add all epsilon and lambda values
    result.update(epsilon_flat)
    result.update(lambda_flat)
    
    return result


if __name__ == '__main__':
    print("🧪 Testing data generator...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sharded_system_dir = os.path.join(script_dir, '..')
    
    print(f"Sharded system directory: {sharded_system_dir}")
    
    sample = generate_single_training_sample(0, sharded_system_dir)
    
    if sample:
        print("\n✅ Test successful!")
        print("Sample:", sample)
    else:
        print("\n❌ Test failed")
