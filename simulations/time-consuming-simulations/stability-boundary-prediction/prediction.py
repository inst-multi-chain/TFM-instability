#!/usr/bin/env python3
"""
Figure 7: Neural Network Based Phase Map Prediction

This experiment validates the neural network's ability to predict phase map boundaries.
Unlike Figure5 which calculates kappa from simulation data, this script:
1. Uses the trained neural network to PREDICT kappa_def and kappa_att
2. Calculates Gi and Ri from predicted kappa values
3. Compares predicted phase boundaries with empirical convergence results

Key Difference from Figure5:
- Figure5: Runs simulation → Calculates kappa from price data → Validates theory
- Figure7: Uses NN to predict kappa → Validates NN prediction capability

This demonstrates the practical applicability of the neural network model
for predicting system stability without running expensive simulations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import subprocess
import time
import shutil
import os
import glob
import torch
import pickle
import sys
from typing import Tuple, Dict

# Add nn-model to path for importing model
sys.path.append('../../../nn-model')

# Import model architecture (will be loaded from saved weights)
# We define NUM_SHARDS here to avoid import issues
NUM_SHARDS = 7
SCALAR_FEATURES = ['delta', 'alpha_total', 'alpha_inbound', 'alpha_outbound', 'd_max']

# --- Configuration identical to Figure5 ---
DELETE_LOG_AFTER_PARSE = True  # Keep logs for debugging
SAVE_PARSED_TIMESERIES = False
PARSED_TS_DIR = 'parsed_timeseries'

# Delay distribution - use Spike as in Figure5
DELAY_DISTRIBUTIONS = {
    'Spike': [0, 0, 0, 0, 1],
}

# Simulation parameters
G_MAX = 2000000
TARGET_TOTAL_DEMAND = G_MAX / 2

# Experiment parameter ranges (same as Figure5)
EPSILON_MIN = 0.0
EPSILON_MAX = 16
EPSILON_STEP = 0.1

ALPHA_MIN = 0.01
ALPHA_MAX = 0.99
ALPHA_STEP = 0.01
ALPHA_IJ_OUT = 0

# Phase map parameters
DELTA = 0.125
EQUIL_P = 1.0
L_TARGET = 0.5
START_STEP = 100
END_STEP = 4999

# Neural Network Model Path
NN_MODEL_PATH = '../../../nn-model/models_weighted_v2/best_model.pth'
NN_SCALERS_PATH = '../../../nn-model/models_weighted_v2/scalers.pkl'

# Calculate total experiment points
EPSILON_POINTS = int((EPSILON_MAX - EPSILON_MIN) / EPSILON_STEP) + 1
ALPHA_POINTS = round((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP) + 1
TOTAL_POINTS_PER_DISTRIBUTION = EPSILON_POINTS * ALPHA_POINTS


# ===================== NEURAL NETWORK PREDICTION =====================

class KappaPredictorWrapper:
    """Wrapper for neural network kappa prediction"""
    
    def __init__(self, model_path: str, scalers_path: str):
        """Load trained model and scalers"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load scalers
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        
        # Import model architecture
        from train_nn_weighted_v2 import KappaPredictor
        
        # Load model
        self.model = KappaPredictor(scalar_input_dim=len(SCALAR_FEATURES)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"✅ Loaded neural network model from {model_path}")
        print(f"✅ Model device: {self.device}")
    
    def predict_kappa(self, delta: float, alpha_total: float, alpha_inbound: float, 
                      alpha_outbound: float, d_max: int, 
                      epsilon_matrix: np.ndarray, lambda_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Predict kappa_def_95 and kappa_att_95 using neural network
        
        Args:
            delta: EIP-1559 update rate
            alpha_total: Total cross-shard ratio
            alpha_inbound: Inbound cross-shard ratio
            alpha_outbound: Outbound cross-shard ratio
            d_max: Maximum delay
            epsilon_matrix: 7x7 epsilon elasticity matrix
            lambda_matrix: 7x7 lambda elasticity matrix
        
        Returns:
            (kappa_def_95, kappa_att_95)
        """
        # Prepare scalar features (MUST match SCALAR_FEATURES order in training)
        X_scalar = np.array([[
            delta, 
            alpha_total, 
            alpha_inbound, 
            alpha_outbound, 
            d_max
        ]], dtype=np.float32)
        
        # Prepare matrix features
        epsilon_flat = epsilon_matrix.flatten()
        lambda_flat = lambda_matrix.flatten()
        X_matrix = np.concatenate([epsilon_flat, lambda_flat]).reshape(1, -1).astype(np.float32)
        
        # Normalize features
        X_scalar_norm = self.scalers['scaler_scalar'].transform(X_scalar)
        X_matrix_norm = self.scalers['scaler_matrix'].transform(X_matrix)
        
        # Reshape matrix features to 2 x 7 x 7
        epsilon_reshaped = X_matrix_norm[:, :49].reshape(1, 1, NUM_SHARDS, NUM_SHARDS)
        lambda_reshaped = X_matrix_norm[:, 49:].reshape(1, 1, NUM_SHARDS, NUM_SHARDS)
        X_matrix_reshaped = np.concatenate([epsilon_reshaped, lambda_reshaped], axis=1)
        
        # Convert to tensors
        X_scalar_tensor = torch.FloatTensor(X_scalar_norm).to(self.device)
        X_matrix_tensor = torch.FloatTensor(X_matrix_reshaped).to(self.device)
        
        # Predict (NOTE: model expects x_scalar FIRST, then x_matrix)
        with torch.no_grad():
            y_pred_scaled = self.model(X_scalar_tensor, X_matrix_tensor)
            y_pred_scaled_np = y_pred_scaled.cpu().numpy()
        
        # Inverse transform: NO expm1 because we don't use log1p anymore!
        y_pred = self.scalers['scaler_y'].inverse_transform(y_pred_scaled_np)
        # y_pred = np.expm1(y_pred_normalized)  # DISABLED - no log1p in training!
        
        kappa_def_95 = float(y_pred[0, 0])
        kappa_att_95 = float(y_pred[0, 1])
        
        # Ensure non-negative (model might predict small negative values)
        kappa_def_95 = max(0.0, kappa_def_95)
        kappa_att_95 = max(0.0, kappa_att_95)
        
        return kappa_def_95, kappa_att_95


# ===================== SIMULATION FUNCTIONS (from Figure5) =====================

def load_config():
    """Load configuration from config.yml"""
    try:
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print("❌ config.yml not found")
        return None

def backup_config():
    """Create a backup of the original config"""
    try:
        shutil.copy('config.yml', 'config_backup.yml')
        print("✅ Created config backup")
        return True
    except Exception as e:
        print(f"❌ Failed to backup config: {e}")
        return False

def restore_config():
    """Restore config from backup"""
    try:
        shutil.copy('config_backup.yml', 'config.yml')
        print("✅ Restored original config")
        return True
    except Exception as e:
        print(f"❌ Failed to restore config: {e}")
        return False

def calculate_demand_matrix(alpha_inflow_to_0):
    """
    Calculate base_demand_matrix for 7-shard system
    Only shard 0 is analyzed, shards 1-6 provide inflow
    
    Args:
        alpha_inflow_to_0: Total inflow ratio to shard 0 from all other shards
    
    Returns:
        7x7 demand matrix
    """
    # Initialize 7x7 matrix with zeros
    matrix = [[0.0 for _ in range(7)] for _ in range(7)]
    
    # Step 1: Outflow from shard 0 (α_0→j,out = 0)
    total_outflow_from_0 = TARGET_TOTAL_DEMAND * ALPHA_IJ_OUT  # = 0
    # Distribute evenly to shards 1-6
    for j in range(1, 7):
        matrix[0][j] = total_outflow_from_0 / 6

    # Step 2: Inflow to shard 0 from shards 1-6
    total_inflow_to_0 = TARGET_TOTAL_DEMAND * alpha_inflow_to_0
    # Distribute evenly from each of shards 1-6
    inflow_per_shard = total_inflow_to_0 / 6
    for j in range(1, 7):
        matrix[j][0] = inflow_per_shard

    # Step 3: Local demand for shard 0
    total_cross_to_0 = sum(matrix[j][0] for j in range(1, 7))
    total_cross_from_0 = sum(matrix[0][j] for j in range(1, 7))
    matrix[0][0] = TARGET_TOTAL_DEMAND - total_cross_from_0 - total_cross_to_0

    # Step 4: Set all diagonal elements equal to shard 0's local demand
    diagonal_value = matrix[0][0]
    for i in range(1, 7):
        matrix[i][i] = diagonal_value

    # Step 5: Calculate remaining cross-shard flows for shards 1-6
    # Each shard j needs: total_demand = matrix[j][j] + matrix[j][0] + (sum of other cross-shard)
    for j in range(1, 7):
        already_allocated = matrix[j][j] + matrix[j][0] + matrix[0][j]
        remaining = TARGET_TOTAL_DEMAND - already_allocated
        
        # Distribute remaining demand evenly to other shards (excluding 0 and self)
        other_shards = [k for k in range(1, 7) if k != j]
        per_other = remaining / len(other_shards) if other_shards else 0
        
        for k in other_shards:
            matrix[j][k] = per_other / 2  # Split evenly between j→k and k→j
            matrix[k][j] = per_other / 2
    
    return matrix

def update_config_for_experiment(epsilon_j0, alpha_inflow, delay_weights):
    """Update config.yml for experiment parameters"""
    try:
        config = load_config()
        if config is None:
            return False

        # Ensure epsilon_matrix is 7x7
        if 'epsilon_matrix' not in config['demand'] or len(config['demand']['epsilon_matrix']) != 7:
            # Initialize 7x7 epsilon matrix with default value 1.5
            config['demand']['epsilon_matrix'] = [[1.5 for _ in range(7)] for _ in range(7)]
        
        # Update epsilon for all cross-shard flows to shard 0 (j→0, j=1..6)
        for j in range(1, 7):
            config['demand']['epsilon_matrix'][j][0] = epsilon_j0

        # Ensure lambda_matrix is 7x7
        if 'lambda_matrix' not in config['demand'] or len(config['demand']['lambda_matrix']) != 7:
            config['demand']['lambda_matrix'] = [[1.5 for _ in range(7)] for _ in range(7)]

        # Calculate and update demand matrix (now returns 7x7)
        demand_matrix = calculate_demand_matrix(alpha_inflow)
        config['demand']['base_demand_matrix'] = demand_matrix

        # Update delay weights
        config['delay']['weights'] = delay_weights

        with open('config.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return True

    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def run_simulation():
    """Run the Go simulation"""
    try:
        result = subprocess.run(['go', 'run', '../../../main.go'],
                              capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print("❌ Simulation failed:", result.stderr[:400])
            return False
        time.sleep(0.05)
        return True
    except subprocess.TimeoutExpired:
        print("❌ Simulation timed out")
        return False
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return False

def parse_latest_log(delete_after=DELETE_LOG_AFTER_PARSE):
    """Parse latest log file and optionally delete it"""
    files = glob.glob('enhanced_simulation_analysis_*.log')
    if not files:
        return None, None
    latest = max(files, key=os.path.getctime)
    data = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    try:
        with open(latest, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                parts = line.split(',')
                try:
                    # For 7-shard system: step, fee0-6, load0-6 (1 + 7 + 7 = 15 columns)
                    if len(parts) >= 15:
                        step = int(parts[0])
                        fees = [float(parts[i]) for i in range(1, 8)]  # parts[1-7]
                        loads = [float(parts[i]) for i in range(8, 15)]  # parts[8-14]
                        for sid in range(7):
                            data[sid].append({'Step': step, 'Fee': fees[sid], 'Load': loads[sid]})
                    # Fallback for 3-shard format (backward compatibility)
                    elif len(parts) >= 7:
                        step = int(parts[0])
                        fees = [float(parts[1]), float(parts[2]), float(parts[3])]
                        loads = [float(parts[4]), float(parts[5]), float(parts[6])]
                        for sid in (0, 1, 2):
                            data[sid].append({'Step': step, 'Fee': fees[sid], 'Load': loads[sid]})
                    # Fallback for single shard
                    elif len(parts) >= 3:
                        step = int(parts[0])
                        fee = float(parts[1])
                        load = float(parts[2])
                        data[0].append({'Step': step, 'Fee': fee, 'Load': load})
                except:
                    continue
    finally:
        if delete_after:
            try:
                os.remove(latest)
                print(f"🧹 Deleted log: {latest}")
            except Exception as e:
                print(f"⚠️ Failed to delete log {latest}: {e}")

    out = {}
    for sid in range(7):
        if data[sid]:
            out[sid] = pd.DataFrame(data[sid])
    return (out if out else None), latest

def check_convergence(load_stats, tolerance=0.00001):
    """Check if system converged to target load"""
    if load_stats is None or not isinstance(load_stats, dict):
        return False
    return abs(load_stats['load0'] - 0.5) < tolerance


# ===================== PHASE MAP CALCULATION =====================

def calculate_phase_map_parameters_from_predicted_kappa(
    epsilon_j0: float, 
    alpha_inflow: float, 
    kappa_def_95: float, 
    kappa_att_95: float
) -> Tuple[float, float, dict]:
    """
    Calculate Gi and Ri using PREDICTED kappa values from neural network
    
    For 7-shard system: shard 0 receives inflow from shards 1-6
    
    Args:
        epsilon_j0: Inbound elasticity parameter (ε_j→0 for j=1..6)
        alpha_inflow: Total inbound cross-shard ratio to shard 0
        kappa_def_95: PREDICTED defense amplification factor
        kappa_att_95: PREDICTED attack amplification factor
    
    Returns:
        (Gi, Ri, details_dict)
    """
    # Load composition for shard 0 in 7-shard system
    alpha_0_local = 1.0 - alpha_inflow
    alpha_j_to_0_in = alpha_inflow / 6  # Evenly distributed from 6 shards
    alpha_0_to_j_out = 0.0  # No outflow from shard 0

    # Elasticities
    lambda_00 = 1.5      # Local elasticity
    lambda_0j = 1.5      # Outbound elasticity (not used)
    epsilon_j0_param = epsilon_j0  # Inbound elasticity (swept parameter)
    epsilon_0j = 1.5     # Outbound elasticity (not used)
    lambda_j0 = 1.5      # Delayed attack elasticity

    # Calculate forces for shard 0
    # Sum over all 6 shards providing inflow
    Lambda_0_local = lambda_00 * alpha_0_local
    Lambda_0_self_delay = epsilon_j0_param * (6 * alpha_j_to_0_in)  # Sum over j=1..6
    Lambda_0j_inst = epsilon_0j * alpha_0_to_j_out  # = 0
    Lambda_j0_delay = lambda_j0 * (6 * alpha_j_to_0_in)  # Sum over j=1..6

    # Calculate Gi and Ri using PREDICTED kappa
    defense_total = Lambda_0_local + kappa_def_95 * Lambda_0_self_delay
    attack_total = kappa_att_95 * Lambda_j0_delay

    Gi = DELTA * defense_total
    Ri = attack_total / defense_total if defense_total > 0 else float('inf')

    return Gi, Ri, {
        'alpha_0_local': alpha_0_local,
        'alpha_j_to_0_in': alpha_j_to_0_in,
        'Lambda_0_local': Lambda_0_local,
        'Lambda_0_self_delay': Lambda_0_self_delay,
        'Lambda_j0_delay': Lambda_j0_delay,
        'defense_total': defense_total,
        'attack_total': attack_total,
        'kappa_def_predicted': kappa_def_95,
        'kappa_att_predicted': kappa_att_95
    }


def prepare_nn_input_from_config(epsilon_j0: float, alpha_inflow: float, 
                                  delay_weights: list) -> dict:
    """
    Prepare input features for neural network prediction
    
    Args:
        epsilon_j0: Epsilon parameter for j→0 cross-shard flows
        alpha_inflow: Inbound cross-shard ratio
        delay_weights: Delay distribution weights
    
    Returns:
        Dictionary with all required NN input features
    """
    # Scalar features
    delta = DELTA
    alpha_total = alpha_inflow  # For 3-shard with symmetric inflow
    alpha_outbound = ALPHA_IJ_OUT
    d_max = len(delay_weights)
    
    # For 3-shard system, need to pad to 7x7 for the neural network
    # The NN was trained on 7-shard data, so we need to provide 7x7 matrices
    epsilon_matrix_3x3 = np.ones((3, 3)) * 1.5  # Base elasticity
    epsilon_matrix_3x3[1, 0] = epsilon_j0  # ε_1→0
    epsilon_matrix_3x3[2, 0] = epsilon_j0  # ε_2→0
    
    lambda_matrix_3x3 = np.ones((3, 3)) * 1.5  # Base elasticity
    
    # Pad to 7x7 (fill extra shards with average values)
    epsilon_matrix = np.ones((7, 7)) * 1.5
    epsilon_matrix[:3, :3] = epsilon_matrix_3x3
    
    lambda_matrix = np.ones((7, 7)) * 1.5
    lambda_matrix[:3, :3] = lambda_matrix_3x3
    
    return {
        'delta': delta,
        'alpha_total': alpha_total,
        'alpha_inbound': alpha_inflow,
        'alpha_outbound': alpha_outbound,
        'd_max': d_max,
        'epsilon_matrix': epsilon_matrix,
        'lambda_matrix': lambda_matrix
    }


# ===================== EXPERIMENT EXECUTION =====================

def run_single_experiment(
    epsilon_j0: float, 
    alpha_inflow: float, 
    delay_weights: list, 
    distribution_name: str, 
    exp_id: int,
    nn_predictor: KappaPredictorWrapper
) -> dict:
    """
    Run single experiment with NN-predicted kappa values
    
    KEY DIFFERENCE from Figure5:
    - Uses neural network to predict kappa BEFORE running simulation
    - Uses predicted kappa to calculate phase map parameters
    - Compares NN prediction with empirical convergence
    """
    
    # Step 1: Prepare NN input features
    nn_input = prepare_nn_input_from_config(epsilon_j0, alpha_inflow, delay_weights)
    
    # Step 2: Predict kappa using neural network
    kappa_def_pred, kappa_att_pred = nn_predictor.predict_kappa(
        delta=nn_input['delta'],
        alpha_total=nn_input['alpha_total'],
        alpha_inbound=nn_input['alpha_inbound'],
        alpha_outbound=nn_input['alpha_outbound'],
        d_max=nn_input['d_max'],
        epsilon_matrix=nn_input['epsilon_matrix'],
        lambda_matrix=nn_input['lambda_matrix']
    )
    
    print(f"  🔮 NN Predicted: κ_def={kappa_def_pred:.4f}, κ_att={kappa_att_pred:.4f}")
    
    # Step 3: Calculate phase map parameters using PREDICTED kappa
    Gi, Ri, details = calculate_phase_map_parameters_from_predicted_kappa(
        epsilon_j0, alpha_inflow, kappa_def_pred, kappa_att_pred
    )
    
    # Phase map prediction: stable if Gi(1 + Ri) < 2 AND Ri < 1
    phase_map_stable = (Gi * (1 + Ri) < 2) and (Ri < 1)
    
    print(f"  📊 Phase Map (NN): Gi={Gi:.4f}, Ri={Ri:.4f}, Gi(1+Ri)={Gi*(1+Ri):.4f}, Stable={phase_map_stable}")
    
    # Step 4: Run simulation to get empirical convergence
    if not update_config_for_experiment(epsilon_j0, alpha_inflow, delay_weights):
        return None
    
    if not run_simulation():
        return None
    
    # Step 5: Parse simulation results
    shard_data, log_path = parse_latest_log()
    if not shard_data or 0 not in shard_data:
        print(f"  ❌ Failed to parse shard data")
        return None
    
    df0 = shard_data[0]
    if df0.empty:
        return None
    
    # Step 6: Check empirical convergence
    final_load = df0['Load'].iloc[-1]
    converged = check_convergence({'load0': final_load})
    
    print(f"  🎯 Empirical: Load={final_load:.6f}, Converged={converged}")
    
    # Step 7: Compare prediction with reality
    prediction_correct = (converged == phase_map_stable)
    
    if converged and phase_map_stable:
        prediction_category = "True Positive"
    elif not converged and not phase_map_stable:
        prediction_category = "True Negative"
    elif not converged and phase_map_stable:
        prediction_category = "False Positive"
    else:
        prediction_category = "False Negative"
    
    status_symbol = "✅" if prediction_correct else "❌"
    print(f"  {status_symbol} Prediction: {prediction_category}")
    
    return {
        'epsilon_j0': epsilon_j0,
        'alpha_inflow': alpha_inflow,
        'distribution': distribution_name,
        'final_load': final_load,
        'converged': converged,
        'phase_map_stable': phase_map_stable,
        'prediction_correct': prediction_correct,
        'prediction_category': prediction_category,
        'kappa_def_predicted': kappa_def_pred,
        'kappa_att_predicted': kappa_att_pred,
        'Gi': Gi,
        'Ri': Ri,
        **details
    }


def run_nn_phase_map_validation():
    """Run complete NN-based phase map validation experiment"""
    print("🤖 Figure 7: Neural Network Phase Map Prediction")
    print("=" * 80)
    
    # Check if NN model exists
    if not os.path.exists(NN_MODEL_PATH):
        print(f"❌ Neural network model not found: {NN_MODEL_PATH}")
        print("Please train the model first using train_nn.py")
        return
    
    if not os.path.exists(NN_SCALERS_PATH):
        print(f"❌ Scalers not found: {NN_SCALERS_PATH}")
        return
    
    # Load neural network
    nn_predictor = KappaPredictorWrapper(NN_MODEL_PATH, NN_SCALERS_PATH)
    
    total_experiments = TOTAL_POINTS_PER_DISTRIBUTION * len(DELAY_DISTRIBUTIONS)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Epsilon range: {EPSILON_MIN} to {EPSILON_MAX}, step {EPSILON_STEP}")
    print(f"Alpha range: {ALPHA_MIN} to {ALPHA_MAX}, step {ALPHA_STEP}")
    print(f"Delay distributions: {list(DELAY_DISTRIBUTIONS.keys())}")
    
    # Backup config
    if not backup_config():
        print("❌ Failed to backup config")
        return
    
    try:
        experiment_count = 0
        all_results = []
        
        for dist_name, delay_weights in DELAY_DISTRIBUTIONS.items():
            print(f"\n🧪 Testing: {dist_name}")
            print(f"Weights: {delay_weights}")
            print("-" * 60)
            
            distribution_results = []
            
            for i in range(EPSILON_POINTS):
                epsilon_j0 = EPSILON_MIN + i * EPSILON_STEP
                
                for j in range(ALPHA_POINTS):
                    alpha_inflow = ALPHA_MIN + j * ALPHA_STEP
                    experiment_count += 1
                    
                    print(f"\nExp {experiment_count}/{total_experiments}: " +
                          f"{dist_name}, ε={epsilon_j0:.2f}, α={alpha_inflow:.2f}")
                    
                    result = run_single_experiment(
                        epsilon_j0, alpha_inflow, delay_weights, 
                        dist_name, experiment_count, nn_predictor
                    )
                    
                    if result is not None:
                        distribution_results.append(result)
                        all_results.append(result)
                    else:
                        print(f"  ❌ Experiment failed")
                        failed_result = {
                            'epsilon_j0': epsilon_j0,
                            'alpha_inflow': alpha_inflow,
                            'distribution': dist_name,
                            'final_load': None,
                            'converged': False,
                            'phase_map_stable': False,
                            'prediction_correct': False,
                            'prediction_category': 'Failed',
                            'kappa_def_predicted': np.nan,
                            'kappa_att_predicted': np.nan,
                            'Gi': np.nan,
                            'Ri': np.nan
                        }
                        distribution_results.append(failed_result)
                        all_results.append(failed_result)
                    
                    # Save intermediate results
                    if experiment_count % 100 == 0:
                        df_temp = pd.DataFrame(all_results)
                        df_temp.to_csv(f'nn_phase_map_validation_temp.csv', index=False)
                        print(f"💾 Saved intermediate results at {experiment_count}")
            
            # Save distribution results
            df_dist = pd.DataFrame(distribution_results)
            df_dist.to_csv(f'nn_phase_map_validation_{dist_name}.csv', index=False)
            print(f"✅ Completed {dist_name}: {len(distribution_results)} experiments")
        
        # Save all results
        df_all = pd.DataFrame(all_results)
        df_all.to_csv('nn_phase_map_validation_all.csv', index=False)
        
        # Analyze results
        analyze_nn_validation_results(df_all)
        
        print(f"\n🎉 All experiments completed!")
        print(f"Total: {experiment_count}")
        
    finally:
        restore_config()
        print("🔄 Config restored")


def analyze_nn_validation_results(df):
    """Analyze NN validation results"""
    if df is None or len(df) == 0:
        print("❌ No data for analysis")
        return
    
    print("\n📊 NEURAL NETWORK PHASE MAP PREDICTION ANALYSIS")
    print("=" * 80)
    
    total_exp = len(df)
    valid_exp = len(df[~df['Gi'].isna()])
    
    if valid_exp == 0:
        print("❌ No valid experiments")
        return
    
    df_valid = df[~df['Gi'].isna()]
    
    correct_pred = len(df_valid[df_valid['prediction_correct'] == True])
    accuracy = correct_pred / valid_exp * 100
    
    print(f"📈 Overall Results:")
    print(f"   Total: {total_exp}")
    print(f"   Valid: {valid_exp}")
    print(f"   Correct: {correct_pred}")
    print(f"   NN Prediction Accuracy: {accuracy:.1f}%")
    
    # Per-distribution analysis
    for dist_name in df['distribution'].unique():
        dist_data = df_valid[df_valid['distribution'] == dist_name]
        if len(dist_data) == 0:
            continue
        
        dist_correct = len(dist_data[dist_data['prediction_correct'] == True])
        dist_accuracy = dist_correct / len(dist_data) * 100
        
        empirical_converged = len(dist_data[dist_data['converged'] == True])
        nn_predicted_stable = len(dist_data[dist_data['phase_map_stable'] == True])
        
        tp = len(dist_data[dist_data['prediction_category'] == 'True Positive'])
        tn = len(dist_data[dist_data['prediction_category'] == 'True Negative'])
        fp = len(dist_data[dist_data['prediction_category'] == 'False Positive'])
        fn = len(dist_data[dist_data['prediction_category'] == 'False Negative'])
        
        print(f"\n🔬 {dist_name}:")
        print(f"   Valid: {len(dist_data)}")
        print(f"   NN Accuracy: {dist_accuracy:.1f}%")
        print(f"   Empirical converged: {empirical_converged} ({empirical_converged/len(dist_data)*100:.1f}%)")
        print(f"   NN predicted stable: {nn_predicted_stable} ({nn_predicted_stable/len(dist_data)*100:.1f}%)")
        print(f"\n   Prediction Categories:")
        print(f"     ✅ TP (实际收敛, NN预测收敛): {tp} ({tp/len(dist_data)*100:.1f}%)")
        print(f"     ✅ TN (实际发散, NN预测发散): {tn} ({tn/len(dist_data)*100:.1f}%)")
        print(f"     ❌ FP (实际发散, NN预测收敛): {fp} ({fp/len(dist_data)*100:.1f}%)")
        print(f"     ❌ FN (实际收敛, NN预测发散): {fn} ({fn/len(dist_data)*100:.1f}%)")
        
        # Kappa statistics
        print(f"\n   Predicted Kappa Statistics:")
        print(f"     Mean κ_def: {dist_data['kappa_def_predicted'].mean():.4f}")
        print(f"     Mean κ_att: {dist_data['kappa_att_predicted'].mean():.4f}")
        print(f"     Max κ_def: {dist_data['kappa_def_predicted'].max():.4f}")
        print(f"     Max κ_att: {dist_data['kappa_att_predicted'].max():.4f}")
    
    # Save summary
    summary_stats = []
    for dist_name in df['distribution'].unique():
        dist_data = df_valid[df_valid['distribution'] == dist_name]
        if len(dist_data) == 0:
            continue
        
        dist_correct = len(dist_data[dist_data['prediction_correct'] == True])
        
        summary_stats.append({
            'distribution': dist_name,
            'total_experiments': len(dist_data),
            'correct_predictions': dist_correct,
            'accuracy': dist_correct / len(dist_data) * 100,
            'empirical_converged': len(dist_data[dist_data['converged'] == True]),
            'nn_predicted_stable': len(dist_data[dist_data['phase_map_stable'] == True]),
            'true_positive': len(dist_data[dist_data['prediction_category'] == 'True Positive']),
            'true_negative': len(dist_data[dist_data['prediction_category'] == 'True Negative']),
            'false_positive': len(dist_data[dist_data['prediction_category'] == 'False Positive']),
            'false_negative': len(dist_data[dist_data['prediction_category'] == 'False Negative']),
            'mean_kappa_def': dist_data['kappa_def_predicted'].mean(),
            'mean_kappa_att': dist_data['kappa_att_predicted'].mean()
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('nn_phase_map_validation_summary.csv', index=False)
    print(f"\n💾 Summary saved to nn_phase_map_validation_summary.csv")


def create_nn_validation_plots(df):
    """Create visualization plots for NN validation"""
    if df is None or len(df) == 0:
        print("❌ No data for plotting")
        return
    
    df_valid = df[~df['Gi'].isna()]
    if len(df_valid) == 0:
        print("❌ No valid data for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for dist_name in df['distribution'].unique():
        dist_data = df_valid[df_valid['distribution'] == dist_name]
        if len(dist_data) == 0:
            continue
        
        correct = dist_data[dist_data['prediction_correct'] == True]
        incorrect = dist_data[dist_data['prediction_correct'] == False]
        
        if len(correct) > 0:
            ax.scatter(correct['Gi'], correct['Ri'], c='green', marker='o',
                      s=30, alpha=0.7, label=f'Correct ({len(correct)})')
        
        if len(incorrect) > 0:
            ax.scatter(incorrect['Gi'], incorrect['Ri'], c='red', marker='x',
                      s=30, alpha=0.7, label=f'Incorrect ({len(incorrect)})')
        
        # Phase boundary
        gi_range = np.linspace(0, max(2, dist_data['Gi'].max()), 100)
        ri_boundary = 1 - np.abs(1 - gi_range)
        ri_boundary = np.maximum(ri_boundary, 0)
        
        ax.plot(gi_range, ri_boundary, 'b--', linewidth=2, label='Phase Boundary')
        
        ax.set_xlabel('Gi (Intensity)', fontsize=12)
        ax.set_ylabel('Ri (Coupling Ratio)', fontsize=12)
        ax.set_title(f'Neural Network Phase Map Prediction\n{dist_name} Distribution', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nn_phase_map_validation_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig('nn_phase_map_validation_scatter.pdf', dpi=300, bbox_inches='tight')
    print("📈 Plots saved as nn_phase_map_validation_scatter.png/pdf")
    plt.show()


def main():
    """Main function"""
    print("🤖 Figure 7: Neural Network Phase Map Prediction")
    print("=" * 60)
    
    print("Choose operation:")
    print("1. Run NN phase map prediction experiment")
    print("2. Analyze existing results")
    print("3. Create validation plots")
    print("4. Exit")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print(f"\nReady to run {TOTAL_POINTS_PER_DISTRIBUTION * len(DELAY_DISTRIBUTIONS)} experiments")
        print("Using NEURAL NETWORK to predict kappa values")
        print("This validates the NN's ability to predict phase boundaries\n")
        
        user_input = input("Proceed? (y/N): ").strip().lower()
        if user_input in ['y', 'yes']:
            run_nn_phase_map_validation()
        else:
            print("Cancelled")
    
    elif choice == "2":
        try:
            df = pd.read_csv('nn_phase_map_validation_all.csv')
            analyze_nn_validation_results(df)
        except FileNotFoundError:
            print("❌ Results not found. Run experiment first.")
    
    elif choice == "3":
        try:
            df = pd.read_csv('nn_phase_map_validation_all.csv')
            create_nn_validation_plots(df)
        except FileNotFoundError:
            print("❌ Results not found. Run experiment first.")
    
    elif choice == "4":
        print("Goodbye!")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
