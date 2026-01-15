#!/usr/bin/env python3
"""
Prediction wrapper with physics-informed features

Uses the same feature engineering as training:
- force_elasticity
- force_coupling  
- max_risk_factor
"""

import numpy as np
import torch
import pickle
from typing import Tuple
import os


class KappaPredictorWithPhysics:
    """
    Wrapper for kappa prediction with physics features
    """
    
    def __init__(self, model_dir: str = 'models_weighted_v2'):
        """
        Load model and scalers
        
        Args:
            model_dir: Directory containing best_model.pth and scalers.pkl
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load scalers
        scaler_path = os.path.join(model_dir, 'scalers.pkl')
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        
        self.scaler_scalar = scalers['scaler_scalar']
        self.scaler_matrix = scalers['scaler_matrix']
        self.scaler_y = scalers['scaler_y']
        
        # Load v2 model architecture
        from train_nn_weighted_v2 import KappaPredictor, SCALAR_FEATURES
        
        self.model = KappaPredictor(scalar_input_dim=len(SCALAR_FEATURES))
        model_path = os.path.join(model_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded from {model_dir}")
    
    def prepare_features(self, delta, alpha_total, alpha_inbound, alpha_outbound, 
                        d_max, epsilon_matrix, lambda_matrix) -> dict:
        """
        Prepare basic scalar features (no physics features)
        
        Returns:
            dict with 5 basic scalar features
        """
        return {
            'delta': delta,
            'alpha_total': alpha_total,
            'alpha_inbound': alpha_inbound,
            'alpha_outbound': alpha_outbound,
            'd_max': d_max
        }
    
    def predict(self, delta, alpha_total, alpha_inbound, alpha_outbound, d_max,
                epsilon_matrix, lambda_matrix) -> Tuple[float, float]:
        """
        Predict kappa values with physics features
        
        Args:
            Same as original predict_kappa function
        
        Returns:
            (kappa_def_95, kappa_att_95)
        """
        # Prepare basic features
        features = self.prepare_features(
            delta, alpha_total, alpha_inbound, alpha_outbound, d_max,
            epsilon_matrix, lambda_matrix
        )
        
        # Prepare scalar input (MUST match SCALAR_FEATURES order in training)
        # Order: delta, alpha_total, alpha_inbound, alpha_outbound, d_max
        X_scalar = np.array([[
            features['delta'],
            features['alpha_total'],
            features['alpha_inbound'],
            features['alpha_outbound'],
            features['d_max']
        ]], dtype=np.float32)
        
        # Prepare matrix input
        epsilon_flat = epsilon_matrix.flatten()
        lambda_flat = lambda_matrix.flatten()
        X_matrix = np.concatenate([epsilon_flat, lambda_flat]).reshape(1, -1)
        
        # Normalize
        X_scalar = self.scaler_scalar.transform(X_scalar)
        X_matrix = self.scaler_matrix.transform(X_matrix)
        
        # Reshape matrices
        epsilon_reshaped = X_matrix[:, :49].reshape(1, 1, 7, 7)
        lambda_reshaped = X_matrix[:, 49:].reshape(1, 1, 7, 7)
        X_matrix_reshaped = np.concatenate([epsilon_reshaped, lambda_reshaped], axis=1)
        
        # Convert to tensors
        X_scalar_tensor = torch.FloatTensor(X_scalar).to(self.device)
        X_matrix_tensor = torch.FloatTensor(X_matrix_reshaped).to(self.device)
        
        # Predict
        with torch.no_grad():
            y_pred = self.model(X_scalar_tensor, X_matrix_tensor)
        
        # Inverse transform
        y_pred_np = y_pred.cpu().numpy()
        y_pred_original = self.scaler_y.inverse_transform(y_pred_np)
        # NO log1p in training anymore, so NO expm1 here!
        # y_pred_original = np.expm1(y_pred_original)  # DISABLED!
        
        kappa_def_95 = float(y_pred_original[0, 0])
        kappa_att_95 = float(y_pred_original[0, 1])
        
        # Ensure non-negative (model might predict small negative values)
        kappa_def_95 = max(0.0, kappa_def_95)
        kappa_att_95 = max(0.0, kappa_att_95)
        
        return kappa_def_95, kappa_att_95


# ===================== STANDALONE TEST =====================

if __name__ == '__main__':
    """Test the predictor with sample configurations"""
    
    predictor = KappaPredictorWithPhysics(model_dir='models_weighted_v2')
    
    print("\n" + "="*70)
    print("🧪 Testing Stratified Model (v2)")
    print("="*70)
    
    # Test 1: Low epsilon, high alpha (should predict LOW kappa)
    print("\n[Test 1] Low ε (1.5), High α (0.90) - Should predict LOW κ")
    epsilon_matrix = np.ones((7, 7)) * 1.5
    epsilon_matrix[0, 1:] = 1.5
    lambda_matrix = np.ones((7, 7)) * 1.5
    
    kappa_def, kappa_att = predictor.predict(
        0.125, 0.51, 0.90, 0.0, 1.0,
        epsilon_matrix, lambda_matrix
    )
    print(f"   κ_def = {kappa_def:.4f}, κ_att = {kappa_att:.4f}")
    print(f"   {'✅ PASS' if kappa_def < 0.5 else '❌ FAIL - Too high!'}")
    
    # Test 2: High epsilon, low alpha (should predict HIGH kappa)
    print("\n[Test 2] High ε (10.0), Low α (0.12) - Should predict HIGH κ")
    epsilon_matrix = np.ones((7, 7)) * 1.5
    epsilon_matrix[0, 1:] = 10.0
    lambda_matrix = np.ones((7, 7)) * 1.5
    
    kappa_def, kappa_att = predictor.predict(
        0.125, 0.51, 0.12, 0.0, 1.0,
        epsilon_matrix, lambda_matrix
    )
    print(f"   κ_def = {kappa_def:.4f}, κ_att = {kappa_att:.4f}")
    print(f"   {'✅ PASS' if kappa_def > 1.0 else '❌ FAIL - Too low!'}")
    
    print("\n" + "="*70)
