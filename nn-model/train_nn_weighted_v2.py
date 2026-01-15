#!/usr/bin/env python3
"""
Neural Network Training with Stratified Importance Weighting (v2)

Key improvements over v1:
1. Stratified weighting: Only boost high-ε scenarios (ε > 5)
2. Conditional asymmetric loss: Heavy penalty only for high-ε underestimation
3. Physical constraint: Low-ε scenarios (ε < 3) should have low κ regardless of α

This avoids over-conservatism in low-ε, high-α scenarios which are actually stable.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import argparse
from typing import Tuple, Dict
import pickle


# ===================== CONFIGURATION =====================

NUM_SHARDS = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Scalar features (basic features only)
SCALAR_FEATURES = ['delta', 'alpha_total', 'alpha_inbound', 'alpha_outbound', 'd_max']

# Matrix features
EPSILON_FEATURES = [f'epsilon_{i}_{j}' for i in range(NUM_SHARDS) for j in range(NUM_SHARDS)]
LAMBDA_FEATURES = [f'lambda_{i}_{j}' for i in range(NUM_SHARDS) for j in range(NUM_SHARDS)]

TARGET_FEATURES = ['kappa_def_95', 'kappa_att_95']


# ===================== HELPER FUNCTIONS =====================

def compute_epsilon_avg(df: pd.DataFrame) -> pd.Series:
    """
    Compute average epsilon for stratified weighting
    """
    return df[[f'epsilon_0_{j}' for j in range(1, NUM_SHARDS)]].mean(axis=1)


# ===================== STRATIFIED SAMPLE WEIGHTING =====================

def calculate_stratified_sample_weights(
    y_train: np.ndarray, 
    epsilon_avg: np.ndarray,
    low_epsilon_threshold: float = 2.0,
    high_epsilon_threshold: float = 8.0,
    boost_factor_high: float = 50.0,
    boost_factor_medium: float = 10.0
) -> np.ndarray:
    """
    Stratified importance weighting based on epsilon only
    
    Strategy:
    - Low ε (<3): 1x (stable, no extra weighting needed)
    - Medium ε (3-5): 10x (moderately unstable)
    - High ε (>5): 50x (highly unstable, critical for safety)
    
    Args:
        y_train: Training targets (N, 2)
        epsilon_avg: Average epsilon (N,)
        low_epsilon_threshold: Threshold for low epsilon
        high_epsilon_threshold: Threshold for high epsilon
        boost_factor_high: Weight for high epsilon samples
        boost_factor_medium: Weight for medium epsilon samples
    
    Returns:
        weights: (N,) sample weights
    """
    weights = np.ones(len(epsilon_avg))
    
    # Stratify by epsilon only
    low_epsilon_mask = epsilon_avg < low_epsilon_threshold
    medium_epsilon_mask = (epsilon_avg >= low_epsilon_threshold) & (epsilon_avg < high_epsilon_threshold)
    high_epsilon_mask = epsilon_avg >= high_epsilon_threshold
    
    # Apply epsilon-based weighting
    weights[low_epsilon_mask] = 1.0
    weights[medium_epsilon_mask] = boost_factor_medium
    weights[high_epsilon_mask] = boost_factor_high
    
    print(f"\n📊 Stratified Sample Weighting (ε only):")
    print(f"   Total samples: {len(weights)}")
    print(f"\n   Epsilon-based stratification:")
    print(f"   - Low ε (<{low_epsilon_threshold}): {low_epsilon_mask.sum()} → 1x")
    print(f"   - Medium ε ({low_epsilon_threshold}-{high_epsilon_threshold}): {medium_epsilon_mask.sum()} → {boost_factor_medium}x")
    print(f"   - High ε (>{high_epsilon_threshold}): {high_epsilon_mask.sum()} → {boost_factor_high}x")
    print(f"\n   Effective dataset size: {weights.sum():.0f} (vs actual {len(weights)})")
    
    return weights


# ===================== ENVELOPE LOSS (Zero-Tolerance for Underestimation) =====================

class EnvelopeLoss(nn.Module):
    """
    Envelope Loss: Create a safety envelope that floats above all data points
    
    Philosophy:
    - Underestimation (pred < target): DEADLY → Penalty 20x
    - Overestimation (pred >= target): ACCEPTABLE → Penalty 1x (normal MSE)
    
    Balanced approach: Punish underestimation more, but don't ignore overestimation completely.
    """
    
    def __init__(self, 
                 underestimate_penalty: float = 15.0,
                 overestimate_penalty: float = 2.0):
        super().__init__()
        self.underestimate_penalty = underestimate_penalty
        self.overestimate_penalty = overestimate_penalty
    
    def forward(self, pred, target, sample_weight=None, epsilon_avg=None):
        """
        Args:
            pred: (batch, 2) predictions
            target: (batch, 2) ground truth
            sample_weight: (batch,) importance weights
            epsilon_avg: (batch,) average epsilon (unused, kept for compatibility)
        """
        squared_error = (target - pred) ** 2
        
        # Asymmetric penalty: Crush underestimation, ignore overestimation
        underestimate_mask = (pred < target).float()
        penalty = torch.where(
            underestimate_mask.bool(),
            torch.tensor(self.underestimate_penalty, device=pred.device),
            torch.tensor(self.overestimate_penalty, device=pred.device)
        )
        
        weighted_error = penalty * squared_error
        
        # Apply sample weights
        if sample_weight is not None:
            sample_weight = sample_weight.unsqueeze(1)
            weighted_error = sample_weight * weighted_error
        
        return weighted_error.mean()


# ===================== CONDITIONAL ASYMMETRIC LOSS (LEGACY - Keep for reference) =====================

class ConditionalAsymmetricMSELoss(nn.Module):
    """
    Conditional asymmetric MSE: penalty varies by epsilon regime
    
    For low ε (< 3): Symmetric loss (conservative predictions not needed)
    For high ε (> 5): Strong asymmetric penalty (safety-critical)
    For medium ε: Moderate asymmetric penalty
    """
    
    def __init__(self, 
                 low_epsilon_threshold: float = 2.0,
                 high_epsilon_threshold: float = 8.0,
                 high_epsilon_penalty: float = 30.0,
                 medium_epsilon_penalty: float = 10.0):
        super().__init__()
        self.low_epsilon_threshold = low_epsilon_threshold
        self.high_epsilon_threshold = high_epsilon_threshold
        self.high_epsilon_penalty = high_epsilon_penalty
        self.medium_epsilon_penalty = medium_epsilon_penalty
    
    def forward(self, pred, target, sample_weight=None, epsilon_avg=None):
        """
        Args:
            pred: (batch, 2) predictions
            target: (batch, 2) ground truth
            sample_weight: (batch,) importance weights
            epsilon_avg: (batch,) average epsilon for conditional penalty
        """
        squared_error = (target - pred) ** 2
        
        if epsilon_avg is not None:
            # Stratified penalty
            underestimate_mask = (pred < target).float()
            
            # Initialize penalty as 1.0 (symmetric)
            penalty = torch.ones_like(underestimate_mask)
            
            # Low ε: Symmetric (no extra penalty)
            low_epsilon_mask = epsilon_avg.unsqueeze(1) < self.low_epsilon_threshold
            
            # Medium ε: Moderate penalty
            medium_epsilon_mask = (
                (epsilon_avg.unsqueeze(1) >= self.low_epsilon_threshold) &
                (epsilon_avg.unsqueeze(1) < self.high_epsilon_threshold)
            )
            penalty = torch.where(
                medium_epsilon_mask & underestimate_mask.bool(),
                torch.tensor(self.medium_epsilon_penalty, device=pred.device),
                penalty
            )
            
            # High ε: Strong penalty
            high_epsilon_mask = epsilon_avg.unsqueeze(1) >= self.high_epsilon_threshold
            penalty = torch.where(
                high_epsilon_mask & underestimate_mask.bool(),
                torch.tensor(self.high_epsilon_penalty, device=pred.device),
                penalty
            )
        else:
            # Fallback: standard asymmetric
            underestimate_mask = (pred < target).float()
            penalty = torch.where(
                underestimate_mask.bool(),
                torch.tensor(self.high_epsilon_penalty, device=pred.device),
                torch.tensor(1.0, device=pred.device)
            )
        
        weighted_error = penalty * squared_error
        
        # Apply sample weights
        if sample_weight is not None:
            sample_weight = sample_weight.unsqueeze(1)
            weighted_error = sample_weight * weighted_error
        
        return weighted_error.mean()


# ===================== DATASET =====================

class KappaDataset(Dataset):
    """Dataset with epsilon and alpha tracking for stratified training"""
    
    def __init__(self, data_path: str, scaler_scalar=None, scaler_matrix=None, 
                 scaler_y=None, train_mode=True):
        df = pd.read_csv(data_path)
        
        # Compute epsilon_avg for stratified training
        epsilon_avg_series = compute_epsilon_avg(df)
        self.epsilon_avg = epsilon_avg_series.values.astype(np.float32)
        self.alpha_inbound = df['alpha_inbound'].values.astype(np.float32)
        
        # Extract features
        self.X_scalar = df[SCALAR_FEATURES].values.astype(np.float32)
        epsilon_vals = df[EPSILON_FEATURES].values.astype(np.float32)
        lambda_vals = df[LAMBDA_FEATURES].values.astype(np.float32)
        self.y = df[TARGET_FEATURES].values.astype(np.float32)
        
        # CRITICAL FIX: NO Log transform! Train on linear kappa values
        # Log-transform compresses 10x difference into 1.7x, killing the signal
        # self.y = np.log1p(self.y)  # DISABLED!
        
        # Normalize
        if train_mode:
            self.scaler_scalar = StandardScaler()
            self.scaler_matrix = StandardScaler()
            self.scaler_y = StandardScaler()
            
            self.X_scalar = self.scaler_scalar.fit_transform(self.X_scalar)
            X_matrix = np.concatenate([epsilon_vals, lambda_vals], axis=1)
            X_matrix = self.scaler_matrix.fit_transform(X_matrix)
            self.y = self.scaler_y.fit_transform(self.y)
        else:
            self.scaler_scalar = scaler_scalar
            self.scaler_matrix = scaler_matrix
            self.scaler_y = scaler_y
            
            self.X_scalar = self.scaler_scalar.transform(self.X_scalar)
            X_matrix = np.concatenate([epsilon_vals, lambda_vals], axis=1)
            X_matrix = self.scaler_matrix.transform(X_matrix)
            self.y = self.scaler_y.transform(self.y)
        
        # Reshape matrices
        epsilon_reshaped = X_matrix[:, :49].reshape(-1, 1, NUM_SHARDS, NUM_SHARDS)
        lambda_reshaped = X_matrix[:, 49:].reshape(-1, 1, NUM_SHARDS, NUM_SHARDS)
        self.X_matrix = np.concatenate([epsilon_reshaped, lambda_reshaped], axis=1).astype(np.float32)
        
        print(f"Loaded {len(self)} samples")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X_scalar[idx]),
            torch.FloatTensor(self.X_matrix[idx]),
            torch.FloatTensor(self.y[idx]),
            torch.FloatTensor([self.epsilon_avg[idx]])
        )


# ===================== MODEL (Same architecture) =====================

class KappaPredictor(nn.Module):
    """Hybrid CNN + MLP model"""
    
    def __init__(self, scalar_input_dim=5, hidden_dims=[256, 128], dropout=0.4):
        super().__init__()
        
        # CNN for matrix features
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # MLP
        cnn_output_dim = 64
        mlp_input_dim = cnn_output_dim + scalar_input_dim
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(prev_dim, 2))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x_scalar, x_matrix):
        cnn_features = self.cnn(x_matrix)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        combined = torch.cat([cnn_features, x_scalar], dim=1)
        return self.mlp(combined)


# ===================== TRAINING =====================

def train_with_stratified_weighting(
    data_path: str,
    output_dir: str = 'models_weighted_v2',
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 0.0001,  # REDUCED: No log transform -> huge gradients
    low_epsilon_threshold: float = 2.0,
    high_epsilon_threshold: float = 8.0,
    boost_factor_high: float = 50.0,
    boost_factor_medium: float = 10.0,
    underestimate_penalty: float = 15.0,  # Balanced: punish underestimation but not crazy
    overestimate_penalty: float = 2.0  # Normal MSE for overestimation
):
    """
    Train model with Envelope Loss (Zero-Tolerance for Underestimation)
    
    Key changes:
    1. NO log transform - train on linear kappa values
    2. Envelope Loss - crush underestimation (100x), ignore overestimation (0.01x)
    3. Lower LR (1e-4) - gradients are huge without log compression
    4. Explicit critical_zone feature - gives model a "cheat code"
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 Training with Envelope Loss (Zero-Tolerance Underestimation)")
    print("="*70)
    print(f"ε thresholds: Low<{low_epsilon_threshold}, Med {low_epsilon_threshold}-{high_epsilon_threshold}, High>{high_epsilon_threshold}")
    print(f"Loss: Underestimate penalty={underestimate_penalty}x, Overestimate penalty={overestimate_penalty}x")
    print(f"NO LOG TRANSFORM - Training on linear kappa values!")
    print(f"Device: {DEVICE}")
    
    # Load data
    full_dataset = KappaDataset(data_path, train_mode=True)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Calculate stratified weights for training set
    train_targets = full_dataset.y[train_dataset.indices]
    train_epsilon_avg = full_dataset.epsilon_avg[train_dataset.indices]
    
    sample_weights = calculate_stratified_sample_weights(
        train_targets, train_epsilon_avg,
        low_epsilon_threshold, high_epsilon_threshold,
        boost_factor_high, boost_factor_medium
    )
    
    # Weighted sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = KappaPredictor(scalar_input_dim=len(SCALAR_FEATURES)).to(DEVICE)
    print(f"\n📊 Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Envelope Loss (Zero-Tolerance for Underestimation)
    criterion = EnvelopeLoss(
        underestimate_penalty=underestimate_penalty,
        overestimate_penalty=overestimate_penalty
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for x_scalar, x_matrix, y, epsilon_avg in train_loader:
            x_scalar = x_scalar.to(DEVICE)
            x_matrix = x_matrix.to(DEVICE)
            y = y.to(DEVICE)
            epsilon_avg = epsilon_avg.squeeze().to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(x_scalar, x_matrix)
            loss = criterion(pred, y)  # Envelope loss doesn't need epsilon_avg
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_scalar, x_matrix, y, epsilon_avg in val_loader:
                x_scalar = x_scalar.to(DEVICE)
                x_matrix = x_matrix.to(DEVICE)
                y = y.to(DEVICE)
                epsilon_avg = epsilon_avg.squeeze().to(DEVICE)
                
                pred = model(x_scalar, x_matrix)
                loss = criterion(pred, y)  # Envelope loss doesn't need epsilon_avg
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    
    # Save final
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # Save scalers
    scalers = {
        'scaler_scalar': full_dataset.scaler_scalar,
        'scaler_matrix': full_dataset.scaler_matrix,
        'scaler_y': full_dataset.scaler_y
    }
    with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"💾 Models saved to {output_dir}/")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History (Envelope Loss - Zero-Tolerance Underestimation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='training_data_clean.csv')
    parser.add_argument('--output-dir', type=str, default='models_weighted_v2')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)  # Lower LR for linear kappa
    parser.add_argument('--low-epsilon-threshold', type=float, default=1.0)
    parser.add_argument('--high-epsilon-threshold', type=float, default=10.0)
    parser.add_argument('--boost-factor-high', type=float, default=5.0)
    parser.add_argument('--boost-factor-medium', type=float, default=1.0)
    parser.add_argument('--underestimate-penalty', type=float, default=50)
    parser.add_argument('--overestimate-penalty', type=float, default=0.5)
    
    args = parser.parse_args()
    
    train_with_stratified_weighting(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        low_epsilon_threshold=args.low_epsilon_threshold,
        high_epsilon_threshold=args.high_epsilon_threshold,
        boost_factor_high=args.boost_factor_high,
        boost_factor_medium=args.boost_factor_medium,
        underestimate_penalty=args.underestimate_penalty,
        overestimate_penalty=args.overestimate_penalty
    )
