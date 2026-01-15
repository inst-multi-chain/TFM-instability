#!/bin/bash
# Retrain neural network with Envelope Loss (Zero-Tolerance for Underestimation)
# 
# Key improvements:
# - NO log transform - train on linear kappa values (10x difference preserved)
# - Envelope Loss - crush underestimation (100x), ignore overestimation (0.01x)
# - Explicit critical_zone feature - model gets a "cheat code"
# - Lower LR (1e-4) - gradients are huge without log compression

set -e

echo "=========================================="
echo "🚀 Retraining with Envelope Loss"
echo "=========================================="

# Get script directory and change to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if training data exists
TRAINING_DATA="nn-model/training_data_clean.csv"
if [ ! -f "$TRAINING_DATA" ]; then
    echo "❌ Training data not found: $TRAINING_DATA"
    exit 1
fi

NUM_SAMPLES=$(wc -l < "$TRAINING_DATA")
echo "📊 Found $NUM_SAMPLES training samples"

# Change to nn-model directory
cd nn-model

# Activate Python environment if needed
if [ -d "venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Run training with Envelope Loss
echo ""
echo "🔥 Starting training with Envelope Loss..."
echo "   Strategy:"
echo "   - Sample weights: Low ε=1x, Med ε=10x, High ε=50x"
echo "   - Loss penalty: Underestimate=20x, Overestimate=1x (balanced)"
echo "   - NO LOG TRANSFORM (linear kappa values)"
echo "   - NO critical zone feature (rely on physics features only)"
echo ""

python3 train_nn_weighted_v2.py \
    --data training_data_clean.csv \
    --output-dir models_weighted_v2 \
    --epochs 200 \
    --batch-size 256 \
    --lr 0.0001 \
    --low-epsilon-threshold 2.0 \
    --high-epsilon-threshold 8.0 \
    --boost-factor-high 50.0 \
    --boost-factor-medium 10.0 \
    --underestimate-penalty 20.0 \
    --overestimate-penalty 1.0

echo ""
echo "✅ Training complete!"
echo "📁 Model saved to: models_weighted_v2/"
echo ""
echo "Next steps:"
echo "1. Test predictions - should see HIGHER kappa in critical zone"
echo "2. Verify model is 'paranoid' (prefers overestimation)"


