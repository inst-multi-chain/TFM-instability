#!/usr/bin/env python3
"""
Batch training data generation
"""

import argparse
import os
import pandas as pd
from data_generator import generate_single_training_sample, generate_boundary_training_sample
import time

def generate_training_dataset(num_samples: int, output_csv: str, 
                              checkpoint_interval: int = 50, 
                              boundary_mode: bool = False):
    """
    Generate training dataset
    
    Args:
        num_samples: Number of samples to generate
        output_csv: Output CSV path
        checkpoint_interval: Save checkpoint every N samples (default: 50)
        boundary_mode: If True, generate epsilon=16, lambda=16 samples
    """
    mode_str = "BOUNDARY (epsilon=16, lambda=16)" if boundary_mode else "NORMAL (random elasticity)"
    print("\n" + "="*70)
    print(f"Generating {num_samples} training samples - {mode_str}")
    print(f"Checkpoint interval: every {checkpoint_interval} samples")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sharded_system_dir = os.path.join(script_dir, '..')
    output_dir = os.path.dirname(output_csv) or '.'
    
    samples = []
    failed_count = 0
    start_time = time.time()
    sample_id = 0
    
    # Choose generator function based on mode
    generator_func = generate_boundary_training_sample if boundary_mode else generate_single_training_sample
    
    while len(samples) < num_samples:
        sample = generator_func(sample_id, sharded_system_dir)
        
        if sample is not None:
            samples.append(sample)
            
            # Print progress every 10 samples
            if len(samples) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / len(samples)
                eta = avg_time * (num_samples - len(samples))
                print(f"Progress: {len(samples)}/{num_samples} | "
                      f"Avg: {avg_time:.1f}s/sample | "
                      f"ETA: {eta/60:.1f}min | "
                      f"Failures: {failed_count}")
            
            # Save checkpoint every checkpoint_interval samples to same CSV
            if len(samples) % checkpoint_interval == 0:
                df_checkpoint = pd.DataFrame(samples)
                df_checkpoint.to_csv(output_csv, index=False)
                print(f"💾 Checkpoint saved to {output_csv} ({len(samples)}/{num_samples})")
        else:
            failed_count += 1
            if failed_count % 10 == 0:
                print(f"⚠️  Failed samples: {failed_count}")
        
        sample_id += 1
        
        # Safety limit
        if sample_id > num_samples * 3:
            print(f"\n❌ Exceeded max attempts ({num_samples * 3})")
            break
    
    elapsed_time = time.time() - start_time
    
    # Save final CSV
    if len(samples) > 0:
        df = pd.DataFrame(samples)
        df.to_csv(output_csv, index=False)
        
        print("\n" + "="*70)
        print("Dataset Generation Summary")
        print("="*70)
        print(f"✅ Generated: {len(samples)} samples")
        print(f"❌ Failed: {failed_count}")
        print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
        print(f"   Avg time/sample: {elapsed_time/len(samples):.1f} seconds")
        print(f"💾 Final output: {output_csv}")
        print(f"📊 Shape: {df.shape}")
        print("\nFeature Statistics:")
        print(df.describe())
        print("="*70)
    else:
        print("\n❌ No samples generated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training dataset')
    parser.add_argument('--num-samples', type=int, default=10000, 
                       help='Number of samples (default: 10000)')
    parser.add_argument('--output', type=str, default='training_data_10k.csv',
                       help='Output CSV file (default: training_data_10k.csv)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Checkpoint interval (default: 50)')
    parser.add_argument('--boundary', action='store_true',
                       help='Generate boundary samples with epsilon=16, lambda=16')
    
    args = parser.parse_args()
    
    generate_training_dataset(
        num_samples=args.num_samples,
        output_csv=args.output,
        checkpoint_interval=args.checkpoint_interval,
        boundary_mode=args.boundary
    )

