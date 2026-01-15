#!/usr/bin/env python3
"""
Simplified Bubble Chart Visualization
Only the first subplot from bubble chart: size=std dev, color=inflation
Generates separate PDF files for each delay distribution.
With convergence boundary curve instead of individual markers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DISTRIBUTIONS = ['Spike', 'Uniform']

def load_convergence_data():
    """Load convergence data for all three delay distributions"""
    data = {}
    for dist in DISTRIBUTIONS:
        filename = f'convergence_results_{dist}.csv'
        try:
            df = pd.read_csv(filename)
            print(f"✅ Loaded {filename}: {len(df)} records")
            data[dist] = df
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            return None
    return data

def create_single_bubble_chart(df, distribution_name):
    """Create single bubble chart: size=std dev, color=inflation"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10.3))

    ax.set_rasterization_zorder(2)


    size_scale = 1500.0
    std_max_local = float(df['load_std'].max())

    scatter = ax.scatter(
        df['epsilon_j0'], 
        df['alpha_inflow'],
        s=df['load_std'] * size_scale,
        c=df['load_inflation'],
        alpha=0.6,
        cmap='viridis',
        edgecolors='none',      
        linewidths=0,
        rasterized=True,       
        zorder=1
    )


    epsilon_values = sorted(df['epsilon_j0'].unique())
    boundary_points = []
    for eps in epsilon_values:
        eps_data = df[df['epsilon_j0'] == eps].sort_values('alpha_inflow')
        converged_alphas = eps_data[eps_data['converged']]['alpha_inflow'].values
        diverged_alphas  = eps_data[~eps_data['converged']]['alpha_inflow'].values

        if len(converged_alphas) > 0 and len(diverged_alphas) > 0:
            max_converged_alpha = converged_alphas.max()
            min_diverged_alpha  = diverged_alphas.min()
            boundary_alpha = (max_converged_alpha + min_diverged_alpha) / 2
            boundary_points.append((eps, boundary_alpha))
        elif len(converged_alphas) > 0:
            boundary_points.append((eps, converged_alphas.max()))
        elif len(diverged_alphas) > 0:
            boundary_points.append((eps, diverged_alphas.min()))

    plotted_boundary = False
    if len(boundary_points) >= 2:
        boundary_points.sort(key=lambda t: t[0])  # 按 eps 排序
        boundary_x, boundary_y = map(np.array, zip(*boundary_points))
        
        # Apply smoothing using UnivariateSpline
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(boundary_x, boundary_y, s=0.001, k=3)
        boundary_x_smooth = np.linspace(boundary_x.min(), boundary_x.max(), 300)
        boundary_y_smooth = spline(boundary_x_smooth)
        
        ax.plot(boundary_x_smooth, boundary_y_smooth, color='darkred', linewidth=4,
                linestyle='-', alpha=0.9, zorder=3, label='Boundary')
        plotted_boundary = True
    else:
        print(f"[WARN] {distribution_name}: 边界点不足（{len(boundary_points)}）无法绘制边界曲线。")


    ax.set_xlabel('$ε_{j0}$', fontsize=34, fontweight='bold')
    ax.set_ylabel('$\sumα_{j0,in}$', fontsize=34, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 16)
    ax.set_ylim(0.01, 0.99)
    y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x_ticks = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(y*100)}%' for y in y_ticks])
    ax.tick_params(axis='both', which='major', labelsize=27)


    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Infl', rotation=270, labelpad=35, fontsize=32, fontweight='bold')
    cbar.ax.tick_params(labelsize=25)


    infl_max = float(df['load_inflation'].max())

    cbar.ax.text(
        0.5, 1.02,
        f"{infl_max:.3f}",
        transform=cbar.ax.transAxes,
        ha='center', va='bottom',
        fontsize=27.5, fontweight='bold'
    )


    legend_elements = []
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    if plotted_boundary:
        legend_elements.append(Line2D([0], [0], color='darkred', linewidth=4, label='Boundary'))

    diverged_df  = df[~df['converged']]

    if len(diverged_df) > 0:
        idx = (diverged_df[['epsilon_j0', 'alpha_inflow']].sum(axis=1)).idxmin()
        min_diverged = diverged_df.loc[idx]
        legend_elements.append(mpatches.Patch(
            color='none',
            label=f"Minimum Inst: ε={min_diverged['epsilon_j0']:.2f}, α={min_diverged['alpha_inflow']:.2f}"
        ))

    if legend_elements:
        ax.legend(handles=legend_elements, fontsize=32, loc='lower center', bbox_to_anchor=(0.46, 1))

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output_filename = f'over-correction-bubble-chart-{distribution_name.lower()}.pdf'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Bubble chart saved as: {output_filename}")
    plt.show()

def analyze_bubble_metrics(df, distribution_name):
    """Analyze metrics for the bubble chart"""
    print(f"\n📊 {distribution_name} Distribution - Bubble Chart Metrics")
    print("="*60)
    
    converged_df = df[df['converged']]
    diverged_df  = df[~df['converged']]
    
    # Find the minimum diverged combination
    if len(diverged_df) > 0:
        min_diverged = diverged_df.loc[diverged_df[['epsilon_j0', 'alpha_inflow']].sum(axis=1).idxmin()]
        print(f"\n🔴 Minimum Non-Convergent Combination:")
        print(f"  ε_j0: {min_diverged['epsilon_j0']:.3f}")
        print(f"  α_j0,in: {min_diverged['alpha_inflow']:.3f} ({min_diverged['alpha_inflow']*100:.1f}%)")
        print(f"  Load Std: {min_diverged['load_std']:.6f}")
        print(f"  Load Inflation: {min_diverged['load_inflation']:.6f}")
    else:
        print(f"\n✅ All combinations converged for {distribution_name}")
    
    print(f"\nLoad Standard Deviation:")
    print(f"  Range: {df['load_std'].min():.6f} - {df['load_std'].max():.6f}")
    print(f"  Std max (this figure): {df['load_std'].max():.6f}")  # ← 明确打印本图的 std max
    print(f"  Overall: {df['load_std'].mean():.6f} ± {df['load_std'].std():.6f}")
    if len(converged_df) > 0:
        print(f"  Converged: {converged_df['load_std'].mean():.6f} ± {converged_df['load_std'].std():.6f}")
    if len(diverged_df) > 0:
        print(f"  Diverged: {diverged_df['load_std'].mean():.6f} ± {diverged_df['load_std'].std():.6f}")
    
    print(f"\nLoad Inflation:")
    print(f"  Range: {df['load_inflation'].min():.6f} - {df['load_inflation'].max():.6f}")
    print(f"  Overall: {df['load_inflation'].mean():.6f} ± {df['load_inflation'].std():.6f}")
    if len(converged_df) > 0:
        print(f"  Converged: {converged_df['load_inflation'].mean():.6f} ± {converged_df['load_inflation'].std():.6f}")
    if len(diverged_df) > 0:
        print(f"  Diverged: {diverged_df['load_inflation'].mean():.6f} ± {diverged_df['load_inflation'].std():.6f}")
    
    # Calculate correlation
    correlation = df['load_std'].corr(df['load_inflation'])
    print(f"\nCorrelation between Std Dev and Inflation: {correlation:.4f}")

def main():
    """Main function"""
    print("🔬 Simplified Bubble Chart Visualization")
    print("Bubble Size = Load Standard Deviation, Color = Load Inflation")
    print("="*70)
    
    # Load data
    data_dict = load_convergence_data()
    if data_dict is None:
        print("❌ Data loading failed")
        return
    
    # Process each distribution separately
    for dist in DISTRIBUTIONS:
        print(f"\n🎨 Processing {dist} distribution...")
        df = data_dict[dist]
        
        # Analyze metrics
        analyze_bubble_metrics(df, dist)
        
        # Create visualization
        print(f"  Creating bubble chart...")
        create_single_bubble_chart(df, dist)
    
    print("\n✅ Bubble chart visualization complete!")
    print("\n📝 Generated Files:")
    print("  - over-correction-bubble-chart-spike.pdf")
    print("  - over-correction-bubble-chart-uniform.pdf")
    print("  - over-correction-bubble-chart-bimodal.pdf")
    print("\n📊 Chart Legend:")
    print("  - X-axis: ε_j0 (epsilon parameter)")
    print("  - Y-axis: α_j→i,in (alpha inflow parameter)")
    print("  - Bubble Size: Load Standard Deviation (larger = more variation)")
    print("  - Bubble Color: Load Inflation (viridis colormap)")
    print("  - Dark red line: Convergence Boundary")

if __name__ == '__main__':
    main()
