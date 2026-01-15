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
from scipy.interpolate import UnivariateSpline

# Configuration
DISTRIBUTIONS = ['Spike', 'Uniform', 'Bimodal']

def load_convergence_data():
    """Load convergence data for all three delay distributions"""
    data = {}
    for dist in DISTRIBUTIONS:
        filename = f'convergence_results_delta_{dist}.csv'
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
        df['delta'], 
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


    delta_values = sorted(df['delta'].unique())
    boundary_points = []
    for delta in delta_values:
        delta_data = df[df['delta'] == delta].sort_values('alpha_inflow')
        converged_alphas = delta_data[delta_data['converged']]['alpha_inflow'].values
        diverged_alphas  = delta_data[~delta_data['converged']]['alpha_inflow'].values

        if len(converged_alphas) > 0 and len(diverged_alphas) > 0:
            max_converged_alpha = converged_alphas.max()
            min_diverged_alpha  = diverged_alphas.min()
            boundary_alpha = (max_converged_alpha + min_diverged_alpha) / 2
            boundary_points.append((delta, boundary_alpha))
        elif len(converged_alphas) > 0:
            boundary_points.append((delta, converged_alphas.max()))
        elif len(diverged_alphas) > 0:
            boundary_points.append((delta, diverged_alphas.min()))



    plotted_boundary = False
    if len(boundary_points) >= 2:
        boundary_points.sort(key=lambda t: t[0]) 
        boundary_x, boundary_y = map(np.array, zip(*boundary_points))
        
        # 应用样条平滑使边界线更顺滑
        # s参数控制平滑程度，值越大越平滑
        try:
            # 创建样条插值，s=0.001提供轻度平滑
            spline = UnivariateSpline(boundary_x, boundary_y, s=0.001, k=3)
            # 在更密集的点上评估样条
            x_smooth = np.linspace(boundary_x.min(), boundary_x.max(), 500)
            y_smooth = spline(x_smooth)
            # 确保平滑后的值在合理范围内
            y_smooth = np.clip(y_smooth, 0.01, 0.99)
            ax.plot(x_smooth, y_smooth, color='darkred', linewidth=4,
                    linestyle='-', alpha=0.9, zorder=3, label='Boundary')
            plotted_boundary = True
        except Exception as e:
            # 如果平滑失败，使用原始点
            print(f"[WARN] {distribution_name}: 平滑失败 ({e})，使用原始边界点")
            ax.plot(boundary_x, boundary_y, color='darkred', linewidth=4,
                    linestyle='-', alpha=0.9, zorder=3, label='Boundary')
            plotted_boundary = True
    else:
        print(f"[WARN] {distribution_name}: 边界点不足（{len(boundary_points)}）无法绘制边界曲线。")


    ax.set_xlabel('$\\delta$', fontsize=34, fontweight='bold')
    ax.set_ylabel('$\\sumα_{j0,in}$', fontsize=34, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.01, 0.99)  # 显示1%-99%的完整范围
    y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x_ticks = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.125]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t:.2f}" if t < 0.125 else "0.125" for t in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(y*100)}%' for y in y_ticks])
    ax.tick_params(axis='both', which='major', labelsize=27)


    # 创建colorbar，调整尺寸和位置使其与主图对齐
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('Infl', rotation=270, labelpad=35, fontsize=32, fontweight='bold')
    cbar.ax.tick_params(labelsize=25)


    infl_max = float(df['load_inflation'].max())

    cbar.ax.text(
        1.15, 0.96,
        f"{infl_max:.3f}",
        transform=cbar.ax.transAxes,
        ha='left', va='bottom',
        fontsize=27.5, fontweight='bold'
    )


    legend_elements = []
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    if plotted_boundary:
        legend_elements.append(Line2D([0], [0], color='darkred', linewidth=4, label='Boundary'))

    total_points = len(df)
    converged_df = df[df['converged']]
    diverged_df  = df[~df['converged']]

    if len(diverged_df) > 0:
        diverged_delta = diverged_df.dropna(subset=['delta'])
        if len(diverged_delta) > 0:
            idx = diverged_delta['delta'].idxmin()
            min_diverged = diverged_delta.loc[idx]
            delta_safe_val = min_diverged['delta']
            shrinkage_pct = max(0.0, (1.0 - delta_safe_val / 0.125) * 100.0)
            legend_elements.append(mpatches.Patch(
                color='none',
                label=f"$\\Delta_{{safe}}$: [0, {delta_safe_val:.3f})"
            ))
            legend_elements.append(mpatches.Patch(
                color='none',
                label=f"Shrinkage: {shrinkage_pct:.1f}%"
            ))


    # Removed Std max legend item per user request

    if legend_elements:
        # Legend just above the plot, aligned to the left, avoid overlap with data
        ax.legend(handles=legend_elements, fontsize=32, loc='lower left',
                 bbox_to_anchor=(0.0, 1.02), frameon=True,
                 handlelength=1.5, handleheight=1.2, labelspacing=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_filename = f'over-correction-bubble-chart-delta-{distribution_name.lower()}.pdf'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Bubble chart saved as: {output_filename}")
    plt.show()

def analyze_bubble_metrics(df, distribution_name):
    """Analyze metrics for the bubble chart"""
    print(f"\n📊 {distribution_name} Distribution - Bubble Chart Metrics")
    print("="*60)
    
    converged_df = df[df['converged']]
    diverged_df  = df[~df['converged']]
    
    # Find the minimum diverged combination (smallest δ that diverges)
    if len(diverged_df) > 0:
        diverged_delta = diverged_df.dropna(subset=['delta'])
        if len(diverged_delta) > 0:
            min_diverged = diverged_delta.loc[diverged_delta['delta'].idxmin()]
            print(f"\n🔴 Minimum Non-Convergent Combination:")
            print(f"  δ: {min_diverged['delta']:.3f}")
            print(f"  α_j0,in: {min_diverged['alpha_inflow']:.3f} ({min_diverged['alpha_inflow']*100:.1f}%)")
            print(f"  Load Std: {min_diverged['load_std']:.6f}")
            print(f"  Infl: {min_diverged['load_inflation']:.6f}")
            delta_safe_val = min_diverged['delta']
            shrinkage_pct = max(0.0, (1.0 - delta_safe_val / 0.125) * 100.0)
            print(f"  Δ_safe: [0, {delta_safe_val:.3f})")
            print(f"  Shrinkage: {shrinkage_pct:.1f}%")
        else:
            print(f"\n✅ All combinations converged for {distribution_name} (no valid δ values in diverged set)")
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
    
    print(f"\nInfl:")
    print(f"  Range: {df['load_inflation'].min():.6f} - {df['load_inflation'].max():.6f}")
    print(f"  Overall: {df['load_inflation'].mean():.6f} ± {df['load_inflation'].std():.6f}")
    if len(converged_df) > 0:
        print(f"  Converged: {converged_df['load_inflation'].mean():.6f} ± {converged_df['load_inflation'].std():.6f}")
    if len(diverged_df) > 0:
        print(f"  Diverged: {diverged_df['load_inflation'].mean():.6f} ± {diverged_df['load_inflation'].std():.6f}")
    
    # Calculate correlation
    correlation = df['load_std'].corr(df['load_inflation'])
    print(f"\nCorrelation between Std Dev and Infl: {correlation:.4f}")

def main():
    """Main function"""
    print("🔬 Simplified Bubble Chart Visualization")
    print("Bubble Size = Load Standard Deviation, Color = Infl")
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
    print("  - over-correction-bubble-chart-delta-spike.pdf")
    print("  - over-correction-bubble-chart-delta-uniform.pdf")
    print("  - over-correction-bubble-chart-delta-bimodal.pdf")
    print("\n📊 Chart Legend:")
    print("  - X-axis: δ (base fee update rate)")
    print("  - Y-axis: α_j→i,in (alpha inflow parameter)")
    print("  - Bubble Size: Load Standard Deviation (larger = more variation)")
    print("  - Bubble Color: Infl (viridis colormap)")
    print("  - Dark red line: Convergence Boundary")
    print("  - Legend Δ_safe: [0, δ_min) and shrinkage = 1 - δ_min/0.125")

if __name__ == '__main__':
    main()
