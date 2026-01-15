import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
import matplotlib.patches as mpatches

# Read data from CSV files
critical_delta_df = pd.read_csv('critical_delta_results.csv')
update_rate_data = pd.read_csv('update_rate_data.csv')

# Build data dictionary from CSV
data = {}
for _, row in critical_delta_df.iterrows():
    name = row['config_name']
    mode = row['mode']
    
    # Get current delta from update_rate_data.csv
    current_delta_row = update_rate_data[update_rate_data['name'] == name]
    if len(current_delta_row) > 0:
        current_delta = current_delta_row.iloc[0]['update_rate']
    else:
        current_delta = 0.000075 if mode == 'Polkadot' else 0.125
    
    # Set color based on mode
    if mode == 'Polkadot':
        color = '#6B9B7D'
    elif name == 'Cosmos Hub':
        color = '#D98B8B'
    else:  # Osmosis
        color = '#C77777'
    
    data[name] = {
        'current_delta': current_delta,
        'critical_delta': row['critical_delta'],
        'inbound_ratio': row['shard0_inbound'],
        'total_ratio': row['shard0_total'],
        'color': color,
        'category': mode
    }

# Create figure
fig, ax = plt.subplots(figsize=(18, 9))

# Prepare data
names = list(data.keys())
x_pos = np.arange(len(names))
bar_width = 0.55

# Plot bars
for i, name in enumerate(names):
    current = data[name]['current_delta']
    critical = data[name]['critical_delta']
    color = data[name]['color']
    
    # Determine if safe or unsafe
    is_safe = current < critical
    
    if is_safe:
        # Safe: show current delta bar + safety margin
        # Current delta bar (solid)
        ax.bar(i, current, bar_width, color=color, alpha=0.9, 
               edgecolor='black', linewidth=2, label='_nolegend_')
        
        # Safety margin (hatched pattern on top)
        safety_margin = critical - current
        ax.bar(i, safety_margin, bar_width, bottom=current, 
               color=color, alpha=0.3, edgecolor='green', linewidth=2.5,
               hatch='///', label='_nolegend_')
        
    else:
        # Unsafe: show critical delta bar + danger extension
        # Critical delta bar (solid)
        ax.bar(i, critical, bar_width, color=color, alpha=0.5,
               edgecolor='black', linewidth=2, label='_nolegend_')
        
        # Danger zone (red cross-hatch above critical)
        danger_zone = current - critical
        ax.bar(i, danger_zone, bar_width, bottom=critical,
               color='red', alpha=0.4, edgecolor='red', linewidth=2.5,
               hatch='xxx', label='_nolegend_')
        
        # Add prominent red warning line at current delta level
        ax.hlines(current, i - bar_width/2, i + bar_width/2, 
                 colors='red', linewidth=5, linestyles='solid', zorder=10)
        
        # Add red warning marker
        ax.plot(i, current + 0.008, 'rv', markersize=15,
                markeredgecolor='darkred', markeredgewidth=2)

# Add labels for current and critical delta values
# Slight per-chain offsets to reduce collisions, plus larger critical labels
critical_label_offsets = {
    'Asset Hub': (-0.05, -0.026),
    'Bifrost': (0.05, -0.006),
    'Hydration': (-0.05, -0.023),
    'Moonbeam': (0.05, -0.036),
    'Cosmos Hub': (-0.05, -0.024),
    'Osmosis': (0.05, -0.002),
}

for i, name in enumerate(names):
    current = data[name]['current_delta']
    critical = data[name]['critical_delta']
    is_safe = current < critical
    
    # Current delta label (bottom)
    if current >= 0.01:
        current_label = f'$\\delta={current:.3f}$'
        if name in ('Cosmos Hub', 'Osmosis'):
            label_y_current = current - 0.01
        else:
            label_y_current = current * 0.3
    elif current >= 0.001:
        current_label = f'$\\delta={current:.3f}$'
        label_y_current = current + 0.0015
    else:
        mantissa, exponent = f"{current:.1e}".split("e")
        current_label = rf'$\delta={mantissa}*10^{{{int(exponent)}}}$'
        label_y_current = current + 0.002
    
    ax.text(i, label_y_current, current_label, 
            ha='center', va='bottom', fontsize=24, 
            fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # Critical delta label at bar top
    critical_label = f'$\\delta_{{max}}={critical:.3f}$'
    x_offset, y_offset = critical_label_offsets.get(name, (0.0, 0.003))
    critical_label_y = critical + y_offset
    
    ax.text(i + x_offset, critical_label_y, critical_label,
            ha='center', va='bottom', fontsize=26,
            fontweight='bold', color='darkgreen' if is_safe else 'darkred',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='lightgreen' if is_safe else 'lightcoral',
                     alpha=0.9, edgecolor='darkgreen' if is_safe else 'darkred', 
                     linewidth=2))

# Add inbound ratio and total ratio annotations above the plot
for i, name in enumerate(names):
    inbound = data[name]['inbound_ratio']
    total = data[name]['total_ratio']
    
    # Position above the plot area
    ax.text(i, 0.150, f'Total: {total:.1%}', 
            ha='center', va='center', fontsize=22, 
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#2E5090', 
                     alpha=0.9, edgecolor='black', linewidth=1.5))
    
    ax.text(i, 0.165, f'In: {inbound:.1%}', 
            ha='center', va='center', fontsize=22, 
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E07A5F', 
                     alpha=0.9, edgecolor='black', linewidth=1.5))

# Configure axes
ax.set_ylim([0, 0.175])
ax.set_ylabel('Delta Value (δ)', fontsize=28, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=25, ha='right', fontsize=24, fontweight='bold')
ax.tick_params(axis='y', labelsize=20)
ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

# Create comprehensive legend
legend_elements = [
    Patch(facecolor='#6B9B7D', edgecolor='black', linewidth=1.5, label='Polkadot Parachains'),
    Patch(facecolor='#D98B8B', edgecolor='black', linewidth=1.5, label='Cosmos Zones'),
    Patch(facecolor='gray', alpha=0.3, edgecolor='green', linewidth=2, hatch='///',
          label='Safe Zone'),
    Patch(facecolor='red', alpha=0.4, edgecolor='red', linewidth=2, hatch='xxx',
          label='Danger Zone'),
]

ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fontsize=24, framealpha=0.95, ncol=4)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('delta-safety-margin.png', dpi=300, bbox_inches='tight')
plt.savefig('delta-safety-margin.pdf', bbox_inches='tight')

print("✅ Figure saved as 'delta-safety-margin.png' and 'delta-safety-margin.pdf'")
print("\n" + "="*80)
print("SAFETY ANALYSIS SUMMARY")
print("="*80)

for name in names:
    current = data[name]['current_delta']
    critical = data[name]['critical_delta']
    is_safe = current < critical
    
    if is_safe:
        margin = critical - current
        margin_pct = (margin / critical) * 100
        status = f"✅ SAFE - Safety margin: {margin:.4f} ({margin_pct:.1f}%)"
    else:
        excess = current - critical
        excess_pct = (excess / critical) * 100
        status = f"⚠️  UNSAFE - Exceeds by: {excess:.4f} ({excess_pct:.1f}%)"
    
    print(f"{name:15s}: δ={current:.5f}, δ_max={critical:.3f} | {status}")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print("• Polkadot: All parachains are SAFE (δ≈7.5*10^-5 << δ_max)")
print("  → Extremely conservative, sacrificing dynamic pricing capability")
print("• Cosmos Hub: UNSAFE (δ=0.125 > δ_max=0.105)")
print("  → Naively adopts Ethereum's δ without considering cross-shard traffic")
print("• Osmosis: SAFE (δ=0.100 < δ_max=0.119)")
print("  → Lower δ makes it safe, but close to critical threshold")
print("="*80)

plt.show()
