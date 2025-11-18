import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    print("Creating Parameter Sensitivity Visualizations")
    print("="*60)

    # Data from Milestone 2 experiments
    
    # CFG Scale experiment
    cfg_scales = [3.0, 5.0, 7.5, 10.0, 15.0]
    cfg_times = [31.8, 30.1, 31.4, 30.0, 222.5]
    
    # Inference Steps experiment
    steps = [10, 20, 30, 50]
    step_times = [17.1, 30.0, 43.5, 139.4]
    
    # Scheduler experiment
    schedulers = ['PNDM', 'DDIM', 'LMS', 'Euler']
    scheduler_times = [31.9, 29.9, 38.4, 50.3]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Milestone 2: Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: CFG Scale vs Time
    axes[0, 0].plot(cfg_scales, cfg_times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('CFG Scale', fontsize=12)
    axes[0, 0].set_ylabel('Generation Time (seconds)', fontsize=12)
    axes[0, 0].set_title('CFG Scale Impact on Generation Time', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=31.4, color='green', linestyle='--', alpha=0.5, label='Optimal (CFG=7.5)')
    axes[0, 0].legend()
    
    # Plot 2: Inference Steps vs Time
    axes[0, 1].plot(steps, step_times, 's-', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('Inference Steps', fontsize=12)
    axes[0, 1].set_ylabel('Generation Time (seconds)', fontsize=12)
    axes[0, 1].set_title('Inference Steps Impact on Generation Time', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=30.0, color='green', linestyle='--', alpha=0.5, label='Optimal (20 steps)')
    axes[0, 1].legend()
    
    # Plot 3: Scheduler Comparison
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = axes[1, 0].bar(schedulers, scheduler_times, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Scheduler Type', fontsize=12)
    axes[1, 0].set_ylabel('Generation Time (seconds)', fontsize=12)
    axes[1, 0].set_title('Scheduler Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axhline(y=29.9, color='green', linestyle='--', alpha=0.5, label='Fastest (DDIM)')
    axes[1, 0].legend()
    
    # Plot 4: Metrics Summary
    metrics = ['FID\n(lower better)', 'Inception\nScore', 'CLIP\nSimilarity']
    values = [374.47, 5.08, 31.85]
    thresholds = [150, 5.0, 0.30]
    
    bars = axes[1, 1].bar(metrics, values, color=['#C73E1D', '#2E86AB', '#06A77D'], alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Final Evaluation Metrics', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add threshold lines
    for i, (val, thresh) in enumerate(zip(values, thresholds)):
        axes[1, 1].axhline(y=thresh, color='red', linestyle='--', alpha=0.3)
        axes[1, 1].text(i, thresh, f'Threshold: {thresh}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('parameter_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: parameter_analysis.png")
    
    # Create a second figure: Time efficiency comparison
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    all_configs = ['CFG 3.0', 'CFG 5.0', 'CFG 7.5\n(Optimal)', 'CFG 10.0', 'CFG 15.0',
                   'Steps 10', 'Steps 20\n(Optimal)', 'Steps 30', 'Steps 50']
    all_times = cfg_times + step_times
    colors_config = ['#2E86AB']*5 + ['#A23B72']*4
    
    bars = ax.bar(range(len(all_configs)), all_times, color=colors_config, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(all_configs)))
    ax.set_xticklabels(all_configs, rotation=45, ha='right')
    ax.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax.set_title('Configuration Impact on Generation Speed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target: ~30s')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: time_comparison.png")
    
    print("\n" + "="*60)
    print("PARAMETER ANALYSIS COMPLETE")
    print("="*60)
    print("Generated visualizations:")
    print("  1. parameter_analysis.png (4-panel analysis)")
    print("  2. time_comparison.png (generation speed comparison)")