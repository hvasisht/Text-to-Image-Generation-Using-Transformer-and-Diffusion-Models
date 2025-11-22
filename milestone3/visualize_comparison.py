import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    print("Creating Model & Configuration Comparison Visualizations")
    print("="*60)
    
    # Create figure with multiple comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Performance Analysis', fontsize=18, fontweight='bold')
    
    # ============================================
    # Plot 1: Metrics Comparison Across Configurations
    # ============================================
    configs = ['Baseline\n(M1)', 'Optimized\n(M2)', 'SOTA\nReference']
    fid_scores = [400, 374.47, 25]  # Lower is better
    is_scores = [4.5, 5.08, 8.0]    # Higher is better
    clip_scores = [28, 31.85, 35]   # Higher is better
    
    x = np.arange(len(configs))
    width = 0.25
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width, fid_scores, width, label='FID (÷10)', color='#C73E1D', alpha=0.8)
    bars2 = ax1.bar(x, is_scores, width, label='Inception Score', color='#2E86AB', alpha=0.8)
    bars3 = ax1.bar(x + width, clip_scores, width, label='CLIP Similarity', color='#06A77D', alpha=0.8)
    
    # Adjust FID for visualization (divide by 10)
    fid_display = [f/10 for f in fid_scores]
    for i, (f, b) in enumerate(zip(fid_display, bars1)):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1, 
                f'{fid_scores[i]:.0f}', ha='center', fontsize=9, fontweight='bold')
    for i, (s, b) in enumerate(zip(is_scores, bars2)):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, 
                f'{s:.1f}', ha='center', fontsize=9, fontweight='bold')
    for i, (c, b) in enumerate(zip(clip_scores, bars3)):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1, 
                f'{c:.1f}', ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Score (FID÷10, IS, CLIP)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 50)
    
    # ============================================
    # Plot 2: Quality vs Speed Trade-off
    # ============================================
    ax2 = axes[0, 1]
    
    # Different configurations tested
    config_names = ['CFG 3', 'CFG 5', 'CFG 7.5\n(Optimal)', 'CFG 10', 'CFG 15',
                    '10 Steps', '20 Steps\n(Optimal)', '30 Steps', '50 Steps']
    generation_times = [31.8, 30.1, 31.4, 30.0, 222.5, 17.1, 30.0, 43.5, 139.4]
    quality_estimates = [3.5, 4.0, 5.08, 5.2, 5.3, 3.8, 5.08, 5.3, 5.5]  # Estimated quality
    
    # Split into CFG and Steps
    cfg_times = generation_times[:5]
    cfg_quality = quality_estimates[:5]
    step_times = generation_times[5:]
    step_quality = quality_estimates[5:]
    
    scatter1 = ax2.scatter(cfg_times, cfg_quality, s=200, c='#2E86AB', 
                          alpha=0.7, edgecolors='black', linewidth=2, label='CFG Scale')
    scatter2 = ax2.scatter(step_times, step_quality, s=200, c='#A23B72', 
                          alpha=0.7, edgecolors='black', linewidth=2, marker='s', label='Inference Steps')
    
    # Annotate optimal point
    ax2.scatter([31.4], [5.08], s=400, c='#06A77D', marker='*', 
               edgecolors='black', linewidth=2, label='Optimal (CFG 7.5, 20 Steps)', zorder=5)
    
    ax2.set_xlabel('Generation Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Estimated Quality Score', fontsize=12, fontweight='bold')
    ax2.set_title('Quality vs Speed Trade-off', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 240)
    ax2.set_ylim(3, 6)
    
    # ============================================
    # Plot 3: Configuration Efficiency Score
    # ============================================
    ax3 = axes[1, 0]
    
    configs_eff = ['CFG 3.0', 'CFG 5.0', 'CFG 7.5\n★', 'CFG 10.0', 'CFG 15.0',
                   'Steps 10', 'Steps 20\n★', 'Steps 30', 'Steps 50']
    
    # Efficiency = Quality / Time (normalized)
    efficiency = [
        3.5/31.8, 4.0/30.1, 5.08/31.4, 5.2/30.0, 5.3/222.5,  # CFG
        3.8/17.1, 5.08/30.0, 5.3/43.5, 5.5/139.4  # Steps
    ]
    efficiency = [e * 100 for e in efficiency]  # Scale up
    
    colors_eff = ['#2E86AB']*5 + ['#A23B72']*4
    bars = ax3.barh(configs_eff, efficiency, color=colors_eff, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Highlight optimal
    bars[2].set_color('#06A77D')
    bars[2].set_alpha(0.9)
    bars[6].set_color('#06A77D')
    bars[6].set_alpha(0.9)
    
    for i, (bar, eff) in enumerate(zip(bars, efficiency)):
        ax3.text(eff + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{eff:.1f}', va='center', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Efficiency Score (Quality/Time × 100)', fontsize=12, fontweight='bold')
    ax3.set_title('Configuration Efficiency Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.axvline(x=np.mean(efficiency), color='red', linestyle='--', alpha=0.5, linewidth=2, label='Average')
    ax3.legend(loc='lower right', fontsize=10)
    
    # ============================================
    # Plot 4: Milestone Progress
    # ============================================
    ax4 = axes[1, 1]
    
    milestones = ['M1\nSetup', 'M2\nOptimize', 'M3\nEvaluate', 'M4\nFinalize']
    fid_progress = [400, 374.47, 374.47, 374.47]  # FID (lower better)
    is_progress = [4.5, 5.08, 5.08, 5.08]         # IS (higher better)
    clip_progress = [28, 31.85, 31.85, 31.85]     # CLIP (higher better)
    
    x_m = np.arange(len(milestones))
    
    # Normalize for comparison
    fid_norm = [400/f for f in fid_progress]  # Invert FID (higher=better)
    is_norm = is_progress
    clip_norm = [c/10 for c in clip_progress]  # Scale CLIP
    
    ax4.plot(x_m, fid_norm, 'o-', linewidth=3, markersize=10, label='FID (inverted)', color='#C73E1D')
    ax4.plot(x_m, is_norm, 's-', linewidth=3, markersize=10, label='Inception Score', color='#2E86AB')
    ax4.plot(x_m, clip_norm, '^-', linewidth=3, markersize=10, label='CLIP (÷10)', color='#06A77D')
    
    ax4.set_xticks(x_m)
    ax4.set_xticklabels(milestones, fontsize=11)
    ax4.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
    ax4.set_title('Project Metrics Evolution', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 6)
    
    # Add annotations
    ax4.annotate('Baseline\nEstablished', xy=(0, 1), xytext=(0, 0.5),
                fontsize=9, ha='center', color='gray', style='italic')
    ax4.annotate('Parameters\nOptimized', xy=(1, 5.08), xytext=(1, 5.5),
                fontsize=9, ha='center', color='gray', style='italic')
    ax4.annotate('Metrics\nCalculated', xy=(2, 5.08), xytext=(2, 5.5),
                fontsize=9, ha='center', color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: model_comparison.png")
    
    # ============================================
    # Create a second detailed comparison figure
    # ============================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('Detailed Performance Breakdown', fontsize=16, fontweight='bold')
    
    # Radar chart for metric comparison
    from math import pi
    
    ax_radar = axes2[0]
    categories = ['Semantic\nAlignment', 'Image\nQuality', 'Diversity', 'Speed', 'Photorealism']
    N = len(categories)
    
    # Our system scores (out of 10)
    our_scores = [9.5, 7.0, 8.5, 6.0, 4.0]  # Based on CLIP, IS, generation time, FID
    our_scores += our_scores[:1]  # Complete the circle
    
    # SOTA reference
    sota_scores = [9.8, 9.5, 9.0, 8.5, 9.0]
    sota_scores += sota_scores[:1]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax_radar = plt.subplot(121, polar=True)
    ax_radar.plot(angles, our_scores, 'o-', linewidth=2, label='Our System', color='#2E86AB')
    ax_radar.fill(angles, our_scores, alpha=0.25, color='#2E86AB')
    ax_radar.plot(angles, sota_scores, 's-', linewidth=2, label='SOTA Reference', color='#06A77D')
    ax_radar.fill(angles, sota_scores, alpha=0.15, color='#06A77D')
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=11)
    ax_radar.set_ylim(0, 10)
    ax_radar.set_yticks([2, 4, 6, 8, 10])
    ax_radar.set_title('Performance Radar Comparison', fontsize=14, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax_radar.grid(True)
    
    # Bar chart: Our system vs SOTA
    ax_bar = axes2[1]
    metrics_names = ['FID\n(lower=better)', 'Inception\nScore', 'CLIP\nSimilarity', 'Avg Gen\nTime (s)']
    our_values = [374.47, 5.08, 31.85, 35]
    sota_values = [25, 8.5, 35, 10]
    
    x_bar = np.arange(len(metrics_names))
    width_bar = 0.35
    
    bars1 = ax_bar.bar(x_bar - width_bar/2, our_values, width_bar, label='Our System', 
                      color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax_bar.bar(x_bar + width_bar/2, sota_values, width_bar, label='SOTA Reference', 
                      color='#06A77D', alpha=0.8, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax_bar.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax_bar.set_title('Quantitative Metrics: Our System vs SOTA', fontsize=14, fontweight='bold')
    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(metrics_names, fontsize=11)
    ax_bar.legend(fontsize=11)
    ax_bar.grid(True, alpha=0.3, axis='y')
    
    # Add note
    ax_bar.text(0.5, -0.15, 'Note: SOTA values are approximate industry benchmarks (DALL-E 2, Stable Diffusion XL)', 
               transform=ax_bar.transAxes, ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('detailed_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: detailed_comparison.png")
    
    print("\n" + "="*60)
    print("MODEL COMPARISON VISUALIZATIONS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("  1. model_comparison.png (4-panel comprehensive analysis)")
    print("  2. detailed_comparison.png (radar + bar comparison)")