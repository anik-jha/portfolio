"""
Generate images for the Quantile Regression blog series.
Run this script to create all required visualizations.

Requirements:
    pip install matplotlib numpy scipy scikit-learn
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for consistent, professional plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Ensure we're in the assets directory
output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)


def generate_ols_vs_qr_median():
    """Blog 1: Comparison of OLS vs QR median regression"""
    np.random.seed(42)
    
    # Generate heteroscedastic data
    n = 200
    X = np.linspace(0, 10, n)
    noise = np.random.normal(0, 1 + 0.3 * X)
    y = 2 + 1.2 * X + noise
    
    # Add outliers
    n_outliers = 8
    outlier_idx = np.random.choice(n, n_outliers, replace=False)
    y[outlier_idx] += np.random.normal(8, 2, n_outliers)
    
    # Fit OLS (mean regression)
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression()
    ols.fit(X.reshape(-1, 1), y)
    y_pred_ols = ols.predict(X.reshape(-1, 1))
    
    # Simple median regression (approximate with statsmodels would be better)
    # For visualization, we'll use a robust line
    from scipy.stats import theilslopes
    median_slope, median_intercept, _, _ = theilslopes(y, X)
    y_pred_median = median_intercept + median_slope * X
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # OLS plot
    ax1.scatter(X, y, alpha=0.6, s=30, color='steelblue', label='Data')
    ax1.plot(X, y_pred_ols, color='red', lw=2.5, label='OLS (Mean)', linestyle='--')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('OLS: Pulled by Outliers')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # QR Median plot
    ax2.scatter(X, y, alpha=0.6, s=30, color='steelblue', label='Data')
    ax2.plot(X, y_pred_median, color='green', lw=2.5, label='Median Regression', linestyle='-')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('QR Median: Robust to Outliers')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ols_vs_qr_median.png', bbox_inches='tight')
    print('✓ Generated: ols_vs_qr_median.png')
    plt.close()


def generate_qr_lines():
    """Blog 1: Multiple quantile regression lines"""
    np.random.seed(42)
    
    # Generate heteroscedastic data
    n = 200
    X = np.linspace(0, 10, n)
    noise = np.random.normal(0, 1 + 0.3 * X)
    y = 2 + 1.2 * X + noise
    
    # Approximate quantile lines (simple percentile-based for visualization)
    X_sorted_idx = np.argsort(X)
    X_sorted = X[X_sorted_idx]
    y_sorted = y[X_sorted_idx]
    
    # Calculate running quantiles
    window = 40
    quantiles = [0.1, 0.5, 0.9]
    colors = {0.1: '#3498db', 0.5: '#2ecc71', 0.9: '#e74c3c'}
    
    plt.figure(figsize=(12, 7))
    plt.scatter(X, y, alpha=0.4, s=30, color='gray', label='Data')
    
    for q in quantiles:
        q_line = []
        x_line = []
        for i in range(window, len(X_sorted) - window):
            window_data = y_sorted[i-window:i+window]
            q_val = np.percentile(window_data, q * 100)
            q_line.append(q_val)
            x_line.append(X_sorted[i])
        
        plt.plot(x_line, q_line, color=colors[q], lw=2.5, 
                label=f'QR(τ={q})', linestyle='-')
    
    # Add annotation
    plt.annotate('Lines diverge\n(heteroscedasticity)', 
                xy=(8, 15), xytext=(6, 17),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Quantile Regression Lines (10th, 50th, 90th Percentiles)', fontsize=14)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'qr_lines.png', bbox_inches='tight')
    print('✓ Generated: qr_lines.png')
    plt.close()


def generate_ols_vs_qr_flowchart():
    """Blog 1: Decision flowchart for OLS vs QR"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Define boxes and arrows
    boxes = [
        {'text': 'Start:\nNeed to model Y ~ X', 'pos': (0.5, 0.95), 'color': '#34495e'},
        {'text': 'Is variance constant\nacross X values?', 'pos': (0.5, 0.82), 'color': '#3498db'},
        {'text': 'Do you only care\nabout the mean?', 'pos': (0.25, 0.68), 'color': '#3498db'},
        {'text': 'Do you need\nprediction intervals?', 'pos': (0.75, 0.68), 'color': '#3498db'},
        {'text': 'Use OLS', 'pos': (0.15, 0.54), 'color': '#2ecc71'},
        {'text': 'Are outliers\nimportant/common?', 'pos': (0.35, 0.54), 'color': '#3498db'},
        {'text': 'Use Quantile\nRegression', 'pos': (0.75, 0.54), 'color': '#e74c3c'},
        {'text': 'Use OLS', 'pos': (0.25, 0.40), 'color': '#2ecc71'},
        {'text': 'Use QR\n(robust to outliers)', 'pos': (0.45, 0.40), 'color': '#e74c3c'},
        {'text': 'Consider:\n• Asymmetric costs?\n• Tail events matter?\n• Risk assessment?', 'pos': (0.5, 0.22), 'color': '#f39c12'},
        {'text': 'If YES to any:\nUse Quantile Regression', 'pos': (0.5, 0.08), 'color': '#e74c3c'},
    ]
    
    for box in boxes:
        bbox_props = dict(boxstyle='round,pad=0.8', facecolor=box['color'], 
                         edgecolor='white', linewidth=2, alpha=0.9)
        ax.text(box['pos'][0], box['pos'][1], box['text'], 
               ha='center', va='center', fontsize=10, color='white',
               weight='bold', bbox=bbox_props, wrap=True)
    
    # Add arrows with labels
    arrows = [
        {'start': (0.5, 0.92), 'end': (0.5, 0.86), 'label': ''},
        {'start': (0.4, 0.78), 'end': (0.25, 0.72), 'label': 'YES'},
        {'start': (0.6, 0.78), 'end': (0.75, 0.72), 'label': 'NO'},
        {'start': (0.25, 0.64), 'end': (0.15, 0.58), 'label': 'YES'},
        {'start': (0.28, 0.64), 'end': (0.35, 0.58), 'label': 'NO'},
        {'start': (0.75, 0.64), 'end': (0.75, 0.58), 'label': 'YES'},
        {'start': (0.35, 0.50), 'end': (0.25, 0.44), 'label': 'NO'},
        {'start': (0.38, 0.50), 'end': (0.45, 0.44), 'label': 'YES'},
    ]
    
    for arrow in arrows:
        ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='white'))
        if arrow['label']:
            mid_x = (arrow['start'][0] + arrow['end'][0]) / 2
            mid_y = (arrow['start'][1] + arrow['end'][1]) / 2
            ax.text(mid_x + 0.02, mid_y, arrow['label'], fontsize=9, 
                   color='white', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='#7f8c8d', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title('Decision Flow: OLS vs Quantile Regression', fontsize=16, 
             weight='bold', color='#2c3e50', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'ols_vs_qr_flow_detailed.png', bbox_inches='tight', 
               facecolor='#ecf0f1')
    print('✓ Generated: ols_vs_qr_flow_detailed.png')
    plt.close()


def generate_pinball_loss_shapes():
    """Blog 2: Pinball loss function shapes"""
    u = np.linspace(-3, 3, 1000)
    quantiles = [0.1, 0.5, 0.9]
    colors = {'0.1': '#e74c3c', '0.5': '#3498db', '0.9': '#2ecc71'}
    
    def pinball_loss(u, tau):
        return np.where(u >= 0, tau * u, (tau - 1) * u)
    
    plt.figure(figsize=(12, 7))
    
    for tau in quantiles:
        loss = pinball_loss(u, tau)
        label = f'τ={tau}'
        if tau == 0.1:
            label += ' (10th percentile)'
        elif tau == 0.5:
            label += ' (Median)'
        elif tau == 0.9:
            label += ' (90th percentile)'
        
        plt.plot(u, loss, lw=2.5, label=label, color=colors[str(tau)])
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add annotations
    plt.annotate('Steep slope\n(penalize under-predictions)', 
                xy=(1.5, 1.35), xytext=(2, 2),
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
                fontsize=10, color='#2ecc71', fontweight='bold')
    
    plt.annotate('Gentle slope\n(tolerate over-predictions)', 
                xy=(-1.5, 0.15), xytext=(-2.5, 0.8),
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
                fontsize=10, color='#2ecc71', fontweight='bold')
    
    plt.annotate('Symmetric\n(balanced)', 
                xy=(0, 0), xytext=(-0.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2),
                fontsize=10, color='#3498db', fontweight='bold')
    
    plt.xlabel('Residual u = y - ŷ (actual - predicted)', fontsize=12)
    plt.ylabel('Loss ρ_τ(u)', fontsize=12)
    plt.title('Pinball Loss Function for Different Quantiles', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pinball_loss_shapes.png', bbox_inches='tight')
    print('✓ Generated: pinball_loss_shapes.png')
    plt.close()


if __name__ == '__main__':
    print('\n🎨 Generating blog images...\n')
    
    try:
        generate_ols_vs_qr_median()
        generate_qr_lines()
        generate_ols_vs_qr_flowchart()
        generate_pinball_loss_shapes()
        
        print('\n✅ All images generated successfully!')
        print(f'📁 Images saved to: {output_dir.absolute()}\n')
        
    except Exception as e:
        print(f'\n❌ Error generating images: {e}')
        import traceback
        traceback.print_exc()
