# Your First Quantile Regression Model: A Hands-On Python Guide

*From raw data to production-ready QR pipelines with statsmodels*

---

**Reading time: ~20 minutes** ☕️☕️

---

Theory is great. Code is better. In this post, we'll build a complete quantile regression system from scratch.

You'll learn how to:
- Fit QR models for multiple quantiles with `statsmodels`
- Interpret coefficients across the distribution
- Build and evaluate 80% prediction intervals
- Compare QR to OLS on robustness and informativeness
- Avoid common pitfalls (coverage evaluation, quantile crossing, feature scaling)

By the end, you'll have **production-ready code** you can adapt to your own datasets.

Let's dive in. 🚀

---

## The Game Plan

We'll walk through a complete workflow:

1. **Generate data** with heteroscedasticity and outliers (to showcase QR's strengths)
2. **Fit OLS** as a baseline
3. **Fit QR** for τ ∈ {0.1, 0.5, 0.9}
4. **Visualize** quantile lines and interpret heterogeneous effects
5. **Build prediction intervals** (80% coverage: 10th to 90th percentile)
6. **Evaluate** with pinball loss and coverage metrics
7. **Real-world example**: Bike-sharing demand forecasting
8. **Common pitfalls** and how to avoid them

Grab coffee. Fire up Jupyter. Let's build. ☕️

---

## Setup: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn
```

**💡 Installation Troubleshooting**:
- If `statsmodels` fails on Mac M1/M2, try: `conda install statsmodels` (conda handles ARM binaries better)
- If you get "Microsoft Visual C++ required" on Windows, install it from [Microsoft's site](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Still stuck? Create a fresh virtual environment:
  - macOS/Linux: `python -m venv qr_env && source qr_env/bin/activate`
  - Windows: `python -m venv qr_env && qr_env\Scripts\activate`

---

## Step 1: Generate Synthetic Data

We'll simulate a scenario where:
- Mean relationship is linear: $y = 2 + 1.2x + \epsilon$
- Variance grows with $x$ (heteroscedasticity)
- A few outliers exist (high leverage points)

This mirrors real data: income vs. education, house prices vs. square footage, server latency vs. load.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
np.random.seed(42)

# Generate base data
n = 200
X = np.linspace(0, 10, n)
noise = np.random.normal(0, 1 + 0.2 * X)  # Variance grows with X
y = 2 + 1.2 * X + noise

# Add outliers (8 high-y points)
n_outliers = 8
outlier_x = np.random.uniform(2, 8, n_outliers)
outlier_y = 2 + 1.2 * outlier_x + np.random.normal(12, 2, n_outliers)

# Combine
X_full = np.concatenate([X, outlier_x])
y_full = np.concatenate([y, outlier_y])
df = pd.DataFrame({'x': X_full, 'y': y_full})

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], alpha=0.6, s=30, color='steelblue')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Synthetic Data: Heteroscedastic + Outliers', fontsize=14)
plt.tight_layout()
plt.show()
```

**What you'll see**: A "funnel" shape—points spread out as $x$ increases. Plus a few extreme outliers pulling upward.

---

## Step 2: Fit OLS (Baseline)

Let's see how ordinary least squares handles this.

```python
from sklearn.linear_model import LinearRegression

X_train = df[['x']].values
y_train = df['y'].values

# Fit OLS
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_train)

print(f"OLS: β₀={ols.intercept_:.3f}, β₁={ols.coef_[0]:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], alpha=0.6, s=30, label='Data', color='steelblue')
plt.plot(df['x'], y_pred_ols, color='red', lw=2.5, label='OLS', linestyle='--')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('OLS Fit (Pulled by Outliers)', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

**Output**:
```
OLS: β₀=2.019, β₁=1.295
```

**Observation**: The line is pulled *upward* by outliers. It doesn't represent the "typical" relationship well. No uncertainty quantification.

---

## Step 3: Fit Quantile Regression for Multiple τ

Now fit QR for three quantiles: **0.1** (10th percentile), **0.5** (median), **0.9** (90th percentile).

We'll use `statsmodels.regression.quantile_regression.QuantReg`.

```python
import statsmodels.api as sm

# Add intercept (statsmodels requires explicit constant term, unlike sklearn)
X_with_const = sm.add_constant(X_train)  # Adds column of 1's for β₀

# Fit quantiles
quantiles = [0.1, 0.5, 0.9]
models = {}
predictions = {}

for q in quantiles:
    # QuantReg: y ~ X, optimize for quantile q
    model = sm.QuantReg(y_train, X_with_const)
    result = model.fit(q=q)  # Uses linear programming under the hood
    models[q] = result
    predictions[q] = result.predict(X_with_const)
    print(f"QR(τ={q}): β₀={result.params[0]:.3f}, β₁={result.params[1]:.3f}")
```

**Output**:
```
QR(τ=0.1): β₀=0.502, β₁=1.007
QR(τ=0.5): β₀=1.734, β₁=1.280
QR(τ=0.9): β₀=3.215, β₁=1.559
```

**Key insight**: Notice how the **intercept increases** and the **slope changes** as τ increases. This reflects:
- Lower quantiles (10th): Conservative predictions (low intercept)
- Upper quantiles (90th): Optimistic predictions (high intercept)
- Slope variation: The effect of $X$ on $Y$ differs across the distribution

This is **heterogeneous treatment effects**—the impact of $X$ is not constant!

---

## Step 4: Visualize All Quantile Lines

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], alpha=0.4, s=30, label='Data', color='gray')
plt.plot(df['x'], y_pred_ols, color='red', lw=2.5, label='OLS', linestyle='--')

colors = {0.1: 'blue', 0.5: 'green', 0.9: 'orange'}
for q in quantiles:
    plt.plot(df['x'], predictions[q], color=colors[q], lw=2.5, 
             label=f'QR(τ={q})', linestyle='-')

# Annotate the key insight
plt.annotate('Lines diverge here\n(heteroscedasticity)', 
             xy=(8, 15), xytext=(6, 18),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=11, color='red')

plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('OLS vs. Quantile Regression', fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.show()
```

**What you'll see**:
- **QR lines diverge** as $x$ increases → captures heteroscedasticity
- **OLS sits between** 50th and 90th percentiles (pulled by outliers)
- **10th percentile** hugs the lower edge of the data
- **90th percentile** hugs the upper edge (including outliers)

The **shaded region between 10th and 90th** = 80% prediction interval. This is *information*—not just a line.

---

## Step 5: Build and Evaluate Prediction Intervals

An 80% prediction interval: [10th percentile, 90th percentile].

For each $x$, we predict:
- Lower bound: $\hat{Q}_{0.1}(Y \mid x)$
- Upper bound: $\hat{Q}_{0.9}(Y \mid x)$

We expect **80% of observations** to fall within this interval.

### Compute Coverage

```python
# Predictions
lower = predictions[0.1]
upper = predictions[0.9]

# Check coverage
in_interval = (y_train >= lower) & (y_train <= upper)
coverage = in_interval.mean()

print(f"Empirical coverage: {coverage:.2%} (target: 80%)")
```

**Output**:
```
Empirical coverage: 78.85% (target: 80%)
```

**Great!** Close to the nominal 80%. This means our QR model is **well-calibrated**.

### Visualize Intervals

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], alpha=0.4, s=30, color='gray', label='Data')
plt.fill_between(df['x'], lower, upper, alpha=0.3, color='skyblue', label='80% PI')
plt.plot(df['x'], predictions[0.5], color='green', lw=2.5, label='Median (τ=0.5)')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('80% Prediction Interval (10th to 90th Percentile)', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

**What you'll see**: A widening band around the median—narrow at low $x$, wide at high $x$. This is **adaptive uncertainty**.

---

## Step 6: Evaluate with Pinball Loss

The pinball loss measures how well predictions minimize the quantile-specific loss.

```python
def pinball_loss(y_true, y_pred, tau):
    residual = y_true - y_pred
    return np.where(residual >= 0, tau * residual, (tau - 1) * residual).mean()

# Compute losses
for q in quantiles:
    loss = pinball_loss(y_train, predictions[q], q)
    print(f"Pinball loss (τ={q}): {loss:.4f}")

# Compare to OLS (using MSE for reference, not direct comparison)
# Note: MSE and pinball loss have different scales/units
mse_ols = ((y_train - y_pred_ols) ** 2).mean()
print(f"\nOLS MSE (for reference): {mse_ols:.4f}")
print("Note: MSE and pinball loss aren't directly comparable (different loss functions)")
```

**Output**:
```
Pinball loss (τ=0.1): 0.3585
Pinball loss (τ=0.5): 0.9694
Pinball loss (τ=0.9): 0.7152

OLS MSE (for reference): 11.0148
Note: MSE and pinball loss aren't directly comparable (different loss functions)
```

**Interpretation**: 
- **Lower pinball loss = better quantile fit** for that specific τ
- **Compare pinball losses across quantiles** to see which is hardest to predict
- **MSE is shown for reference only**—it measures mean squared error (different objective)
- To fairly compare OLS vs. QR, evaluate both on the **same metric** (e.g., both on pinball loss for τ=0.5)

---

## Step 7: Interpret Coefficients Across Quantiles

Let's extract and compare slopes across τ:

```python
# Extract coefficients
coefs = pd.DataFrame({
    'tau': quantiles,
    'intercept': [models[q].params[0] for q in quantiles],
    'slope': [models[q].params[1] for q in quantiles]
})

print(coefs)
```

**Output**:
```
   tau  intercept     slope
0  0.1   0.502408  1.006868
1  0.5   1.734078  1.279678
2  0.9   3.215210  1.559328
```

**Interpretation**:
- **Intercept**: Baseline $Y$ when $X=0$. Increases from 0.50 (10th) to 3.22 (90th) → higher quantiles have higher baseline.
- **Slope**: For each unit increase in $X$:
  - At the 10th percentile: $Y$ increases by 1.01
  - At the median: $Y$ increases by 1.28
  - At the 90th percentile: $Y$ increases by 1.56

**Insight**: The effect of $X$ is **stronger at the upper tail**. This is common in:
- Income inequality (education's impact is larger for high earners)
- House prices (square footage matters more for luxury homes)
- Server performance (load impacts tail latency more than median)

This is **distributional heterogeneity**—something OLS *cannot* capture.

---

## Step 8: Compare QR to OLS on Robustness

Let's see how OLS and median regression (τ=0.5) respond to outliers.

### Remove Outliers and Refit

```python
# Identify outliers (simple heuristic: y > mean + 3*std)
df['is_outlier'] = df['y'] > (df['y'].mean() + 3 * df['y'].std())
df_clean = df[~df['is_outlier']]

print(f"Removed {df['is_outlier'].sum()} outliers")

# Refit OLS
X_clean = df_clean[['x']].values
y_clean = df_clean['y'].values
ols_clean = LinearRegression()
ols_clean.fit(X_clean, y_clean)

# Refit QR (median)
X_clean_const = sm.add_constant(X_clean)
qr_clean = sm.QuantReg(y_clean, X_clean_const).fit(q=0.5)

print(f"\nWith outliers:")
print(f"  OLS: β₁={ols.coef_[0]:.3f}")
print(f"  QR(0.5): β₁={models[0.5].params[1]:.3f}")

print(f"\nWithout outliers:")
print(f"  OLS: β₁={ols_clean.coef_[0]:.3f} (changed by {abs(ols_clean.coef_[0] - ols.coef_[0]):.3f})")
print(f"  QR(0.5): β₁={qr_clean.params[1]:.3f} (changed by {abs(qr_clean.params[1] - models[0.5].params[1]):.3f})")
```

**Output**:
```
Removed 4 outliers

With outliers:
  OLS: β₁=1.295
  QR(0.5): β₁=1.280

Without outliers:
  OLS: β₁=1.238 (changed by 0.057)
  QR(0.5): β₁=1.273 (changed by 0.007)
```

**Insight**: Median regression is **~10× more robust** than OLS. The slope barely changes when outliers are removed. OLS shifts significantly.

---

## Real-World Example: Bike-Sharing Demand

Let's apply QR to a realistic bike-sharing demand forecasting scenario inspired by [Capital Bikeshare](https://www.capitalbikeshare.com/) data patterns.

> **📝 Note**: For tutorial clarity, we're using **simulated data** that mimics real bike-sharing patterns (heteroscedasticity, rush hour effects, weather dependence). The actual Capital Bikeshare dataset is available [here](https://www.capitalbikeshare.com/system-data) if you want to apply these techniques to real data.

**Scenario**: Predict hourly bike rentals based on temperature, hour, season. We want:
- **Median** (typical demand) for reporting
- **90th percentile** (surge demand) for rebalancing trucks
- **10th percentile** (low demand) for maintenance scheduling

### Load Data (Simulated Mini-Version)

```python
# Simulate bike demand data
np.random.seed(123)
n = 500
hour = np.random.randint(0, 24, n)
temp = np.random.uniform(0, 35, n)  # Celsius
is_weekend = np.random.binomial(1, 0.3, n)

# Demand model (heteroscedastic)
demand = (
    50 
    + 3 * temp 
    + 10 * (hour >= 7) * (hour <= 9)  # Morning rush
    + 15 * (hour >= 17) * (hour <= 19)  # Evening rush
    + 20 * is_weekend  # Weekends higher
    + np.random.normal(0, 5 + 0.5 * temp)  # Variance grows with temp
)
demand = np.maximum(demand, 0)  # No negative rentals

df_bikes = pd.DataFrame({
    'temp': temp,
    'hour': hour,
    'is_weekend': is_weekend,
    'demand': demand
})

# Split train/test
from sklearn.model_selection import train_test_split

# 80/20 split is standard for n > 500. For smaller data (n < 200), use 70/30 or cross-validation.
train, test = train_test_split(df_bikes, test_size=0.2, random_state=42)
```

### Fit Multi-Quantile Model

```python
X_train = train[['temp', 'hour', 'is_weekend']].values
y_train = train['demand'].values
X_test = test[['temp', 'hour', 'is_weekend']].values
y_test = test['demand'].values

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit quantiles
quantiles_bike = [0.1, 0.5, 0.9]
models_bike = {}
preds_test = {}

for q in quantiles_bike:
    model = sm.QuantReg(y_train, X_train_const).fit(q=q)
    models_bike[q] = model
    preds_test[q] = model.predict(X_test_const)
    print(f"τ={q} | β_temp={model.params[1]:.2f}, β_hour={model.params[2]:.2f}, β_weekend={model.params[3]:.2f}")
```

**Output**:
```
τ=0.1 | β_temp=2.67, β_hour=0.28, β_weekend=23.27
τ=0.5 | β_temp=3.08, β_hour=0.07, β_weekend=21.48
τ=0.9 | β_temp=3.59, β_hour=0.22, β_weekend=21.30
```

**Insight**:
- **Temperature effect grows** at higher quantiles (3.59 at 90th vs. 2.67 at 10th) → hot days cause *surges*, not just increases
- **Weekend effect is strong and consistent** (around 21-23) across quantiles → weekend demand is elevated but stable
- **Hour effect is small but positive** → rush hour matters, but less than temp/weekend

### Evaluate Coverage on Test Set

```python
lower_test = preds_test[0.1]
upper_test = preds_test[0.9]
in_interval_test = (y_test >= lower_test) & (y_test <= upper_test)
coverage_test = in_interval_test.mean()

print(f"Test set coverage (80% PI): {coverage_test:.2%}")
```

**Output**:
```
Test set coverage (80% PI): 75.00%
```

Close to 80%! Slightly under, but within statistical noise.

---

## Common Pitfalls and How to Avoid Them

### 0. Insufficient Sample Size

**Rule of thumb**: For reliable quantile estimation, you need **at least 30-50 observations per quantile** you're modeling.

- **Example**: Modeling τ ∈ {0.1, 0.5, 0.9} with 3 quantiles? You need n ≥ 150-250.
- **Why**: Extreme quantiles (0.05, 0.95) require even more data—you're estimating tails where data is sparse.
- **What if you have less?** Consider:
  - Reducing the number of quantiles (just use 0.25, 0.5, 0.75)
  - Using cross-validation to assess stability
  - Collecting more data before deploying to production

> **💡 Quick check**: Run bootstrap resampling (200 iterations). If coefficient estimates vary wildly (CV > 50%), you likely need more data.

### 1. Evaluating Coverage on Training Data

**Wrong**:
```python
# Don't do this!
coverage_train = ((y_train >= lower_train) & (y_train <= upper_train)).mean()
```

**Why**: Training set coverage is always optimistic (overfitting). Always evaluate on a **held-out test set**.

### 2. Forgetting to Scale Features

If features have very different scales (e.g., temperature in [0, 35] and population in [0, 1M]), use standardization:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**⚙️ Computational Note**: `statsmodels` QR uses linear programming (interior point method), which is:
- **Slower than OLS** (roughly 5-10× for moderate datasets)
- **Scales well to n ≈ 100k** (but gets slow beyond that)
- **Not parallelized** (fit each quantile sequentially)

For very large datasets (n > 1M), consider:
- **Sampling** for exploratory analysis
- **LightGBM** (Blog 4) for production (100× faster with GPU)
- **Distributed QR** frameworks like Spark MLlib

### 3. Quantile Crossing

For some $x$, you might get $\hat{Q}_{0.6}(x) < \hat{Q}_{0.5}(x)$—illogical!

**Solution**: Post-process with isotonic regression (sort predictions for each sample across τ). Or use joint QR models (advanced).

**How to detect and fix**:

```python
# Check for quantile crossing
crossing_mask = predictions[0.5] < predictions[0.1]  # Median below 10th? Wrong!
if crossing_mask.any():
    n_crossings = crossing_mask.sum()
    print(f"Warning: {n_crossings} samples have quantile crossing. Consider post-processing.")
    
    # Simple fix: enforce monotonicity (create corrected copies)
    preds_corrected = {q: predictions[q].copy() for q in [0.1, 0.5, 0.9]}
    
    for i in range(len(preds_corrected[0.5])):
        if preds_corrected[0.5][i] < preds_corrected[0.1][i]:
            preds_corrected[0.5][i] = preds_corrected[0.1][i]  # Push median up
        if preds_corrected[0.9][i] < preds_corrected[0.5][i]:
            preds_corrected[0.9][i] = preds_corrected[0.5][i]  # Push 90th up
    
    # Use preds_corrected for downstream tasks
    predictions = preds_corrected
```

**💡 Pro tip**: For production systems, consider using `sklearn.isotonic.IsotonicRegression` to enforce monotonicity more robustly across all quantiles.

### 4. Not Checking Residual Patterns

Even with QR, check if residuals are random:

```python
residuals_median = y_test - preds_test[0.5]
plt.scatter(preds_test[0.5], residuals_median, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residual Plot (Median QR)')
plt.show()
```

If you see patterns (e.g., funnel, curve), add non-linear terms or switch to GBM (Blog 4).

---

## Bootstrap Confidence Intervals for Coefficients

Want to know if the slope difference between τ=0.1 and τ=0.9 is statistically significant? Use bootstrap:

```python
from sklearn.utils import resample

n_bootstrap = 200
slopes_bootstrap = {q: [] for q in quantiles}

for _ in range(n_bootstrap):
    # Resample with replacement
    indices = resample(range(len(X_train)), replace=True)
    X_boot = X_train[indices]
    y_boot = y_train[indices]
    X_boot_const = sm.add_constant(X_boot)
    
    for q in quantiles:
        model_boot = sm.QuantReg(y_boot, X_boot_const).fit(q=q, max_iter=1000)
        slopes_bootstrap[q].append(model_boot.params[1])

# Compute 95% CIs
for q in quantiles:
    ci_lower = np.percentile(slopes_bootstrap[q], 2.5)
    ci_upper = np.percentile(slopes_bootstrap[q], 97.5)
    print(f"τ={q}: Slope 95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")
```

**Output**:
```
τ=0.1: Slope 95% CI = [2.461, 2.845]
τ=0.5: Slope 95% CI = [2.865, 3.315]
τ=0.9: Slope 95% CI = [3.413, 3.776]
```

**Interpretation**: The CIs don't overlap much → slopes are **significantly different** across quantiles. Strong evidence of heterogeneous effects!

---

## Putting It All Together: Reusable Function

Here's a production-ready function you can copy-paste:

```python
def fit_evaluate_qr(X_train, y_train, X_test, y_test, quantiles=[0.1, 0.5, 0.9]):
    """
    Fit and evaluate quantile regression for multiple quantiles.
    
    Args:
        X_train, y_train: Training data (arrays)
        X_test, y_test: Test data (arrays)
        quantiles: List of quantiles to fit (default: [0.1, 0.5, 0.9])
    
    Returns:
        models: dict of fitted models
        predictions: dict of test predictions
        metrics: dict of evaluation metrics
    
    Raises:
        ValueError: If input shapes mismatch or contain NaNs
    """
    import statsmodels.api as sm
    import numpy as np
    
    # Validation
    if X_train.shape[0] != len(y_train):
        raise ValueError(f"X_train and y_train shape mismatch: {X_train.shape[0]} vs {len(y_train)}")
    if X_test.shape[0] != len(y_test):
        raise ValueError(f"X_test and y_test shape mismatch: {X_test.shape[0]} vs {len(y_test)}")
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("Training data contains NaNs. Handle missing values before fitting.")
    
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    
    models = {}
    preds_train = {}
    preds_test = {}
    
    for q in quantiles:
        model = sm.QuantReg(y_train, X_train_const).fit(q=q)
        models[q] = model
        preds_train[q] = model.predict(X_train_const)
        preds_test[q] = model.predict(X_test_const)
    
    # Evaluate
    lower, upper = min(quantiles), max(quantiles)
    interval_width = upper - lower
    
    in_interval = (y_test >= preds_test[lower]) & (y_test <= preds_test[upper])
    coverage = in_interval.mean()
    
    interval_widths = preds_test[upper] - preds_test[lower]
    sharpness = interval_widths.mean()
    
    metrics = {
        'coverage': coverage,
        'target_coverage': interval_width,
        'sharpness': sharpness
    }
    
    print(f"Coverage: {coverage:.2%} (target: {interval_width:.0%})")
    print(f"Mean interval width: {sharpness:.2f}")
    
    return models, preds_test, metrics

# Usage
models, preds, metrics = fit_evaluate_qr(X_train, y_train, X_test, y_test)
```
```
Coverage: 75.00% (target: 80%)
Mean interval width: 35.42
```
---

## When to Use QR vs. OLS: A Decision Framework

Not sure if you need quantile regression? Use this guide:

### ✅ **Use Quantile Regression when:**
- **Heteroscedasticity is present**: Variance changes with X (funnel shapes in residuals)
- **You need prediction intervals**: OLS gives point estimates only; QR gives full distribution
- **Outliers are common**: QR is robust to extreme values (especially median regression)
- **Tail behavior matters**: Risk management, SLA compliance, extreme event forecasting
- **Distributional effects exist**: Impact of X varies across Y's distribution (income inequality, housing markets)

**Examples**: Server latency (p99 matters), insurance claims (tail risk), income prediction (distributional fairness)

### ⚠️ **Use OLS when:**
- **You only need the mean**: Simple forecasting, averaging over many samples
- **Data is well-behaved**: Homoscedastic, Gaussian errors, few outliers
- **Speed is critical**: OLS is 5-10× faster than QR
- **Interpretability is paramount**: OLS coefficients have clear "average effect" interpretation
- **Sample size is tiny**: n < 100 and you need a single robust estimate (use median regression, not full QR)

**Examples**: A/B testing (average treatment effect), simple trend analysis, dashboards with real-time updates

### 🤔 **Not sure? Try this:**
1. Fit OLS and plot residuals vs. fitted values
2. If you see a funnel shape → **use QR**
3. If residuals look random and you have outliers → **use median regression (τ=0.5)**
4. If you need uncertainty quantification → **use QR for prediction intervals**

---

## TL;DR

- **Fit QR with statsmodels**: `sm.QuantReg(y, X).fit(q=tau)`
- **Interpret coefficients**: Slopes/intercepts vary across quantiles → heterogeneous effects
- **Build prediction intervals**: [Q_0.1, Q_0.9] for 80% coverage
- **Evaluate**: Coverage (calibration), pinball loss (sharpness), bootstrap CIs (significance)
- **Compare to OLS**: QR is more robust, captures distributional effects, provides intervals
- **Real-world**: Bike-sharing example shows how to apply QR to demand forecasting
- **Pitfalls**: Evaluate on test set, scale features, watch for quantile crossing

---

## Beyond Linear: When Straight Lines Aren't Enough

You've just built a linear quantile regression model. But what if your data has non-linear patterns—curves, interactions, threshold effects?

Linear models will underfit. Manual feature engineering ($X^2$, $\sin(X)$, interactions) is tedious, domain-specific, and misses patterns you didn't think of.

In **Blog 4**, we'll level up to **Gradient Boosting Machines**—models that automatically discover non-linear patterns, feature interactions, and complex decision boundaries. You'll learn:
- How to fit non-linear QR with LightGBM (3 lines of code)
- Bayesian hyperparameter tuning with Optuna (10× faster than grid search)
- Feature importance: what drives tails vs. center?
- When to use GBM vs. linear QR (decision framework)

---

## Series Navigation

**Part 3 of 5: Your First Quantile Regression Model**

← **Previous:** [Part 2 - The Math Behind the Magic: Understanding the Pinball Loss](#blog/blog2-medium)

**Next:** [Part 4 - Leveling Up: Gradient Boosting for Quantile Regression](#blog/blog4-medium) →

---

### Complete Series

1. [Part 1 - Beyond the Average: Why Quantile Regression is a Game-Changer](#blog/blog1-medium)
2. [Part 2 - The Math Behind the Magic: Understanding the Pinball Loss](#blog/blog2-medium)
3. **[Part 3 - Your First Quantile Regression Model: A Hands-On Python Guide](#blog/blog3-medium)** (Current)
4. [Part 4 - Leveling Up: Gradient Boosting for Quantile Regression](#blog/blog4-medium)
5. [Part 5 - The State of the Art: Probabilistic Forecasting](#blog/blog5-medium)

---

*This is Part 3 of a 5-part series on mastering quantile regression. [Read the full series](README.md).*
