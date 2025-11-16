# The Math Behind the Magic: Understanding the Pinball Loss

*Why asymmetric loss functions find quantiles, and how to optimize them*

---

![Pinball loss shapes for different quantiles](assets/pinball_loss_shapes.png)

**Reading time: ~17 minutes** ☕️

---

In [Blog 1](blog1_medium.md), you learned *why* quantile regression is powerful. Now it's time to understand *how* it works.

At the heart of quantile regression is a deceptively simple function called the **pinball loss** (also known as the *check loss* or *quantile loss*). It's the secret sauce that makes QR find quantiles instead of means.

**Why study the math?** You might be thinking: "Can't I just call `statsmodels.QuantReg` or `xgboost.QuantileRegressor` and move on?"

You could. But understanding pinball loss gives you superpowers:
- **Debug failures**: When your 90th percentile predictions are off, you'll know whether to adjust τ, your loss function, or your data
- **Customize for your domain**: The formula \( \tau = \frac{C_{\text{under}}}{C_{\text{under}} + C_{\text{over}}} \) only makes sense if you understand *why* asymmetric loss creates quantiles
- **Explain to stakeholders**: "We're minimizing pinball loss with τ=0.9" is gibberish. "We're penalizing under-predictions 9× more because outages cost $100K" is actionable.

Let's dive in.

In this post, you'll learn:
- How minimizing a loss function finds the quantile
- The mathematical structure of the pinball loss
- Why it's asymmetric (and why that matters)
- How to optimize it (subgradient descent, linear programming)
- How to evaluate quantile predictions (pinball loss, coverage, sharpness)
- How to choose τ based on business costs

By the end, you'll have a deep understanding of the mathematical machinery powering quantile regression.

---

## Finding Quantiles via Loss Minimization

### Quick Recap: What's a Quantile?

The **τ-quantile** (or \(100 \tau\)-th percentile) of a distribution is the value \(q\) such that:
- Probability of being below \(q\): \(P(Y \leq q) = \tau\)
- Probability of being above \(q\): \(P(Y > q) = 1 - \tau\)

Examples:
- **τ=0.5**: Median (50th percentile)—half above, half below
- **τ=0.9**: 90th percentile—90% below, 10% above
- **τ=0.1**: 10th percentile—10% below, 90% above

### The Minimization View

Here's the key insight: **quantiles minimize expected loss** for a specific loss function.

For the mean:
$$\mu = \arg\min_{c} E[(Y - c)^2]$$
The mean \(\mu\) minimizes squared loss (MSE).

For the τ-quantile:
$$q_{\tau} = \arg\min_{c} E[\rho_{\tau}(Y - c)]$$
The quantile \(q_{\tau}\) minimizes **pinball loss** \(\rho_{\tau}\).

This is profound: *changing the loss function changes the optimal predictor* from mean to quantile.

---

## Comparing Loss Functions: MSE vs. Pinball

Before diving deeper, let's contrast the two losses:

| Property | MSE \((y - \hat{y})^2\) | Pinball \(\rho_{\tau}(y - \hat{y})\) |
|----------|----------------------|-----------------------------------|
| **Finds** | Mean | τ-quantile |
| **Symmetry** | Symmetric | Asymmetric |
| **Outlier sensitivity** | High (quadratic) | Moderate (linear) |
| **Differentiability** | Smooth | Piecewise (subgradient at 0) |
| **Optimization** | Closed-form (OLS) | Linear programming or subgradient |
| **Interpretability** | "Typical value" | "Value with P(Y ≤ v) = τ" |

**When MSE is better**: You truly care about the mean, outliers are noise, simplicity matters.

**When pinball is better**: You need quantiles, outliers are informative, asymmetric costs exist.

---

## The Pinball Loss Function

### Definition

$$\rho_{\tau}(u) = \begin{cases} 
\tau \cdot u & \text{if } u \geq 0 \text{ (under-prediction)} \\
(1 - \tau) \cdot |u| & \text{if } u < 0 \text{ (over-prediction)}
\end{cases}$$

Or equivalently:
$$\rho_{\tau}(u) = u \cdot (\tau - \mathbb{1}_{u < 0})$$

where \(u = y - \hat{y}\) is the residual (error).

### Intuition

The pinball loss is **piecewise linear** with two slopes:
- **Slope above zero** (under-prediction): \(\tau\)
- **Slope below zero** (over-prediction): \(1 - \tau\)

For **τ=0.9** (90th percentile):
- Under-predict by 1 unit → loss = \(0.9 \times 1 = 0.9\)
- Over-predict by 1 unit → loss = \(0.1 \times 1 = 0.1\)

Under-predictions are penalized **9× more** than over-predictions. This asymmetry pushes the model to predict *high* so that 90% of observations fall below.

For **τ=0.5** (median):
- Under-predict by 1 unit → loss = \(0.5 \times 1 = 0.5\)
- Over-predict by 1 unit → loss = \(0.5 \times 1 = 0.5\)

Balanced penalties → finds the median.

---

## Visualizing the Pinball Loss

![Pinball loss shapes](assets/pinball_loss_shapes.png)

This plot shows $\rho_{\tau}(u)$ for three values of τ:

1. **τ=0.1** (10th percentile, red): Steep slope for $u < 0$ (over-predictions penalized heavily). Gentle slope for $u > 0$ (under-predictions tolerated). Result: low predictions.

2. **τ=0.5** (median, blue): Symmetric V-shape (like absolute loss). Balanced penalties. Result: median.

3. **τ=0.9** (90th percentile, green): Steep slope for $u > 0$ (under-predictions penalized heavily). Gentle slope for $u < 0$ (over-predictions tolerated). Result: high predictions.

**Key observation**: The pinball loss is **convex** (V-shaped), which means optimization is well-behaved (no local minima).

---

## Why Does This Work? (The Intuitive Explanation)

Let's understand *why* minimizing pinball loss finds the quantile, using intuition instead of heavy math.

### The Balancing Act

Imagine you're trying to find the right prediction value $c$. The pinball loss creates two opposing forces:

1. **Points above your prediction** ($Y > c$): Each one "pushes" your prediction upward with force τ
2. **Points below your prediction** ($Y < c$): Each one "pushes" your prediction downward with force $(1 - \tau)$

Your prediction settles at the point where these forces balance out.

### Concrete Example: Finding the 90th Percentile (τ=0.9)

Let's say you have 100 data points and you're trying different prediction values:

**Try prediction = 50 (too low)**
- 95 points are above 50 → they push upward with total force: $95 \times 0.9 = 85.5$
- 5 points are below 50 → they push downward with total force: $5 \times 0.1 = 0.5$
- Net force: **85 upward** → prediction needs to go higher!

**Try prediction = 95 (too high)**
- 5 points are above 95 → they push upward with total force: $5 \times 0.9 = 4.5$
- 95 points are below 95 → they push downward with total force: $95 \times 0.1 = 9.5$
- Net force: **5 downward** → prediction needs to go lower!

**Try prediction = 90th percentile (just right)**
- 10 points are above it → they push upward with total force: $10 \times 0.9 = 9$
- 90 points are below it → they push downward with total force: $90 \times 0.1 = 9$
- Net force: **0 (balanced!)** ✅

The forces balance exactly when $c$ splits the data so that the fraction below equals τ—which is the definition of the τ-quantile!

### The General Pattern

For any quantile τ:
- When forces balance: (fraction below) × \(1-\tau\) = (fraction above) × \(\tau\)
- Rearranging: (fraction below) = \(\tau\)
- That's exactly the τ-quantile!

**Why the asymmetry matters**: If τ=0.9:
- Under-predictions (when actual > predicted): penalty = 0.9 per unit
- Over-predictions (when actual < predicted): penalty = 0.1 per unit

The penalty ratio is 0.9 : 0.1 = 9 : 1. This 9× heavier penalty on under-predictions *forces* your prediction high. Why? Because if you predict too low, you pay a huge price (0.9 per unit). But if you predict too high, the price is mild (0.1 per unit). So the model errs on the side of predicting high enough that ~90% of actual values fall below it. **The asymmetry in the loss directly creates the asymmetry in the data split.**

**The median is special**: When τ=0.5:
- Under-predictions: penalty = 0.5 per unit
- Over-predictions: penalty = 0.5 per unit

Equal penalties → the model doesn't care if it's high or low, it just wants to split the data in half → median. This is why median regression is the "balanced" version of pinball loss.

---

## From Sample to Conditional Quantile

So far, we've talked about the **unconditional quantile** (quantile of \(Y\) without considering \(X\)). Quantile regression extends this to **conditional quantiles**: \(Q_{\tau}(Y \mid X)\).

**In plain English**: Instead of just asking "what's the 90th percentile of all heights?", we now ask "what's the 90th percentile of heights *given that someone is 6 feet tall*?" We're conditioning on information ($X$).

### Linear Quantile Regression

Assume:
$$Q_{\tau}(Y \mid X) = X^T \beta(\tau)$$

**Translation**: "We're guessing that the 90th percentile is just a straight line. The line depends on \(X\) (like height), and we write it as \(X^T \beta(\tau)\), which just means 'multiply \(X\) by some numbers (\(\beta\)) and add them up'."

We find \(\beta(\tau)\) by minimizing the sample pinball loss:

$$\min_{\beta} \sum_{i=1}^{n} \rho_{\tau}(y_i - x_i^T \beta)$$

**Translation**: "We adjust our numbers ($\beta$) over and over until we find the best straight line that minimizes the total pinball loss across all our data points ($i=1$ to $n$)."

This is a **linear programming problem** (convex, piecewise linear objective). Can be solved efficiently with LP solvers or specialized QR algorithms.

**Key insight**: We get *different coefficients* $\beta(\tau)$ for each τ. This means:
- The effect of $X$ on the 10th percentile may differ from its effect on the 90th percentile.
- **Concrete example**: Income → spending. Someone earning \$200K might spend money very differently at the 90th percentile (luxury goods) vs. the 10th percentile (essentials). The relationship between income and spending *changes* depending on which part of the spending distribution you're looking at.

This is called **heterogeneous treatment effects** or **distributional effects**. (Fancy way of saying: "The effect of $X$ depends on which quantile you're looking at.")

---

## Optimization: How to Minimize Pinball Loss

### Method 1: Linear Programming (Exact)

The QR problem can be reformulated as:

$$\min_{\beta, r^+, r^-} \tau \sum_{i} r_i^+ + (1 - \tau) \sum_{i} r_i^-$$

**Translation**: "We want to minimize the total cost, where points above our line cost $\tau$ each and points below cost $(1-\tau)$ each."

subject to:
$$y_i - x_i^T \beta = r_i^+ - r_i^-, \quad r_i^+, r_i^- \geq 0$$

**Translation**: "$r^+$ tracks how far each point is *above* our line, and $r^-$ tracks how far each point is *below* our line. The difference between them is just the error (actual - predicted)."

where $r^+$ and $r^-$ are positive and negative parts of residuals. This is a standard LP. Solvers: `scipy.optimize.linprog`, specialized QR packages.

**Pros**: Exact solution, globally optimal.  
**Cons**: Scales poorly for huge datasets (n > 1M).

### Method 2: Subgradient Descent (Approximate)

The pinball loss is non-differentiable at $u=0$, but we can use **subgradients** (a generalization of gradients for non-smooth functions—think "gradient-like direction that still works"):

$$\partial \rho_{\tau}(u) = \begin{cases}
\tau & \text{if } u > 0 \\
-(1 - \tau) & \text{if } u < 0 \\
[-(1-\tau), \tau] & \text{if } u = 0
\end{cases}$$

**Translation**: "At each point, we compute a direction (the subgradient) that tells us how to adjust our parameters. For points above the line (u > 0), the direction is $\tau$. For points below (u < 0), it's $-(1-\tau)$. Right on the line? Could be either, so we say it's somewhere in between."

Subgradient descent update:

$$\beta^{(t+1)} = \beta^{(t)} - \eta \cdot \frac{1}{n} \sum_{i=1}^{n} x_i \cdot g_i$$

**Translation**: "Take our current parameters ($\beta^{(t)}$), compute the average direction from all data points ($\sum x_i \cdot g_i / n$), move in that direction by a tiny step size ($\eta$), and repeat. Each round, we get closer to the best answer."

where $g_i = \tau$ if $y_i \geq x_i^T \beta^{(t)}$ (under-prediction), else $g_i = -(1 - \tau)$ (over-prediction).

**Pros**: Scales to large data (stochastic version), simple to implement.  
**Cons**: Approximate (stops near optimum, not exactly at optimum).

### Method 3: Gradient Boosting (Practical)

For non-linear QR (Blog 4), we use gradient boosting with pinball loss. GBM libraries (LightGBM, XGBoost) handle the optimization internally.

**In plain English**: "Instead of a straight line, we build a forest of trees. Each tree learns from the mistakes of previous trees, and they all work together to predict quantiles. You don't need to understand how it works—just plug in `objective='quantile'` and it handles everything."

---

## Worked Example: Finding the 90th Percentile by Hand

Let's apply subgradient descent to a tiny dataset.

### Data

| X  | Y  |
|----|----|
| 1  | 2  |
| 2  | 3  |
| 3  | 7  |
| 4  | 8  |
| 5  | 12 |

Task: Find $\beta_0$ and $\beta_1$ such that $Q_{0.9}(Y \mid X) = \beta_0 + \beta_1 X$.

**In plain English**: "We have 5 data points. We want to fit a straight line $(\beta_0 + \beta_1 X)$ that predicts the 90th percentile. $\beta_0$ is the starting point (y-intercept) and $\beta_1$ is the slope (how much $Y$ changes when $X$ increases by 1)."

### Initialization

Start with $\beta_0 = 0$, $\beta_1 = 1$. Learning rate $\eta = 0.1$, $\tau = 0.9$.

**In plain English**: "We make an initial guess: start at 0 and increase by 1 for each unit of $X$. The learning rate (0.1) controls how big our steps are—smaller steps are safer but slower."

### Iteration 1

Predictions: $\hat{y}_i = \beta_0 + \beta_1 x_i = 0 + 1 \cdot x_i = x_i$

**Translation**: "Using our current guess, predict each $y$ value. Since $\beta_0 = 0$ and $\beta_1 = 1$, our predictions are just $x_i$ itself."

Residuals: $u_i = y_i - \hat{y}_i$

**Translation**: "Compute the error (actual - predicted) for each point."

| i | $x_i$ | $y_i$ | $\hat{y}_i$ | $u_i$ | $g_i$ (subgrad) |
|---|-------|-------|-------------|-------|-----------------|
| 1 | 1 | 2 | 1 | 1 | 0.9 |
| 2 | 2 | 3 | 2 | 1 | 0.9 |
| 3 | 3 | 7 | 3 | 4 | 0.9 |
| 4 | 4 | 8 | 4 | 4 | 0.9 |
| 5 | 5 | 12 | 5 | 7 | 0.9 |

All residuals positive → all $g_i = 0.9$.

**Translation**: "All actual values are higher than our predictions (all errors are positive). This means we predicted too low. Since we're targeting τ=0.9, we penalize under-predictions heavily, so all $g_i = 0.9$."

Gradient:
$$\nabla \beta_0 = -\frac{1}{5} \sum g_i = -0.9$$

**Translation**: "Average the penalty directions across all 5 points: $(0.9 + 0.9 + 0.9 + 0.9 + 0.9) / 5 = 0.9$. The negative sign means 'move in the opposite direction to reduce loss.'"

$$\nabla \beta_1 = -\frac{1}{5} \sum g_i x_i = -\frac{1}{5}(0.9 \cdot 1 + 0.9 \cdot 2 + \cdots + 0.9 \cdot 5) = -0.9 \cdot 3 = -2.7$$

**Translation**: "For the slope, we weight each penalty by the corresponding $x$ value (1, 2, 3, 4, 5), average them: $0.9 \times (1+2+3+4+5) / 5 = 0.9 \times 3 = 2.7$. Again, negative sign means move opposite."

Update:
$$\beta_0^{(1)} = 0 - 0.1 \cdot (-0.9) = 0.09$$

**Translation**: "Move $\beta_0$ by step size (0.1) × gradient direction (-0.9). Since the direction is negative, we subtract a negative, which means we add: $0 + 0.1 \times 0.9 = 0.09$. Slightly higher!"

$$\beta_1^{(1)} = 1 - 0.1 \cdot (-2.7) = 1.27$$

**Translation**: "Same idea for the slope: $1 + 0.1 \times 2.7 = 1.27$. Steeper slope!"

Continue for 100+ iterations → converges to $\beta_0 \approx 0.5$, $\beta_1 \approx 2.1$.

**Translation**: "Keep repeating this process. After many iterations, we find our best-fit line: $Q_{0.9}(Y \mid X) \approx 0.5 + 2.1X$."

Prediction for $x=3$: $Q_{0.9}(Y \mid x=3) \approx 0.5 + 2.1 \cdot 3 = 6.8$.

**Translation**: "Our 90th percentile prediction when $X=3$ is approximately 6.8."

In our data, 4 out of 5 points (80%) have $Y \leq 6.8$ at $x=3$ neighborhood. Close to 90% (small sample variability).

**Key takeaway**: Notice how *all* residuals are positive (all $g_i = 0.9$), meaning our initial line is too low. The gradient pushes the line *upward* more aggressively than MSE would (which would use $g_i \propto u_i$, not constant 0.9). This aggressive upward push is how QR finds the 90th percentile.

---

## Evaluating Quantile Predictions

Unlike MSE (mean squared error) for OLS, we need different metrics for QR.

### Metric 1: Pinball Loss (Sharpness)

$$\text{Pinball Loss} = \frac{1}{n} \sum_{i=1}^{n} \rho_{\tau}(y_i - \hat{q}_{\tau,i})$$

**Translation**: "For each point, compute the pinball loss, then average across all points. Lower is better."

**Use**: Compare models for the same τ. Not comparable across different τ.

### Metric 2: Coverage (Calibration)

$$\text{Coverage} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}_{y_i \leq \hat{q}_{\tau,i}}$$

**Translation**: "$\mathbb{1}_{y_i \leq \hat{q}_{\tau,i}}$ is a fancy way to write: 'Does the actual value fall below the predicted quantile? If yes, count it (1). If no, count it (0).' Then average across all points."

For a well-calibrated model predicting the τ-quantile, coverage should be ≈ τ.

Examples:
- Predict 90th percentile → expect 90% of $y_i \leq \hat{q}_{0.9,i}$
- Predict 10th percentile → expect 10% of $y_i \leq \hat{q}_{0.1,i}$

**Use**: Check calibration. If coverage ≠ τ, model is biased (systematically over/under-predicting the quantile).

### Metric 3: Interval Sharpness

For prediction intervals (e.g., 80% interval from 10th to 90th percentile):

$$\text{Sharpness} = \frac{1}{n} \sum_{i=1}^{n} (\hat{q}_{0.9,i} - \hat{q}_{0.1,i})$$

**Translation**: "For each point, compute the width of the prediction interval (upper bound minus lower bound), then average. Lower is better because narrow intervals show confidence."

**Trade-off**: Can always get 100% coverage with infinite-width intervals. Good models have *narrow intervals with correct coverage*.

### Metric 4: Winkler Score (Combined Metric)

$$W_{\alpha} = \frac{1}{n} \sum_{i=1}^{n} \left[ (U_i - L_i) + \frac{2}{\alpha}(L_i - y_i) \mathbb{1}_{y_i < L_i} + \frac{2}{\alpha}(y_i - U_i) \mathbb{1}_{y_i > U_i} \right]$$

**Translation in plain English**: 
- First part $(U_i - L_i)$: "How wide is the interval?" (narrower is better)
- Second part: "If the actual value falls below the lower bound, add a penalty proportional to how far below it is"
- Third part: "If the actual value exceeds the upper bound, add a penalty proportional to how far above it is"
- $\alpha$ controls the penalty strength

where $L_i$ and $U_i$ are lower and upper bounds of a $(1-\alpha)$ prediction interval (e.g., $\alpha=0.2$ for 80% interval).

Balances sharpness (first term) and coverage violations (second and third terms).

**Lower is better**. The Winkler score has the same units as your target variable (e.g., dollars, bikes, seconds). A score of 10.5 means "on average, each prediction's interval width plus coverage penalties sum to 10.5 units."

---

## Choosing τ Based on Business Costs

The magic of quantile regression: you can *align your model with business costs*.

### Framework: Asymmetric Cost Function

Suppose:
- **Cost of under-predicting** by 1 unit: $C_{\text{under}}$
- **Cost of over-predicting** by 1 unit: $C_{\text{over}}$

Optimal quantile:
$$\tau^* = \frac{C_{\text{under}}}{C_{\text{under}} + C_{\text{over}}}$$

**Proof sketch**: This is the quantile that minimizes expected cost under the asymmetric cost function.

### Example 1: Restaurant Reservations

- **Under-booking** (predict 50 diners, 70 show up): Turn away 20 customers → lost revenue + bad reviews → $C_{\text{under}} = \$500$
- **Over-booking** (predict 70, 50 show up): Wasted prep + idle staff → $C_{\text{over}} = \$50$

Optimal τ:
$$\tau^* = \frac{500}{500 + 50} = 0.91$$

Predict the **91st percentile** of demand. Accept that you'll over-prepare 9% of the time to avoid under-booking disasters.

### Example 2: Inventory for Perishable Goods

- **Under-stock** (lost sales): $C_{\text{under}} = \$20$ per unit
- **Over-stock** (spoilage): $C_{\text{over}} = \$15$ per unit

Optimal τ:
$$\tau^* = \frac{20}{20 + 15} = 0.57$$

Stock for the **57th percentile** of demand. Slight bias toward over-stocking (spoilage is costly but less than lost sales).

### Example 3: Server Capacity

- **Under-provision** (outage): $C_{\text{under}} = \$100K$ (downtime, customer churn)
- **Over-provision** (idle servers): $C_{\text{over}} = \$1K$ (wasted compute)

Optimal τ:
$$\tau^* = \frac{100000}{100000 + 1000} \approx 0.99$$

Provision for the **99th percentile**. Accept 1% wasted capacity to avoid catastrophic outages.

**Callback to Blog 1**: Remember the hospital readmission scenario? If readmitting a patient costs $50K (emergency care + reputation damage) and unnecessary home visits cost $500, then:

$$\tau^* = \frac{50000}{50000 + 500} \approx 0.99$$

**Translation**: "The cost of missing a readmission (50K) is 100× higher than the cost of unnecessary home visits (500), so we target the 99th percentile of risk. This means we accept that 1% of patients won't get visits (over-provision) to avoid the catastrophic cost of missing someone who needs readmission (under-provision)."

Predict the 99th percentile of readmission risk—err on the side of caution.

---

---

## Practical Implementation Tips

### 1. Use Existing Libraries

- **Python**: `statsmodels.regression.quantile_regression.QuantReg`
- **R**: `quantreg::rq`
- **Gradient boosting**: LightGBM (`objective='quantile'`), XGBoost (`objective='reg:quantileerror'`)

Don't implement from scratch unless you have a reason (learning, custom loss, research).

### 2. Start with Median (τ=0.5)

Median regression is a good sanity check:
- Robust to outliers (unlike OLS)
- Easier to interpret than extreme quantiles
- If median QR ≈ OLS, data is symmetric (OLS is fine)

### 3. Plot the Pinball Loss

Visualize $\rho_{\tau}(u)$ for your chosen τ. Does the asymmetry make sense? For τ=0.95, under-predictions should be penalized 19× more—does that align with your domain?

### 4. Monitor Coverage in Production

Track: "What % of observations fall below our 90th percentile forecast?" Should be ≈90%. If it drifts (e.g., drops to 70%), retrain or recalibrate.

---

## TL;DR

- **Pinball loss** is piecewise linear with asymmetric slopes: $\rho_{\tau}(u) = u \cdot (\tau - \mathbb{1}_{u < 0})$
- Minimizing pinball loss finds the **τ-quantile** (not the mean)
- **Asymmetry** encodes business costs: $\tau = \frac{C_{\text{under}}}{C_{\text{under}} + C_{\text{over}}}$
- **Optimization**: Linear programming (exact), subgradient descent (scalable), or GBM (non-linear)
- **Evaluation**: Pinball loss (sharpness), coverage (calibration), Winkler score (combined)
- **Key insight**: Changing the loss function changes the optimal predictor—QR is *not* just "OLS with extra steps"

---

---

## Series Navigation

**Part 2 of 5: The Math Behind the Magic**

← **Previous:** [Part 1 - Beyond the Average: Why Quantile Regression is a Game-Changer](#blog/blog1-medium)

**Next:** [Part 3 - Your First Quantile Regression Model: A Hands-On Python Guide](#blog/blog3-medium) →

---

### Complete Series

1. [Part 1 - Beyond the Average: Why Quantile Regression is a Game-Changer](#blog/blog1-medium)
2. **[Part 2 - The Math Behind the Magic: Understanding the Pinball Loss](#blog/blog2-medium)** (Current)
3. [Part 3 - Your First Quantile Regression Model: A Hands-On Python Guide](#blog/blog3-medium)
4. [Part 4 - Leveling Up: Gradient Boosting for Quantile Regression](#blog/blog4-medium)
5. [Part 5 - The State of the Art: Probabilistic Forecasting](#blog/blog5-medium)

---

*This is Part 2 of a 5-part series on mastering quantile regression. [Read the full series](README.md).*
