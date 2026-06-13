import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, linregress
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/processed/merged_dataset.csv")

# X and Y
x = df["AHI_1_B"].values
y = df["brain_health_score"].values

# -----------------------------
# Pearson Correlation
# -----------------------------
r, p = pearsonr(x, y)

print("=" * 50)
print("Pearson Correlation")
print("=" * 50)
print(f"r       = {r:.4f}")
print(f"p-value = {p:.6f}")

# -----------------------------
# Linear Regression (OLS)
# -----------------------------
result = linregress(x, y)

slope = result.slope
intercept = result.intercept

print("\n" + "=" * 50)
print("Linear Regression")
print("=" * 50)
print(f"Slope (w)     = {slope:.6f}")
print(f"Intercept (b) = {intercept:.6f}")

print(
    f"\nEquation:\n"
    f"Brain Health Score = "
    f"{slope:.6f} * AHI_1_B + "
    f"{intercept:.6f}"
)

# -----------------------------
# Polynomial Regression
# -----------------------------
X = x.reshape(-1, 1)

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

print("\n" + "=" * 50)
print("Polynomial Regression (Degree = 2)")
print("=" * 50)

print(
    f"y = "
    f"{poly_model.coef_[2]:.6f}x² + "
    f"{poly_model.coef_[1]:.6f}x + "
    f"{poly_model.intercept_:.6f}"
)

# -----------------------------
# Plot
# -----------------------------
x_plot = np.linspace(
    x.min(),
    x.max(),
    500
)

# Linear fit
y_linear = slope * x_plot + intercept

# Polynomial fit
y_poly = poly_model.predict(
    poly.transform(
        x_plot.reshape(-1, 1)
    )
)

plt.figure(figsize=(10, 6))

# Scatter points
plt.scatter(
    x,
    y,
    s=60,
    alpha=0.7,
    label="Subjects"
)

# Linear fit
plt.plot(
    x_plot,
    y_linear,
    linewidth=2,
    label="Linear Fit (OLS)"
)

# Polynomial fit
plt.plot(
    x_plot,
    y_poly,
    "--",
    linewidth=2,
    label="Polynomial Fit (Degree 2)"
)

plt.xlabel("AHI_1_B", fontsize=12)
plt.ylabel("Brain Health Score", fontsize=12)

plt.title(
    f"Brain Health Score vs AHI_1_B\n"
    f"Pearson r = {r:.3f}, p = {p:.4f}",
    fontsize=13
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/plots/brain_health_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPlot saved successfully!")
print("Location: results/plots/brain_health_analysis.png")