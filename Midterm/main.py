"""
Name:       Kaushik Madiraju
Library:    SciPy
URL:        https://scipy.org/

Description:
This library provides advanced mathematical, scientific, and statistical functionality for Python.
In this project, we demonstrate how a data scientist can use SciPy to solve a real-world digital marketing problem involving budget optimization under nonlinear ROI curves, A/B testing for campaign effectiveness, and long-term revenue forecasting. The use of `scipy.optimize`, `scipy.stats`, and `scipy.integrate` enables efficient, modular, and data-driven decision making.
"""



import numpy as np
from scipy.optimize import minimize
from scipy.stats import ttest_ind
from scipy.integrate import quad


# -------- Step 1: Simulate campaign performance data --------
def generate_campaign_data(n=100, seed=42):
    np.random.seed(seed)
    campaign_A = np.random.normal(loc=0.15, scale=0.02, size=n)
    campaign_B = np.random.normal(loc=0.10, scale=0.02, size=n)
    return campaign_A, campaign_B


# -------- Step 2: Define ROI curves for each campaign --------
def roi_curve_A(budget):
    return 2000 * (1 - np.exp(-0.0003 * budget))


def roi_curve_B(budget):
    return 1500 * (1 - np.exp(-0.0004 * budget))


# -------- Step 3: Optimization function --------
def optimize_budget(total_budget):
    def neg_total_roi(x):
        return -(roi_curve_A(x[0]) + roi_curve_B(x[1]))

    constraints = {'type': 'eq', 'fun': lambda x: total_budget - sum(x)}
    bounds = [(0, total_budget), (0, total_budget)]
    initial_guess = [total_budget / 2, total_budget / 2]

    result = minimize(neg_total_roi, initial_guess, bounds=bounds, constraints=constraints)
    return result.x, -result.fun


# -------- Step 4: A/B testing --------
def ab_test(data_A, data_B):
    t_stat, p_val = ttest_ind(data_A, data_B)
    return t_stat, p_val


# -------- Step 5: Revenue Forecast --------
def forecast_revenue(budget_A):
    monthly_budget = budget_A / 12
    revenue, _ = quad(lambda t: roi_curve_A(monthly_budget), 0, 12)
    return revenue


# --------- Main Program ---------
def main():
    try:
        user_input = float(input("Enter your total digital marketing budget (in dollars): $"))
        if user_input <= 0:
            print("Please enter a positive number.")
            return
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return

    # Simulate campaign performance
    campaign_A_data, campaign_B_data = generate_campaign_data()

    # Optimize allocation
    (budget_A, budget_B), max_roi = optimize_budget(user_input)

    # A/B Test
    t_stat, p_val = ab_test(campaign_A_data, campaign_B_data)

    # Forecast 12-month revenue from Campaign A
    forecast = forecast_revenue(budget_A)

    # Print Results
    print("\n--- Budget Allocation Results ---")
    print(f"Total Budget: ${user_input:,.2f}")
    print(f"→ Campaign A: ${budget_A:,.2f}")
    print(f"→ Campaign B: ${budget_B:,.2f}")
    print(f"Max Expected ROI: ${max_roi:,.2f}")

    print("\n--- A/B Test Result ---")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("✅ Statistically significant difference — likely Campaign A performs better.")
    else:
        print("ℹ️ No statistically significant difference between campaigns.")

    print("\n--- Revenue Forecast ---")
    print(f"Projected 12-month Revenue (Campaign A): ${forecast:,.2f}")


if __name__ == "__main__":
    main()

