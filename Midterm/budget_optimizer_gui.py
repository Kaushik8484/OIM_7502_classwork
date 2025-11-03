import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ttest_ind
from scipy.integrate import quad

# -------- Simulate campaign data --------
def generate_campaign_data(n=100, seed=42):
    np.random.seed(seed)
    campaign_A = np.random.normal(loc=0.15, scale=0.02, size=n)
    campaign_B = np.random.normal(loc=0.10, scale=0.02, size=n)
    return campaign_A, campaign_B

# -------- ROI curves --------
def roi_curve_A(budget):
    return 2000 * (1 - np.exp(-0.0003 * budget))

def roi_curve_B(budget):
    return 1500 * (1 - np.exp(-0.0004 * budget))

# -------- Optimization --------
def optimize_budget(total_budget):
    def neg_total_roi(x):
        return -(roi_curve_A(x[0]) + roi_curve_B(x[1]))

    constraints = {'type': 'eq', 'fun': lambda x: total_budget - sum(x)}
    bounds = [(0, total_budget), (0, total_budget)]
    guess = [total_budget / 2, total_budget / 2]

    result = minimize(neg_total_roi, guess, bounds=bounds, constraints=constraints)
    return result.x, -result.fun

# -------- A/B test --------
def ab_test(data_A, data_B):
    t_stat, p_val = ttest_ind(data_A, data_B)
    return t_stat, p_val

# -------- Revenue forecast --------
def forecast_revenue(budget_A):
    monthly_budget = budget_A / 12
    revenue, _ = quad(lambda t: roi_curve_A(monthly_budget), 0, 12)
    return revenue

# -------- Main function --------
def run_optimizer():
    try:
        budget = float(entry.get())
        if budget <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid positive number.")
        return

    campaign_A, campaign_B = generate_campaign_data()
    (bA, bB), max_roi = optimize_budget(budget)
    t_stat, p_val = ab_test(campaign_A, campaign_B)
    forecast = forecast_revenue(bA)

    result = f"""--- Optimization Results ---

Total Budget: ${budget:,.2f}

→ Campaign A: ${bA:,.2f}
→ Campaign B: ${bB:,.2f}
Max Projected ROI: ${max_roi:,.2f}

A/B Test p-value: {p_val:.4f}
{"✓ Campaign A likely better" if p_val < 0.05 else "No significant difference"}

12-Month Revenue Forecast: ${forecast:,.2f}
"""
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, result)

# -------- GUI Setup --------
BABSON_GREEN = "#1E4D2B"
BABSON_LIGHT = "#D9EAD3"

root = tk.Tk()
root.title("Digital Marketing ROI Optimizer – Babson")
root.geometry("650x580")
root.configure(bg=BABSON_LIGHT)

# --- Header ---
header = tk.Label(
    root,
    text="Digital Marketing ROI Optimizer",
    bg=BABSON_GREEN,
    fg="white",
    font=("Helvetica", 18, "bold"),
    pady=12
)
header.pack(fill="x")

# --- Subheader ---
subheader = tk.Label(
    root,
    text="Created by Kaushik Madiraju – MBA Candidate, Babson College",
    bg=BABSON_LIGHT,
    fg=BABSON_GREEN,
    font=("Helvetica", 12, "italic"),
    pady=6
)
subheader.pack()

# --- Input label and field ---
label = tk.Label(
    root,
    text="Enter Total Marketing Budget ($):",
    bg=BABSON_LIGHT,
    fg=BABSON_GREEN,
    font=("Helvetica", 12)
)
label.pack(pady=(15, 5))

entry = tk.Entry(root, width=25, font=("Helvetica", 14), fg=BABSON_GREEN, bg="white", insertbackground=BABSON_GREEN)
entry.pack()

# --- Run Button ---
run_btn = tk.Button(
    root,
    text="Run Optimization",
    command=run_optimizer,
    bg=BABSON_GREEN,
    fg="white",
    font=("Helvetica", 12, "bold"),
    padx=12,
    pady=6
)
run_btn.pack(pady=18)

# --- Output Display ---
output_text = tk.Text(
    root,
    height=18,
    width=70,
    font=("Courier", 10),
    fg=BABSON_GREEN,
    bg="white",
    insertbackground=BABSON_GREEN,
    wrap="word",
    padx=8,
    pady=8
)
output_text.pack(padx=10, pady=(0, 10))

# --- Launch App ---
root.mainloop()
