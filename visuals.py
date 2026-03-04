import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner execution
warnings.filterwarnings('ignore')

# ==========================================
# IMPORT CONFIGURATION FROM MAIN ALGORITHM
# ==========================================
# This ensures that if variables are tweaked in algorithm.py,
# the visualisations automatically update to match without code duplication.
from algorithm import (
    TARGET_COMPLETES,
    MAX_TIME_SECONDS,
    MAX_QUALIFIERS,
    Z_SCORE,
    SIMULATIONS,
    COST_PER_RESPONDENT,
    adjust_demographics
)


def generate_visuals():
    print("--- Tracksuit Algorithm Visualisations & Data Summary ---")
    print("Running algorithm and generating visual proofs... please wait.\n")

    # 1. RUN THE CORE MATH
    df = pd.read_csv("fake_category_data.csv")
    df["incidence_rate"] = df["incidence_rate"].astype(float)
    df["category_length_seconds"] = df["category_length_seconds"].astype(float)

    # Use the exact demographic function imported from algorithm.py
    df['demographic_multiplier'] = df.apply(adjust_demographics, axis=1)
    df['effective_incidence_rate'] = df['incidence_rate'] * df['demographic_multiplier']
    df["expected_time_load"] = df["effective_incidence_rate"] * df["category_length_seconds"]

    df = df.sort_values("effective_incidence_rate")
    categories = df.to_dict("records")

    # Greedy Bin Packing
    surveys = []
    for cat in categories:
        placed = False
        for survey in surveys:
            current_load = sum(c["expected_time_load"] for c in survey)
            new_load = current_load + cat["expected_time_load"]
            if new_load <= MAX_TIME_SECONDS and len(survey) < MAX_QUALIFIERS:
                survey.append(cat)
                placed = True
                break
        if not placed:
            surveys.append([cat])

    # Respondent Allocation
    survey_allocations = []
    total_respondents = 0
    for survey in surveys:
        required_n = 0
        for cat in survey:
            p = cat["effective_incidence_rate"]
            base_n = TARGET_COMPLETES / p
            buffer = Z_SCORE * math.sqrt((TARGET_COMPLETES * (1 - p)) / (p ** 2))
            required_n = max(required_n, base_n + buffer)
        required_n = math.ceil(required_n)
        survey_allocations.append(required_n)
        total_respondents += required_n

    naive_total = sum(math.ceil(TARGET_COMPLETES / row["effective_incidence_rate"]) for _, row in df.iterrows())
    savings_per_month = (naive_total - total_respondents) * COST_PER_RESPONDENT

    # Print core data metrics
    print("=== DATA SUMMARY ===")
    print(f"Total respondents required (Optimised): {total_respondents:,}")
    print(f"Total respondents required (Naive): {naive_total:,}")
    print(f"Cost reduction: {(1 - total_respondents / naive_total) * 100:.2f}%")
    print(f"Estimated Monthly Savings: ${savings_per_month:,.2f}\n")

    # 2. RUN MONTE CARLO FOR PLOT DATA
    completes_sim = {}
    total_time_sim = np.zeros(SIMULATIONS)
    for survey, n_people in zip(surveys, survey_allocations):
        for cat in survey:
            p = cat["effective_incidence_rate"]
            L = cat["category_length_seconds"]
            c = np.random.binomial(n_people, p, size=SIMULATIONS)
            completes_sim[cat["category_id"]] = c
            total_time_sim += (c * L)

    avg_time_sim = total_time_sim / total_respondents

    print("=== SIMULATION RESULTS ===")
    print(f"Mean average survey time: {np.mean(avg_time_sim):.2f} seconds")
    print(f"Time constraint satisfied (<{MAX_TIME_SECONDS}s): {np.all(avg_time_sim < MAX_TIME_SECONDS)}")

    prob_meeting_target = {cat_id: np.mean(completes >= TARGET_COMPLETES) for cat_id, completes in
                           completes_sim.items()}
    lowest_cat = min(prob_meeting_target, key=prob_meeting_target.get)
    print(
        f"Worst category probability of hitting {TARGET_COMPLETES} completes: {prob_meeting_target[lowest_cat] * 100:.1f}%\n")

    # ==========================================
    # 3. GENERATE AND SAVE PLOTS
    # ==========================================
    print("=== GENERATING PLOTS ===")
    sns.set_theme(style="whitegrid")

    # Plot 1: Cost Reduction (Bar Chart)
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=["Naive Baseline", "Optimised Algorithm"], y=[naive_total, total_respondents], palette="viridis")
    plt.title("Total Respondents Required (Cost)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Number of Respondents", fontsize=12)

    # Add data labels on top of the bars
    for i, v in enumerate([naive_total, total_respondents]):
        ax.text(i, v + 1000, f"{v:,}", ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig("algo_1_cost_comparison.png", dpi=300, bbox_inches='tight')
    print(" -> Saved 'algo_1_cost_comparison.png'")
    plt.close()

    # Plot 2: Average Time Distribution (Histogram)
    plt.figure(figsize=(8, 6))
    sns.histplot(avg_time_sim, bins=30, kde=True, color='teal')
    plt.axvline(MAX_TIME_SECONDS, color='red', linestyle='--', linewidth=2.5,
                label=f"SLA Time Limit ({MAX_TIME_SECONDS}s)")
    plt.title("Simulated Average Survey Time", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Average Time per Respondent (seconds)", fontsize=12)
    plt.ylabel(f"Frequency (Over {SIMULATIONS:,} Months)", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("algo_2_time_proof.png", dpi=300, bbox_inches='tight')
    print(" -> Saved 'algo_2_time_proof.png'")
    plt.close()

    # Plot 3: Target Completes for Hardest Category (Histogram)
    plt.figure(figsize=(8, 6))
    sns.histplot(completes_sim[lowest_cat], bins=25, color='coral', discrete=True)
    plt.axvline(TARGET_COMPLETES, color='red', linestyle='--', linewidth=2.5,
                label=f"SLA Target ({TARGET_COMPLETES} Completes)")
    plt.title(f"Completes Distribution: Hardest Category (ID: {lowest_cat})", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Number of Qualified Completes", fontsize=12)
    plt.ylabel(f"Frequency (Over {SIMULATIONS:,} Months)", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("algo_3_target_proof.png", dpi=300, bbox_inches='tight')
    print(" -> Saved 'algo_3_target_proof.png'")
    plt.close()

    print("\nSuccess! 3 algorithmic visual proofs have been saved to your directory.")


if __name__ == "__main__":
    generate_visuals()