import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner terminal execution
warnings.filterwarnings('ignore')


def run_eda(filepath="fake_category_data.csv"):
    print("--- Tracksuit Exploratory Data Analysis ---")
    print("Generating visualisations... please wait a moment.\n")

    # 1. Load and prep the data
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Make sure it is in the same directory.")
        return

    df["incidence_rate"] = df["incidence_rate"].astype(float)
    df["category_length_seconds"] = df["category_length_seconds"].astype(float)

    # 2. Set professional visual theme
    sns.set_theme(style="whitegrid")

    # ==========================================
    # PLOT 1: Distribution Histograms
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Incidence Rate Distribution
    sns.histplot(df["incidence_rate"], bins=20, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("Distribution of Incidence Rates", fontsize=14, pad=10, fontweight='bold')
    axes[0].set_xlabel("Incidence Rate", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)

    # Category Length Distribution
    sns.histplot(df["category_length_seconds"], bins=20, kde=True, ax=axes[1], color='salmon')
    axes[1].set_title("Distribution of Category Lengths", fontsize=14, pad=10, fontweight='bold')
    axes[1].set_xlabel("Length (seconds)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)

    plt.tight_layout()
    plt.savefig("eda_distributions.png", dpi=300, bbox_inches='tight')
    print(" -> Saved 'eda_distributions.png'")
    plt.close()

    # ==========================================
    # PLOT 2: Scatter Plot (Correlation Check)
    # ==========================================
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="incidence_rate",
        y="category_length_seconds",
        alpha=0.7,
        color='purple',
        s=60
    )
    plt.title("Incidence Rate vs. Category Length", fontsize=14, pad=10, fontweight='bold')
    plt.xlabel("Incidence Rate", fontsize=12)
    plt.ylabel("Category Length (seconds)", fontsize=12)

    plt.tight_layout()
    plt.savefig("eda_scatter.png", dpi=300, bbox_inches='tight')
    print(" -> Saved 'eda_scatter.png'")
    plt.close()

    print("\nSuccess! EDA visuals generated. You can now embed these into your README.")


if __name__ == "__main__":
    run_eda()