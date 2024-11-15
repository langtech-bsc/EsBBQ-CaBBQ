import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS_DIR = "results/bias_score"

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model_name', help="name of the model")
args = parser.parse_args()

# Results file path
MODEL_SCORES = os.path.join(RESULTS_DIR,f"{args.model_name}.csv")

# Open results file and set 'category' column as index
df = pd.read_csv(MODEL_SCORES,index_col="category")

# Custom colormaps
cmaps = {
        'acc': mcolors.LinearSegmentedColormap.from_list("green_red", ["red", "yellow", "green"], N=100),
        'diff_bias':mcolors.LinearSegmentedColormap.from_list("orange_blue", ["orange", "white", "blue"], N=100)
        }

# Create a single figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(9, 8))

# Iterate over the scores and plot each heatmap in a subplot
for i, score in enumerate(["acc", "diff_bias"]):
    
    center = 0.5
    if score.startswith("diff_bias"):
        center = 0

    # Select only columns corresponding to the score
    df_score = df.filter(regex=f'^{score}')
    
    # Plot each heatmap on the corresponding subplot
    sns.heatmap(df_score, annot=True, cmap=cmaps[score], fmt=".2f", cbar=True, center=center, ax=axes[i])
    
    # Customize each subplot
    axes[i].xaxis.tick_top()  # Move x-axis ticks to the top
    axes[i].set(xlabel="", ylabel="")
    axes[i].set_aspect('equal')
    axes[i].tick_params(labelsize=10)

fig.suptitle(args.model_name, fontsize=14, fontweight='bold',y=0.95)

# Adjust layout for better spacing
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.3)

# Save the final plot
plt.savefig(os.path.join(RESULTS_DIR, f"{args.model_name}.png"))