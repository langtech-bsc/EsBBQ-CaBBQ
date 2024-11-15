import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Results file path
RESULTS_DIR = "results/bias_score"

parser = argparse.ArgumentParser()
parser.add_argument('-m','--models', help="name of the models to plot, separated by commas")
parser.add_argument('-t','--title', help="custom title for the plot. Will be also the name of the output png")
parser.add_argument('-n','--model_names',help="name of the models in the plot, separated by commas and in the same order as given in -m", required=False)
args = parser.parse_args()

def get_model_scores(model_name):
    file_path = os.path.join(RESULTS_DIR,f"{model_name}.csv")
    # Open results file and set 'category' column as index
    df = pd.read_csv(file_path,index_col="category")
    return df

# Get model names
models = [model.strip() for model in args.models.split(",")]

# Get scores for every model
scores_dict = {model:get_model_scores(model) for model in models}

# Names shown in the plot
model_names = [model_name.strip() for model_name in args.model_names.split(",")] if args.model_names else models

# Combine the data into a single DataFrame with multi-index
combined_data = pd.concat({model: df for model, df in scores_dict.items()}, axis=1)

# Custom colormaps
cmaps = {
        'acc': mcolors.LinearSegmentedColormap.from_list("green_red", ["red", "yellow", "green"], N=100),
        'diff_bias':mcolors.LinearSegmentedColormap.from_list("orange_blue", ["orange", "white", "blue"], N=100)
        }

# Create a single figure with 2 subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes = axes.flatten()  # Flatten the array of axes for easy iteration

# Iterate over the scores and plot each heatmap in a subplot
for i, score in enumerate(['acc_a', 'acc_d', 'diff_bias_a', 'diff_bias_d']):

    # Extract data for the specific score, with models as columns and categories as rows
    plot_data = combined_data.xs(score, level=1, axis=1)

    cmap = cmaps['acc']
    center = 0.5
    vmin = 0
    vmax = 1
    if score.startswith("diff_bias"):
        cmap = cmaps['diff_bias']
        center = 0
        # WARNING!!!!! be careful with these settings
        vmin = -0.4
        vmax = 0.6

    # Plot each heatmap on the corresponding subplot
    sns.heatmap(plot_data, annot=True, cmap=cmap, cbar=True, fmt=".2f", center=center, ax=axes[i], vmin=vmin, vmax=vmax)
    
    # Customize each subplot
    axes[i].set_title(f'{score}',pad=15)
    axes[i].set_xticklabels(model_names)
    axes[i].xaxis.tick_top()  # Move x-axis ticks to the top
    axes[i].set(xlabel="", ylabel="")
    axes[i].set_aspect('equal')
    axes[i].tick_params(labelsize=10)

fig.suptitle(args.title, fontsize=14, fontweight='bold',y=0.95)

# Adjust layout for better spacing
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, hspace=0.3, wspace=0.2)

# Save the final plot
plt.savefig(os.path.join(RESULTS_DIR, f"{args.title}.png"))