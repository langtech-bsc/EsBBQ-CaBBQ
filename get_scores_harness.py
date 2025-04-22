import os
import argparse
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from matplotlib.font_manager import FontProperties

# Define monospace font properties
monospace_font = FontProperties(family="monospace")

parser = argparse.ArgumentParser()
parser.add_argument('--models', help="model name(s)",nargs="+")
parser.add_argument('--models_to_compare',nargs="*")
parser.add_argument('--model_names', help="model name(s) in the plot. If more than one, in the same order as given in --models", required=False, nargs="*")
parser.add_argument('--model_names_to_compare',help="model name(s) in the plot. If more than one, in the same order as given in --models_to_compare", required=False, nargs="*")
parser.add_argument('--title', help="custom title for the plot. Will be also the name of the output png", required=False, default="base-instruct")

args = parser.parse_args()

RESULTS_DIR = "results/harness/prod-harness"
OUTPUT_DIR = "results/bias_score/prod-harness"

score_names = {"acc_ambig":f"Acc{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "acc_disambig":f"Acc{r'$_d$'}{r'$_i$'}{r'$_s$'}{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "bias_score_ambig":f"Bias{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "bias_score_disambig":f"Bias{r'$_d$'}{r'$_i$'}{r'$_s$'}{r'$_a$'}{r'$_m$'}{r'$_b$'}"}

categories = [
    "Age",
    "Disability Status",
    "Gender",
    "LGBTQIA",
    "Nationality",
    "Physical Appearance",
    "Race/Ethnicity",
    "Religion",
    "SES",
    "Spanish Region"
]

def get_clean_cat_scores(data_dict):
    scores = {score_name.split(",")[0]:score_val for score_name,score_val in data_dict.items() if score_name.split(",")[0] in score_names.keys()}
    return scores

def get_model_scores(model):
    results_dir = os.path.join(RESULTS_DIR, f"{model}/esbbq")
    results_file = [f for f in os.listdir(results_dir) if f.startswith("results")][0]
    data = open(f"{results_dir}/{results_file}")
    data = json.load(data)['results']
    scores = {cat:get_clean_cat_scores(scores) for cat,scores in data.items() if cat != "esbbq"}
    return scores

def get_df_score(data,score):
    rows = []
    for model, categories in data.items():
        for category, metrics in categories.items():
            rows.append({'Model': model, 'Category': category, 'Score': metrics[score]})
    df = pd.DataFrame(rows)
    return df

# Custom colormaps for heatmap
cmaps = {
    'acc': mcolors.LinearSegmentedColormap.from_list("green_red", ["#e31a1c", "#FFFF99", "#33a02c"], N=100),
    'bias_score': mcolors.LinearSegmentedColormap.from_list("orange_blue", ["#ff7f00", "white", "#1f78b4"], N=100)
}

##########################################
## HEATMAP COMPARISON Base vs. Instruct ##
##########################################

# Get scores for every model
scores_dict = {model: get_model_scores(model) for model in args.models}
scores_dict_to_compare = {model: get_model_scores(model) for model in args.models_to_compare}

# Names shown in the plot
model_names = args.model_names if args.model_names else args.models
model_names_to_compare = args.model_names_to_compare if args.model_names_to_compare else args.models_to_compare

# Iterate over the scores and plot each heatmap in a subplot
for score in list(score_names.keys()):

    # Create a single figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes = axes.flatten()  # Flatten the array of axes for easy iteration
    
    cmap = cmaps['acc']
    center = 0.5
    vmin = 0
    vmax = 1

    if score.startswith("bias_score"):
        cmap = cmaps['bias_score']
        center = 0
        vmin = -0.5
        vmax = 0.5

    # Create a shared colorbar normalization
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy data for colorbar
    
    for i in range(2):

        # Plot each heatmap on the corresponding subplot
        if i == 0:
            df_scores = get_df_score(scores_dict,score)
            heatmap_data = df_scores.pivot(index='Category', columns='Model', values='Score')
            sns.heatmap(heatmap_data, annot=True, cmap=cmap, cbar=False,fmt=".2f", center=center, ax=axes[i], vmin=vmin, vmax=vmax)
            axes[i].set_title(f'Base Models', pad=15, fontweight="bold",fontsize=14)
            axes[i].set_yticklabels(categories, fontstyle='oblique')
        else: 
            df_scores_to_compare = get_df_score(scores_dict_to_compare,score)
            heatmap_data_to_compare = df_scores_to_compare.pivot(index='Category', columns='Model', values='Score')
            sns.heatmap(heatmap_data_to_compare, annot=True, cmap=cmap, cbar=False,fmt=".2f", center=center, ax=axes[i], vmin=vmin, vmax=vmax)
            axes[i].set_title(f'Instructed Models', pad=15, fontweight="bold",fontsize=14)
            axes[i].set_yticklabels("")
        
        axes[i].set_xticklabels(model_names,rotation=90, fontproperties=monospace_font)
        axes[i].xaxis.tick_top()  # Move x-axis ticks to the top
        axes[i].set(xlabel="", ylabel="")
        axes[i].set_aspect('equal')
        axes[i].tick_params(labelsize=12)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.37, 0.07, 0.37, 0.02])  # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")

    fig.suptitle(score_names[score], fontsize=20, fontweight='bold',x=0.54,y=0.99,fontstyle="oblique")

    # Adjust layout for better spacing
    plt.subplots_adjust(top=0.65,left=0.2,wspace=0.05)

    # Save the final plot
    plt.savefig(os.path.join(OUTPUT_DIR, f"{args.title}_{score}.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{args.title}_{score}.pdf"), format="pdf")