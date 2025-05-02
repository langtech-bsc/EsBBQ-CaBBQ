# Get avg scores in a csv file and
# Plot heatmap for every score type

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

# Names shown in the plot
model_names = {m:m for m in args.models}
model_names_to_compare = {m:m for m in args.models_to_compare}
if args.model_names:
    model_names = {m:n for (m,n) in list(zip(args.models,args.model_names))}
if args.model_names_to_compare:
    model_names_to_compare = {m:n for (m,n) in list(zip(args.models_to_compare,args.model_names_to_compare))}   

def open_results_file(model):
    results_dir = os.path.join(RESULTS_DIR, f"{model}/esbbq")
    results_file = [f for f in os.listdir(results_dir) if f.startswith("results")][0]
    data = open(f"{results_dir}/{results_file}")
    data = json.load(data)['results']
    return data

def get_clean_cat_scores(data_dict):
    scores = {score_name.split(",")[0]:score_val for score_name,score_val in data_dict.items() if score_name.split(",")[0] in score_names.keys()}
    return scores

def get_model_avg_scores(model):
    data = open_results_file(model)
    scores = get_clean_cat_scores(data['esbbq'])
    return scores

def get_model_cat_scores(model):
    data = open_results_file(model)
    scores = {cat:get_clean_cat_scores(scores) for cat,scores in data.items() if cat != "esbbq"}
    return scores

def get_df_score(data,score):
    rows = []
    for model, categories in data.items():
        for category, metrics in categories.items():
            rows.append({'Model': model, 'Category': category, 'Score': metrics[score]})
    df = pd.DataFrame(rows)
    return df

################################
### SAVE CSV WITH AVG SCORES ###
################################

all_scores_dict = {model_names[model]: get_model_avg_scores(model) for model in args.models}
df_avg_tmp = pd.DataFrame.from_dict(all_scores_dict, orient='index')[score_names.keys()]
df_avg_tmp['model_type'] = "base"

all_scores_dict_to_compare = {model_names_to_compare[model]: get_model_avg_scores(model) for model in args.models_to_compare}
df_avg_tmp_2 = pd.DataFrame.from_dict(all_scores_dict_to_compare, orient='index')[score_names.keys()]
df_avg_tmp_2['model_type'] = "instruct"

df_avg = pd.concat([df_avg_tmp,df_avg_tmp_2],ignore_index=False)
df_avg.to_csv((os.path.join(OUTPUT_DIR, f"avg_scores.csv")))

###############
### HEATMAP ###
###############

# Custom colormaps for heatmap
cmaps = {
    'acc': mcolors.LinearSegmentedColormap.from_list("green_red", ["#e31a1c", "#FFFF99", "#33a02c"], N=100),
    'bias_score': mcolors.LinearSegmentedColormap.from_list("orange_blue", ["#ff7f00", "white", "#1f78b4"], N=100)
}

# Get cat scores for every model
scores_dict = {model_names[model]: get_model_cat_scores(model) for model in args.models}
scores_dict_to_compare = {model_names_to_compare[model]: get_model_cat_scores(model) for model in args.models_to_compare}

# Iterate over the scores and plot each heatmap in a subplot
for score in list(score_names.keys()):

    # Create a single figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes = axes.flatten()  # Flatten the array of axes for easy iteration
    
    cmap = cmaps['acc']
    center = 0.5
    vmin = 0
    vmax = 1

    if score.startswith("bias_score"):
        cmap = cmaps['bias_score']
        center = 0
        vmin = -0.3
        vmax = 0.3

    # Create a shared colorbar normalization
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy data for colorbar
    
    for i in range(2):

        # Plot each heatmap on the corresponding subplot
        if i == 0:
            df_scores = get_df_score(scores_dict,score)
            heatmap_data = df_scores.pivot(index='Category', columns='Model', values='Score')
            # Reorder models
            heatmap_data = heatmap_data[model_names.values()]
            sns.heatmap(heatmap_data, annot=True, cmap=cmap, cbar=False,fmt=".2f", center=center, ax=axes[i], vmin=vmin, vmax=vmax)
            axes[i].set_title(f'Base Models', pad=15, fontweight="bold",fontsize=14)
            axes[i].set_yticklabels(categories, fontstyle='oblique')
        else: 
            df_scores_to_compare = get_df_score(scores_dict_to_compare,score)
            heatmap_data_to_compare = df_scores_to_compare.pivot(index='Category', columns='Model', values='Score')
            # Reorder models
            heatmap_data_to_compare = heatmap_data_to_compare[model_names_to_compare.values()]
            sns.heatmap(heatmap_data_to_compare, annot=True, cmap=cmap, cbar=False,fmt=".2f", center=center, ax=axes[i], vmin=vmin, vmax=vmax)
            axes[i].set_title(f'Instructed Models', pad=15, fontweight="bold",fontsize=14)
            axes[i].set_yticklabels("")
        
        # axes[i].set_xticklabels(model_names,rotation=90, fontproperties=monospace_font)
        axes[i].tick_params(axis='x', labelrotation=90, labelfontfamily="monospace")
        axes[i].xaxis.tick_top()  # Move x-axis ticks to the top
        axes[i].set(xlabel="", ylabel="")
        axes[i].set_aspect('equal')
        axes[i].tick_params(labelsize=12)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.37, 0.07, 0.37, 0.02])  # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")

    fig.suptitle(score_names[score], fontsize=20, fontweight='bold',x=0.54,y=0.99,fontstyle="oblique")

    # Adjust layout for better spacing
    plt.subplots_adjust(top=0.65,left=0.2,wspace=0.1)

    # Save the final plot
    plt.savefig(os.path.join(OUTPUT_DIR, f"{args.title}_{score}.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{args.title}_{score}.pdf"), format="pdf")