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
import matplotlib.patches as patches
from bias_score import *

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir',help="directory with Harness results")
parser.add_argument('--output_dir',help="output directory")
parser.add_argument('--language',choices=['ca','es'], required=True)
parser.add_argument('--models_base', help="model name(s)",nargs="+")
parser.add_argument('--models_instruct',nargs="*")
parser.add_argument('--model_names_base', help="model name(s) in the plot. If more than one, in the same order as given in --models_base", required=False, nargs="*")
parser.add_argument('--model_names_instruct',help="model name(s) in the plot. If more than one, in the same order as given in --models_instruct", required=False, nargs="*")
parser.add_argument('--title', help="custom title for the plot. Will be also the name of the output png", required=False, default="base-instruct")

args = parser.parse_args()

score_names = {"acc_ambig":f"Acc{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "acc_disambig":f"Acc{r'$_d$'}{r'$_i$'}{r'$_s$'}{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "bias_score_ambig":f"Bias{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "bias_score_disambig":f"Bias{r'$_d$'}{r'$_i$'}{r'$_s$'}{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "normalized_bias_ambig":f"Bias{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "normalized_bias_disambig":f"Bias{r'$_d$'}{r'$_i$'}{r'$_s$'}{r'$_a$'}{r'$_m$'}{r'$_b$'}"}

categories = [
    # "Avg.",
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
model_names_base = {m:m for m in args.models_base}
model_names_instruct = {m:m for m in args.models_instruct}
if args.model_names_base:
    model_names_base = {m:n for (m,n) in list(zip(args.models_base,args.model_names_base))}
if args.model_names_instruct:
    model_names_instruct = {m:n for (m,n) in list(zip(args.models_instruct,args.model_names_instruct))}   

def open_results_file(model,language):
    results_dir = os.path.join(args.results_dir, f"{model}/{language}bbq")
    results_file = [f for f in os.listdir(results_dir) if f.startswith("results")][0]
    data = open(f"{results_dir}/{results_file}")
    data = json.load(data)['results']
    return data

def get_clean_cat_results(data_dict):
    scores = {score_name.split(",")[0]:score_val for score_name,score_val in data_dict.items() if score_name.split(",")[0] in score_names.keys()}
    return scores

def get_model_results(model,language):
    data = open_results_file(model,language)
    scores = {cat:get_clean_cat_results(scores) for cat,scores in data.items()}
    # rename key with avg score
    scores['avg'] = scores[f'{language}bbq']
    del scores[f'{language}bbq']
    # get max bias score
    for cat in scores.keys():
        scores[cat]['upper_bound_bias_ambig'] = upper_bound_bias_score(scores[cat]['acc_ambig'],'ambig')
        scores[cat]['upper_bound_bias_disambig'] = upper_bound_bias_score(scores[cat]['acc_disambig'],'disambig')
        # scores[cat]['normalized_bias_ambig'] = normalized_bias_score(scores[cat]['bias_score_ambig'],scores[cat]['upper_bound_bias_ambig'])
        # scores[cat]['normalized_bias_disambig'] = normalized_bias_score(scores[cat]['bias_score_disambig'],scores[cat]['upper_bound_bias_disambig'])
    return scores

def get_df_cat_score(data,score):
    rows = []
    for model, categories in data.items():
        for category, metrics in categories.items():
            # do not include avg. scores
            if category != "avg":
                rows.append({'Model': model, 'Category': category, 'Score': metrics[score]})
    df = pd.DataFrame(rows)
    return df

# Get scores for every model
scores_dict_base = {model_names_base[model]: get_model_results(model,args.language) for model in args.models_base}
scores_dict_instruct = {model_names_instruct[model]: get_model_results(model,args.language) for model in args.models_instruct}

# Save csv with avg scores
df_tmp = pd.DataFrame({model: metrics['avg'] for model, metrics in scores_dict_base.items()}).T
df_tmp['model_type'] = 'base'
df_tmp_2 = pd.DataFrame({model: metrics['avg'] for model, metrics in scores_dict_instruct.items()}).T
df_tmp_2['model_type'] = 'instruct'
df_avg = pd.concat([df_tmp,df_tmp_2])
df_avg.to_csv(os.path.join(args.output_dir, f"{args.language}bbq_avg_scores.csv"))

# Get heatmap

# Custom colormaps for heatmap
cmaps = {
    'acc': mcolors.LinearSegmentedColormap.from_list("green_red", ["#e31a1c", "#FFFF99", "#33a02c"], N=100),
    'bias_score': mcolors.LinearSegmentedColormap.from_list("orange_blue", ["#ff7f00", "white", "#1f78b4"], N=100)
}

# Iterate over the scores and plot each heatmap in a subplot
for score in ['acc_ambig','acc_disambig','bias_score_ambig','bias_score_disambig']:
#for score in ['acc_ambig','acc_disambig','normalized_bias_ambig','normalized_bias_disambig']:

    # Create a single figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(25,8))
    axes = axes.flatten()  # Flatten the array of axes for easy iteration
    
    cmap = cmaps['acc']
    center = 0.5
    vmin = 0
    vmax = 1

    if 'bias' in score:
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
            df_scores_base = get_df_cat_score(scores_dict_base,score)
            heatmap_data_base = df_scores_base.pivot(index='Category', columns='Model', values='Score')
            # Reorder models
            heatmap_data_base = heatmap_data_base[model_names_base.values()]
            sns.heatmap(heatmap_data_base, annot=True, cmap=cmap, cbar=False,fmt=".2f", center=center, ax=axes[i], vmin=vmin, vmax=vmax,annot_kws={"size": 14})
            axes[i].set_title(f'Base Models', pad=15, fontweight="bold",fontsize=20)
            axes[i].set_yticklabels(categories, fontstyle='oblique')
            # Change the first label (avg)
            # axes[i].get_yticklabels()[0].set_fontweight("demi")  # Make bold
            # axes[i].get_yticklabels()[0].set_fontstyle('normal')  # Make normal (not oblique)
        else: 
            df_scores_instruct = get_df_cat_score(scores_dict_instruct,score)
            heatmap_data_instruct = df_scores_instruct.pivot(index='Category', columns='Model', values='Score')
            # Reorder models
            heatmap_data_instruct = heatmap_data_instruct[model_names_instruct.values()]
            sns.heatmap(heatmap_data_instruct, annot=True, cmap=cmap, cbar=False,fmt=".2f", center=center, ax=axes[i], vmin=vmin, vmax=vmax,annot_kws={"size": 14})
            axes[i].set_title(f'Instructed Models', pad=15, fontweight="bold",fontsize=20)
            axes[i].set_yticklabels("")
        
        # axes[i].set_xticklabels(model_names_base,rotation=90, fontproperties=monospace_font)
        axes[i].tick_params(axis='x', labelrotation=90, labelfontfamily="monospace")
        axes[i].xaxis.tick_top()  # Move x-axis ticks to the top
        axes[i].set(xlabel="", ylabel="")
        # axes[i].set_aspect('equal')
        axes[i].tick_params(labelsize=16)

        # Make first row (avg score) bold and black
        # for text in axes[i].texts:
        #     x, y = text.get_position()
        #     if int(y) == 0: 
        #         text.set_color('black')
        #         text.set_fontweight("demi")
        
        # # Add a horizontal white line to separate the first row from the rest
        # axes[i].hlines(y=1, xmin=0, xmax=heatmap_data.shape[1], colors='white', linewidth=3)

    # Add shared colorbar
    # cbar_ax = fig.add_axes([0.34, 0.05, 0.37, 0.02])  # [left, bottom, width, height]
    # fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    
    # Set score as title
    # fig.suptitle(score_names[score], fontsize=20, fontweight='bold',x=0.52,y=0.99,fontstyle="oblique")

    # Adjust layout for better spacing
    # plt.subplots_adjust(top=0.65,left=0.15,wspace=0.03)
    plt.subplots_adjust(top=0.6,left=0.15,bottom=0.08,wspace=0.03)
    
    # Save the final plot
    plt.savefig(os.path.join(args.output_dir, f"{args.language}bbq_{args.title}_{score}.png"))
    plt.savefig(os.path.join(args.output_dir, f"{args.language}bbq_{args.title}_{score}.pdf"), format="pdf")

    # Save scores per category
    # df_scores['model_type'] = "base"
    # df_scores_instruct['model_type'] = "instruct"
    # final_df = pd.concat([df_scores,df_scores_instruct],ignore_index=True)
    # # Set model and model types in column axis
    # pivot_df = final_df.pivot(index='Category', columns=['Model', 'model_type'], values='Score')
    # # Sort the column MultiIndex: by Model, then model_type (base first, instruct second)
    # pivot_df = pivot_df.sort_index(axis=1, level=[0, 1], sort_remaining=False)
    # pivot_df.to_csv(os.path.join(args.output_dir, f"{score}_scores.csv"))