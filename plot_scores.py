import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from matplotlib.font_manager import FontProperties

# Define monospace font properties
monospace_font = FontProperties(family="monospace")

metrics = ["kobbq","basqbbq"]

parser = argparse.ArgumentParser()
parser.add_argument('--models', help="model name(s)",nargs="+")
parser.add_argument('--models_to_compare',nargs="*")
parser.add_argument('--model_names', help="model name(s) in the plot. If more than one, in the same order as given in -m", required=False, nargs="*")
parser.add_argument('--model_names_to_compare',nargs="*")
parser.add_argument('--title', default="", help="custom title for the plot. Will be also the name of the output png", required=False)
parser.add_argument("--metric", nargs="+", choices=metrics, default=metrics)
parser.add_argument("--old_harness", action="store_true")
parser.add_argument("--heatmap", action="store_true")
parser.add_argument("--heatmap_comparison", action="store_true")
parser.add_argument("--stripplot", action="store_true")
parser.add_argument("--v05", action="store_true")
parser.add_argument("--bbq_original",action="store_true")

args = parser.parse_args()

RESULTS_DIR = "results/bias_score"
if args.old_harness:
    RESULTS_DIR = "results/bias_score/old-harness"
if args.v05:
    RESULTS_DIR = "results/v0.5/bias_score/old-harness"
if args.bbq_original:
    RESULTS_DIR = "results/bbq_original/bias_score/old-harness"

OUTPUT_DIR = f"{RESULTS_DIR}/plots"

def get_model_scores(metric,model_name):
    file_path = os.path.join(RESULTS_DIR, f"{metric}_{model_name}.csv")
    # Open results file and set 'category' column as index
    df = pd.read_csv(file_path, index_col="category")
    return df

score_names = {"acc_a":f"Acc{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "acc_d":f"Acc{r'$_d$'}{r'$_i$'}{r'$_s$'}{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "diff_bias_a":f"Bias{r'$_a$'}{r'$_m$'}{r'$_b$'}",
                "diff_bias_d":f"Bias{r'$_d$'}{r'$_i$'}{r'$_s$'}{r'$_a$'}{r'$_m$'}{r'$_b$'}"}

# score_names = {"acc_a":f"Acc{r'$_a$'}",
#                 "acc_d":f"Acc{r'$_d$'}",
#                 "diff_bias_a":f"Difference{r'$_a$'}",
#                 "diff_bias_d":f"Difference{r'$_d$'}"}

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

# Custom colormaps for heatmap
# cmaps = {
#     'acc': mcolors.LinearSegmentedColormap.from_list("green_red", ["red", "yellow", "green"], N=100),
#     'diff_bias': mcolors.LinearSegmentedColormap.from_list("orange_blue", ["orange", "white", "blue"], N=100)
# }

# HTML codes for paired palette
# ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
cmaps = {
    'acc': mcolors.LinearSegmentedColormap.from_list("green_red", ["#e31a1c", "#FFFF99", "#33a02c"], N=100),
    'diff_bias': mcolors.LinearSegmentedColormap.from_list("orange_blue", ["#ff7f00", "white", "#1f78b4"], N=100)
}

# Colors for technical report
# set2_palette = plt.get_cmap("Set2", 8)
# cmaps = {
#     'acc': mcolors.LinearSegmentedColormap.from_list("set2_acc", [set2_palette(1), set2_palette(5), set2_palette(4)], N=100),
#     'diff_bias': mcolors.LinearSegmentedColormap.from_list("set2_diff_bias", [set2_palette(0), "white", set2_palette(1)], N=100)
# }

#############
## HEATMAP ##
#############

if args.heatmap: 

    # If a single model's heatmap
    if len(args.models) == 1:
        model = args.models[0]

        for metric in args.metric:
            
            df_model = get_model_scores(metric,model)

            # Create a single figure with 2 subplots (1 row, 2 columns)
            fig, axes = plt.subplots(1, 2, figsize=(9, 8))

            for i, score in enumerate(list(score_names.keys())):
                cmap = cmaps['acc']
                center = 0.5
                vmin = 0
                vmax = 0.8

                if score.startswith("diff_bias"):
                    cmap = cmaps['diff_bias']
                    center = 0
                    vmin = -0.5
                    vmax = 0.5

                # Select only columns corresponding to the score
                df_score = df_model.filter(regex=f'^{score}')
                
                # Plot each heatmap on the corresponding subplot
                sns.heatmap(df_score, annot=True, cmap=cmap, fmt=".2f", cbar=True, center=center, ax=axes[i], vmin=vmin, vmax=vmax)
                
                # Customize each subplot
                axes[i].xaxis.tick_top()  # Move x-axis ticks to the top
                axes[i].set(xlabel="", ylabel="")
                axes[i].set_xticklabels(srotation=45)
                axes[i].set_aspect('equal')
                axes[i].tick_params(labelsize=10)

            # Adjust layout for better spacing
            plt.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.3)

            fig.suptitle(model, fontsize=14, fontweight='bold', y=0.95)

            # Save the final plot
            plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_{model}.png"))
            plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_{model}.pdf"),format="pdf")

    # If a multiple model heatmap
    else:
        
        for metric in args.metric:

            # Get scores for every model
            scores_dict = {model: get_model_scores(metric,model) for model in args.models}

            # Names shown in the plot
            model_names = args.model_names if args.model_names else args.models

            # Combine the data into a single DataFrame with multi-index
            combined_data = pd.concat({model: df for model, df in scores_dict.items()}, axis=1)

            # Create a single figure with 2 subplots (2 rows, 2 columns)
            fig, axes = plt.subplots(2, 2, figsize=(13, 13)) # before 10, 10
            axes = axes.flatten()  # Flatten the array of axes for easy iteration

            # Iterate over the scores and plot each heatmap in a subplot
            for i, score in enumerate(list(score_names.keys())):

                # Extract data for the specific score, with models as columns and categories as rows
                plot_data = combined_data.xs(score, level=1, axis=1)

                cmap = cmaps['acc']
                center = 0.5
                vmin = 0
                vmax = 1

                if score.startswith("diff_bias"):
                    cmap = cmaps['diff_bias']
                    center = 0
                    vmin = -0.5
                    vmax = 0.5

                # Plot each heatmap on the corresponding subplot
                sns.heatmap(plot_data, annot=True, cmap=cmap, cbar=True,fmt=".2f", center=center, ax=axes[i], vmin=vmin, vmax=vmax)

                # Customize each subplot
                axes[i].set_title(f'{score_names[score]}', pad=15, fontweight="bold",fontsize=16)
                axes[i].set_xticklabels(model_names,rotation=90)
                axes[i].xaxis.tick_top()  # Move x-axis ticks to the top
                axes[i].set(xlabel="", ylabel="")
                axes[i].set_aspect('equal')
                axes[i].tick_params(labelsize=12)

            fig.suptitle(args.title, fontsize=14, fontweight='bold', y=0.95)

            # Adjust layout for better spacing
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.4, wspace=0.01)

            # Save the final plot
            plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_{args.title}.png"))
            plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_{args.title}.pdf"), format="pdf")

            # Save all scores
            combined_data.to_csv(os.path.join(RESULTS_DIR, f"{metric}_{args.title}_all_scores.csv"))

########################
## HEATMAP COMPARISON ##
########################

if args.heatmap_comparison:

    if len(args.models) == 1:

        pass
        #TODO

    else: 
        
        for metric in args.metric:

            # Get scores for every model
            scores_dict = {model: get_model_scores(metric,model) for model in args.models}
            scores_dict_to_compare = {model: get_model_scores(metric,model) for model in args.models_to_compare}

            # Names shown in the plot
            model_names = args.model_names if args.model_names else args.models
            model_names_to_compare = args.model_names_to_compare if args.model_names_to_compare else args.models_to_compare

            # Combine the data into a single DataFrame with multi-index
            combined_data = pd.concat({model: df for model, df in scores_dict.items()}, axis=1)
            combined_data_to_compare = pd.concat({model: df for model, df in scores_dict_to_compare.items()}, axis=1)

            # Iterate over the scores and plot each heatmap in a subplot
            for score in list(score_names.keys()):

                # Create a single figure with 4 subplots
                fig, axes = plt.subplots(1, 2, figsize=(12, 8))
                axes = axes.flatten()  # Flatten the array of axes for easy iteration

                # Extract data for the specific score, with models as columns and categories as rows
                plot_data = combined_data.xs(score, level=1, axis=1)
                plot_data_to_compare = combined_data_to_compare.xs(score, level=1, axis=1)
                
                cmap = cmaps['acc']
                center = 0.5
                vmin = 0
                vmax = 1

                if score.startswith("diff_bias"):
                    cmap = cmaps['diff_bias']
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
                        sns.heatmap(plot_data, annot=True, cmap=cmap, cbar=False,fmt=".2f", center=center, ax=axes[i], vmin=vmin, vmax=vmax)
                        axes[i].set_title(f'Base Models', pad=15, fontweight="bold",fontsize=14)
                        axes[i].set_yticklabels(categories, fontstyle='oblique')
                    else: 
                        sns.heatmap(plot_data_to_compare, annot=True, cmap=cmap, cbar=False,fmt=".2f", center=center, ax=axes[i], vmin=vmin, vmax=vmax)
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
                plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_{args.title}_{score}.png"))
                plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_{args.title}_{score}.pdf"), format="pdf")

#################
## SCATTERPLOT ##
#################

custom = {"axes.edgecolor": "black", "axes.grid":False}
sns.set_style("whitegrid",rc=custom)

if args.stripplot:

    for metric in args.metric:
        
        mean_scores = pd.DataFrame()
        for i,model in enumerate(args.models):
            model_mean_scores = pd.read_csv(os.path.join(RESULTS_DIR,f"{metric}_mean_{model}.csv"))
            model_mean_scores['model_name'] = args.model_names[i]
            model_mean_scores['model_type'] = "instruct" if "instruct" in model.lower() else "base"
            mean_scores = pd.concat([mean_scores,model_mean_scores],ignore_index=True)
        
        df_long = mean_scores.melt(id_vars=['model', 'model_name', 'model_type'], value_vars=list(score_names.keys()), var_name='score_type', value_name='score')
        palette = sns.color_palette(palette='Paired')

        fig, axes = plt.subplots(3, 1, figsize=(20, 8))
        axes = axes.flatten() 

        for i, score_type in enumerate(['acc','diff']):

            ax = axes[i]
            base = df_long[(df_long['model_type'] == "base") & (df_long['score_type'].str.startswith(score_type))]
            instruct = df_long[(df_long['model_type'] == "instruct") & (df_long['score_type'].str.startswith(score_type))]
            
            sns.stripplot(y='score_type', x='score', data=base, hue='model_name', 
                            dodge=False, jitter=False, marker="o", alpha=1, palette=palette, 
                            legend=True, size=15, ax=ax)
            sns.stripplot(y='score_type', x='score', data=instruct, hue='model_name', 
                            dodge=False, jitter=False, marker="X", alpha=1, palette=palette, 
                            legend=True, size=15, ax=ax)
                    
            ax.set_xlabel('') 
            ax.set_ylabel('')
            ax.tick_params(axis='x', labelsize=16)

            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

            separation_lines = [0.5]
            for line in separation_lines:
                ax.axhline(line, color='black', linewidth=1)

            if score_type == "acc":
                ax.set_yticklabels(list(score_names.values())[0:2],fontweight="bold",fontsize=20,fontstyle="oblique")
                ax.set_xlim(0, 1)
            else:
                ax.set_yticklabels(list(score_names.values())[2:4],fontweight="bold",fontsize=20,fontstyle="oblique")
                ax.set_xlim(-1, 1)
        
        ax = axes[2]
        sns.stripplot(y='score_type', x='score', data=base, hue='model_name', 
                        dodge=False, jitter=False, marker="o", alpha=1, palette=palette, 
                        legend=True, size=15, ax=ax)
        sns.stripplot(y='score_type', x='score', data=instruct, hue='model_name', 
                        dodge=False, jitter=False, marker="X", alpha=1, palette=palette, 
                        legend=True, size=15, ax=ax)

        ax.set_xlabel('') 
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=16)

        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        separation_lines = [0.5]
        for line in separation_lines:
            ax.axhline(line, color='black', linewidth=1)

        ax.set_yticklabels(list(score_names.values())[2:4],fontweight="bold",fontsize=20,fontstyle="oblique")
        ax.set_xlim(0, 0.3)

        # Add a shared legend
        fig.legend(handles[:9], labels[:9], loc="upper center", bbox_to_anchor=(0.5,0.95), ncols=5, edgecolor="black", prop={'family': 'monospace', 'size': 20})

        plt.tight_layout(rect=[0, 0, 1, 0.75])
        plt.subplots_adjust(hspace=0.8)
        
        plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_{args.title}_mean.png"))
        plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_{args.title}_mean.pdf"), format="pdf")
