import os 
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import ast
from collections import Counter
import numpy as np

current_date = datetime.now().date()

gender_category_dict = {}
groups_category_dict = {}

if not os.path.exists("stats/template_stats"):
    os.makedirs("stats/template_stats")

# Iterate over each category file 
for file in sorted(os.listdir("templates_es")):
    if file.startswith("final"):
        
        category = file.split("_")[1][:-5]
        
        df = pd.read_excel(f"templates_es/{file}")
        
        # Not include removed templates
        df = df[df['final_stereotyped_groups']!="-"]

        ######################
        ## Get gender info ###
        ######################

        # Replace NaN values in 'Stated_gender_info' with 'n' (neutral), except in Gender Identity
        if category != 'GenderIdentity':
            df['Stated_gender_info'] = df['Stated_gender_info'].fillna('n')
        else:
            # in Gender Identity, replace for 'm vs. f' except for the cases in which m or f is not the stereotyped group
            df['Stated_gender_info'] = np.where(
                (df['Stated_gender_info'].isna()) & (df['final_stereotyped_groups'].str.contains("trans",na=False)), 
                'n', 
                df['Stated_gender_info'].fillna('m vs. f')
            )
        # Replace 'fake-f' for 'n'
        df['Stated_gender_info'] = df['Stated_gender_info'].replace('fake-f', 'n')
        
        # Group all gender info. of every id
        ids = df.groupby('Q_id')['Stated_gender_info'].apply(lambda x: set(x)).to_dict()
        
        gender_category_dict[category] = {
                                    'm':len([id for id,gender in ids.items() if len(gender) == 1 and 'm' in gender]),
                                    'f':len([id for id,gender in ids.items() if len(gender) == 1 and 'f' in gender]),
                                    'n':len([id for id,gender in ids.items() if len(gender) == 1 and 'n' in gender]),
                                    'm vs. f':len([id for id,gender in ids.items() if len(gender) == 1 and 'm vs. f' in gender]),
                                    'm_and_f':len([id for id,gender in ids.items() if len(gender) == 2 and 'm' in gender and 'f' in gender]),
                                    'm_and_f_and_n':len([id for id,gender in ids.items() if len(gender) == 3]),
                                    'total':len(ids)
                                    }
        
        # Sum of all values in the dictionary must be equal to the total number of templates
        if gender_category_dict[category]['total'] != sum(gender_category_dict[category].values())-gender_category_dict[category]['total']:
            print(f"WARNING: There is an error in {category}.")

        ############################
        ## Get social groups info ##
        ############################

        # Convert strings to actual lists
        df['final_stereotyped_groups'] = df['final_stereotyped_groups'].apply(ast.literal_eval)

        # Group and create sets from lists by flattening each group
        groups = (
            df.groupby('Q_id')['final_stereotyped_groups']
            .apply(lambda x: set(item for sublist in x for item in sublist))
            .to_dict()
        )

        # Count occurrences of each group
        groups_category_dict[category] = dict(Counter(group for groups in groups.values() for group in groups))

        # Write social group info for each category to CSV
        with open(f"stats/template_stats/{current_date}_social-groups_{category}.csv",mode="w",newline='') as groups_file:
            writer = csv.DictWriter(groups_file,fieldnames=['social_group','templates'])
            for group, value in groups_category_dict[category].items():
                writer.writerow({'social_group':group, 'templates':value})

# Write all gender info to CSV 
with open(f"stats/template_stats/{current_date}_gender-info.csv", mode='w', newline='') as gender_file:
    writer = csv.DictWriter(gender_file, fieldnames=['category', 'm', 'f', 'n', 'm vs. f', 'm_and_f', 'm_and_f_and_n', 'total'])
    writer.writeheader()
    for category, values in gender_category_dict.items():
        writer.writerow({'category': category, **values})

##############################
## Get plot for gender info ##
##############################

fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()

# Function to format autopct labels and avoid 0% labels
def func(pct, allvalues):
    absolute = round(pct/100.*sum(allvalues), 0)  # Calculate the actual number
    if absolute == 0:  # Do not show labels if the value is 0
        return ''
    return f"{pct:.1f}%"

# Loop through each category and plot the pie chart
for i, (category, values) in enumerate(gender_category_dict.items()):
    labels = [key for key in values if key != 'total']
    sizes = [values[key] for key in labels]
    wedges, texts, autotexts = axes[i].pie(sizes, 
                                           autopct=lambda pct: func(pct, sizes), 
                                           startangle=90, 
                                           colors=plt.cm.tab20.colors[:len(labels)], 
                                           labels=None,
                                           pctdistance=0.8, 
                                           labeldistance=1.15,
                                           )
    
    # Ensure that the percentage labels do not overlap
    for autotext in autotexts:
        autotext.set_fontsize(12)

    axes[i].set_title(category)

# Add a single legend for all pie charts
legend_labels = ['m', 'f', 'neutral', 'm vs. f', 'm and f', 'm, f and neutral']
fig.legend(legend_labels, fontsize=14, ncol=6, loc='upper center')

plt.tight_layout()

# Save plot
plt.savefig(f"stats/template_stats/{current_date}_gender-info.png", format='png', bbox_inches='tight', dpi=300)

#####################################
## Get plot for social groups info ##
#####################################

fig, axs = plt.subplots(4, 2, figsize=(16, 20))

# Loop through each category and plot the pie chart
for i, (category, values) in enumerate(groups_category_dict.items()):
    labels = list(values.keys())
    sizes = list(values.values())
    colors = plt.cm.tab20.colors[:len(labels)] 
    ax = axs[i // 2, i % 2] 
    wedges = ax.pie(sizes, 
                    labels=None, 
                    colors=colors, 
                    autopct='%1.1f%%', 
                    startangle=140,
                    pctdistance=0.8, 
                    labeldistance=1.15)[0]
    
    # Ensure that the percentage labels do not overlap
    for autotext in autotexts:
        autotext.set_fontsize(10)
    
    ax.set_title(category, fontsize=14)
    
    # Add legend for each pie chart
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.92, right=0.8, hspace=0.4, wspace=0.2)

# Save plot
plt.savefig(f"stats/template_stats/{current_date}_social-groups.png", format='png', bbox_inches='tight', dpi=300)
