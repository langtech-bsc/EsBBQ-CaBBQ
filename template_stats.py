import os
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import ast
from collections import Counter
import numpy as np

OUTPUT_DIR = "stats/template_stats"
TEMPLATE_DIR = "templates_es"
current_date = datetime.now().date()

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to format pie chart percentage labels and avoid 0% labels
def format_pct_labels(pct, allvalues):
    absolute = round(pct / 100.0 * sum(allvalues), 0)
    return f"{pct:.1f}%" if absolute > 0 else ''

# Initialize dictionaries for gender and social group data
gender_category_dict = {}
groups_category_dict = {}

# Loop through each category
for file in sorted(os.listdir(TEMPLATE_DIR)):
    if not file.startswith("final"):
        continue

    category = file.split("_")[1][:-5]
    df = pd.read_excel(os.path.join(TEMPLATE_DIR, file))

    # Filter out removed templates
    df = df[df['final_stereotyped_groups'] != "-"]

    ######################
    ## Get gender info ###
    ######################

    # Replace NaN values in 'Stated_gender_info' with 'n' (neutral), except in Gender Identity
    if category != 'GenderIdentity':
        df['Stated_gender_info'] = df['Stated_gender_info'].fillna('n')
    else:
        # in Gender Identity, replace for 'm vs. f' except for the cases in which m or f is not the stereotyped group
        df['Stated_gender_info'] = np.where(
            df['Stated_gender_info'].isna() & df['final_stereotyped_groups'].str.contains("trans", na=False),
            'n',
            df['Stated_gender_info'].fillna('m vs. f')
        )
    
    # Replace 'fake-f' for 'n'
    df['Stated_gender_info'] = df['Stated_gender_info'].replace('fake-f', 'n')

    # Group gender information by Q_id
    ids = df.groupby('Q_id')['Stated_gender_info'].apply(lambda x: set(x)).to_dict()

    gender_category_dict[category] = {
        'm': sum(1 for gender in ids.values() if len(gender) == 1 and 'm' in gender),
        'f': sum(1 for gender in ids.values() if len(gender) == 1 and 'f' in gender),
        'n': sum(1 for gender in ids.values() if len(gender) == 1 and 'n' in gender),
        'm vs. f': sum(1 for gender in ids.values() if len(gender) == 1 and 'm vs. f' in gender),
        'm_and_f': sum(1 for gender in ids.values() if len(gender) == 2 and {'m', 'f'}.issubset(gender)),
        'm_and_f_and_n': sum(1 for gender in ids.values() if len(gender) == 3),
        'total': len(ids)
    }

    if gender_category_dict[category]['total'] != sum(gender_category_dict[category].values()) - gender_category_dict[category]['total']:
        print(f"WARNING: Mismatch in category {category}.")

    ############################
    ## Get social groups info ##
    ############################
    
    # Convert strings to actual lists
    df['final_stereotyped_groups'] = df['final_stereotyped_groups'].apply(ast.literal_eval)
    
    # Group and create sets from lists by flattening each group
    groups = df.groupby('Q_id')['final_stereotyped_groups'].apply(lambda x: set(item for sublist in x for item in sublist)).to_dict()
    
    # Count occurrences of each group
    groups_category_dict[category] = dict(Counter(group for groups in groups.values() for group in groups))

    # Write social group data to CSV
    with open(os.path.join(OUTPUT_DIR, f"{current_date}_social-groups_{category}.csv"), mode="w", newline='') as groups_file:
        writer = csv.DictWriter(groups_file, fieldnames=['social_group', 'templates'])
        writer.writeheader()
        for group, count in groups_category_dict[category].items():
            writer.writerow({'social_group': group, 'templates': count})

# Write all gender data to CSV
with open(os.path.join(OUTPUT_DIR, f"{current_date}_gender-info.csv"), mode='w', newline='') as gender_file:
    writer = csv.DictWriter(gender_file, fieldnames=['category', 'm', 'f', 'n', 'm vs. f', 'm_and_f', 'm_and_f_and_n', 'total'])
    writer.writeheader()
    for category, data in gender_category_dict.items():
        writer.writerow({'category': category, **data})

######################
## Plot gender info ##
######################

fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()
for i, (category, values) in enumerate(gender_category_dict.items()):
    labels, sizes = zip(*[(k, v) for k, v in values.items() if k != 'total'])
    wedges, _, autotexts = axes[i].pie(sizes,
                                       startangle=90, 
                                       autopct=lambda pct: format_pct_labels(pct, sizes),
                                       colors=plt.cm.tab20.colors[:len(labels)], 
                                       pctdistance=0.8,
                                       labeldistance=1.15
                                       )
    # Add category title
    axes[i].set_title(category, fontsize=14)
    # Ensure that the percentage labels do not overlap
    for autotext in autotexts:
        autotext.set_fontsize(10)

# Add a single legend for all pie charts
fig.legend(labels, fontsize=12, ncol=6, loc='upper center')
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(OUTPUT_DIR, f"{current_date}_gender-info.png"), format='png', bbox_inches='tight', dpi=300)

#############################
## Plot social groups info ##
#############################

fig, axs = plt.subplots(4, 2, figsize=(16, 20))
for i, (category, values) in enumerate(groups_category_dict.items()):
    labels, sizes = zip(*values.items())
    colors = plt.cm.tab20.colors[:len(labels)]
    ax = axs[i // 2, i % 2]
    wedges, _, autotexts = ax.pie(sizes, 
                                labels=None, 
                                colors=colors, 
                                autopct='%1.1f%%', 
                                startangle=140, 
                                pctdistance=0.8
                                )
    # Add category title and legend for each pie chart
    ax.set_title(category, fontsize=14)
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.92, right=0.8, hspace=0.4, wspace=0.2)

# Save plot
plt.savefig(os.path.join(OUTPUT_DIR, f"{current_date}_social-groups.png"), format='png', bbox_inches='tight', dpi=300)