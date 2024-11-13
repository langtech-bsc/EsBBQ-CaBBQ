import os 
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np

OUTPUT_DIR = "stats/instance_stats"
DATA_DIR = "data_es"
FERTILITY_DIR = "stats/template_fertility"
current_date = datetime.now().date()

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to format pie chart percentage labels and avoid 0% labels
def format_pct_labels(pct, allvalues):
    absolute = round(pct / 100.0 * sum(allvalues), 0)
    return f"{pct:.1f}%" if absolute > 0 else ''

#############################
## Plot template fertility ##
#############################

data_frames = []

# Iterate over each stats file
for file in sorted(os.listdir(FERTILITY_DIR)):
    if not file.endswith("csv"):
        continue
    
    category = file.split(".")[0]
    data = pd.read_csv(os.path.join(FERTILITY_DIR, file))
    # Fill missing 'version' values with '-'
    data['version'] = data['version'].fillna('-')
    # Add a column with the category name
    data['category'] = category 
    data_frames.append(data)

# Concatenate all data into a single DataFrame
all_data = pd.concat(data_frames)

# Group instances by category question_index 
all_data = all_data.groupby(['category', 'question_index'], as_index=False)['instances'].sum()

plt.figure(figsize=(13, 10))
sns.stripplot(x='category', y='instances', data=all_data, jitter=False, dodge=False)

# Define the upper limit for the y-axis
y_max = 1500
plt.ylim(0, y_max)

# Add labels for points with instances > 500
for _, row in all_data[all_data['instances'] > 500].iterrows():
    plt.text(x=row['category'], 
            y=min(row['instances'], 1500), 
            s=f"{row['question_index']} - {row['instances']}", 
            ha='right', 
            va='bottom', 
            fontsize=8
            )

# Save plot
plt.savefig(os.path.join(FERTILITY_DIR,f"{current_date}_template-fertility.png"), format='png', bbox_inches='tight', dpi=300)

########################################
## Get gender and social groups info ###
########################################

# Initialize dictionaries for gender and social group data, and total count
gender_category_dict = {}
groups_category_dict = {}
len_dict = {}

def get_instance_gender(row,gender_combinations):
    genders = gender_combinations[row['question_index']]
    if genders == {'m'}:
        return 'm'
    elif genders == {'f'}:
        return 'f'
    elif genders == {'n'}:
        return 'n'
    elif genders == {'m vs. f'}:
        return 'm vs. f'
    elif genders == {'m', 'f'}:
        return 'm_and_f'
    elif genders == {'m', 'f', 'n'}:
        return 'm_and_f_and_n'
    return print(f"** WARNING: Error in {category} {row['question_index']} ({genders}).")

# Iterate over each category instance file 
for file in sorted(os.listdir(DATA_DIR)):

    if not file.endswith(".full.csv"):
        continue
        
    category = file.split(".")[0]

    df = pd.read_csv(os.path.join(DATA_DIR, file),low_memory=False)
    
    # Save total number of instances
    len_dict[category] = len(df)
    
    # Replace NaN values in 'Stated_gender_info' with 'n', except in Gender Identity
    if category != 'GenderIdentity':
        df['stated_gender_info'] = df['stated_gender_info'].fillna('n')
    else:
        # in Gender Identity, replace for 'm vs. f' except for the cases in which m or f is not the stereotyped group
        df['stated_gender_info'] = np.where(
            (df['stated_gender_info'].isna()) & (df['stereotyped_groups'].str.contains("trans",na=False)), 
            'n', 
            df['stated_gender_info'].fillna('m vs. f')
        )
    # Replace 'fake-f' for 'n'
    df['stated_gender_info'] = df['stated_gender_info'].replace('fake-f', 'n')
    
    # Group gender info by "question_index" level (i.e. at template level)
    gender_combinations = df.groupby('question_index')['stated_gender_info'].apply(lambda x: set(x)).to_dict()

    # Store in a new column the gender info of the instance at template level
    df['instance_gender_info'] = df.apply(lambda row: get_instance_gender(row, gender_combinations), axis=1)

    # Store counts of gender at instance level
    gender_counts = df['instance_gender_info'].value_counts().reindex(['m', 'f', 'n', 'm vs. f', 'm_and_f', 'm_and_f_and_n'], fill_value=0).to_dict()
    gender_counts['total'] = len(df)
    gender_category_dict[category] = gender_counts
    
    if gender_category_dict[category]['total'] != sum(gender_category_dict[category].values())-gender_category_dict[category]['total']:
        print(f"WARNING: There is an error in {category}.")

    # Get stereotyped group of the instance from second element of the list in 'answer_info.ans0' 
    # Convert string representations to actual lists and keep only last element
    df['answer_info.ans0'] = df['answer_info.ans0'].str.strip("[]").str.replace("'", "").str.split(", ").str[-1]

    # Count occurrences of each group
    groups_category_dict[category] = df['answer_info.ans0'].value_counts().to_dict()

    # Write social group info to CSV
    with open(os.path.join(OUTPUT_DIR, f"{current_date}_social-groups_{category}.csv"),mode="w",newline='') as groups_file:
        writer = csv.DictWriter(groups_file,fieldnames=['social_group','templates'])
        for group, value in groups_category_dict[category].items():
            writer.writerow({'social_group':group, 'templates':value})

# Write len info to CSV
with open(os.path.join(OUTPUT_DIR,f"{current_date}_len-info.csv"), mode='w', newline='') as len_file:
    writer = csv.DictWriter(len_file, fieldnames=['category', 'instances'])
    writer.writeheader()
    for category, values in len_dict.items():
        writer.writerow({'category': category, 'instances':values})

# Write gender info to CSV
with open(os.path.join(OUTPUT_DIR,f"{current_date}_gender-info.csv"), mode='w', newline='') as gender_file:
    writer = csv.DictWriter(gender_file, fieldnames=['category', 'm', 'f', 'n', 'm vs. f', 'm_and_f', 'm_and_f_and_n', 'total'])
    writer.writeheader()
    for category, values in gender_category_dict.items():
        writer.writerow({'category': category, **values})

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