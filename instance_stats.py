import os 
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np

current_date = datetime.now().date()

#####################################
## Get plot for template fertility ##
#####################################

data_frames = []

if not os.path.exists("stats/instance_stats"):
    os.makedirs("stats/instance_stats")

# Iterate over each stats file
for file in sorted(os.listdir("stats/template_fertility")):
    if file.endswith("csv"):
        file_path = f"stats/template_fertility/{file}"
        category = file.split(".")[0]
        data = pd.read_csv(file_path)
        # Fill missing 'version' values with '-'
        data['version'] = data['version'].fillna('-')
        # Add a column with the category name
        data['category'] = category 
        data_frames.append(data)

# Concatenate all data into a single DataFrame
all_data = pd.concat(data_frames)

# Group instances by category question_index 
all_data = all_data.groupby(['category', 'question_index'], as_index=False)['instances'].sum()

plt.figure(figsize=(13, 20))
sns.stripplot(x='category', y='instances', data=all_data, jitter=False, dodge=False)

# Define the upper limit for the y-axis
y_max = 1500
plt.ylim(0, y_max)

# Add labels to points with instances > 500
for i, row in all_data.iterrows():
    if row['instances'] > 500:
        # If instances exceed y_max, set y position to y_max for display purposes
        y_pos = min(row['instances'], y_max)
        plt.text(
            x=row['category'], 
            y=y_pos, 
            s=f"{row['question_index']} - {row['instances']}", 
            ha='right', va='bottom', fontsize=8, color='black'
        )

# Save plot
plt.savefig(f"stats/template_fertility/{current_date}_template-fertility.png", format='png', bbox_inches='tight', dpi=300)

########################################
## Get gender and social groups info ###
########################################

gender_category_dict = {}
groups_category_dict = {}
len_dict = {}

def instance_gender_info(row):
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
    return print(f"** WARNING: There is an error in {category} {row['question_index']} ({genders}).")

# Iterate over each category instance file 
for file in sorted(os.listdir("data_es")):
    if file.endswith(".full.csv"):
        
        category = file.split(".")[0]
    
        df = pd.read_csv(f"data_es/{file}",low_memory=False)
        
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
        
        # Group gender info at "question_index" level (i.e. at template level)
        gender_combinations = df.groupby('question_index')['stated_gender_info'].apply(lambda x: set(x)).to_dict()

        # Store in a new column the gender info of the instance at template level
        df['instance_gender_info'] = df.apply(instance_gender_info, axis=1)

        # Store the result in the dictionary
        gender_category_dict[category] = {
                                            'm': (df['instance_gender_info'] == 'm').sum(),
                                            'f': (df['instance_gender_info'] == 'f').sum(),
                                            'n': (df['instance_gender_info'] == 'n').sum(),
                                            'm vs. f': (df['instance_gender_info'] == 'm vs. f').sum(),
                                            'm_and_f': (df['instance_gender_info'] == 'm_and_f').sum(),
                                            'm_and_f_and_n': (df['instance_gender_info'] == 'm_and_f_and_n').sum(),
                                            'total': len(df)
                                        }
        
        if gender_category_dict[category]['total'] != sum(gender_category_dict[category].values())-gender_category_dict[category]['total']:
            print(f"WARNING: There is an error in {category}.")

        # Get stereotyped group of the instance from second element of the list in 'answer_info.ans0' 
        # Convert string representations to actual lists and keep only 2nd element
        df['answer_info.ans0'] = df['answer_info.ans0'].str.strip("[]").str.replace("'", "").str.split(", ")
        df['answer_info.ans0'] = df['answer_info.ans0'].apply(lambda x: x[-1])

        # Count occurrences of each group
        groups_category_dict[category] = df['answer_info.ans0'].value_counts().to_dict()

        # Write social group info to CSV
        with open(f"stats/instance_stats/{current_date}_social-groups_{category}.csv",mode="w",newline='') as groups_file:
            writer = csv.DictWriter(groups_file,fieldnames=['social_group','templates'])
            for group, value in groups_category_dict[category].items():
                writer.writerow({'social_group':group, 'templates':value})

# Write len info to CSV
with open(f"stats/instance_stats/{current_date}_len-info.csv", mode='w', newline='') as len_file:
    writer = csv.DictWriter(len_file, fieldnames=['category', 'instances'])
    writer.writeheader()
    for category, values in len_dict.items():
        writer.writerow({'category': category, 'instances':values})

# Write gender info to CSV
with open(f"stats/instance_stats/{current_date}_gender-info.csv", mode='w', newline='') as gender_file:
    writer = csv.DictWriter(gender_file, fieldnames=['category', 'm', 'f', 'n', 'm vs. f', 'm_and_f', 'm_and_f_and_n', 'total'])
    writer.writeheader()
    for category, values in gender_category_dict.items():
        row = {'category': category, **values}
        writer.writerow(row)

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

# Adjust layout to avoid overlapping subplots and make room for the legend
plt.tight_layout()

# Save plot
plt.savefig(f"stats/instance_stats/{current_date}_gender-info.png", format='png', bbox_inches='tight', dpi=300)

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
                    autopct=lambda pct: func(pct, sizes), 
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
plt.savefig(f"stats/instance_stats/{current_date}_social-groups.png", format='png', bbox_inches='tight', dpi=300)