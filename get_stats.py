import os
import argparse
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import ast
from collections import Counter
import numpy as np
import warnings
import seaborn as sns

languages = ["es","ca"]

parser = argparse.ArgumentParser()
parser.add_argument("--language", choices=languages, help="language of the instances.")
parser.add_argument("--templates", action="store_true", help="Process templates.")
parser.add_argument("--instances", action="store_true", help="Process instances.")

args = parser.parse_args()
if not (args.templates or args.instances):
    parser.error("At least one of --templates or --instances is required.")

lang = args.language

TEMPLATE_DIR = "templates"
DATA_DIR = f"data_{lang}"
FERTILITY_DIR = f"stats/{lang}/template_fertility"
OUTPUT_DIR_TEMPLATES = f"stats/{lang}/template_stats"
OUTPUT_DIR_INSTANCES = f"stats/{lang}/instance_stats"

current_date = datetime.now().date()

# rename columns to remove lang specification
def rename_columns(df, lang):
    return df.rename(columns={c:c.rsplit("_",1)[0] for c in df.columns if c.endswith(f"_{lang}")})

# Replace NaN values in 'Stated_gender_info' with 'n' (neutral), except in Gender
def update_stated_gender_info(df):
    gender_info_col = "stated_gender_info"
    if category == 'Gender':
        df[gender_info_col] = df[gender_info_col].fillna('m vs. f')
    else:
        df[gender_info_col] = df[gender_info_col].fillna('n')
    # Replace 'fake-f' for 'n'
    df[gender_info_col] = df[gender_info_col].replace('fake-f', 'n')
    return df

def get_instance_gender(row,gender_combinations):
    genders = gender_combinations[row['template_id']]
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
    elif genders == {'m', 'n'}:
        return 'm_and_n'
    elif genders == {'f', 'n'}:
        return 'f_and_n'    
    elif genders == {'m', 'f', 'n'}:
        return 'm_and_f_and_n'
    return warnings.warn(f"Error in {category} {row['template_id']} ({genders}).")

# Function to format pie chart percentage labels and avoid 0% labels
def format_pct_labels(pct, allvalues):
    absolute = round(pct / 100.0 * sum(allvalues), 0)
    return f"{pct:.1f}%" if absolute > 0 else ''

def plot_gender_info(output_dir,gender_category_dict):
    fig, axes = plt.subplots(2, 5, figsize=(15, 9))
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
    fig.legend(labels, fontsize=12, ncol=9, loc='upper center')

    # plt.subplots_adjust(top=0.8, right=0.8, hspace=0.1, wspace=0.1)
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(output_dir, f"{current_date}_gender-info.png"), format='png', bbox_inches='tight', dpi=300)

def plot_social_groups_info(output_dir,groups_category_dict):
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    axes = axes.flatten()
    for i, (category, values) in enumerate(groups_category_dict.items()):
        labels, sizes = zip(*values.items())
        colors = plt.cm.tab20.colors[:len(labels)]
        wedges, _, autotexts = axes[i].pie(sizes, 
                                    labels=None, 
                                    colors=colors, 
                                    autopct='%1.1f%%', 
                                    startangle=140, 
                                    pctdistance=0.8
                                    )
        # Add category title and legend for each pie chart
        axes[i].set_title(category, fontsize=14)
        axes[i].legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.8, right=0.8, hspace=0.1, wspace=0.1)

    # Save plot
    plt.savefig(os.path.join(output_dir, f"{current_date}_social-groups.png"), format='png', bbox_inches='tight', dpi=300)

####################
## TEMPLATE STATS ##
####################

if args.templates:

    print("\nGetting template stats.\n")

    # Initialize dictionaries for gender and social group data
    gender_category_dict = {}
    groups_category_dict = {}

    # Loop through each category
    for file in sorted(os.listdir(TEMPLATE_DIR)):
        if file.startswith("vocabulary"):
            continue

        category = file.split(".")[0]
        df = pd.read_excel(os.path.join(TEMPLATE_DIR, file))

        # filter columns according to language
        if lang == "es":
            df = df.drop(columns=[column for column in df.columns if column.endswith("_ca")])
        elif lang == "ca":
            df = df.drop(columns=[column for column in df.columns if column.endswith("_es")])
        # rename columns
        df = rename_columns(df,lang)
        
        print(f"Processing {category} category.")

        ######################
        ## Get gender info ###
        ######################

        # Update stated_gender_info
        df = update_stated_gender_info(df)

        # Group gender information by template_id
        ids = df.groupby('esbbq_template_id')['stated_gender_info'].apply(lambda x: sorted(set(x))).to_dict()

        gender_category_dict[category] = {
            'm': sum(1 for gender in ids.values() if len(gender) == 1 and 'm' in gender),
            'f': sum(1 for gender in ids.values() if len(gender) == 1 and 'f' in gender),
            'n': sum(1 for gender in ids.values() if len(gender) == 1 and 'n' in gender),
            'm_vs_f': sum(1 for gender in ids.values() if len(gender) == 1 and 'm vs. f' in gender),
            'm_and_f': sum(1 for gender in ids.values() if len(gender) == 2 and {'m', 'f'}.issubset(gender)),
            'm_and_n': sum(1 for gender in ids.values() if len(gender) == 2 and {'m', 'n'}.issubset(gender)),
            'f_and_n': sum(1 for gender in ids.values() if len(gender) == 2 and {'f', 'n'}.issubset(gender)),
            'm_and_f_and_n': sum(1 for gender in ids.values() if len(gender) == 3 and {'m','f','n'}.issubset(gender)),
            # 'm_and_f_and_nb': sum(1 for gender in ids.values() if len(gender) == 3 and {'m','f','nb'}.issubset(gender)),
            # 'nb_vs_m/f': sum(1 for gender in ids.values() if len(gender) == 1 and 'nb' in gender),
            'total': len(ids)
        }

        if gender_category_dict[category]['total'] != sum(gender_category_dict[category].values()) - gender_category_dict[category]['total']:
            warnings.warn(f"Mismatch in category {category}.")

        ############################
        ## Get social groups info ##
        ############################
        
        # Convert strings to actual lists
        df['stereotyped_groups'] = df['stereotyped_groups'].apply(lambda x: ast.literal_eval(x))
        
        # Remove "disability" when not necessary
        if category == "DisabilityStatus":
            df['stereotyped_groups'].apply(lambda x: x if len(x) == 1 else x.remove("disability"))

        # Group and create sets from lists by flattening each group
        groups = df.groupby('esbbq_template_id')['stereotyped_groups'].apply(lambda x: set(item for sublist in x for item in sublist)).to_dict()

        # Count occurrences of each group
        groups_category_dict[category] = dict(Counter(group for groups in groups.values() for group in groups))

        # Write social group data to CSV
        with open(os.path.join(OUTPUT_DIR_TEMPLATES, f"{current_date}_social-groups_{category}.csv"), mode="w", newline='') as groups_file:
            writer = csv.DictWriter(groups_file, fieldnames=['social_group', 'templates'])
            writer.writeheader()
            for group, count in groups_category_dict[category].items():
                writer.writerow({'social_group': group, 'templates': count})

    # Write all gender data to CSV
    with open(os.path.join(OUTPUT_DIR_TEMPLATES, f"{current_date}_gender-info.csv"), mode='w', newline='') as gender_file:
        writer = csv.DictWriter(gender_file, fieldnames=['category', 'm', 'f', 'n', 'm_vs_f', 'm_and_f', 'm_and_n', 'f_and_n', 'm_and_f_and_n', 'total'])
        writer.writeheader()
        for category, data in gender_category_dict.items():
            writer.writerow({'category': category, **data})

    # Plot gender info
    plot_gender_info(OUTPUT_DIR_TEMPLATES,gender_category_dict)

    # Plot social groups info
    plot_social_groups_info(OUTPUT_DIR_TEMPLATES,groups_category_dict)

####################
## INSTANCE STATS ##
####################

if args.instances:

    print("\nGetting instance stats.\n")

    #############################
    ## Plot template fertility ##
    #############################

    data_frames = []

    # Iterate over each stats file
    for file in sorted(os.listdir(FERTILITY_DIR)):
        if not file.endswith(".fertility.csv"):
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
    
    # Group instances by category template_id
    all_data = all_data.groupby(['category', 'template_id'], as_index=False)['instances'].sum()
    all_data.to_csv(os.path.join(FERTILITY_DIR,f"{current_date}_template-fertility_avg.csv"))

    plt.figure(figsize=(13, 10))
    sns.stripplot(x='category', y='instances', data=all_data, jitter=False, dodge=False)

    # Define the upper limit for the y-axis
    y_max = 1500
    plt.ylim(0, y_max)

    # Add labels for points with instances > 500
    for _, row in all_data[all_data['instances'] > 500].iterrows():
        plt.text(x=row['category'], 
                y=min(row['instances'], 1500), 
                s=f"{row['template_id']} - {row['instances']}", 
                ha='right', 
                va='bottom', 
                fontsize=8
                )

    # Save plot
    plt.savefig(os.path.join(FERTILITY_DIR,f"{current_date}_template-fertility.png"), format='png', bbox_inches='tight', dpi=300)

    print("Template fertility plot saved.\n")

    ########################################
    ## Get gender and social groups info ###
    ########################################

    # Initialize dictionaries for gender and social group data, and total count
    gender_category_dict = {}
    groups_category_dict = {}
    len_dict = {}

    # Iterate over each category instance file 
    for file in sorted(os.listdir(DATA_DIR)):

        if not file.endswith(".full.csv"):
            continue
            
        category = file.split(".")[0]

        df = pd.read_csv(os.path.join(DATA_DIR, file),low_memory=False)

        print(f"Processing {category} category.")
        
        # Save total number of instances
        len_dict[category] = len(df)
        
        # Update stated_gender_info
        df = update_stated_gender_info(df)

        # Group gender info at template level
        gender_combinations = df.groupby('template_id')['stated_gender_info'].apply(lambda x: set(x)).to_dict()

        # Store in a new column the gender info of the instance at template level
        df['instance_gender_info'] = df.apply(lambda row: get_instance_gender(row, gender_combinations), axis=1)

        # Store counts of gender at instance level
        gender_counts = df['instance_gender_info'].value_counts().reindex(['m', 'f', 'n', 'm vs. f', 'm_and_f', 'm_and_n', 'f_and_n', 'm_and_f_and_n'], fill_value=0).to_dict()
        gender_counts['total'] = len(df)
        gender_category_dict[category] = gender_counts
    
        if gender_category_dict[category]['total'] != sum(gender_category_dict[category].values())-gender_category_dict[category]['total']:
            warnings.warn(f"There is an error in {category}.")

        # Get stereotyped group of the instance
        if category in ["Age","DisabilityStatus","Gender","Physical","SES"]:
            stereotyped_groups = df['stereotyped_groups'].str.strip("[]").str.replace("'", "")
            # Remove "disability" when not necessary
            if category == "DisabilityStatus":
                stereotyped_groups = stereotyped_groups.apply(lambda x: x.split(",")[-1].strip().replace('''"''',''''''))
        else: # stereotyped group is in ans0
            # Convert string representations to actual lists and keep only last element
            stereotyped_groups = df['answer_info.ans0'].apply(lambda x: ast.literal_eval(x)[-1].split(",")[-1].strip())

        # Count occurrences of each group
        groups_category_dict[category] = stereotyped_groups.value_counts().to_dict()

        # Write social group info to CSV
        with open(os.path.join(OUTPUT_DIR_INSTANCES, f"{current_date}_social-groups_{category}.csv"),mode="w",newline='') as groups_file:
            writer = csv.DictWriter(groups_file,fieldnames=['social_group','templates'])
            for group, value in groups_category_dict[category].items():
                writer.writerow({'social_group':group, 'templates':value})

    # Write len info to CSV
    with open(os.path.join(OUTPUT_DIR_INSTANCES,f"{current_date}_len-info.csv"), mode='w', newline='') as len_file:
        writer = csv.DictWriter(len_file, fieldnames=['category', 'instances'])
        writer.writeheader()
        for category, values in len_dict.items():
            writer.writerow({'category': category, 'instances':values})

    # Write gender info to CSV
    with open(os.path.join(OUTPUT_DIR_INSTANCES,f"{current_date}_gender-info.csv"), mode='w', newline='') as gender_file:
        writer = csv.DictWriter(gender_file, fieldnames=['category', 'm', 'f', 'n', 'm vs. f', 'm_and_f', 'm_and_n', 'f_and_n', 'm_and_f_and_n', 'total'])
        writer.writeheader()
        for category, values in gender_category_dict.items():
            writer.writerow({'category': category, **values})

    # Plot gender info
    plot_gender_info(OUTPUT_DIR_INSTANCES,gender_category_dict)

    # Plot social groups info
    plot_social_groups_info(OUTPUT_DIR_INSTANCES,groups_category_dict)