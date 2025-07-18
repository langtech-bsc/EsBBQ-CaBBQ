import os
import pandas as pd
import language_tool_python
import argparse

# languages available
languages = ["es","ca"]

parser = argparse.ArgumentParser(prog="Revise Instances", description="This script will read the instance files in the data folder and revise them for linguistic errors.")
parser.add_argument("--language", choices=languages, help="language of the instances.")
args = parser.parse_args()

# get language
lang = args.language

# Initialize language tool
tool = language_tool_python.LanguageTool(lang)

OUTPUT_DIR = f"instance_language-revision/{lang}"
DATA_DIR = f"data_{lang}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function for error checking
def check_text(text):
    matches = tool.check(text)
    return [{
        'error_category': match.category,
        'error_type': match.ruleIssueType,
        'sentence': match.context,
        'matched_text': match.matchedText,
        'error_message': match.message,
        'error_replacement': match.replacements
    } for match in matches] if matches else None

# Iterate over each category instances
for file in sorted(os.listdir(DATA_DIR)):
    if not file.endswith(".full.csv"):
        continue

    category = file.split(".")[0]

    print("------------------------------")
    print(f"{category} revision started.")

    df = pd.read_csv(os.path.join(DATA_DIR, file), low_memory=False)
    
    # Check text columns for errors and save results in another column
    error_columns = ['errors_context', 'errors_question', 'errors_ans0', 'errors_ans1']
    text_columns = ['context', 'question', 'ans0', 'ans1']
    for text_col, error_col in zip(text_columns, error_columns):
        df[error_col] = df[text_col].apply(check_text)

    # Filter rows with at least one error in the error columns and drop the ones without errors
    df.dropna(how='all', subset=error_columns, inplace=True)
    
    # Drop exact duplicates
    for col in error_columns:
        df[col] = df[col].astype(str)
    df.drop_duplicates(subset=['template_id'] + error_columns, inplace=True)
    
    # Select only relevant columns and save the revised DataFrame
    output_columns = ['template_id', 'version'] + error_columns
    output_path = os.path.join(OUTPUT_DIR, f"{category}_revision.csv")
    df[output_columns].to_csv(output_path, index=False)

    print(f"{category} revision completed.")
    print("------------------------------")