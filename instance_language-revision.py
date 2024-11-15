import os
import pandas as pd
import language_tool_python

# Initialize language tool for Spanish
tool = language_tool_python.LanguageTool('es')

OUTPUT_DIR = "instance_language-revision"
DATA_DIR = "data_es"

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
    df.drop_duplicates(subset=['question_index'] + error_columns, inplace=True)
    
    # Select only relevant columns and save the revised DataFrame
    output_columns = ['question_index', 'version'] + error_columns
    output_path = os.path.join(OUTPUT_DIR, f"{category}_revision.csv")
    df[output_columns].to_csv(output_path, index=False)

    print(f"{category} revision completed.")