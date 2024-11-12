import language_tool_python
import os
import pandas as pd

tool = language_tool_python.LanguageTool('es')

if not os.path.exists("instance_language-revision"):
    os.makedirs("instance_language-revision")

def check_text(text):
    matches = tool.check(text)
    if matches:
        return [{
            'error_category':match.category,
            'error_type':match.ruleIssueType,
            'sentence':match.context,
            'matched_text':match.matchedText,
            'error_message':match.message,
            'error_replacement':match.replacements
            } for match in matches] 
    return None

for file in sorted(os.listdir("data_es")):
    if file.endswith(".full.csv"):

        category = file.split(".")[0]

        df = pd.read_csv(f"data_es/{file}", low_memory=False)
        
        # Check all text columns for errors
        for column in ['context', 'question', 'ans0', 'ans1']:
            df[f'errors_{column}'] = df[column].apply(lambda x: check_text(x))

        # Drop rows where all values in error columns are None
        error_columns = ['errors_context', 'errors_question', 'errors_ans0', 'errors_ans1']
        df = df.dropna(how='all', subset=error_columns)
        
        # Convert lists/dictionaries to strings for deduplication
        for column in error_columns:
            df[f'string_{column}'] = df[column].apply(str)

        # Drop duplicates based on the stringified error columns
        df = df.drop_duplicates(subset=['question_index', 'string_errors_context', 'string_errors_question', 'string_errors_ans0', 'string_errors_ans1'])
        
        # Keep only the necessary columns
        df = df[['question_index', 'version', 'errors_context', 'errors_question', 'errors_ans0', 'errors_ans1']]
    
        # Save data
        output_path = f"instance_language-revision/{category}_revision.csv"
        with open(output_path, 'w') as f:
            df.to_csv(output_path,index=False)

        print(f"{category} revision completed.")