import pandas as pd
import json

def safe_load_jsonl_to_excel(jsonl_filename, excel_filename):
    data = []
    with open(jsonl_filename, 'r') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping a line due to an error: {e}")
                continue
    
    # Only create the DataFrame from successfully parsed data
    if data:
        df = pd.DataFrame(data)
        df.to_excel(excel_filename, index=False, engine='openpyxl')
    else:
        print("No data could be loaded.")

jsonl_filename = 'functions/paths_outputv41class0.jsonl'
excel_filename = 'sharedv3class.xlsx'
safe_load_jsonl_to_excel(jsonl_filename, excel_filename)