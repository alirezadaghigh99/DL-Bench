import os
import json
import glob
import re

# Define the allowed keys.
ALLOWED_KEYS = {"stage", "task", "prompt", "ground Truth", "repo", "function", "test_cases", "class"}

def parse_file(file_path):
    """
    Reads a file and parses its contents into a dictionary.
    Only lines where the key (before the colon) is in ALLOWED_KEYS are used.
    Lines not starting with an allowed key are assumed to be a continuation of the current key's value.
    """
    data = {}
    current_key = None
    data["file"] = file_path.split("/")[-1]
    # Regex to capture "key: value" format
    # We use a regex that takes the text before the first colon as key, then the rest as value.
    line_pattern = re.compile(r'^(?P<key>[^:]+):\s*(?P<value>.*)$')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()  # Remove trailing newline and whitespace
            if not line:
                continue  # skip empty lines
            
            match = line_pattern.match(line)
            if match:
                key_candidate = match.group("key").strip()
                value_part = match.group("value").strip()
                # Check if this key is one of the allowed keys.
                if key_candidate in ALLOWED_KEYS:
                    # Start a new key-value pair.
                    current_key = key_candidate
                    # If key already exists, we add a newline separator before appending.
                    if current_key in data:
                        data[current_key] += "\n" + value_part
                    else:
                        data[current_key] = value_part
                    continue  # move to next line

            # If the line doesn't match a new allowed key, and we have an active key, 
            # treat it as a continuation of the current key's value.
            if current_key is not None:
                data[current_key] += "\n" + line.strip()
    return data

def process_folder(folder_path, output_jsonl):
    """
    Processes all .txt files in the given folder,
    parses each file into a dictionary (using allowed keys), and writes all dictionaries
    to an output file in JSONL format.
    """
    all_dicts = []
    
    # Get all .txt files in the folder
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    for file_path in txt_files:
        parsed_data = parse_file(file_path)
        all_dicts.append(parsed_data)
    
    # Write each dictionary as a JSON object in one line
    with open(output_jsonl, 'w', encoding='utf-8') as out_file:
        for item in all_dicts:
            json_line = json.dumps(item)
            out_file.write(json_line + "\n")


import json
import re
from sklearn.model_selection import train_test_split

def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_task_data(task_str):
    """
    Given a task string (which might contain a newline with 'data: ...'),
    returns a tuple: (main_task, data_type). If no data_type is found,
    data_type will be an empty string.
    """
    # Split the task string by lines
    lines = task_str.splitlines()
    main_task = lines[0].strip() if lines else ""
    data_type = ""
    # Look for a line that starts with 'data:'
    for line in lines[1:]:
        match = re.match(r'^data\s*:\s*(.+)$', line.strip(), flags=re.IGNORECASE)
        if match:
            data_type = match.group(1).strip()
            break
    return main_task, data_type

def add_stratify_key(record):
    """
    Create a stratification key based on 'stage', the main part of 'task',
    and the data type (if any) extracted from 'task'. These three parts are
    concatenated to form a unique key.
    """
    stage = record.get("stage", "").strip()
    task_full = record.get("task", "")
    main_task, data_type = extract_task_data(task_full)
    # Create a stratification key that joins the three components
    key = f"{stage}||{main_task}||{data_type}"
    return key

def stratified_split(data, train_frac=0.2, random_state=42):
    """
    Splits the data into train and test sets using stratified sampling
    based on the combined key of stage, task, and data type.
    """
    # Create the stratification keys for all records
    strat_keys = [add_stratify_key(record) for record in data]
    
    train, test = train_test_split(
        data,
        train_size=train_frac,
        stratify=strat_keys,
        random_state=random_state
    )
    return train, test

def write_jsonl(records, file_path):
    """Writes a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    # Path to your input JSONL file
    input_file = "../../data/dataset.jsonl"
    
    # Read all records from the JSONL file
    records = read_jsonl(input_file)
    
    # Perform a stratified split (20% train, 80% test)
    train_records, test_records = stratified_split(records, train_frac=0.2)
    
    # Write the resulting splits to separate JSONL files
    write_jsonl(train_records, "train.jsonl")
    write_jsonl(test_records, "test.jsonl")
    
    print(f"Splitting completed: {len(train_records)} train records and {len(test_records)} test records.")

# if __name__ == "__main__":
#     # Specify the folder containing your txt files and the output file name.
#     folder_path = "/home/aliredaq/Desktop/DeepAgent/ICSE/data/DLEval-20240920T201632Z-001/DLEval"   # <-- Change this to your folder path
#     output_jsonl = "dataset.jsonl"
    
#     process_folder(folder_path, output_jsonl)
#     print(f"All files processed and saved to {output_jsonl}")
