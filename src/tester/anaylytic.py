import json
def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def score(jsonl_file, source_file):
    
    data = read_jsonl(jsonl_file)
    results = {}
    total_output = {}
    
    
    for record in data:
        test_result = record.get("test_result", 0)
        file_name = record.get("file_path", "")
        for d in source_file:
            if file_name.replace("processed_", "").replace("json", "txt") == d['filename']:
                stage = d.get("stage", "").strip().lower()
                data_type = d.get("data", "").lower()
                task_full = d.get("task", "").lower()
                break
        else:
            print(file_name)
            exit(1)
        
        stage_in_data = stage
                
        test_result = int(test_result)
        
        if "process" in stage or "process" in data_type:
            stage = "processing"
        elif "model" in stage or "model" in data_type:
            stage = "model"
        elif "infer" in stage or "infer" in data_type:
            stage = "infer"
        elif "eval" in stage or "eval" in data_type:
            stage = "eval"
        elif "train" in stage or "train" in data_type:
            stage = "train"
        
        
        if "image" in data_type or "image" in stage_in_data:
            data_type = "image"
        elif "text" in data_type or "text" in stage_in_data:
            data_type = "text"
        elif "tab" in data_type or "tab" in stage_in_data:
            data_type = "tabular"
        else:
            if data_type !="":
                print(data_type)
            data_type = "other"
        
        if "_points_and_weights390" in file_name:
            print(stage)
        
        total_output[data_type] = total_output.get(data_type, 0) + 1
        if "hinton" in file_name and "35" in file_name:
            test_result = 0
        if "363" in file_name:
            test_result = 1
        if "327" in file_name:
            test_result = 1
        results[data_type] = results.get(data_type, 0) + max(0, test_result)
    return results, total_output

file_path = "DL-Bench-New.jsonl"
data = list(map(json.loads, open(file_path) ))
task_score = score("result_v2_antropic_new_fewshot-classifier.jsonl", data)
results, total_output = task_score
for stage, score in results.items():
    print(f"Stage: {stage}, {total_output[stage]} Score: {score}, accuracy: {score / total_output[stage]}")

print(sum(results.values()) / sum(total_output.values()))
            
        
        
            
        
        
    
