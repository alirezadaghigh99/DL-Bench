
import os
import json
import re
from test_runner_class import runner
repos = [ "pytorch3d" ]
llms = [ "geminai"]

for r in repos:
        
    for llm in llms:
        original_data_params = f"LLMs/output_mistral/{r}/"
        folder_path = f"LLMs/output_{llm}/{r}/"
        # BASE_PATH = "/home/x/Desktop/CG-DeepLearning/CGBench/repo_test_v4/"
        # BASE_PATH = "/local/data0/moved_data/publishablew/"
        BASE_PATH = "/local/data0/moved_data/"
        # BASE_PATH = "/local/data0/moved_data/"
        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            print("##################################")
            print(filename)
            print("##################################")
            if filename.endswith('.json') and "processed_classes" in filename:
                # if not "compute_transformation" in filename:
                #     continue
                file_path = os.path.join(folder_path, filename)
                original_param_path = os.path.join(original_data_params, filename)
                print(original_param_path)
                with open(original_param_path, "r") as f:
                    data = json.load(f)
                    class_name = data["class"]
                    # if "Translate" not in class_name:
                    #     continue
                    p = data["ground_truth"].split("#")[0]
                    f_name = data["function_name"]
                    tests = data["test"]
                    repo = f"{r}"
                    path_to_fn = os.path.join(BASE_PATH, repo, p)
                    tests = os.path.join(BASE_PATH, repo, tests)
                    print(path_to_fn)
                    print(filename)
                    
                with open(file_path, 'r') as f:
                    # Load the JSON data
                    data = json.load(f)
                    
                    
                    result = data.get('result', '')
                    
                    
                    # Use regex to extract the code starting from '```python\n'
                    match = re.search(r'```python\n(.*?)(?:\n```|$)', result, re.DOTALL)
                    if match:
                        code = match.group(1)
                        code = code.split("# Example")
                        code = code[0]
                        print(code)
                        
                        with open("t.txt", "w") as f:
                            f.write(code)
                            print(code)
                        runner(file_path=path_to_fn, function_name=f_name, test_file=tests, class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm)
                    elif re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL):
                        match1 = re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL)
                        if match1:
                            code = match1.group(1)
                            print(code)
                            with open("t.txt", "w") as f:
                                f.write(code)
                                print(code)
                            runner(file_path=path_to_fn, function_name=f_name, test_file=tests,class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm)
                    else:
                        if "def" in result:
                            code = result
                            with open("t.txt", "w") as f:
                                f.write(code)
                                print(code)
                            runner(file_path=path_to_fn, function_name=f_name, test_file=tests,class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm)
                        else:
                            print(f"No code found in {filename}")
                        
                    
