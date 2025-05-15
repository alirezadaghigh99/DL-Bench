
import os
import json
import re
from test_runner_class import runner
technique = "fewshot"
techniques = [  "zeroshot"]
r_s = [ "kornia", "pytorch3d"]
r_s2 = [  "scikit-learn", "neurodiffeq",  "umap", "vision", "small-text", "GPflow", "recommenders", "Laplace", "pyod", "pfrl", "pennylane", "nncf", "neupy", "emukit", "DeepReg", "deepchem"]
r_s3 = ["nlp-architecht", "imagededup","cleanlab", "pytorch-forecasting", "pytorch-widedeep","torchgeo", "lightly"]
r_s +=   r_s2 + r_s3
llms = [ "openai_new"]
techniques = [  "fewshot-same-cat","fewshot", "zeroshot-guided"]
techniques = [  "fewshot-same-cat-guided","fewshot-guided"]
llms = [  "antropic_new", "mistral_new" ]
versions = [ "v1"]
techniques = [  "zeroshot", "fewshot", "fewshot-guided", "fewshot-different", "fewshot-same-cat", "fewshot-same-cat-guided"]
for v in versions:
    for technique in techniques:
        for r in r_s:
                
            for llm in llms:
                # original_data_params = f"LLMs/output_mistral/{r}/"
                folder_path = f"../../results/llm-output/{technique}/output_{llm}/{v}/{r}"
                # BASE_PATH = "/home/aliredaq/Desktop/CG-DeepLearning/CGBench/repo_test_v4/"
                if r != "pytorch3d":
                    BASE_PATH = "/local/data0/moved_data/publishablew/"
                else:
                    
                    BASE_PATH = "/local/data0/moved_data/"
                # BASE_PATH = "/local/data0/moved_data/"
                # Loop through each file in the folder
                for filename in os.listdir(folder_path):
                    print("##################################")
                    print(filename)
                    print("##################################")
                    if filename.endswith('.json') and "processed_class" in filename:
                        if "korniaforward97" in filename or "korniaforward97" in filename:
                            continue
                        file_path = os.path.join(folder_path, filename)
                        original_param_path = os.path.join(folder_path, filename)
                        print(original_param_path)
                        with open(original_param_path, "r") as f:
                            data = json.load(f)
                            class_name = data["class"]
                            # if "Translate" not in class_name:
                            #     continue
                            p = data["ground_truth"].split("#")[0]
                            f_name = data["function_name"]
                            tests = data["test"]
                            prompt = data["prompt"]
                            stage = data.get('stage', '')
                            task = data.get('task', '')
                            data_value = data.get('data', '')
                            repo = f"{r}"
                            if repo != "pytorch3d":
                                path_to_fn = os.path.join(BASE_PATH, repo, repo, p)
                                tests = os.path.join(BASE_PATH, repo, repo, tests)
                            else:
                                path_to_fn = os.path.join(BASE_PATH, repo, p)
                                tests = os.path.join(BASE_PATH,repo, tests)
                            print(path_to_fn)
                            print(filename)
                            
                        with open(file_path, 'r') as f:
                            # Load the JSON data
                            data = json.load(f)
                            
                            
                            result = data.get('result', '')
                            if result and "429 RESOURCE_EXHAUSTED" not in result:
                            
                            # Use regex to extract the code starting from '```python\n'
                                # First try to match Python code blocks
                                match = re.search(r'```python\n(.*?)(?:\n```|$)', result, re.DOTALL)
                                if match:
                                    code = match.group(1)
                                    # Remove example usage if present
                                    if "# Example usage" in code:
                                        code = code.split("# Example usage")[0]
                                    print(code)
                                    
                                    with open("t.txt", "w") as f:
                                        f.write(code)
                                    test_result = runner(file_path=path_to_fn, function_name=f_name, test_file=tests, class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm, prompt=prompt, technique=technique)
                                
                                # Try to match code blocks without language specification
                                elif re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL):
                                    match1 = re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL)
                                    if match1:
                                        code = match1.group(1)
                                        print(code)
                                        with open("t.txt", "w") as f:
                                            f.write(code)
                                        test_result = runner(file_path=path_to_fn, function_name=f_name, test_file=tests, class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm, prompt=prompt, technique=technique)
                                
                                # If no code blocks, check for class or function definitions
                                elif "class " + class_name in result:
                                    # This is a class definition
                                    code = result
                                    print(code)
                                    with open("t.txt", "w") as f:
                                        f.write(code)
                                    test_result = runner(file_path=path_to_fn, function_name=f_name, test_file=tests, class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm, prompt=prompt, technique=technique)
                                
                                elif "def " + f_name in result:
                                    # This is a function definition
                                    code = result
                                    print(code)
                                    with open("t.txt", "w") as f:
                                        f.write(code)
                                    test_result = runner(file_path=path_to_fn, function_name=f_name, test_file=tests, class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm, prompt=prompt, technique=technique)
                                
                                # As a last resort, try direct class and method patterns
                                elif re.search(r'class\s+' + re.escape(class_name) + r'\s*\(', result, re.DOTALL) and \
                                     re.search(r'def\s+' + re.escape(f_name) + r'\s*\(', result, re.DOTALL):
                                    code = result
                                    print(code)
                                    with open("t.txt", "w") as f:
                                        f.write(code)
                                    test_result = runner(file_path=path_to_fn, function_name=f_name, test_file=tests, class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm, prompt=prompt, technique=technique)
                                
                                else:
                                    print(f"No code found in {filename}")
                            else:
                                test_result = "-1"
                        data_to_save = {
                                "test_result" :test_result, 
                                "file_path" : filename,
                                "stage": stage,
                                "task": task, 
                                "data": data_value
                            }
                        with open(f"result_{v}_{llm}_{technique}.jsonl", "a") as f: 
                            print("I am saving", data_to_save, f)
                            json.dump(data_to_save, f)
                            f.write("\n")
                                
                            
