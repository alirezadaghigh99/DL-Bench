
import os
import json
import re
import ast

def extract_functions_and_imports(code):
    """
    Extracts import statements and function definitions from Python code.
    
    Parameters:
    code (str): The input Python code as a string.
    
    Returns:
    str: Filtered code containing only imports and function definitions.
    """
    try:
        tree = ast.parse(code)
        extracted_lines = []
        
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):  # Keep import statements
                extracted_lines.append(code.splitlines()[node.lineno - 1])
            elif isinstance(node, ast.FunctionDef):  # Keep function definitions
                extracted_lines.append("\n".join(code.splitlines()[node.lineno - 1:node.end_lineno]))

        return "\n\n".join(extracted_lines)
    except Exception as e:
        # print(code)
        print(e)
        return "error code"

from test_runner_function import runner
# techniques = [   "fewshot-same-cat"]

r_s1 = [ "kornia", "pyro"]
r_s2 = [  "scikit-learn", "neurodiffeq",  "umap", "vision", "small-text", "inference", "GPflow", "recommenders", "Laplace", "pyod", "pfrl", "pennylane", "nncf", "neupy", "emukit", "DeepReg", "deepchem", "cleanlab", "pytorch-forecasting", "pytorch-widedeep","torchgeo", "lightly"]
r_s3 = ["nlp-architecht", "imagededup", "pytorch-forecasting", "pytorch-widedeep", "ignite"]
r_s = ["pytorch3d"] + r_s1 + r_s2 + r_s3
remaining2 = ["litgpt" , "avalanche"]
versions = [ "v1"]
techniques = [  "zeroshot", "fewshot", "fewshot-guided", "fewshot-different", "fewshot-same-cat", "fewshot-same-cat-guided"]
repos = ["nncf", "kornia", "pennylane", "vision"]
# r_s = repos
llms = [  "antropic_new", "mistral_new" ]
# versions = ["v1_temperature_0.5", "v2_temperature_0.5", "v3_temperature_0.5", "v4_temperature_0.5"]
for version in versions:
    for technique in techniques:
        for llm in llms:
            for r in r_s:
                folder_path = f"../../results/llm-output/{technique}/output_{llm}/{version}/{r}"
                print(folder_path)
                if r != "pytorch3d":
                    BASE_PATH = "/local/data0/moved_data/publishablew/"
                else:
                    BASE_PATH = "/local/data0/moved_data/"
                for filename in os.listdir(folder_path):
                    # print("##################################")
                    # print(filename)
                    # print("##################################")
                    if filename.endswith('.json') and not "processed_classes" in filename:
                        # if not "train" in filename:
                        #     continue
                        file_path = os.path.join(folder_path, filename)
                        with open(file_path, 'r') as f:
                            # if  not "_compute_occlusion_layers" in file_path:
                            #     continue
                            data = json.load(f)
                            repo = f"{r}"
                            p = data["ground_truth"].split("#")[0]
                            f_name = data["function_name"]
                            prompt = data["prompt"]
                            tests = data["test"]
                            result = data.get('result', '')
                            result = result.replace("#code","\n")
                            stage = data.get('stage', '')
                            task = data.get('task', '')
                            data = data.get('data', '')
                            # print(result)
                            # exit(1)
                            if result and "429 RESOURCE_EXHAUSTED" not in result:
                                if repo != "pytorch3d":
                                    path_to_fn = os.path.join(BASE_PATH, repo, repo, p)
                                    tests = os.path.join(BASE_PATH, repo, repo, tests)
                                else:
                                    path_to_fn = os.path.join(BASE_PATH, repo, p)
                                    tests = os.path.join(BASE_PATH,repo, tests)
                                match = re.search(r'```python[^\n]*\n(.*?)(?:\n```|$)', result, re.DOTALL)
                                if match:
                                    code = match.group(1)
                                    print("first match")
                                    print(code)
                                    code = extract_functions_and_imports(code)
                                    if "error code" in code:
                                        test_result = -1
                                    else:
                                    # exit(1)
                                    # exit(1)
                                        with open("t.txt", "w") as f:
                                            f.write(code)
                                        test_result = runner(path_to_fn, f_name, tests, filename, llm, r, code, prompt, technique)
                                    
                                elif re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL):
                                    match1 = re.search(r'```[^\n](.*?)(?:\n```|$)', result, re.DOTALL)
                                    if match1:
                                        print("second match")
                                        code = match1.group(1)
                                        code = extract_functions_and_imports(code)
                                        if "error code" in code:
                                            test_result = -1
                                            
                                        else:
                                            with open("t.txt", "w") as f:
                                                f.write(code)
                                                print(code)
                                            test_result = runner(path_to_fn, f_name, tests, filename, llm, r, code, prompt, technique)
                                else:
                                    if "def" in result:
                                    
                                        code = result
                                        print("third match")
                                        print(code)
                                        code = extract_functions_and_imports(code)
                                        if "error code" in code:
                                            test_result = -1
                                        else:
                                            with open("t.txt", "w") as f:
                                                f.write(code)
                                                # print(code)
                                            test_result = runner(path_to_fn, f_name, tests, filename, llm, r, code, prompt, technique)
                                    else:
                                        test_result = -1
                            else:
                                test_result = -1
                        data_to_save = {
                            "test_result" :test_result, 
                            "file_path" : filename,
                            "stage": stage,
                            "task": task, 
                            "data": data
                        }
                        with open(f"result_{version}_{llm}_{technique}.jsonl", "a") as f: 
                            json.dump(data_to_save, f)
                            f.write("\n")
                                
                                
                            
