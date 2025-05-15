import os
import json
import re
import ast
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_runner_class import runner as class_runner
from test_runner_function import runner as function_runner

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
        print(e)
        return "error code"

def process_class_file(file_path, folder_path, BASE_PATH, r, llm, technique, version):
    """Process a class file and run tests on it."""
    with open(file_path, "r") as f:
        data = json.load(f)
        class_name = data["class"]
        p = data["ground_truth"].split("#")[0]
        f_name = data["function_name"]
        tests = data["test"].split()
        prompt = data["prompt"]
        stage = data.get('stage', '')
        task = data.get('task', '')
        data_value = data.get('data', '')
        repo = f"{r}"
        
        
        if repo != "pytorch3d":
            path_to_fn = os.path.join(BASE_PATH, repo, repo, p)
            temp_tests  = []
            for test in tests:
                temp_tests.append(os.path.join(BASE_PATH, repo, repo, test))
            tests = " ".join(temp_tests)
        else:
            path_to_fn = os.path.join(BASE_PATH, repo, p)
            temp_tests  = []
            for test in tests:
                temp_tests.append(os.path.join(BASE_PATH, repo, test))
            tests = " ".join(temp_tests)
        
        print(path_to_fn)
        filename = os.path.basename(file_path)
        print(filename)
        
    with open(file_path, 'r') as f:
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
                test_result = class_runner(file_path=path_to_fn, function_name=f_name, test_file=tests, 
                                          class_name=class_name, code_str=code, llm_output=filename, 
                                          repository=r, llm=llm, prompt=prompt, technique=technique)
            
            # Try to match code blocks without language specification
            elif re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL):
                match1 = re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL)
                if match1:
                    code = match1.group(1)
                    print(code)
                    with open("t.txt", "w") as f:
                        f.write(code)
                    test_result = class_runner(file_path=path_to_fn, function_name=f_name, test_file=tests, 
                                              class_name=class_name, code_str=code, llm_output=filename, 
                                              repository=r, llm=llm, prompt=prompt, technique=technique)
            
            # If no code blocks, check for class or function definitions
            elif "class " + class_name in result:
                # This is a class definition
                code = result
                print(code)
                with open("t.txt", "w") as f:
                    f.write(code)
                test_result = class_runner(file_path=path_to_fn, function_name=f_name, test_file=tests, 
                                          class_name=class_name, code_str=code, llm_output=filename, 
                                          repository=r, llm=llm, prompt=prompt, technique=technique)
            
            elif "def " + f_name in result:
                # This is a function definition
                code = result
                print(code)
                with open("t.txt", "w") as f:
                    f.write(code)
                test_result = class_runner(file_path=path_to_fn, function_name=f_name, test_file=tests, 
                                          class_name=class_name, code_str=code, llm_output=filename, 
                                          repository=r, llm=llm, prompt=prompt, technique=technique)
            
            # As a last resort, try direct class and method patterns
            elif re.search(r'class\s+' + re.escape(class_name) + r'\s*\(', result, re.DOTALL) and \
                 re.search(r'def\s+' + re.escape(f_name) + r'\s*\(', result, re.DOTALL):
                code = result
                print(code)
                with open("t.txt", "w") as f:
                    f.write(code)
                test_result = class_runner(file_path=path_to_fn, function_name=f_name, test_file=tests, 
                                          class_name=class_name, code_str=code, llm_output=filename, 
                                          repository=r, llm=llm, prompt=prompt, technique=technique)
            
            else:
                print(f"No code found in {filename}")
                test_result = "-1"
        else:
            test_result = "-1"
            
    data_to_save = {
        "test_result": test_result, 
        "file_path": filename,
        "stage": stage,
        "task": task, 
        "data": data_value
    }
    
    with open(f"result_{version}_{llm}_{technique}.jsonl", "a") as f: 
        print("I am saving", data_to_save, f)
        json.dump(data_to_save, f)
        f.write("\n")

def process_function_file(file_path, folder_path, BASE_PATH, r, llm, technique, version):
    """Process a function file and run tests on it."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        repo = f"{r}"
        p = data["ground_truth"].split("#")[0]
        f_name = data["function_name"]
        prompt = data["prompt"]
        tests = data["test"].split()
        result = data.get('result', '')
        result = result.replace("#code","\n")
        stage = data.get('stage', '')
        task = data.get('task', '')
        data_value = data.get('data', '')
        
        
        if result and "429 RESOURCE_EXHAUSTED" not in result:
            if repo != "pytorch3d":
                path_to_fn = os.path.join(BASE_PATH, repo, repo, p)
                temp_tests  = []
                
                for test in tests:
                    temp_tests.append(os.path.join(BASE_PATH, repo, repo, test))
                tests = " ".join(temp_tests)
            else:
                path_to_fn = os.path.join(BASE_PATH, repo, p)
                temp_tests  = []
                for test in tests:
                    temp_tests.append(os.path.join(BASE_PATH, repo, test))
                tests = " ".join(temp_tests)
            print(tests)
            # print(temp_tests)
                
            match = re.search(r'```python[^\n]*\n(.*?)(?:\n```|$)', result, re.DOTALL)
            if match:
                code = match.group(1)
                print("first match")
                print(code)
                code = extract_functions_and_imports(code)
                if "error code" in code:
                    test_result = -1
                else:
                    with open("t.txt", "w") as f:
                        f.write(code)
                    test_result = function_runner(path_to_fn, f_name, tests, os.path.basename(file_path), llm, r, code, prompt, technique)
                
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
                        test_result = function_runner(path_to_fn, f_name, tests, os.path.basename(file_path), llm, r, code, prompt, technique)
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
                        test_result = function_runner(path_to_fn, f_name, tests, os.path.basename(file_path), llm, r, code, prompt, technique)
                else:
                    test_result = -1
        else:
            test_result = -1
            
    data_to_save = {
        "test_result": test_result, 
        "file_path": os.path.basename(file_path),
        "stage": stage,
        "task": task, 
        "data": data_value
    }
    
    with open(f"result_{version}_{llm}_{technique}.jsonl", "a") as f: 
        json.dump(data_to_save, f)
        f.write("\n")

# Main execution
def main():
    # Common repository sets across both files
    r_s1 = [ "pytorch", "pytorch3d"]
    r_s2 = ["scikit-learn", "neurodiffeq", "umap", "vision", "small-text", "inference", 
            "GPflow", "recommenders", "Laplace", "pyod", "pfrl", "pennylane", "nncf", 
            "neupy", "emukit", "DeepReg", "deepchem"]
    r_s3 = ["nlp-architecht", "imagededup", "pytorch-forecasting", "pytorch-widedeep", 
            "torchgeo", "lightly", "ignite"]
    r_s = r_s1 + r_s2 + r_s3
    r_s += ["kornia"]
    r_s += ["cleanlab"]
    print(len(r_s))
    input("Press Enter to continue...")
    
    # Model and technique configuration
    versions = ["v2"]
    techniques = ["fewshot", "fewshot-different", 
                 "fewshot-same-cat", ]
    
    techniques = ["zeroshot"]
    # techniques = ["zeroshot"]
    # r_s = ["lightly"]
    llms = ["openai-4o_new", "mistral_new", "llama_new", "qwen_new", "deepseek_new", "o3-mini_new", "antropic_new"]
    techniques = ["fewshot-classifier"]
    # llms = [ "openai-4o_new"]
    # llms = ["o3-mini_new", "antropic_new"]
    
    for version in versions:
        for technique in techniques:
            for llm in llms:
                for r in r_s:
                    try:
                        folder_path = f"../../results/llm-output/{technique}/output_{llm}/{version}/{r}"
                        print(f"Processing {folder_path}")
                        
                        # Set BASE_PATH based on repository
                        if r != "pytorch3d":
                            BASE_PATH = "/local/data0/moved_data/publishablew/"
                        else:
                            BASE_PATH = "/local/data0/moved_data/"
                        
                        # Process all files in the folder
                        for filename in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, filename)
                            if "nlp-architect" in filename:
                                continue
                            
                            
                            # Process class files
                            if filename.endswith('.json') and "processed_class" in filename:
                                if "korniaforward97" in filename:  # Skip specific files
                                    continue
                                
                                
                                print("##################################")
                                print(f"Processing class file: {filename}")
                                print("##################################")
                                process_class_file(file_path, folder_path, BASE_PATH, r, llm, technique, version)
                            
                            # Process function files
                            elif filename.endswith('.json') and "processed_class" not in filename:
                                print("##################################")
                                print(f"Processing function file: {filename}")
                                print("##################################")
                                process_function_file(file_path, folder_path, BASE_PATH, r, llm, technique, version)
                    except Exception as e:
                            print(f"Error processing {r}: {e}")
                            with open("error_log.txt", "a") as error_log:
                                error_log.write(f"Error processing {r}: {e}\n")
                                error_log.write(f"{r}")
                            continue

if __name__ == "__main__":
    main()