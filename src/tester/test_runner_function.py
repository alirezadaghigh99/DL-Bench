import os
import subprocess
import shutil
from replacer_function import replace_function
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write_to_python_file(file_path: str, content: str, mode: str = 'w') -> None:
    """
    Writes content to a Python file.

    Parameters:
        file_path (str): The path to the Python file.
        content (str): The content to write into the file.
        mode (str): The mode for writing ('w' for overwrite, 'a' for append). Default is 'w'.

    Returns:
        None

    Raises:
        ValueError: If the mode is not 'w' or 'a'.
        IOError: If there is an issue writing to the file.
    """
    if mode not in ('w', 'a'):
        raise ValueError("Invalid mode. Use 'w' to overwrite or 'a' to append.")
    
    try:
        with open(file_path, mode, encoding='utf-8') as file:
            file.write(content)
    except IOError as e:
        raise IOError(f"An error occurred while writing to the file '{file_path}': {e}")

# Example usage
# file_path = "example.py"
# content = "def hello_world():\n    print('Hello, World!')\n"
# write_to_python_file(file_path, content)

# llm = "openai"
def backup_file(file_path):
    backup_path = file_path + ".backup"
    shutil.copy(file_path, backup_path)
    print(f"Backup created at {backup_path}")
    return backup_path

def clear_pycache():
    pycache_dir = "__pycache__"
    if os.path.exists(pycache_dir):
        shutil.rmtree(pycache_dir)
        print(f"{pycache_dir} cleared.")

def run_pytest(test_file, python_path="/local/data0/moved_data/publishablew/", 
               test_case=None, conda_env="/home/aliredaq/anaconda3/envs/myenv/", is_conda = False, rep = None):
    if rep == "DeepReg":
        is_conda = True
    python_path += rep + "/"
    if rep == "pytorch3d":
        python_path = "/local/data0/moved_data/"
    
    if test_case:
        full_test_file = f"{test_file}::{test_case}"
    else:
        full_test_file = test_file
    vi_env = "venv"
    if rep == "nlp-architecht":
        vi_env = "nvenv"

    if is_conda:
        conda_setup = "/home/aliredaq/anaconda3/etc/profile.d/conda.sh"

        command = f'source {conda_setup} && conda activate {rep.lower()} && PYTHONPATH={python_path} && cd {python_path}{rep} && python3 -m pytest {full_test_file} --color=no --cache-clear -v' 
       
    else:
        command =  f'source {python_path}/{rep}/{vi_env}/bin/activate && PYTHONPATH={python_path}{rep} python -m pytest {full_test_file} --color=no --cache-clear -v -s' 
        print("!"*20)
        print(command)
    try:  
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=180)
    except Exception as e:
        return "error", "error"
    stdout_output = result.stdout
    stderr_output = result.stderr

    
    return result.stdout, result.stderr
    
    return result.stdout
def ruin_function(file_path, function_name):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        inside_function = False
        for line in lines:
            if line.strip().startswith(f"def {function_name}("):
                inside_function = True
                f.write(line)  
                f.write('    return ""\n')  # Ruin the function by inserting `return ""`
            elif inside_function and line.startswith(" "):  # Continue writing the rest of the function
                f.write(line)
            else:
                inside_function = False  
                f.write(line) 
def restore_file(backup_path, original_path):
    shutil.copy(backup_path, original_path)
    print(f"File restored from {backup_path}")

def compare_test_results(first_result, second_result):
    first_failed = set(line for line in first_result.splitlines() if "FAILED" in line)
    second_failed = set(line for line in second_result.splitlines() if "FAILED" in line)
    
    return second_failed  - first_failed
import json
def process_test_results(test_cases, test_errs, all_failed, initial_test, file_name, llm_output, function_name, llm, rep, generated_code, function_path = None, prompt=None, technique = "zero-shot"):
    out = file_name.replace(".py",".txt")
    
    out = f"../../results/test_output/{technique}/{llm}/v2/{rep}/" + llm_output.replace(".json", ".txt")
    output_dir = os.path.dirname(out)
    os.makedirs(output_dir, exist_ok=True)
    pass_or_fail = f"{0}" if "FAILED" in str(test_cases) else f"{1}"
    with open(out , 'w') as f:
        f.write("output file:\n")
        f.write(f"{llm_output}\n")
        f.write("function:\n")
        f.write(f"{function_name}\n")
        f.write("Error Cases:\n")
        f.write(str(test_errs))
        f.write("\n")
        f.write("Pass or Failed: ")
        f.write(f"{pass_or_fail}\n")
        f.write("\n")
        f.write("Related Failed Test Cases:\n")
        
        f.write(f"{test_cases}\n")
        f.write("\n")
        f.write("All Test Cases On Generated code:\n")
       
        f.write(f"{all_failed}\n")
        
        f.write("\n\nInitial Result:\n")
        f.write(initial_test)
    return pass_or_fail
        

def ignore_test_cases(repo, test_cases):
    from test_removal import remving
    if not repo in remving.removing.keys():
        return test_cases
    test_to_remove = remving.removing[repo]
    items_to_check = list(test_cases)
    
    for item in items_to_check:
        # Check if any substring from values_list appears in the current item
        if any(substring in item for substring in test_to_remove):
            # Remove the entire item from the original set
            test_cases.remove(item)
    
    return test_cases

def runner(file_path, function_name, test_file, llm_output, llm, rep, code, prompt, technique):
    backup_path = backup_file(file_path)

    initial_test_result, initial_err = run_pytest(test_file, rep=rep)
    print("Initial test run completed.")
    try:
        replace_function(file_path, function_name, "temp.py", "t.txt")
        import time
        
        print(f"{function_name} function has been ruined.")
        clear_pycache()
        ruined_test_result, err = run_pytest(test_file, rep=rep)
        
    except Exception as e:
        print(e)
        ruined_test_result, err = "Error", e
    print("Tests run after ruining the function.")
    related_tests = compare_test_results(initial_test_result, ruined_test_result)
    print("Related test cases identified.")

    restore_file(backup_path, file_path)
    
    
    
    related_tests = ignore_test_cases(rep, related_tests )

    res = process_test_results(test_cases=related_tests, all_failed = ruined_test_result, test_errs=err, initial_test=initial_test_result, file_name=os.path.basename(file_path) , llm_output=llm_output, function_name=function_name, llm=llm, rep=rep, generated_code=code, function_path=file_path, prompt=prompt, technique=technique)
    if "Error" == ruined_test_result:
        return 0
    else:
        return res
# if __name__ == "__main__":
#     file_path = "/home/aliredaq/Desktop/CG-DeepLearning/CGBench/repo_test_v4/pyro/pyro/ops/tensor_utils.py"  
#     function_name = "as_complex"  
#     test_file = "/home/aliredaq/Desktop/CG-DeepLearning/CGBench/repo_test_v4/pyro/tests/ops/test_tensor_utils.py::test_dct_dim"  

#     runner(file_path, function_name, test_file, "")

