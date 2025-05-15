import os
import subprocess
import shutil
from replacer_class import replace_function_in_class
import os
import subprocess
import shutil
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

def run_pytest(test_file, python_path= "/local/data0/moved_data/publishablew/", test_case=None, conda_env="/home/aliredaq/anaconda3/envs/myenv/", is_conda = False, repository="None"):
    if repository == "DeepReg":
        is_conda = True
    
    python_path += repository + "/"
    if repository == "pytorch3d":
        python_path = "/local/data0/moved_data/"
    
    if test_case:
        full_test_file = f"{test_file}::{test_case}"
    else:
        full_test_file = test_file
    vi_env = "venv"
    if repository == "nlp-architecht":
        vi_env = "nvenv"
   
    if is_conda:
        conda_setup = "/home/aliredaq/anaconda3/etc/profile.d/conda.sh"

        command = f'source {conda_setup} && conda activate {conda_env} && PYTHONPATH={python_path} && cd {python_path}{repository} && python -m pytest {full_test_file} --color=no --cache-clear -v' 
        print("!"*20)
        print(command)
    else:
        command =  f'''source {python_path}/{repository}/{vi_env}/bin/activate && PYTHONPATH={python_path}{repository} python3 -m pytest {full_test_file} --color=no --cache-clear -v''' 
        print("!"*20)
        print(command)  
    result = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
    stdout_output = result.stdout
    stderr_output = result.stderr

    # Print the output and error messages (if any)
    if stdout_output:
        print("Output:\n", stdout_output)
    if stderr_output:
        print("Error:\n", stderr_output)
    return result.stdout, stderr_output
    
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
                f.write('    return ""\n')  
            elif inside_function and line.startswith(" "):  
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
    
    return second_failed - first_failed  

import json
def process_test_results(test_cases, test_errs, all_failed, initial_test, file_name, llm_output, function_name, llm, rep, generated_code, function_path = None, prompt=None, technique = "fewshot"):
    out = file_name.replace(".py",".txt")
    
    out = f"../../results/test_output/{technique}/{llm}/v2/{rep}/class_" + llm_output.replace(".json", ".txt")
    output_dir = os.path.dirname(out)
    os.makedirs(output_dir, exist_ok=True)
    pass_or_fail = f"{0}" if "FAILED" in str(test_cases) else f"{1}"
    
    repo_path = function_path.split("/")[:7]
    module_path = function_path.split("/")[7:]
    
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
def runner(file_path, function_name, test_file, class_name, code_str, llm_output, repository, llm, prompt, technique):
    print(class_name)
    
    backup_path = backup_file(file_path)
    try:
        initial_test_result, initial_err = run_pytest(test_file, repository=repository)
    except:
        initial_test_result, initial_err = "initial_error", "initial_error"
    print("Initial test run completed.")
    try:
        replace_function_in_class(file_path, class_name, function_name, code_str)
        print(f"{function_name} function has been ruined.")
        clear_pycache()
        ruined_test_result, ruined_err = run_pytest(test_file, repository=repository)
    except Exception as e:
        ruined_test_result = "Error"
        ruined_err = "errorr"
    print(ruined_test_result)
    print("Tests run after ruining the function.")
    related_tests = compare_test_results(initial_test_result, ruined_test_result)
    print("Related test cases identified.")

    restore_file(backup_path, file_path)
    
    print("Final test run completed.")
    related_tests = ignore_test_cases(repository, related_tests )
    print(os.path.basename(file_path))
    res = process_test_results(test_cases=related_tests, all_failed = ruined_test_result, test_errs=ruined_err, initial_test=initial_test_result, file_name=os.path.basename(file_path) , llm_output=llm_output, function_name=function_name, llm=llm, rep=repository, generated_code=code_str, function_path=file_path, prompt=prompt, technique=technique)
    if ruined_test_result == "Error":
        return 0
    return res
if __name__ == "__main__":
    
    code_str = '''
import torch

class Translate(Transform3d):
    def __init__(
        self,
        x,
        y=None,
        z=None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        return

        super().__init__(dtype=dtype, device=device)

        if x is not None and (y is None and z is None):
            if isinstance(x, (torch.Tensor, list, tuple)):
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)
                self.translation = x.to(self.device)
            else:
                self.translation = torch.tensor([x, y or 0, z or 0], dtype=self.dtype, device=self.device)

        elif y is not None and z is None:
            if isinstance(y, (torch.Tensor, list, tuple)):
                if len(y.shape) == 1:
                    y = y.unsqueeze(0)
                self.translation = torch.cat([self.translation, y.to(self.device)], dim=1)
            else:
                self.translation = torch.cat([self.translation, torch.tensor([y, 0], dtype=self.dtype, device=self.device)], dim=1)

        elif z is not None:
            if isinstance(z, (torch.Tensor, list, tuple)):
                if len(z.shape) == 1:
                    z = z.unsqueeze(0)
                self.translation = torch.cat([self.translation, z.to(self.device)], dim=1)
            else:
                self.translation = torch.cat([self.translation, torch.tensor([0, y, z], dtype=self.dtype, device=self.device)], dim=1)

        self.matrix = torch.eye(4, device=self.device).to(self.dtype)
        self.matrix[0, 3] = self.translation[0]
        self.matrix[1, 3] = self.translation[1]
        self.matrix[2, 3] = self.translation[2]    
    
    '''
    file_path = "/local/data0/moved_data/pytorch3d/pytorch3d/transforms/transform3d.py"  
    function_name = "__init__"
    class_name = "Translate"  
    test_file = "/local/data0/moved_data/pytorch3d/tests/test_transforms.py"  

    runner(file_path, function_name, test_file, class_name, code_str, "")


