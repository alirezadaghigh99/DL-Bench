import os
import subprocess
import shutil
from replace_class import replace_function_in_class

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

def run_pytest(test_file, python_path= "/local/data0/moved_data/", test_case=None, conda_env="/home/z/anaconda3/envs/myenv/", is_conda = False, repository="None"):
    # python_path += repository + "/"
    if test_case:
        full_test_file = f"{test_file}::{test_case}"
    else:
        full_test_file = test_file

    if is_conda:
        conda_setup = "/home/z/anaconda3/etc/profile.d/conda.sh"

        command = f'source {conda_setup} && conda activate {conda_env} && PYTHONPATH={python_path} && cd {python_path}{repository} && python -m pytest {full_test_file} --color=no --cache-clear -v' 
        print("!"*20)
        print(command)
    else:
        command =  f'source {python_path}/{repository}/venv/bin/activate && PYTHONPATH={python_path}{repository} python -m pytest {full_test_file} --color=no --cache-clear -v' 
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

def process_test_results(test_cases, ruined_err, final_result, initial_test, file_name, llm_output, function_name, class_name, repository, llm):
    
    out = file_name.replace(".py",".txt")
    out = f"/local/data0/moved_data/Organized_benchmark/results/{llm}/{repository}/class_" + llm_output.replace(".json", ".txt")
    output_dir = os.path.dirname(out)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{out}" if "FAILED" in final_result else f"{out}"
    print(output_filename, "error", llm_output)
    with open(output_filename , 'w') as f:
        f.write("output file:\n")
        f.write(f"{llm_output}\n")
        f.write("function:\n")
        f.write(f"{function_name}\n")
        f.write("class:\n")
        f.write(f"{class_name}\n")
        f.write("Error Cases:\n")
        f.write(ruined_err)
        f.write("\n")
        f.write("Related Test Cases:\n")
        for test in test_cases:
            f.write(f"{test}\n")
        f.write("\nFinal Test Result:\n")
        f.write(final_result)
        f.write("\n\nInitial Result:\n")
        f.write(initial_test)

def runner(file_path, function_name, test_file, class_name, code_str, llm_output, repository, llm):
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
        ruined_test_result = "Compile Error"
        ruined_err = "errorr"
        ruined_test_result = "ERRRor"
    print(ruined_test_result)
    print("Tests run after ruining the function.")
    related_tests = compare_test_results(initial_test_result, ruined_test_result)
    print("Related test cases identified.")

    restore_file(backup_path, file_path)
    try:
        final_test_result, final_err = run_pytest(test_file, repository=repository)
    except:
        final_test_result, final_err = "final_err", "final_err"
    print("Final test run completed.")

    print(os.path.basename(file_path))
    process_test_results(test_cases=related_tests, ruined_err = ruined_err, final_result=final_test_result, initial_test=initial_test_result, file_name=os.path.basename(file_path) , llm_output=llm_output, function_name=function_name, class_name=class_name, repository=repository, llm=llm)

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


