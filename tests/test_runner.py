import os
import subprocess
import shutil
from replacer import replace_function
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
               test_case=None, conda_env="/home/z/anaconda3/envs/myenv/", is_conda = False, rep = None):
    # Prepare the command
    if rep == "DeepReg":
        is_conda = True
    python_path += rep + "/"
    if test_case:
        full_test_file = f"{test_file}::{test_case}"
    else:
        full_test_file = test_file

    # Use `source` to activate conda environment and run the pytest command
    if is_conda:
        conda_setup = "/home/z/anaconda3/etc/profile.d/conda.sh"

        command = f'source {conda_setup} && conda activate {rep.lower()} && PYTHONPATH={python_path} && cd {python_path}{rep} && python -m pytest {full_test_file} --color=no --cache-clear -v' 
        print("!"*20)
        print(command)
    else:
        command =  f'source {python_path}/{rep}/venv/bin/activate && PYTHONPATH={python_path}{rep} python -m pytest {full_test_file} --color=no --cache-clear -v -s' 
        print("!"*20)
        print(command)  
    # Run the command with bash -c
    result = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
    stdout_output = result.stdout
    stderr_output = result.stderr

    # Print the output and error messages (if any)
    if stdout_output:
        print("Output:\n", stdout_output)
    if stderr_output:
        print("Error:\n", stderr_output)
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
                f.write(line)  # Write the function definition
                f.write('    return ""\n')  # Ruin the function by inserting `return ""`
            elif inside_function and line.startswith(" "):  # Continue writing the rest of the function
                f.write(line)
            else:
                inside_function = False  
                f.write(line) 
    exit(1)
def restore_file(backup_path, original_path):
    shutil.copy(backup_path, original_path)
    print(f"File restored from {backup_path}")

def compare_test_results(first_result, second_result):
    first_failed = set(line for line in first_result.splitlines() if "FAILED" in line)
    second_failed = set(line for line in second_result.splitlines() if "FAILED" in line)
    
    return second_failed  

def process_test_results(test_cases, test_errs, final_result, initial_test, file_name, llm_output, function_name, llm, rep):
    out = file_name.replace(".py",".txt")
    out = f"/local/data0/moved_data/Organized_benchmark/results/{llm}/{rep}/" + llm_output.replace(".json", ".txt")
    output_dir = os.path.dirname(out)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{out}" if "FAILED" in final_result else f"{out}"
    print(output_filename, "error", llm_output)
    with open(output_filename , 'w') as f:
        f.write("output file:\n")
        f.write(f"{llm_output}\n")
        f.write("function:\n")
        f.write(f"{function_name}\n")
        f.write("Error Cases:\n")
        f.write(test_errs)
        f.write("\n")
        f.write("Related Test Cases:\n")
        for test in test_cases:
            f.write(f"{test}\n")
        f.write("\nFinal Test Result:\n")
        f.write(final_result)
        f.write("\n\nInitial Result:\n")
        f.write(initial_test)

def runner(file_path, function_name, test_file, llm_output, llm, rep):
    backup_path = backup_file(file_path)

    initial_test_result, initial_err = run_pytest(test_file, rep=rep)
    print("Initial test run completed.")
    try:
        replace_function(file_path, function_name, "temp.py", "/local/data0/moved_data/Organized_benchmark/t.txt")
        import time
        
        print(f"{function_name} function has been ruined.")
        clear_pycache()
        ruined_test_result, err = run_pytest(test_file, rep=rep)
        
        print(ruined_test_result)
    except Exception as e:
        print(e)
        ruined_test_result, err = "ERRor", "Error"
    print(err)
    print("Tests run after ruining the function.")
    related_tests = compare_test_results(initial_test_result, ruined_test_result)
    print("Related test cases identified.")

    restore_file(backup_path, file_path)
    
    final_test_result, final_err = run_pytest(test_file, rep=rep)
    print("Final test run completed.")

    print(os.path.basename(file_path))
    process_test_results(test_cases=related_tests, test_errs=err, final_result=final_test_result, initial_test=initial_test_result, file_name=os.path.basename(file_path) , llm_output=llm_output, function_name=function_name, llm=llm, rep=rep)

# if __name__ == "__main__":
#     file_path = "/home/z/Desktop/CG-DeepLearning/CGBench/repo_test_v4/pyro/pyro/ops/tensor_utils.py"  
#     function_name = "as_complex"  
#     test_file = "/home/z/Desktop/CG-DeepLearning/CGBench/repo_test_v4/pyro/tests/ops/test_tensor_utils.py::test_dct_dim"  

#     runner(file_path, function_name, test_file, "")


