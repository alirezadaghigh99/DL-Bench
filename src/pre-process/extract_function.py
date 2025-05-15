import ast
import os
import csv
import json
import sys
sys.setrecursionlimit(10000)
import subprocess

from radon.complexity import cc_visit
from radon.visitors import ComplexityVisitor
from radon.metrics import h_visit
from radon.metrics import mi_visit
from cognitive_complexity.api import get_cognitive_complexity\

def count_physical_loc(code_string):
   # Split the input string into lines
   lines = code_string.split('\n')
   
   # Filter out empty lines and count the remaining lines
   non_empty_lines = [line for line in lines if line.strip() != '']
   
   return len(non_empty_lines)

def calculate_cyclomatic_complexity(code):
   # Analyze the code
   blocks = cc_visit(code)
   # for block in blocks:
   #     print(f'{block.name}: {block.complexity}')

   # Calculate the average Cyclomatic Complexity
   total_complexity = sum(block.complexity for block in blocks)
   average_complexity = total_complexity / len(blocks) if blocks else 0
   # print(f'Average Cyclomatic Complexity: {average_complexity}')
   return average_complexity

def calculate_halstead_complexity(code):
   results = h_visit(code)
   return results[0].vocabulary

def calculate_mi(code_string):
   mi_score = mi_visit(code_string, True)
   return mi_score

def calculate_cognitive_complexity(code):
   parsed_code = ast.parse(code)
   new_body = [node for node in parsed_code.body if not isinstance(node, ast.Import) and not isinstance(node, ast.ImportFrom) and not isinstance(node, ast.Assign)]
   # print(new_body[0])
   funcdef = new_body[0]
   cc_score = get_cognitive_complexity(funcdef)
   return cc_score
def calculate_complexity(function):
    return calculate_cyclomatic_complexity(function) + calculate_mi(function) + count_physical_loc(function) + calculate_cognitive_complexity(function) + calculate_halstead_complexity(function)


def read_and_modified_csv(csv_path, repo_base_path, existing_headers, new_headers, github_token, github_repo_url):
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file, fieldnames=existing_headers)
        rows = list(reader)

    i = 0
    from tqdm import tqdm
    for row in tqdm(rows):
        i += 1
        print(i)
        parts = row['Function Name'].split('_')
        repo = parts[0]
        module_and_name = row["Function Name"].replace(f"{repo}_", "")
        parts = module_and_name.split(".")
        class_or_func_name = parts[-1]  # Last part is the class or function name
        module_path = '.'.join(parts[:-1])

        path = os.path.join(repo_base_path, repo)
        try:
            function_definition, definition_path = extract_from_module(path, module_path, class_or_func_name)
            function_definition = function_definition.strip()
            if function_definition.startswith("def") or repo_base_path.startswith("pytorch") or repo_base_path.startswith("pennylane"):
                continue
            if function_definition:
                relative_path = definition_path[len(repo_base_path):].lstrip(os.sep)
                print(relative_path)
                save_path = os.path.join(external_repo_path, relative_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'a') as f:
                    f.write(function_definition)
                    f.write("\n\n")
                # git_commit_and_push(external_repo_path, f"Updated {class_or_func_name} in {save_path}", github_token, github_repo_url)

                github_url = f"{github_repo_url}/blob/master/{relative_path.replace(os.sep, '/')}"
                with open(f"functions/paths_outputv41class{int(i//5000)}.jsonl", "a") as f:
                    json.dump({
                        'repo': repo,
                        'function_name': class_or_func_name,
                        'path_to_function_in_repo': github_url,
                        'function_definition' : function_definition,
                    }, f)
                    f.write("\n")
        except Exception as e:
            print(f"Failed to extract: {str(e)}")
            continue
        
        with open(f"functions/new_outputv41class{int(i//5000)}.jsonl" , "a") as f:
            json.dump({
                'repo': repo,
                'function_name': class_or_func_name,
                'function_definition': function_definition,
                'file_path': definition_path  
            }, f)
            f.write("\n")
def git_commit_and_push(repo_path, commit_message, github_token, github_repo_url):
    os.chdir(repo_path)
    subprocess.run(['git', 'add', '.'], check=True)
    try:
        # Attempt to commit; if no changes, this will raise an exception that we catch
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
    except subprocess.CalledProcessError as e:
        if "nothing to commit" in e.stderr.decode():
            print("Nothing to commit. Working tree clean.")
            return
        else:
            raise e  # Re-raise the exception if it's not the "nothing to commit" error
    # Use the token for pushing
    repo_url = f"https://{github_token}@{github_repo_url.split('://')[1]}"
    subprocess.run(['git', 'push', repo_url, 'master'], check=True)

def extract_from_module(repo_path, module_path, class_or_func_name):
   
    parts = module_path.split('.')
    base_path = os.path.join(repo_path, *parts[:-1])
    target_file = parts[-1]
    potential_paths = [
        os.path.join(base_path, target_file + '.py'),  # a/b/c.py
        os.path.join(repo_path, *parts) + '.py',       # a/b/c.py (whole path as file)
        os.path.join(base_path, target_file, '__init__.py')  # a/b/c/__init__.py
    ]

    for path in potential_paths:
        if os.path.isfile(path):
            with open(path, 'r') as file:
                source_code = file.read()
                result = extract_class_or_function(source_code, class_or_func_name)
                if result:
                    return result, path  # Return both the source code and the path

    # If not found, search through all Python files in the repo
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as file:
                    source_code = file.read()
                    result = extract_class_or_function(source_code, class_or_func_name)
                    if result:
                        return result, full_path  # Return both the source code and the path

    return "Class or function not found in the specified module.", None

def extract_class_or_function(source_code, target_name):
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
            start_lineno, end_lineno = node.lineno, getattr(node, 'end_lineno', node.lineno)
            return '\n'.join(source_code.splitlines()[start_lineno-1:end_lineno])  # -1 because line numbers are 1-based
    return None

# Example usage:
csv_path = 'sorted_data_v4.csv'
external_repo_path = '/home/aliredaq/Desktop/Code-Generation-Benchmark'

repo_base_path = 'repo_test_v4'
existing_headers = ["id", 'Function Name'] + [f"Called in functions with {i} calls" for i in range(1, 6)]
new_headers = ['repo', 'function_name', 'function_definition', 'file_path']
github_token = ''
github_repo_url = "https://github.com/v/b"
read_and_modified_csv(csv_path, repo_base_path, existing_headers, new_headers, github_token, github_repo_url)
