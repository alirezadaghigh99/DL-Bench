import ast
import os

import os
import re
import json
import os
import re



def find_test_files(root_dir):
    # This will store the final results
    results = []
    # Define test libraries to look for
    test_libraries = ['unittest', 'pytest', 'nose', 'doctest']

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if 'test' or 'tests' is part of the directory path
        if 'test' in dirpath.split(os.sep) or 'tests' in dirpath.split(os.sep):
            # Process each file in the directory
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(dirpath, filename)
                    with open(file_path, 'r') as file:
                        file_content = file.read()
                    
                    # Search for test libraries in the file
                    used_libraries = []
                    for lib in test_libraries:
                        if re.search(rf'import {lib}', file_content) or re.search(rf'from {lib} import', file_content):
                            used_libraries.append(lib)
                    
                    if used_libraries:
                        # Check if the project entry already exists
                        project_entry = next((item for item in results if item['project_name'] == os.path.basename(root_dir)), None)
                        if project_entry:
                            # Append file path to existing entry
                            project_entry['test_files'].append(file_path)
                            # Merge unique test libraries used
                            project_entry['test_libraries'] = list(set(project_entry['test_libraries'] + used_libraries))
                        else:
                            # Append new project entry
                            results.append({
                                'project_name': os.path.basename(root_dir),
                                'test_files': [file_path],
                                'test_libraries': used_libraries
                            })

    # Write results to a JSONL file
    print(root_dir.split("/")[-1])
    with open(f'crawl_tests_v4/{root_dir.split("/")[-1]}.jsonl', 'w') as outfile:
        for result in results:
            outfile.write(json.dumps(result) + '\n')

# Example usage:
# find_test_files('/path/to/your/cloned/repo')


import ast
import os

class ProjectImportChecker(ast.NodeVisitor):
    def __init__(self, base_path):
        self.base_path = base_path
        self.internal_imports = set()  # Tracks modules that are part of the project
        self.local_imports = {}  # Tracks specific local names imported from modules
        self.instance_classes = {}  # Maps instance variables to their class names
        self.calls = set()
        self.function_calls_map = {}  # Maps function definitions to their internal calls
        self.current_function = None  # Track the current function context

    def visit_Import(self, node):
        for alias in node.names:
            module_name = alias.name
            self.check_import(module_name, alias.asname or alias.name)

    def visit_ImportFrom(self, node):
        module_name = node.module if node.module else ''
        base_import_path = os.path.join(self.base_path, module_name.replace('.', os.sep))
        for alias in node.names:
            full_import_name = f"{module_name}.{alias.name}"
            local_name = alias.asname or alias.name
            if os.path.exists(base_import_path) or os.path.exists(f"{base_import_path}.py"):
                self.local_imports[local_name] = full_import_name
                self.internal_imports.add(full_import_name)

    def check_import(self, module_name, local_name):
        potential_path = module_name.replace('.', os.sep)
        full_path = os.path.join(self.base_path, potential_path)
        if os.path.exists(full_path) or os.path.exists(f"{full_path}.py"):
            self.local_imports[local_name] = module_name
            self.internal_imports.add(module_name)

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.function_calls_map[node.name] = set()
        self.generic_visit(node)
        self.current_function = None

    def visit_Assign(self, node):
        # Track class instances to resolve method calls correctly
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id in self.local_imports:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.instance_classes[target.id] = self.local_imports[node.value.func.id]
        self.generic_visit(node)

    def visit_Call(self, node):
        full_call = self._resolve_call(node)
        if full_call and any(internal in full_call for internal in self.internal_imports):
            self.calls.add(full_call)
            if self.current_function:
                self.function_calls_map[self.current_function].add(full_call)
        self.generic_visit(node)

    def _resolve_call(self, node):
        if isinstance(node.func, ast.Attribute):
            call_path = []
            current = node.func
            while isinstance(current, ast.Attribute):
                call_path.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                if current.id in self.local_imports:
                    resolved_name = self.local_imports[current.id] + '.' + '.'.join(reversed(call_path))
                elif current.id in self.instance_classes:
                    resolved_name = self.instance_classes[current.id] + '.' + '.'.join(reversed(call_path))
                else:
                    return None
                return resolved_name
        elif isinstance(node.func, ast.Name):
            if node.func.id in self.local_imports:
                return self.local_imports[node.func.id]
        return None

def extract_calls_from_file(filename, project_path):
    with open(filename, 'r') as file:
        source = file.read()
    tree = ast.parse(source)
    visitor = ProjectImportChecker(project_path)
    visitor.visit(tree)
    return visitor.calls, visitor.function_calls_map
# Example usage

def analyze_test_files(jsonl_input, project_path):
    function_occurrences = {}
    function_call_relations = {}

    # Read the test file paths from the input JSONL file
    with open(f"crawl_tests_v4/{jsonl_input}", 'r') as file:
        for line in file:
            data = json.loads(line)
            test_files = data['test_files']
            
            # Process each test file
            for test_file in test_files:
                calls, function_calls_map = extract_calls_from_file(test_file, project_path)
                # Aggregate occurrences of each function call
                for call in calls:
                    if call in function_occurrences:
                        function_occurrences[call] += 1
                    else:
                        function_occurrences[call] = 1

                # Aggregate function to function call mappings
                for func, calls in function_calls_map.items():
                    if func not in function_call_relations:
                        function_call_relations[func] = set()
                    function_call_relations[func].update(calls)

    # Save the aggregated data to JSONL files
    with open(f'crawl_tests_v4/{project_path.split("/")[-1]}_function_occurrences.jsonl', 'w') as file:
        for func, count in function_occurrences.items():
            file.write(json.dumps({func: count}) + '\n')

    with open(f'crawl_tests_v4/{project_path.split("/")[-1]}_function_call_relations.jsonl', 'w') as file:
        for func, calls in function_call_relations.items():
            file.write(json.dumps({func: list(calls)}) + '\n')
            
            
repos = os.listdir("repo_test_v4")
from tqdm import tqdm
for repo in tqdm(repos):
    
    project_directory = f'repo_test_v4/{repo}'

    # print(extract_calls_from_file())
    if os.path.exists(f'crawl_tests_v4/{project_directory.split("/")[-1]}.jsonl'):
        print("#####")
        continue
    try:
        find_test_files(project_directory)
        analyze_test_files(f'{repo}.jsonl', project_directory)
    except Exception as e:
        print(e)
        print(project_directory)
print(len(os.listdir("crawl_tests_v4")))
# file_to_analyze = 'tests/lib/gui/stats/event_reader_test.py'
# calls, function_calls_map = extract_calls_from_file(file_to_analyze, project_directory)
# print("Internal calls found in the file:", calls)
# print("Function to internal calls map:", function_calls_map)

# find_test_files(project_directory)
# Example usagedsafs