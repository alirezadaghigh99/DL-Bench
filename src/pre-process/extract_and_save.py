import ast
import os
import re
import json
import csv

# Directory containing all repositories
base_dir = "repo_test"

# Define test libraries to look for
test_libraries = ['unittest', 'pytest', 'nose', 'doctest']

def is_test_file(file_path):
    """Check if the given file contains any test library imports."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        for lib in test_libraries:
            if re.search(rf'\bimport\s+{lib}\b|\bfrom\s+{lib}\s+import\b', content):
                return True
    return False

def find_test_files(repo_dir):
    """Find all test files in the given repository directory."""
    test_files = []
    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".py") and ("test" in root.split(os.sep) or "tests" in root.split(os.sep)):
                file_path = os.path.join(root, file)
                if is_test_file(file_path):
                    test_files.append(file_path)
    return test_files

class ProjectImportChecker(ast.NodeVisitor):
    def __init__(self, base_path):
        self.base_path = base_path
        self.internal_imports = set()
        self.local_imports = {}
        self.instance_classes = {}
        self.calls = set()
        self.function_calls_map = {}
        self.current_function = None
        self.defined_functions = set()

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
        self.defined_functions.add(node.name)
        self.current_function = node.name
        self.function_calls_map[node.name] = set()
        self.generic_visit(node)
        self.current_function = None

    def visit_Assign(self, node):
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
                return resolved_name.split('.')[0]  # Return only the function or method name
        elif isinstance(node.func, ast.Name):
            if node.func.id in self.local_imports:
                return self.local_imports[node.func.id].split('.')[0]  # Return only the function name
        return None

def extract_calls_from_file(filename, project_path):
    try:
        with open(filename, 'r') as file:
            source = file.read()
        tree = ast.parse(source)
        visitor = ProjectImportChecker(project_path)
        visitor.visit(tree)
        return visitor.calls, visitor.function_calls_map, visitor.defined_functions
    except SyntaxError as e:
        print(f"Syntax error in file {filename}: {e}")
        return set(), {}, set()

def analyze_test_files(test_files, project_path):
    function_call_relations = {}

    for test_file in test_files:
        calls, function_calls_map, defined_functions = extract_calls_from_file(test_file, project_path)
        user_defined_calls = {call for call in calls if call in defined_functions}
        for func, calls in function_calls_map.items():
            if func not in function_call_relations:
                function_call_relations[func] = set()
            function_call_relations[func].update({call for call in calls if call in defined_functions})

    return function_call_relations

def save_to_csv(function_call_relations, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Function', 'Test Cases'])
        for func, tests in function_call_relations.items:
            writer.writerow([func] + list(tests))

def process_repos(base_dir):
    aggregated_function_call_relations = {}
    repos = os.listdir(base_dir)
    for repo in repos:
        repo_dir = os.path.join(base_dir, repo)
        if os.path.isdir(repo_dir):
            test_files = find_test_files(repo_dir)
            function_call_relations = analyze_test_files(test_files, repo_dir)
            for func, tests in function_call_relations.items():
                if func not in aggregated_function_call_relations:
                    aggregated_function_call_relations[func] = set()
                aggregated_function_call_relations[func].update(tests)
    save_to_csv(aggregated_function_call_relations, "test_to_code/function_test_mappings.csv")

if __name__ == "__main__":
    process_repos(base_dir)
