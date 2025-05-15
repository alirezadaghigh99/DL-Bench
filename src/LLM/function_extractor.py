import ast
import os
import re
def get_function_definition(file_path, function_name, class_name=None):
    """
    Returns the full source code (as a string) of the specified function
    from the given file. If class_name is provided, it will look for 
    the function inside that class. Otherwise, it will look for a top-level 
    function definition.

    :param file_path: Path to the Python source file.
    :param function_name: Name of the function to extract.
    :param class_name: Name of the class containing the function (optional).
    :return: The full text of the function definition, or None if not found.
    """
    # Read in the entire source file
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Parse the source into an AST
    tree = ast.parse(source)
    
    class FunctionFinder(ast.NodeVisitor):
        """
        AST NodeVisitor that searches for the target function.
        If class_name is provided, searches within that class only.
        Otherwise, searches for top-level function definition.
        """
        def __init__(self, target_func_name, target_class_name=None):
            self.target_func_name = target_func_name
            self.target_class_name = target_class_name
            self.found_node = None

        def visit_ClassDef(self, node):
            # If we're looking for a function inside a particular class
            if self.target_class_name is not None and node.name == self.target_class_name:
                # Look for the function within this class's body
                for body_node in node.body:
                    if isinstance(body_node, ast.FunctionDef) and body_node.name == self.target_func_name:
                        self.found_node = body_node
                        return  # Stop searching once found
            # Keep searching recursively (in case the class can appear deeper)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            # If class_name is None, we're looking for a top-level function
            if self.target_class_name is None and node.name == self.target_func_name:
                self.found_node = node
                return
            self.generic_visit(node)

    finder = FunctionFinder(function_name, class_name)
    finder.visit(tree)

    if not finder.found_node:
        # Function not found
        return None

    # Grab the line numbers
    func_node = finder.found_node
    start_lineno = func_node.lineno - 1  # AST is 1-based
    end_lineno = func_node.end_lineno    # Requires Python 3.8+

    lines = source.splitlines()
    # Extract the lines corresponding to the function definition
    function_source = "\n".join(lines[start_lineno:end_lineno])

    return function_source


def parse_text_to_json(text):
    lines = text.split('\n')
    result = {}
    current_key = None
    current_value_lines = []
    valid_keys = {
        'stage',
        'task',
        'data',
        'prompt',
        'ground truth',
        'repo',
        'function',
        'test_cases',
        'class'
    }
    
    for line in lines:
        stripped_line = line.strip()
        
        if not stripped_line:
            if current_key is not None:
                current_value_lines.append('')
            continue
        
        key_match = re.match(r'^(\s*)(.*?):\s*(.*)$', line)
        
        if key_match:
            key = key_match.group(2).strip().lower()
            value = key_match.group(3)
            if key in valid_keys:
                if current_key is not None:
                    result[current_key] = '\n'.join(current_value_lines).strip()
                current_key = key_match.group(2).strip()
                current_value_lines = [value] if value else []
                continue  
        if current_key is not None:
            current_value_lines.append(line)
    if current_key is not None:
        result[current_key] = '\n'.join(current_value_lines).strip()
    return result

# print(get_function_definition("/local/data0/moved_data/publishablew/small-text/small-text/small_text/query_strategies/bayesian.py", "score", "BALD"))
# import os
input_folder = "../../data/DLEval-20240920T201632Z-001/DLEval"
BASE_DIR = "/local/data0/moved_data/publishablew/"
all_files = os.listdir(input_folder)

counter = 0
import json
for idx, file in enumerate(all_files):
    
    if file.endswith(".txt"):
        file_path = os.path.join(input_folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        try:
            json_data = parse_text_to_json(text)
            json_data["filename"] = str(file)
            with open("DL-Bench-New.jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            # print(json_data)
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            counter += 1
            continue
        #     function_path = os.path.join(BASE_DIR, json_data["repo"], json_data["repo"], json_data["ground Truth"].split("#")[0])
        #     class_name = json_data.get("class", None)
        #     function_name = json_data["function"]
        #     function_definition = get_function_definition(function_path, function_name, class_name)
        #     if function_definition:
        #         data_dict = {
        #             "idx": idx,
        #             "function": function_definition,
        #         }
        #         with open("f_definitions.jsonl", 'a', encoding='utf-8') as f:
        #             f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
        #     else:
        #         print(function_path, function_name, class_name)
        #         counter += 1
        # except Exception as e:
        #     print(f"Error processing {file}: {e}")
        #     counter += 1
print(counter)