import ast
import os
import shutil

def replace_function(repo_path, function_name, temp_file, generated_code_file):
    backup_path = repo_path + ".backup"
    shutil.copy(repo_path, backup_path)
    print(f"Original file backed up to {backup_path}")
    
    with open(repo_path, 'r') as f:
        repo_code = f.read()
    
    tree = ast.parse(repo_code)
    
    original_function_def = None
    function_parent = None  
    imports_in_original = []

    class RenameFunctionAndCaptureImports(ast.NodeTransformer):
        def __init__(self):
            self.class_stack = []

        def visit_ClassDef(self, node):
            self.class_stack.append(node)
            self.generic_visit(node)
            self.class_stack.pop()
            return node

        def visit_FunctionDef(self, node):
            nonlocal original_function_def, function_parent
            if node.name == function_name:
                original_function_def = node 
                function_parent = self.class_stack[-1] if self.class_stack else None
                node.name = f"{function_name}1"  
            return node

        def visit_Import(self, node):
            imports_in_original.append(ast.unparse(node))
            return node

        def visit_ImportFrom(self, node):
            imports_in_original.append(ast.unparse(node))
            return node

    transformer = RenameFunctionAndCaptureImports()
    tree = transformer.visit(tree)

    if original_function_def is None:
        raise ValueError(f"Function '{function_name}' not found in the repository code.")

    import copy

    wrapper_function_def = ast.FunctionDef(
        name=function_name,
        args=copy.deepcopy(original_function_def.args),
        body=[],  # Will fill this in next
        decorator_list=copy.deepcopy(original_function_def.decorator_list),
        returns=copy.deepcopy(original_function_def.returns),
        type_comment=original_function_def.type_comment
    )

    wrapper_function_def.lineno = original_function_def.lineno
    wrapper_function_def.col_offset = original_function_def.col_offset

    import_stmt = ast.ImportFrom(
        module='.' + os.path.splitext(temp_file)[0],
        names=[ast.alias(name=function_name, asname=None)],
        level=0  # Adjust level as needed
    )
    import_stmt.lineno = original_function_def.lineno + 1
    import_stmt.col_offset = original_function_def.col_offset + 4  

    arg_names = [arg.arg for arg in original_function_def.args.args]

    is_method = function_parent is not None and isinstance(function_parent, ast.ClassDef)

    if is_method and 'self' in arg_names:
        call_arg_names = arg_names[1:]  
    else:
        call_arg_names = arg_names

    function_caller = ast.Name(id=function_name, ctx=ast.Load())

    call_args = [ast.Name(id=arg_name, ctx=ast.Load()) for arg_name in call_arg_names]
    function_call = ast.Call(
        func=function_caller,
        args=call_args,
        keywords=[]
    )
    function_call.lineno = original_function_def.lineno + 2
    function_call.col_offset = original_function_def.col_offset + 8  # Indent inside the return statement

    # Create the return statement
    return_stmt = ast.Return(value=function_call)
    return_stmt.lineno = original_function_def.lineno + 2
    return_stmt.col_offset = original_function_def.col_offset + 4  # Indent inside the function

    # Set the body of the wrapper function
    wrapper_function_def.body = [import_stmt, return_stmt]

    # Fix locations
    ast.fix_missing_locations(wrapper_function_def)

    # Print the wrapper function code for verification
    wrapper_function_code = ast.unparse(wrapper_function_def)
    print("Wrapper Function Code:")
    print(wrapper_function_code)

    class InsertWrapperFunction(ast.NodeTransformer):
        def visit_Module(self, node):
            if function_parent is None:
                new_body = []
                inserted_wrapper = False
                for stmt in node.body:
                    if isinstance(stmt, ast.FunctionDef):
                        if stmt.name == f"{function_name}1":
                            if not inserted_wrapper:
                                print(f"Inserting wrapper function '{function_name}' at module level")
                                new_body.append(wrapper_function_def)
                                inserted_wrapper = True
                    new_body.append(stmt)
                if not inserted_wrapper:
                    # If the wrapper hasn't been inserted, append it at the end
                    print(f"Appending wrapper function '{function_name}' at the end of the module")
                    new_body.append(wrapper_function_def)
                node.body = new_body
            else:
                self.generic_visit(node)
            return node

        def visit_ClassDef(self, node):
            if function_parent is node:
                new_body = []
                inserted_wrapper = False
                for stmt in node.body:
                    if isinstance(stmt, ast.FunctionDef):
                        if stmt.name == f"{function_name}1":
                            if not inserted_wrapper:
                                print(f"Inserting wrapper function '{function_name}' into class '{node.name}'")
                                new_body.append(wrapper_function_def)
                                inserted_wrapper = True
                    new_body.append(stmt)
                if not inserted_wrapper:
                    print(f"Appending wrapper function '{function_name}' at the end of class '{node.name}'")
                    new_body.append(wrapper_function_def)
                node.body = new_body
            else:
                self.generic_visit(node)
            return node

    tree = InsertWrapperFunction().visit(tree)
    ast.fix_missing_locations(tree)

    with open(repo_path, 'w') as f:
        f.write(ast.unparse(tree))

    with open(generated_code_file, 'r') as gen_file:
        generated_code = gen_file.read()

    try:
        generated_tree = ast.parse(generated_code)
    except Exception as e:
        return f"Cannot generate AST because {e}"

    imports_in_generated = []

    class ReplaceFunctionCallsAndCaptureImports(ast.NodeTransformer):
        def __init__(self, repo_file_name):
            self.repo_file_name = os.path.basename(repo_file_name)
        
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == function_name:
                node.func.id = f"{function_name}1"
            elif isinstance(node.func, ast.Attribute) and node.func.attr == function_name:
                node.func.attr = f"{function_name}1"
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if any(alias.name == function_name for alias in node.names):
                for alias in node.names:
                    if alias.name == function_name:
                        alias.name = f"{function_name}1"  # Rename the import to function_name1
            imports_in_generated.append(ast.unparse(node))
            return node

    replace_transformer = ReplaceFunctionCallsAndCaptureImports(repo_path)
    generated_tree = replace_transformer.visit(generated_tree)

    imports_to_add = [imp for imp in imports_in_original if imp not in imports_in_generated]

    temp_code_with_imports = "\n".join(imports_to_add) + "\n" + ast.unparse(generated_tree)
    temp_code_with_imports = temp_code_with_imports.replace("import accimage", "")
    temp_path = os.path.join(os.path.dirname(repo_path), temp_file)
    with open(temp_path, 'w') as f:
        print(temp_code_with_imports)
        f.write(temp_code_with_imports)
