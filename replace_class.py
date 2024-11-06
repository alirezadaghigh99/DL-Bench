import ast
import shutil
import os

def extract_function_from_code(code_str, function_name):
    """
    Extracts a function definition from the provided code string.
    """
    parsed_code = ast.parse(code_str)
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise ValueError(f"Function '{function_name}' not found in the provided code.")

def replace_function_in_class(file_path, class_name, function_name, code_str):
    # Step 1: Create a backup of the original file
    backup_file_path = file_path + '.backup'
    shutil.copyfile(file_path, backup_file_path)
    print(f"Backup created at {backup_file_path}")

    # Step 2: Read the original file
    with open(file_path, 'r') as file:
        original_code = file.read()

    # Step 3: Parse the original code into an AST
    tree = ast.parse(original_code)

    # Step 4: Extract the new function node from the provided code
    try:
        new_function_node = extract_function_from_code(code_str, function_name)
    except ValueError as e:
        print(e)
        return

    # Step 5: Traverse the AST to find the class and replace the function
    class_found = False
    function_replaced = False

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_found = True
            for idx, item in enumerate(node.body):
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    node.body[idx] = new_function_node
                    function_replaced = True
                    print(f"Replaced function '{function_name}' in class '{class_name}'.")
                    break
            if not function_replaced:
                print(f"Function '{function_name}' not found in class '{class_name}'.")
            break

    if not class_found:
        print(f"Class '{class_name}' not found in the file.")
        return

    # Step 6: Convert the modified AST back to code
    try:
        modified_code = ast.unparse(tree)
    except AttributeError:
        # For Python versions < 3.9, use astor
        import astor
        modified_code = astor.to_source(tree)

    # Step 7: Write the modified code back to the original file
    with open(file_path, 'w') as file:
        file.write(modified_code)

    print(f"Updated file saved at {file_path}")
    print("Sleeping for 10 seconds before restoring the original file...")
    # import time
    # time.sleep(10)

    # # Step 9: Restore the original file from the backup
    # shutil.copyfile(backup_file_path, file_path)
    # print(f"Original file restored from backup at {backup_file_path}")
    # Step 8: (Optional) Delete the backup file
    # os.remove(backup_file_path)
    # print(f"Backup file {backup_file_path} deleted.")

# Example usage:
if __name__ == "__main__":
    # Provide the path to your Python file
    file_path = '/local/data0/moved_data/vision/torchvision/transforms/transforms.py'

    # Specify the class and function names
    class_name = 'ToPILImage'
    function_name = '__call__'

    # Provide your code (which may include other parts) as a string
    code_str = '''
class ToPILImage:
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.
        """
        if isinstance(pic, torch.Tensor):
            # Convert torch tensor to numpy array
            pic = pic.mul(255).byte().numpy()
            if pic.ndimension() == 3:
                # Handle 3D tensor (C x H x W)
                pic = np.transpose(pic, (1, 2, 0))
            elif pic.ndimension() == 2:
                # Handle 2D tensor (H x W)
                pass
            else:
                raise ValueError("Unsupported tensor dimension: {}".format(pic.ndimension()))
        elif isinstance(pic, np.ndarray):
            if pic.ndim == 3 and pic.shape[2] == 1:
                # Handle single channel image
                pic = pic[:, :, 0]
        else:
            raise TypeError("pic should be Tensor or ndarray. Got {}".format(type(pic)))

        return Image.fromarray(pic, mode=self.mode)
    '''

    replace_function_in_class(file_path, class_name, function_name, code_str)
