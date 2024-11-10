
import os
import json
import re
from test_runner import runner
r_s = ["ignite"]
llms = [ "geminai"]
for llm in llms:
    for r in r_s:

    
        folder_path = f"LLMs/output_{llm}/{r}/"
        print(folder_path)
        BASE_PATH = "/local/data0/moved_data/publishablew/"
        # BASE_PATH = "/local/data0/moved_data/p1/"
        # BASE_PATH = "/local/data0/moved_data/"
        # BASE_PATH = "/local/data0/moved_data/"
        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            # print("##################################")
            # print(filename)
            # print("##################################")
            if filename.endswith('.json') and not "processed_classes" in filename:
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as f:
                    # Load the JSON data
                    # if not "processed_vision_box_xywh_to_xyxy368" in file_path:
                    #     continue
                    data = json.load(f)
                    repo = f"{r}"
                    p = data["ground_truth"].split("#")[0]
                    f_name = data["function_name"]
                    tests = data["test"]
                    result = data.get('result', '')
                    
                    path_to_fn = os.path.join(BASE_PATH, repo, repo, p)
                    tests = os.path.join(BASE_PATH, repo, repo, tests)
                    print(path_to_fn)
                    print(filename)
                    # Use regex to extract the code starting from '```python\n'
                    match = re.search(r'```python\n(.*?)(?:\n```|$)', result, re.DOTALL)
                    if match:
                        code = match.group(1)
                        code = code.split("# Example")
                        code = code[0]
                        print(code)
                        with open("t.txt", "w") as f:
                            f.write(code)
                            print(code)
                        runner(path_to_fn, f_name, tests, filename, llm, r)
                    elif re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL):
                        match1 = re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL)
                        if match1:
                            code = match1.group(1)
                            print(code)
                            with open("t.txt", "w") as f:
                                f.write(code)
                                print(code)
                            runner(path_to_fn, f_name, tests, filename, llm, r)
                    else:
                        if "def" in result:
                        
                            code = result
                            print(code)
                            with open("t.txt", "w") as f:
                                f.write(code)
                                print(code)
                            runner(path_to_fn, f_name, tests, filename, llm, r)
                        
                        
                    
