import os
import json
def list_files(directory):
    # List all files and directories in the specified directory
    files = os.listdir(directory)
    # Filter out directories, keep only files
    files = [file for file in files if os.path.isfile(os.path.join(directory, file))]
    return files

def aggregate_function_calls(data, call_counts):
    
    # Iterate through each function and its call count in the data dictionary
    for function_name, num_calls in data.items():
        # If the number of calls is already a key in the dictionary, increment its value
        if num_calls in call_counts:
            call_counts[num_calls] += 1
        # Otherwise, add the number of calls as a new key with a value of 1
        else:
            call_counts[num_calls] = 1
    return call_counts


import json

def analyze_function_calls(data, call_counts):

    # Process each line of data, which is a JSON string
    for line in data:
        # Convert the JSON string to a Python dictionary

        # Each dictionary has one key-value pair; extract the number of called functions (length of the list)
        num_called = len(list(data.values())[0])

        # Update the dictionary with the count of functions that have this number of calls
        if num_called in call_counts:
            call_counts[num_called] += 1
        else:
            call_counts[num_called] = 1

    return call_counts

files = list_files("crawl_tests_v4")
call_count = {}
for file in files:
    if "function_occurrences" in file:
        with open(f"crawl_tests_v4/{file}", 'r') as file:
            for line in file:
                data = json.loads(line)
                call_count = aggregate_function_calls(data, call_count)

sorted_dict = dict(sorted(call_count.items(), key=lambda item: int(item[0])))
print(sorted_dict)
for file in files:
    if "function_call" in file:
        with open(f"crawl_tests_v4/{file}", 'r') as file:
            for line in file:
                data = json.loads(line)
                call_count = analyze_function_calls(data, call_count)

sorted_dict = dict(sorted(call_count.items(), key=lambda item: int(item[0])))

print(sorted_dict)
import pandas as pd
from collections import defaultdict


# data = [
#     {"test_get_run": ["tests.integration.functional.conftest.constant_int_output_test_step", "tests.integration.functional.conftest.int_plus_one_test_step"]},
# {"test_get_run_fails_for_non_existent_run": []},
# {"test_get_unlisted_runs": ["tests.integration.functional.conftest.constant_int_output_test_step", "tests.integration.functional.conftest.int_plus_one_test_step"]},
# {"test_basic_crud_for_entity": []}
# ]

test_to_func = {}
function_counts = defaultdict(lambda: defaultdict(int))
for file in files:
    if "function_call" in file:
        with open(f"crawl_tests_v4/{file}", 'r') as f:
            for line in f:
                data = json.loads(line)


                for function, calls in data.items():
                    n_calls = len(calls)
                    for called in calls:
                        
                        function_counts[file.split("_")[0] + "_" + called][n_calls] += 1
                        
                        if file.split("_")[0] + "_" + called not in test_to_func:
                            test_to_func[file.split("_")[0] + "_" + called] = [list(data.keys())[0]]
                        else:
                            test_to_func[file.split("_")[0] + "_" + called].append(list(data.keys())[0])
for d in data:
    if d == "test_dequantize_int8_bias_cuda":
        continue
    print(d)
    for function, calls in d.items():
        n_calls = len(calls)
        for called in calls:
            
            function_counts[called][n_calls] += 1


max_calls = max(max(c.keys()) for c in function_counts.values()) if function_counts else 0
columns = [f'Called in functions with {i} calls' for i in range(1, max_calls+1)]


df = pd.DataFrame(index=function_counts.keys(), columns=columns).fillna(0)

# Fill the DataFrame with actual counts
for func, calls in function_counts.items():
    for n_calls, count in calls.items():
        column_name = f'Called in functions with {n_calls} calls'
        df.at[func, column_name] = count

df.index.name = "Function Name"
df.to_csv("data4.csv")
with open("test_to_func_v4.json", "w") as f:
    json.dump(test_to_func, f)

