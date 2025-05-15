import boto3
import json
import sys
import os
import random

from function_extractor import get_function_definition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompts.shots_for_few import *
from prompts.shots_for_cot import *
from botocore.exceptions import ClientError
from openai import OpenAI
import csv
import json
OPENAI_API_KEY = "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"
Geminay_KEY = "AIzaSyAtEDC9lhEDrx5AojepI3IObJm0XIzVvIA"
from prompts.few_shot_cot import cot_fewshot_data_outsider, cot_fewshot_using_different_categories, cot_few_shot_using_same_or_different_category
from prompts.zero_shot_cot import ZeroShotCoT
from prompts.few_shot import fewshot_data_outsider, fewshot_using_different_categories, few_shot_using_same_or_different_category, fewshot_data_outside_benchmark
zero_shot_cot = ZeroShotCoT("dlbench", "4o", "zero_cot")

data_path = "/home/aliredaq/Desktop/DeepAgent/ICSE/data/DLEval-20240920T201632Z-001/DLEval"
all_files = os.listdir(data_path)
client = OpenAI(api_key=OPENAI_API_KEY)
# import google.generativeai as genai
from fireworks.client import Fireworks

# from google import genai

FIREWORK_API = "fw_3ZWAcGSbWGnWkXzqunnY92FZ"
def call_geminai(prompt):
    
    client = genai.Client(api_key=Geminay_KEY)
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
    )
        
    except Exception as e:
        return str(e)
    return response.text
def get_completion(prompt,tem = 0 , model="gpt-4o"):
    
        c=[
                {'role': 'user', 'content': prompt}]


        response = client.chat.completions.create(model=model,
        messages=c,
        
        )

        return response.choices[0].message.content
   
    
    
def call_deepseek(input_prompt):
    client = Fireworks(api_key=FIREWORK_API)
    
    response = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v3",
    messages=[{
    "role": "user",
    "content": f"{input_prompt}",
}],
    )

    return response.choices[0].message.content
    
    
def call_qwen(input_prompt):
    client = Fireworks(api_key=FIREWORK_API)
    
    response = client.chat.completions.create(
    model="accounts/fireworks/models/mixtral-8x22b-instruct",
    messages=[{
    "role": "user",
    "content": f"{input_prompt}",
}],
    )

    return response.choices[0].message.content
    
    
def call_mistral(prompt):
    client = Fireworks(api_key=FIREWORK_API)
    response = client.chat.completions.create(
    model="accounts/fireworks/models/mixtral-8x22b-instruct",
    temperature=0,
    messages=[{
    "role": "user",
    "content": f"{prompt}",
}],
    )

    return response.choices[0].message.content
    
def call_llama(prompt):
    client = Fireworks(api_key=FIREWORK_API)
    response = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    temperature=0,
    messages=[{
    "role": "user",
    "content": f"{prompt}",
}],
    )

    return response.choices[0].message.content

def call_antropic(prompt):
# Create a Bedrock Runtime client in the AWS Region of your choice.

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    # Set the model ID, e.g., Claude 3 Haiku.
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    # Define the prompt for the model.
    prompt += "\nONLY RETURN THE CODE"
    # Format the request payload using the model's native structure.
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 12000,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    
        # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)

        

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["content"][0]["text"]
    return response_text


import os
import json

def process_file_data(prompt, llm):
    
    if llm == "antropic":
        return call_antropic(prompt)
    if llm == "mistral":
        return call_mistral(prompt)
    if llm == "llama":
        return call_llama(prompt=prompt)
    if llm == "o3-mini" or llm == "openai-4o":
        if llm == "o3-mini":
            return get_completion(prompt=prompt, model="o3-mini")
        return get_completion(prompt=prompt)
    if llm == "geminai":
        return call_geminai(prompt=prompt)
    if llm == "deepseek":
        return call_deepseek(prompt)
    if llm == "qwen":
        return call_qwen(prompt)

import json
import re

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

from tqdm import tqdm
def process_txt(input_file_path):

    with open(input_file_path, 'r') as f:
        file_content = f.read()

    parsed_data = parse_text_to_json(file_content)
    
    return parsed_data
from tqdm import tqdm
def process_folder(input_folder, output_folder, llm, repo, technique):
    """
    Process each text file in the input folder, parse it to JSON, call the process_file_data function,
    and save the results into an output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    counter = 0
    
    for filename in tqdm(os.listdir(input_folder)):
        if not f"{repo}" in filename:
            continue
        
        input_file_path = os.path.join(input_folder, filename)
        counter += 1
        while True:
            if os.path.isfile(input_file_path) and filename.endswith(".txt"):
                

                parsed_data = process_txt(input_file_path)
                repository = parsed_data["repo"]
                if repo == "pytorch" and repository != "pytorch":
                    break
                
                prompt = parsed_data ["prompt"]
                new_prompt = prompt
                if technique == "zeroshot":
                    new_prompt = prompt
                if technique == "fewshot" or technique == "fewshot-guided":
                    shots = 0
                    trial = 0
                    all_shots = []
                    while shots != 4:
                        trial += 1
                        try:
                            shot = random.choice(all_files)
                            data_in_file = process_txt(os.path.join(data_path, shot))
                            if "class" in shot:
                                shot_stage = data_in_file["data"]
                                shot_data = data_in_file["stage"]
                            else:
                                shot_stage = data_in_file["stage"]
                                shot_data = data_in_file["data"]
                            
                            function_shot_path = data_in_file["ground Truth"]
                            repository_shot = data_in_file["repo"]
                            funciton_shot = data_in_file["function"]
                            if "class" in data_in_file.keys():
                                class_shot = data_in_file['class']
                            else:
                                class_shot = None
                            function_shot_path = os.path.join("/local/data0/moved_data/publishablew", repository_shot, repository_shot, function_shot_path)
                            shot_definition = get_function_definition(function_shot_path, funciton_shot, class_shot)
                            if True:
                                all_shots.append(shot_definition)
                                shots += 1
                            if trial == 100:
                                dat = "imag"
                        except:
                            print("Error in makeing shots")
                            pass
                    new_prompt = few_shot_using_same_or_different_category(prompt, all_shots[0], all_shots[1], all_shots[2], all_shots[3], None, False)
                if technique == "zeroshot_cot":
                    new_prompt = zero_shot_cot.form_technique_prompt(prompt)
                if technique == "fewshot-outside-benchmark":
                    new_prompt = fewshot_data_outside_benchmark(prompt)
                
                if technique == "fewshot-different" or technique == "fewshot-guided-different":
                    new_prompt = fewshot_using_different_categories(prompt=prompt)
                step = parsed_data["stage"]
                stage = parsed_data['stage']
                if technique == "fewshot-classifier":
                    
                    fca = 'class_dl.jsonl'

                    classed_data = list(map(json.loads, open(fca) ))
                    for data in classed_data:
                        if data["filename"] == filename:
                            stage = data["predicted_label"]
                    
                    print(stage)
                    
                    
                task = parsed_data["task"]
                data = parsed_data["data"]
                cat = ""
                dat = ""
                if "train" in stage.lower():
                    cat = "train"
                elif "pro" in stage.lower():
                    cat = "pre"
                elif "model" in stage.lower():
                    cat = "model"
                elif "metric" in stage.lower():
                    cat = "eval"
                elif "infer" in stage.lower():
                    cat = "infer"
                else:
                    cat = "pre"
                
                if "imag" in step.lower() or "imag" in task.lower() or "imag" in data.lower():
                    dat = "imag"
                elif "tex" in step.lower() or "tex" in task.lower() or "tex" in data.lower():
                    dat = "tex"
                elif "tab" in step.lower() or "tab" in task.lower() or "tab" in data.lower():
                    dat = "tab"
                if technique == "fewshot-cot-different":
                    new_prompt = cot_fewshot_using_different_categories(prompt)
                
                
                if technique == "fewshot-same-cat" or technique == "fewshot-same-cat-guided" or technique == "fewshot-classifier":
                    print(stage)
                    shots = 0
                    all_shots = []
                    trial = 0
                    while shots != 4:
                        trial += 1
                        try:
                            shot = random.choice(all_files)
                            data_in_file = process_txt(os.path.join(data_path, shot))
                            if "class" in shot:
                                shot_stage = data_in_file["data"]
                                shot_data = data_in_file["stage"]
                            else:
                                shot_stage = data_in_file["stage"]
                                shot_data = data_in_file["data"]
                            
                            function_shot_path = data_in_file["ground Truth"]
                            repository_shot = data_in_file["repo"]
                            funciton_shot = data_in_file["function"]
                            if "class" in data_in_file.keys():
                                class_shot = data_in_file['class']
                            else:
                                class_shot = None
                            function_shot_path = os.path.join("/local/data0/moved_data/publishablew", repository_shot, repository_shot, function_shot_path)
                            shot_definition = get_function_definition(function_shot_path, funciton_shot, class_shot)
                            if cat in shot_stage.lower() and dat in shot_data.lower():
                                all_shots.append(shot_definition)
                                shots += 1
                            if trial == 100:
                                dat = "imag"
                        except:
                            print("Error in making shots")
                            pass
                    new_prompt = few_shot_using_same_or_different_category(prompt, all_shots[0], all_shots[1], all_shots[2], all_shots[3],cat, True )
                    print(cat)
                if technique == "cot-fewshot-same-cat":
                    stage = parsed_data["stage"]
                    task = parsed_data["task"]
                    data = parsed_data["data"]
                    cat = ""
                    if "train" in stage.lower() or "train" in task.lower() or "train" in data.lower():
                        shot1, shot2, shot3 = training
                        shot4 = pre_post[0]
                        cat = "train"
                    elif "pre" in stage.lower() or "pre" in task.lower() or "pre" in data.lower():
                        shot1, shot2, shot3 = pre_post
                        shot4 = model[1]
                        cat = "Processing"
                    elif "model" in stage.lower() or "model" in task.lower() or "model" in data.lower():
                        shot1, shot2, shot3 = model
                        shot4 = pre_post[0]
                        cat = "Model Construction"
                    elif "eval" in stage.lower() or "eval" in task.lower() or "eval" in data.lower():
                        shot1, shot2, shot3 = eval
                        shot4 = pre_post[0]
                        cat = "Evaluation"
                    elif "infer" in stage.lower() or "infer" in task.lower() or "infer" in data.lower():
                        shot1, shot2, shot3 = infer
                        shot4 = pre_post[0]
                        cat = "Inference"
                    else:
                        shot1, shot2, shot3 = pre_post_stage
                        shot4 = model[0]
                    new_prompt = cot_few_shot_using_same_or_different_category(prompt, shot1, shot2, shot3, shot4, cat, True )
                if technique == "fewshot-differ-stage-same-repo":
                    shots = 0
                    all_shots = []
                    while shots != 4:
                        try:
                            shot = random.choice(all_files)
                            data_in_file = process_txt(os.path.join(data_path, shot))
                            if "class" in shot:
                                shot_stage = data_in_file["data"]
                            else:
                                shot_stage = data_in_file["stage"]
                            
                            function_shot_path = data_in_file["ground Truth"]
                            repository_shot = data_in_file["repo"]
                            funciton_shot = data_in_file["function"]
                            if "class" in data_in_file.keys():
                                class_shot = data_in_file['class']
                            else:
                                class_shot = None
                            function_shot_path = os.path.join("/local/data0/moved_data/publishablew", repository_shot, repository_shot, function_shot_path)
                            shot_definition = get_function_definition(function_shot_path, funciton_shot, class_shot)
                            if cat not in shot_stage.lower():
                                all_shots.append(shot_definition)
                                shots += 1
                        except:
                            print("Error in making shots")
                            pass
                    new_prompt = few_shot_using_same_or_different_category(prompt, all_shots[0], all_shots[1], all_shots[2], all_shots[3], False)
                
                if technique == "cot-fewshot-differ-stage-same-repo":
                    shot1, shot2, shot3 = pre_post
                    shot4 = model[0]
                    new_prompt = cot_few_shot_using_same_or_different_category(prompt, shot1, shot2, shot3, shot4)
                if technique == "zeroshot-guided":
                    new_prompt += "\n Be aware of bugs like shape mismatch, value errors, or Calculation problems."
                if technique == "self_planning":
                    from prompts.self_planning import self_planning_implementation, self_planning_plan
                    p = self_planning_plan(prompt=prompt)
                    if llm == "openai":
                        plan = get_completion(prompt=p)
                    if llm == "antropic":
                        plan = call_antropic(prompt=p)
                    
                    new_prompt = self_planning_implementation(prompt, plan)
                new_prompt += "\n you need to return code in ```python  ``` format. please do not include any explanation."
                if "guided" in technique:
                    new_prompt += "Be aware of shape mismatch error, value error, index error, and calculation error"
                try:
                    result = process_file_data(new_prompt, llm)
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    import time
                    time.sleep(5)
                    continue
                if  "function" in parsed_data.keys():
                    
                    function_name = parsed_data ["function"]
                elif "f_name" in parsed_data.keys(): 
                    function_name = parsed_data ["f_name"]
                else:
                    function_name = ""
                if "class" in parsed_data.keys():
                    class_name = parsed_data["class"]
                else:
                    class_name = ""
                    
                ground_truth = parsed_data["ground Truth"]
                test = parsed_data["test_cases"]
                output_file_path = os.path.join(output_folder, f"processed_{filename.replace('.txt', '.json')}")
                output_data = {
                    "result": result,
                    "prompt": new_prompt,
                    "function_name": function_name,
                    "ground_truth": ground_truth,
                    "test": test,
                    "class":class_name,
                    "stage":parsed_data["stage"],
                    "task": parsed_data["task"],
                    "data" : parsed_data["data"]
                }
                output_dir = os.path.dirname(output_file_path)
                os.makedirs(output_dir, exist_ok=True)
                try:
                    with open(output_file_path, 'w') as out_f:
                        json.dump(output_data, out_f, indent=4)
                except Exception as e:
                    import time
                    print(f"Error writing to file {output_file_path}: {e}")
                    time.sleep(5)
                    continue
                    
                    output_data["result"] = "error"
                    with open(output_file_path, 'w') as out_f:
                        json.dump(output_data, out_f, indent=4)
                print(f"Processed and saved result for {filename}")
                break

def call():
    llms = [ "openai-4o", "mistral", "llama", "deepseek", "qwen", "o3-mini", "antropic"]
    # llms = ["llama", "deepseek", "qwen"]
    # llm = ["o3-mini"]
    # llms = ["antropic"]
    repos = list(set(["GPflow","ignite", "kornia", "scikit-learn", "imagededup", "neurodiffeq", "pytorch3d", "umap", "vision", "small-text", "inference", "GPflow", "recommenders", "Laplace", "pyro", "pyod", "pfrl", "pennylane", "nncf", "neupy", "litgpt" , "emukit", "DeepReg", "deepchem", "cleanlab", "pytorch-forecasting", "pytorch-widedeep", "avalanche", "nlp-architecht","torchgeo", "lightly", "emukit", "stanza", "pytorch"]))
    # repos2 = ["kornia", "cleanlab", "scikit-learn", "pyod", "pyro", "inference", "recommenders"]
    # repos3 = ["small-text", "imagededup", "emukit", "Laplace", "pfrl", "pennylane", "neupy", "torchgeo", "stanza", "nncf"]
    # repos4 = ["ignite", "umap", "neurodiffeq", "avalanche", "litgpt", "DeepReg"]
    # repos = ["vision", "pytorch3d", "pytorch-forecasting", "pytorch-widedeep", "deepchem", "pytorch"] + repos2 + repos3 + repos4
    versions = [ "v2"]
    # techniques = [  "zeroshot"]
    repos = ["lightly"]
    techniques = [ "zeroshot"]
    # techniques = ["fewshot-outside-benchmark"]
    for version in versions:
        for technique in techniques:
            for llm in llms:
                for repo in repos:
                    input_folder = "../../data/DLEval-20240920T201632Z-001/DLEval"
                    output_folder = f"../../results/llm-output/{technique}/output_{llm}_new/{version}/{repo}"

                    process_folder(input_folder, output_folder, llm , repo, technique)
call()

# print(call_deepseek("how are you today"))

