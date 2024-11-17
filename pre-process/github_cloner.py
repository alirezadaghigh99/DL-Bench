import requests
import os
import re
import json
from tqdm import tqdm

v = "4" 
repo_file_base = 'repo_test_v'+ v
test_files_base = 'test_files_v'+ v
def get_repo_size(repo_url, token):
    parts = repo_url.split('/')
    owner = parts[-2]
    repo = parts[-1]
    
    # Headers to use the personal access token for authentication
    headers = {'Authorization': f'token {token}'}

    response = requests.get(f'https://api.github.com/repos/{owner}/{repo}', headers=headers)
    if response.status_code == 200:
        repo_data = response.json()
        return repo_data['size']
    else:
        print("Failed to retrieve repository data:", response.json())
        return -1  # Return -1 if there's an issue fetching data
    

def clone_and_extract_tests(repo, token):
    # Get the repository size
    repo_size = get_repo_size(repo, token)
    print(f"Repository size: {repo_size} KB")

    size_limit = 9500000  # example limit: 100 MB in KB

    if repo_size > size_limit or repo_size == -1:
        print(f"{repo} is too large to clone (limit: {size_limit} KB).")
        print (repo)
        return []

    # Clone the repository locally
    repo_path = repo.split('/')[-1].replace('.git', '')
    print(repo_path)
    os.system(f"mkdir {repo_file_base}/{repo_path}")
    os.system(f'git clone {repo} {repo_file_base}/{repo_path}')
    
    # Path where repo is cloned

    # Regex to find typical unit test files
    test_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if re.search(r'test.*\.py$', file):
                test_files.append(os.path.join(root, file))
    
    return test_files



file_path = 'repo_test_v3.jsonl'
with open(file_path, 'r') as json_file:
    json_list = list(json_file)

i = 0
count = 0
token = 'c'
print(len(json_list))
for js in tqdm(json_list):
    data = json.loads(js)
    i += 1

    if os.path.exists(f"{test_files_base}/{data['url'].split('/')[-1]}.text"):
        continue
    data_to_write = str(clone_and_extract_tests(data['url'], token))
    if data_to_write == []:
        continue
    with open(f"{test_files_base}/{data['url'].split('/')[-1]}.text", "w") as f:
        f.write(data_to_write)
        count += 1
    if count == 1200:
        break
    
print(os.listdir(f"{repo_file_base}").__len__())
    

