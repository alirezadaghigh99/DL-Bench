import requests
import json
from json_saver import save_dict_to_jsonl
import os
def get_pull_requests(repo, token):
    
    url = f"https://api.github.com/repos/{repo}/pulls"
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    pull_requests = []
    
    # Paginate through all pull requests
    while url:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        
        # Collect all pull request URLs
        for pr in data:
            pull_requests.append(pr['html_url'])
        
        # Check if there is a next page in the pagination
        if 'next' in response.links:
            url = response.links['next']['url']
        else:
            url = None
    
    return pull_requests

import requests

def get_pull_request_details(repo, pull_number, token):
   
    base_url = f"https://api.github.com/repos/{repo}"
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3.diff'
    }
    
    # Fetch the pull request details
    pr_url = f"{base_url}/pulls/{pull_number}"
    pr_response = requests.get(pr_url, headers={'Authorization': f'token {token}', 'Accept': 'application/json'})
    pr_response.raise_for_status()
    pr_data = pr_response.json()

    # Extract pull request message
    pr_message = pr_data.get('body', '')

    # Fetch code diffs
    diffs_response = requests.get(pr_url, headers=headers)
    diffs_response.raise_for_status()
    diffs = diffs_response.text

    # Fetch commits associated with the pull request
    commits_url = pr_data['commits_url']
    commits_response = requests.get(commits_url, headers={'Authorization': f'token {token}', 'Accept': 'application/json'})
    commits_response.raise_for_status()
    commits_data = commits_response.json()
    commits = [commit['html_url'] for commit in commits_data]

    return {
        'message': pr_message,
        'diffs': diffs,
        'commits': commits
    }


import requests

def fetch_branch_data(repo_owner, repo_name, token):
    base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Get all branches excluding the master branch
    branches = requests.get(f"{base_url}/branches", headers=headers)
    branches_data = [branch for branch in branches.json() if branch['name'] != 'master']

    results = {}

    for branch in branches_data:
        branch_name = branch['name']
        print(f"Processing branch: {branch_name}")

        # Get commits for the branch
        commits = requests.get(f"{base_url}/commits?sha={branch_name}", headers=headers).json()

        for commit in commits:
            commit_sha = commit['sha']
            commit_message = commit['commit']['message']

            # Fetch the commit details
            commit_details = requests.get(f"{base_url}/commits/{commit_sha}", headers=headers).json()

            # Extract files changed and comments if available
            files_changed = [file['filename'] for file in commit_details['files']]
            comments_url = commit_details['comments_url']
            comments = requests.get(comments_url, headers=headers).json()

            # Store data
            results[commit_sha] = {
                'branch': branch_name,
                'commit_message': commit_message,
                'files_changed': files_changed,
                'comments': [comment['body'] for comment in comments]
            }

    return results

def get_file_contents(repo, file_path, ref, token):
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}?ref={ref}"
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3.raw'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def get_pr_details_and_files(repo, pull_number, token):

    pr_details_url = f"https://api.github.com/repos/{repo}/pulls/{pull_number}"
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3.json'
    }
    pr_details_response = requests.get(pr_details_url, headers=headers)
    pr_details_response.raise_for_status()
    pr_details = pr_details_response.json()

    base_commit_sha = pr_details['base']['sha']  # Get the base commit SHA
    files_url = pr_details['url'] + '/files'
    files_response = requests.get(files_url, headers=headers)
    files_response.raise_for_status()
    files_data = files_response.json()

    file_contents = {}
    for file in files_data:
        filename = file['filename']
        sha_after = file['sha']
        
        try:
            content_before = get_file_contents(repo, filename, base_commit_sha, token)  # Content from the base commit
        except requests.exceptions.HTTPError:
            content_before = "Content not available"

        try:
            content_after = get_file_contents(repo, filename, sha_after, token)  # Content from the current file commit
        except requests.exceptions.HTTPError:
            content_after = "Content not available"

        file_contents[filename] = (content_before, content_after)

    return file_contents


token = "" 
file_path = "repos_python.jsonl"

with open(file_path, 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    url = result["url"].replace("https://github.com/", "")
    print(url)
    if os.path.exists(f"c/{url.replace('/', '_')}.jsonl"):
        continue
    # try:
    #     data = fetch_branch_data(url.split("/")[0], url.split("/")[1], token)
    #     save_dict_to_jsonl(data, f"branches/{url.replace('/','_')}.jsonl")
        
    # except Exception as e:
    #     print("error", e)
    repo_pull_requests = get_pull_requests(url, token)
    for pr in repo_pull_requests:
        try:
            details = get_pull_request_details(url, pr.split("/")[-1], token)

            print("Commit Links:", details['commits'])
            details["id"] = pr.split("/")[-1]
            path = url.replace("/", "_")
            save_dict_to_jsonl(details, f"c/{path}.jsonl")
        except Exception as e:
            print("error ", e)
            
    



# for pr in repo_pull_requests:
#     print(pr)
# details = get_pull_request_details('udacity/deep-learning-v2-pytorch', "417", token)
# print("Pull Request Message:", details['message'])
# print("Code Diffs:", details['diffs'])
# print("Commit Links:", details['commits'])
