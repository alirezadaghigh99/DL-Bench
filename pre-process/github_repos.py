import requests
import time
import requests
from json_saver import save_dict_to_jsonl
def search_github_repos(query, language="Jupyter Notebook", min_stars=5, sort="updated", order="desc", max_pages=40, token=None):
    """Search GitHub repositories by query and paginate through results, filtering by minimum stars."""
    url = "https://api.github.com/search/repositories"
    all_repos = []
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    for page in range(1, max_pages):
        time.sleep(1)  # Delay to avoid hitting API rate limit
        # Include stars qualifier in the query
        params = {
            'q': f'{query} language:"{language}" stars:>={min_stars}',
            'sort': sort,
            'order': order,
            'per_page': 10,
            'page': page
        }
        response = requests.get(url, headers=headers, params=params)
        try:
            response.raise_for_status()  # Raises stored HTTPError, if one occurred
            data = response.json()
            all_repos.extend(data['items'])  # Add fetched repos to the list
            if 'next' not in response.links:
                break  # Stop if there are no more pages to fetch
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")  # Print HTTP error message
            break
        except requests.exceptions.RequestException as e:
            print(f"Other error occurred: {e}")  # Print other types of exceptions
            break
    print(all_repos)
    return all_repos

def main():
    deep_learning_tags = "deep-learning"  # Example tag
    deep_learning_tags = [
    "deep-learning",
    "machine-learning",
    "neural-network",
    "tensorflow",
    "keras",
    "pytorch",
    "nlp",
    "computer-vision",
    "data-science",
    "convolutional-neural-networks",
    "natural-language-processing",
    "ai",
    "artificial-intelligence",
    "reinforcement-learning",
    "lstm",
    "image-processing",
    "object-detection",
    "neural-networks",
    "gpu",
    "scikit-learn",
    "transfer-learning",
    "openai-gym",
    "generative-adversarial-network",
    "autoencoders",
    "yolo",
    "image-recognition",
    "speech-recognition",
    "face-recognition",
    "deep-reinforcement-learning",
    "tensorflow2"
]
#     deep_learning_tags = [
#     "deep-learning",
#     "machine-learning",
#     "neural-network",
#     "tensorflow",
#     "keras",
#     "pytorch",
#     "nlp",
#     "computer-vision",
#     "data-science",
#     "convolutional-neural-networks",
#     "natural-language-processing",
#     "ai",
#     "artificial-intelligence",
#     "reinforcement-learning",
#     "lstm",
#     "image-processing",
#     "object-detection",
#     "neural-networks",
#     "gpu",
#     "scikit-learn",
#     "transfer-learning",
#     "generative-adversarial-network",
#     "autoencoders",
#     "yolo",

#     "face-recognition",
#     "deep-reinforcement-learning",
#     "tensorflow2"
# ]
#     deep_learning_tags = [
#     "machine-learning",
#     "neural-network",
#     "computer-vision",
#     "ai",
#     "object-detection",
#     "autoencoders",
#     "deep-reinforcement-learning",
# ]
    token = ''  # Replace with your actual token
    count = 0
    for tag in deep_learning_tags:
        repos = search_github_repos(tag, max_pages=60, token=token, language="python", min_stars=400)
        time.sleep(20)  # Set the number of pages to fetch

        for repo in repos:
            data = {
                "repo_name": repo['name'], 
                'description': repo['description'],
                'url': repo['html_url'],
                'updated_at': repo['updated_at'],
                'stars': repo['stargazers_count']
            }
            save_dict_to_jsonl(data, "repos_python_test.jsonl")
            
            
        print(count)
if __name__ == '__main__':
    main()
