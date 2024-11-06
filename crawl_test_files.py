import os
import re

# Define the directory containing all repositories
base_dir = "repo_test"

# Define the list of test libraries to search for
test_libraries = ["pytest", "unittest", "nose", "doctest"]

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

def save_test_files(repo_dir, test_files):
    """Save the list of test files to a text file."""
    repo_name = os.path.basename(repo_dir)
    output_file = os.path.join("all_test_files", f"{repo_name}_test_files.txt")
    with open(output_file, "w", encoding="utf-8") as file:
        for test_file in test_files:
            file.write(f"{test_file}\n")

def main():
    """Main function to process all repositories in the base directory."""
    for repo_name in os.listdir(base_dir):
        repo_dir = os.path.join(base_dir, repo_name)
        if os.path.isdir(repo_dir):
            test_files = find_test_files(repo_dir)
            save_test_files(repo_dir, test_files)

if __name__ == "__main__":
    main()
