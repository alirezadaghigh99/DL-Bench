import os

def count_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith('txt0'):
                count += 1
    return count
p = "recommenders"
# Specify the directory you want to search
directory_path_output_gemini = f'/local/data0/moved_data/Organized_benchmark/LLMs/output_geminai'
directory_path_result_gemini = f'/local/data0/moved_data/Organized_benchmark/results/geminai/{p}'
directory_path_output_openai = f'/local/data0/moved_data/Organized_benchmark/LLMs/output_openai/{p}'
directory_path_result_openai = f'/local/data0/moved_data/Organized_benchmark/results/openai/{p}'

# Get the number of files
file_count_output_g = count_files(directory_path_output_gemini)
file_count_result_g = count_files(directory_path_result_gemini)
file_count_output_o = count_files(directory_path_output_openai)
file_count_result_o = count_files(directory_path_result_openai)

print(f"Total number of files (excluding those ending with 'txt0'): {file_count_output_g}")
print(f"Total number of files (excluding those ending with 'txt0'): {file_count_result_g}")
print(f"Total number of files (excluding those ending with 'txt0'): {file_count_output_o}")
print(f"Total number of files (excluding those ending with 'txt0'): {file_count_result_o}")
