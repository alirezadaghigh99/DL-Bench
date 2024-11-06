
from radon.complexity import cc_visit
from radon.visitors import ComplexityVisitor
from radon.metrics import h_visit
from radon.metrics import mi_visit
from cognitive_complexity.api import get_cognitive_complexity\

def count_physical_loc(code_string):
   # Split the input string into lines
   lines = code_string.split('\n')
   
   # Filter out empty lines and count the remaining lines
   non_empty_lines = [line for line in lines if line.strip() != '']
   
   return len(non_empty_lines)

def calculate_cyclomatic_complexity(code):
   # Analyze the code
   blocks = cc_visit(code)
   # for block in blocks:
   #     print(f'{block.name}: {block.complexity}')

   # Calculate the average Cyclomatic Complexity
   total_complexity = sum(block.complexity for block in blocks)
   average_complexity = total_complexity / len(blocks) if blocks else 0
   # print(f'Average Cyclomatic Complexity: {average_complexity}')
   return average_complexity

def calculate_halstead_complexity(code):
   results = h_visit(code)
   return results[0].vocabulary

def calculate_mi(code_string):
   mi_score = mi_visit(code_string, True)
   return mi_score

def calculate_cognitive_complexity(code):
   parsed_code = ast.parse(code)
   new_body = [node for node in parsed_code.body if not isinstance(node, ast.Import) and not isinstance(node, ast.ImportFrom) and not isinstance(node, ast.Assign)]
   # print(new_body[0])
   funcdef = new_body[0]
   cc_score = get_cognitive_complexity(funcdef)
   return cc_score