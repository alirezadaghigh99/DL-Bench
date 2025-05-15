# DL-Bench: Deep Learning Benchmark Framework

<div align="center">
  <img src="https://img.shields.io/badge/DL--Bench-Deep%20Learning%20Benchmark-blue?style=for-the-badge" alt="DL-Bench"/>
  <br>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/PyTorch-Supported-orange?style=flat-square&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/TensorFlow-Supported-orange?style=flat-square&logo=tensorflow" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/LLMs-Multiple-green?style=flat-square" alt="Multiple LLMs"/>
</div>

<p align="center">A comprehensive framework for benchmarking Large Language Models on real-world deep learning tasks</p>

---

## 📋 Overview

DL-Bench is a comprehensive evaluation framework designed to benchmark Large Language Models (LLMs) on practical deep learning tasks. Unlike other coding benchmarks, DL-Bench focuses specifically on the deep learning domain, covering the entire ML pipeline from data preprocessing to model deployment.

The benchmark consists of real-world deep learning problems extracted from popular frameworks and libraries, providing a realistic assessment of LLMs' capabilities in addressing deep learning challenges.

## 🏆 Key Features

- **End-to-End Pipeline Coverage**: Tasks spanning preprocessing, model construction, training, evaluation, and inference
- **Multiple LLM Support**: Integration with OpenAI, Anthropic, Mistral, Llama, DeepSeek, Qwen, and more
- **Advanced Prompting Techniques**: Zero-shot, few-shot, chain-of-thought, and self-planning approaches
- **Comprehensive Testing Framework**: Automated evaluation against real-world test cases
- **Cross-Repository Compatibility**: Works with 30+ machine learning repositories
- **Fine-grained Categorization**: ML stage-specific evaluation and comparison

## 📁 Project Structure

### 📂 `/data`
- **DL-Bench-Enriched-Processed-Sorted.csv** - The main dataset containing deep learning tasks, bugs, and solutions, carefully curated and categorized.

### 📂 `/src`

#### 📂 `/LLM`
- **call.py** - Unified interface for multiple LLM providers:
  - OpenAI (GPT-4o, o3-mini)
  - Anthropic (Claude 3.5 Sonnet via AWS Bedrock)
  - Mistral (Mixtral-8x22b)
  - Llama (v3-70B)
  - DeepSeek-v3
  - Qwen
  - Gemini (support included)
- **function_extractor.py** - Extracts function definitions from source code files.

#### 📂 `/bug taxonomy`
- Classification framework for deep learning bugs.

#### 📂 `/category_training`
- **category_training.py** - BERT-based classifier for categorizing tasks into different ML pipeline stages.
- **ds1000_training_result.csv** - Results from training on the DS1000 dataset.

#### 📂 `/prompts`
Implements advanced prompting strategies:

- **few_shot.py** - Few-shot prompting with various selection strategies.
- **few_shot_cot.py** - Chain-of-thought with few-shot examples.
- **self_planning.py** - Two-step approach where LLM plans before implementation.
- **shots_for_cot.py** - Example shots for chain-of-thought prompting.
- **shots_for_few.py** - Example shots for few-shot prompting.
- **techniques.py** - Common prompting technique utilities.
- **zero_shot_cot.py** - Zero-shot chain-of-thought implementation.

#### 📂 `/test_removal`
- **remving.py** - Removes incorrect or problematic test cases from the benchmark.

#### 📂 `/tester`
- **parse_run_combined.py** - **Main entry point** for the benchmark evaluation:
  - Processes both function and class implementations
  - Extracts code from LLM responses
  - Routes to appropriate test runners
  - Records results with comprehensive metadata
- **parse_run_class.py** - Runner specialized for class-based tasks.
- **parse_run_function.py** - Runner specialized for function-based tasks.
- **replacer_class.py** & **replacer_function.py** - Utilities for replacing implementations.
- **test_runner_class.py** & **test_runner_function.py** - Execute tests on LLM-generated code.
- **anaylytic.py** - Analysis utilities for benchmark results.
- **miss.py** - Handling for missing or failed test cases.
- **train_test_splitting.py** - Utilities for dataset splitting.

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Benchmark
```bash
# Main benchmark execution
python src/tester/parse_run_combined.py

# For specific LLM and prompting technique
python src/LLM/call.py
```

## 🧠 Prompting Techniques

DL-Bench implements numerous prompting strategies to evaluate different interaction methods:

### Zero-shot Approaches
- **Standard Zero-shot**: Direct task prompting without examples
- **Zero-shot CoT**: Adds "Let's generate the code step by step" to encourage reasoning

### Few-shot Approaches
- **Standard Few-shot**: Includes 4 examples before the main prompt
- **Category-based Few-shot**: Examples from same ML pipeline stage
- **Cross-category Few-shot**: Examples from different ML stages
- **External Few-shot**: Examples from outside the benchmark
- **Repository-based Few-shot**: Examples from same/different repositories

### Advanced Techniques
- **Chain-of-Thought**: Multiple variations encouraging step-by-step reasoning
- **Self-planning**: Two-stage process with planning then implementation
- **Guided Approaches**: Includes specific warnings about common bugs
- **Classifier-based**: Uses trained classifier to determine optimal examples

## 📊 Benchmark Coverage

DL-Bench evaluates performance across the entire ML pipeline:

| Stage | Description | Examples |
|-------|-------------|----------|
| **Pre/Post-Processing** | Data loading, transformation, augmentation | Image resizing, normalization, data formatting |
| **Model Construction** | Architecture design, layer configuration | Creating network layers, defining model architecture |
| **Training** | Loss functions, optimizers, training loops | Custom loss functions, training procedures |
| **Evaluation** | Metrics, validation procedures | Accuracy calculation, evaluation metrics |
| **Inference** | Prediction, deployment | Model prediction, inference optimization |

## 🧪 Test Execution Process

1. **Input Processing**: Parse JSON files containing problem definitions
2. **LLM Querying**: Generate solution using specified LLM and prompting technique
3. **Code Extraction**: Extract Python code from LLM response
4. **Test Execution**: Run extracted code against test cases
5. **Result Collection**: Record success/failure and metadata

## 📈 Example Results

The benchmark produces detailed results showing how different LLMs and prompting techniques perform across ML stages:

```
# Sample output format
{
  "test_result": "PASS",  // or error code
  "file_path": "example_file.json",
  "stage": "model_construction",
  "task": "create_convolution_layer",
  "data": "image"
}
```

## 💻 Example Usage

### Testing with Different LLMs
```python
from src.LLM.call import process_file_data

# Test with different LLMs
result_openai = process_file_data(prompt, llm="openai-4o")
result_claude = process_file_data(prompt, llm="antropic")
result_llama = process_file_data(prompt, llm="llama")
```

### Using Different Prompting Techniques
```python
from src.prompts.few_shot import few_shot_using_same_or_different_category
from src.prompts.zero_shot_cot import ZeroShotCoT

# Create few-shot prompt
few_shot_prompt = few_shot_using_same_or_different_category(
    prompt, example1, example2, example3, example4, category="train", is_same=True
)

# Create zero-shot CoT prompt
zero_shot_cot = ZeroShotCoT("dlbench", "4o", "zero_cot")
cot_prompt = zero_shot_cot.form_technique_prompt(prompt)
```

### Running Tests for a Specific Repository
```python
from src.tester.parse_run_combined import process_function_file

# Process function files for PyTorch repository
process_function_file(
    file_path="path/to/test.json",
    folder_path="results/output",
    BASE_PATH="/path/to/base",
    r="pytorch",
    llm="openai-4o_new",
    technique="zeroshot",
    version="v2"
)
```

## 🧠 Category Classification

DL-Bench includes a BERT-based classifier for categorizing deep learning tasks:

- Fine-tuned on deep learning function descriptions
- Categorizes tasks into ML pipeline stages
- Used to select relevant few-shot examples
- Improves performance by matching examples with similar tasks

## 🔍 Supported Repositories

DL-Bench works with 30+ machine learning repositories, including:

- PyTorch
- PyTorch3D
- scikit-learn
- Kornia
- neurodiffeq
- GPflow
- DeepReg
- deepchem
- And many more...

## 👥 Contributors

- Project maintained by the DL-Bench team

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">DL-Bench: Advancing LLM evaluation in deep learning</p>