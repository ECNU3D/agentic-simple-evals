```
  ___                   _   _        _____ _                 _        _____           _     
 / _ \                 | | (_)      /  ___(_)               | |      |  ___|         | |    
/ /_\ \ __ _  ___ _ __ | |_ _  ___  \ `--. _ _ __ ___  _ __ | | ___  | |____   ____ _| |___ 
|  _  |/ _` |/ _ \ '_ \| __| |/ __|  `--. \ | '_ ` _ \| '_ \| |/ _ \ |  __\ \ / / _` | / __|
| | | | (_| |  __/ | | | |_| | (__  /\__/ / | | | | | | |_) | |  __/ | |___\ V / (_| | \__ \
\_| |_/\__, |\___|_| |_|\__|_|\___| \____/|_|_| |_| |_| .__/|_|\___| \____/ \_/ \__,_|_|___/
        __/ |                                         | |                                   
       |___/                                          |_|                                                                
```

# Overview
This repository contains a lightweight library for evaluating language models.
We are open sourcing it so we can be transparent about the accuracy numbers we're publishing alongside our latest models.

**Note:** This is a fork of the original [OpenAI simple-evals](https://github.com/openai/simple-evals) repository with several enhancements and additions.

## Environment Setup

This project uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:

```bash
# Google Cloud Project ID for Vertex AI models
GOOGLE_CLOUD_PROJECT_ID=your-project-id
```

You can copy the `.env.example` file as a template:

```bash
cp .env.example .env
```

Then edit the `.env` file with your actual values.

## Fork Changes

This fork extends the original simple-evals repository with the following key improvements:

### Additional Model Support
- **Gemini Models**: Added support for Google's Gemini models (`GeminiSampler`) with both API key and Vertex AI authentication, including support for Gemini grounding capabilities
- **Claude on Vertex AI**: Implemented `ClaudeVertexCompletionSampler` for running Claude models through Google Cloud Vertex AI instead of direct Anthropic API

### Windows Compatibility
- **Windows HumanEval Fix**: Added `human_eval_windows_patch.py` to resolve Windows compatibility issues with the HumanEval benchmark by replacing Unix-specific timeout mechanisms with Windows-compatible threading-based solutions

### Infrastructure Improvements
- **Checkpointing System**: Implemented robust checkpointing functionality across all evaluations to support resuming interrupted evaluation runs, with checkpoint loading and saving capabilities
- **Batch Processing**: Added configurable batch processing to improve memory management and allow for better control over evaluation execution
- **Enhanced Error Handling**: Improved exception handling and retry mechanisms for API calls
- **Progress Tracking**: Better progress reporting and logging throughout evaluation processes

### Configuration Enhancements
- **Environment Variable Handling**: Improved API key and authentication management with fallback mechanisms
- **Configurable Parameters**: Enhanced parameterization for batch sizes, timeouts, and other evaluation settings
- **Flexible Authentication**: Support for multiple authentication methods including API keys, Vertex AI, and Application Default Credentials

All original functionality and evaluation benchmarks remain unchanged, ensuring backward compatibility while adding these new capabilities.

## Agentic Evaluation Generation

This fork introduces a powerful **Agentic Evaluation Generation** system that can automatically create custom evaluations for any dataset. The system uses intelligent agents to analyze datasets, understand their structure, and generate appropriate evaluation code.

### Key Features

- **Automatic Dataset Analysis**: Intelligently analyzes dataset structure, column types, and content patterns
- **Automatic Config Discovery**: Attempts to select the most relevant dataset configuration and split when not provided
- **Smart Template Selection**: Automatically selects the most appropriate evaluation template based on task type (multiple choice, math, reading comprehension, code generation, etc.)
- **Custom Prompt Templates**: Full support for custom prompt templates that can be set during generation or at runtime
- **Flexible Column Mapping**: Handles complex datasets with automatic column mapping and multi-column context extraction
- **Code Validation**: Generated code is automatically validated for syntax, imports, and required patterns
- **Multiple Task Types**: Supports MMLU-style, Math, DROP, HumanEval, SimpleQA, and generic evaluation patterns

### Usage

#### Interactive Evaluation Builder

Use the interactive builder to create custom evaluations:

```python
from example_agentic_builder import create_evaluation_interactively

# Create an evaluation interactively
create_evaluation_interactively()
```

This will guide you through:
1. **Dataset Selection**: Choose from Hugging Face datasets
2. **Task Type Configuration**: Specify the evaluation style (multiple choice, math, etc.)
3. **Custom Prompt Templates**: Define how questions should be formatted
4. **Scoring Methods**: Choose appropriate scoring strategies
5. **Code Generation**: Automatically generate the complete evaluation class

#### Programmatic Generation

You can also generate evaluations programmatically:

```python
from agents.code_generator import CodeGenerator
from data_models import EvalConfig, DatasetAnalysis
from sampler.chat_completion_sampler import ChatCompletionSampler

# Configure the evaluation
config = EvalConfig(
    task_type="factual_qa",
    scoring_method="exact_match",
    prompt_template="Answer the question concisely.\nQuestion: {question}\nAnswer:"
)

# Generate the evaluation code
sampler = ChatCompletionSampler(model="gpt-4")
generator = CodeGenerator(sampler)
eval_code = generator.generate_eval_class(analysis, config)
```

### Supported Task Types

- **Multiple Choice** (`multiple_choice_style`): MMLU-style evaluations with choice extraction
- **Math** (`math_style`): Mathematical problem solving with numerical scoring
- **Reading Comprehension** (`reading_comprehension_style`): DROP-style context-based Q&A
- **Code Generation** (`code_generation_style`): HumanEval-style programming tasks
- **Factual Q&A** (`factual_qa_style`): SimpleQA-style factual questions
- **Text Generation** (`text_generation_style`): Open-ended text generation tasks
- **Classification** (`classification_style`): Category classification tasks
- **Translation** (`text_generation_style`): Translate text from one language to another
- **Summarization** (`text_generation_style`): Summarize text passages

### Custom Prompt Templates

The system fully supports custom prompt templates with variable substitution:

```python
# Custom prompt with context
custom_prompt = """
Based on the following context, answer the question.

Context: {context}
Question: {question}

Provide a concise answer:
"""

# The system will automatically handle variable substitution from dataset columns
```

### Generated Code Structure

All generated evaluations follow a consistent structure:
- Inherit from `Eval` and `EvaluationMixin` base classes
- Support configurable batch processing and checkpointing
- Include proper error handling and validation
- Use the scoring strategies from `eval_utils.ScoringStrategy`
- Support runtime prompt template customization

### Example Generated Evaluation

The system generates complete, runnable evaluation classes like:

```python
class CustomDatasetEval(Eval, EvaluationMixin):
    def __init__(self, dataset_name: str = "custom/dataset",
                 num_examples: Optional[int] = None,
                 batch_size: int = 20,
                 checkpoint_file: Optional[str] = None,
                 prompt_template: Optional[str] = None):
        # Full initialization with dataset loading
        
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        # Complete evaluation logic with batch processing
```

This system makes it easy to create evaluations for new datasets without manual template editing or code writing.

## Benchmark Results

| Model                        | Prompt        | MMLU   | GPQA [^8]   | MATH [^6]| HumanEval | MGSM[^5] | DROP[^5]<br>(F1, 3-shot) | SimpleQA 
|:----------------------------:|:-------------:|:------:|:------:|:--------:|:---------:|:------:|:--------------------------:|:---------:| 
| **o3**                         |               |        |        |          |           |        |                             |                      |           |
| o3-high [^10]                | n/a [^7]      |  93.3  |  83.4  |   98.1   |  88.4     |  92.0  |  89.8                      |  48.6     |
| o3 [^9] [^10]                | n/a           |  92.9  |  82.8  |   97.8   |  87.4     |  92.3  |  80.6                      |  49.4     |
| o3-low [^10]                 | n/a           |  92.8  |  78.6  |   96.9   |  87.3     |  91.9  |  82.3                      |  49.4     |
| **o4-mini**                    |               |        |        |          |           |        |                             |                      | 
| o4-mini-high [^9] [^10]      | n/a           |  90.3  |  81.3  |   98.2   |  99.3     |  93.5  |  78.1                      |  19.3     |
| o4-mini [^9] [^10]           | n/a           |  90.0  |  77.6  |   97.5   |  97.3     |  93.7  |  77.7                      |  20.2     |
| o4-mini-low [^10]            | n/a           |  89.5  |  73.6  |   96.2   |  95.9     |  93.0  |  76.0                      |  20.2     |
| **o3-mini**                    |               |        |        |          |           |        |                             |                      |           |
| o3-mini-high                 | n/a           |  86.9  |  77.2  |   97.9   |  97.6     |  92.0  |  80.6                      |  13.8     |
| o3-mini                      | n/a           |  85.9  |  74.9  |   97.3   |  96.3     |  90.8  |  79.2                      |  13.4     |
| o3-mini-low                  | n/a           |  84.9  |  67.6  |   95.8   |  94.5     |  89.4  |  77.6                      |  13.0     | 
| **o1**                         |               |        |        |          |           |        |                             |                      |
|  o1                          | n/a           |  91.8  |  75.7  |   96.4   |    -      |  89.3  |  90.2                      |  42.6     |
| o1-preview                   | n/a           |  90.8  |  73.3  |   85.5   |  92.4     |  90.8  |  74.8                      |  42.4     | 
| o1-mini                      | n/a           |  85.2  |  60.0  |   90.0   |  92.4     |  89.9  |  83.9                      |  07.6     |  
| **GPT-4.1**                            |               |        |        |          |           |        |                             |                      |           |
| gpt-4.1-2025-04-14           | assistant [^2]|  90.2  |  66.3  |   82.1   |   94.5    |  86.9  |  79.4                      | 41.6      |
| gpt-4.1-mini-2025-04-14      | assistant     |  87.5  |  65.0  |   81.4   |   93.8    |  88.2  |  81.0                      | 16.8      |
| gpt-4.1-nano-2025-04-14      | assistant     |  80.1  |  50.3  |   62.3   |   87.0    |  73.0  |  82.2                      | 07.6      |
| **GPT-4o**                     |               |        |        |          |           |        |                             |                      |           |
| gpt-4o-2024-11-20            | assistant     |  85.7  |  46.0  |   68.5   |   90.2    |  90.3  |  81.5                      | 38.8      |  
| gpt-4o-2024-08-06            | assistant     |  88.7  |  53.1  |   75.9   |   90.2    |  90.0  |  79.8                      | 40.1      |  
| gpt-4o-2024-05-13            | assistant     |  87.2  |  49.9  |   76.6   |   91.0    |  89.9  |  83.7                      | 39.0      |
| gpt-4o-mini-2024-07-18       | assistant     |  82.0  |  40.2  |   70.2   |   87.2    |  87.0  |  79.7                      | 09.5      | 
| **GPT-4.5-preview**          |               |        |        |          |           |        |                            |           |
| gpt-4.5-preview-2025-02-27   | assistant     |  90.8  |  69.5  |   87.1   |   88.6    |  86.9  |  83.4                      | 62.5      |
| **GPT-4 Turbo and GPT-4**    |               |        |        |          |           |        |                            |           |
| gpt-4-turbo-2024-04-09       | assistant     |  86.7  |  49.3  |   73.4   |   88.2    |  89.6  |  86.0                      | 24.2      |
| gpt-4-0125-preview           | assistant     |  85.4  |  41.4  |   64.5   |   86.6    |  85.1  |  81.5                      | n/a       |
| gpt-4-1106-preview           | assistant     |  84.7  |  42.5  |   64.3   |   83.7    |  87.1  |  83.2                      | n/a       |
| **Other Models (Reported)**   |               |        |        |        |           |        |                           |
| [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) | unknown |  88.3  |  59.4  |  71.1  |   92.0    | 91.6 | 87.1 |  28.9 | 
| [Claude 3 Opus](https://www.anthropic.com/news/claude-3-family) | unknown |  86.8  |  50.4  |  60.1  |   84.9    |   90.7   |  83.1 |  23.5 |                   
| [Llama 3.1 405b](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) | unknown |  88.6  |  50.7  |  73.8  |   89.0    | 91.6 |  84.8                   | n/a 
| [Llama 3.1 70b](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) | unknown |  82.0  |  41.7  |  68.0  |   80.5    |  86.9  |  79.6                   | n/a 
| [Llama 3.1 8b](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) | unknown |  68.4  |  30.4  |  51.9  |   72.6    |  68.9  |  59.5                   | n/a 
| [Grok 2](https://x.ai/blog/grok-2) | unknown | 87.5 | 56.0 | 76.1 | 88.4 | n/a | n/a | n/a 
| [Grok 2 mini](https://x.ai/blog/grok-2) | unknown | 86.2 | 51.0 | 73.0 | 85.7 | n/a | n/a | n/a 
| [Gemini 1.0 Ultra](https://goo.gle/GeminiV1-5) | unknown | 83.7 | n/a | 53.2 | 74.4 | 79.0 | 82.4 | n/a 
| [Gemini 1.5 Pro](https://goo.gle/GeminiV1-5) | unknown | 81.9 | n/a | 58.5 | 71.9 | 88.7 | 78.9 | n/a 
| [Gemini 1.5 Flash](https://goo.gle/GeminiV1-5) | unknown | 77.9 | 38.6 | 40.9 | 71.5 | 75.5 | 78.4 | n/a 

## Background

Evals are sensitive to prompting, and there's significant variation in the formulations used in recent publications and libraries.
Some use few-shot prompts or role playing prompts ("You are an expert software programmer...").
These approaches are carryovers from evaluating *base models* (rather than instruction/chat-tuned models) and from models that were worse at following instructions.

For this library, we are emphasizing the *zero-shot, chain-of-thought* setting, with simple instructions like "Solve the following multiple choice problem". We believe that this prompting technique is a better reflection of the models' performance in realistic usage.

**We will not be actively maintaining this repository and monitoring PRs and Issues.** In particular, we're not accepting new evals. Here are the changes we might accept.
- Bug fixes (hopefully not needed!)
- Adding adapters for new models
- Adding new rows to the table below with eval results, given new models and new system prompts.

This repository is NOT intended as a replacement for https://github.com/openai/evals, which is designed to be a comprehensive collection of a large number of evals.

## Evals

This repository currently contains the following evals:

- MMLU: Measuring Massive Multitask Language Understanding, reference: https://arxiv.org/abs/2009.03300, https://github.com/hendrycks/test, [MIT License](https://github.com/hendrycks/test/blob/master/LICENSE)
- MATH: Measuring Mathematical Problem Solving With the MATH Dataset, reference: https://arxiv.org/abs/2103.03874, https://github.com/hendrycks/math, [MIT License](https://github.com/idavidrein/gpqa/blob/main/LICENSE)
- GPQA: A Graduate-Level Google-Proof Q&A Benchmark, reference: https://arxiv.org/abs/2311.12022, https://github.com/idavidrein/gpqa/,  [MIT License](https://github.com/idavidrein/gpqa/blob/main/LICENSE)
- DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs, reference: https://arxiv.org/abs/1903.00161, https://allenai.org/data/drop, [Apache License 2.0](https://github.com/allenai/allennlp-models/blob/main/LICENSE)
- MGSM: Multilingual Grade School Math Benchmark (MGSM), Language Models are Multilingual Chain-of-Thought Reasoners, reference: https://arxiv.org/abs/2210.03057, https://github.com/google-research/url-nlp, [Creative Commons Attribution 4.0 International Public License (CC-BY)](https://github.com/google-research/url-nlp/blob/main/LICENSE)
- HumanEval: Evaluating Large Language Models Trained on Code, reference https://arxiv.org/abs/2107.03374, https://github.com/openai/human-eval, [MIT License](https://github.com/openai/human-eval/blob/master/LICENSE)
- SimpleQA: Measuring short-form factuality in large language models, reference: https://openai.com/index/introducing-simpleqa, [MIT License](https://github.com/openai/simple-evals/blob/main/LICENSE)
- BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents, reference: https://openai.com/index/browsecomp, [MIT License](https://github.com/openai/simple-evals/blob/main/LICENSE)

## Samplers

We have implemented sampling interfaces for the following language model APIs:

- OpenAI: https://platform.openai.com/docs/overview
- Claude: https://www.anthropic.com/api

Make sure to set the `*_API_KEY` environment variables before using these APIs.

## Setup

Due to the optional dependencies, we're not providing a unified setup mechanism. Instead, we're providing instructions for each eval and sampler.

For [HumanEval](https://github.com/openai/human-eval/) (python programming)
```bash
git clone https://github.com/openai/human-eval
pip install -e human-eval
```

For the [OpenAI API](https://pypi.org/project/openai/):
```bash
pip install openai
```

For the [Anthropic API](https://docs.anthropic.com/claude/docs/quickstart-guide):
```bash
pip install anthropic
```

## Running the evals
```bash
python -m simple-evals.simple_evals --list-models
```
This will list all the models that you can evaluate.

To run the evaluations, you can use the following command:
```bash
python -m simple-evals.simple_evals --model <model_name> --examples <num_examples>
```
This will launch evaluations through the OpenAI API.

## Notes

[^1]:chatgpt system message: "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
[^2]:assistant system message in [OpenAI API doc](https://platform.openai.com/docs/api-reference/introduction): "You are a helpful assistant." .
[^3]:claude-3 empty system message: suggested by Anthropic API doc, and we have done limited experiments due to [rate limit](https://docs.anthropic.com/claude/reference/rate-limits) issues, but we welcome PRs with alternative choices.
[^4]:claude-3 lmsys system message: system message in LMSYS [Fast-chat open source code](https://github.com/lm-sys/FastChat/blob/7899355ebe32117fdae83985cf8ee476d2f4243f/fastchat/conversation.py#L894): "The assistant is Claude, created by Anthropic. The current date is {{currentDateTime}}. Claude's knowledge base was last updated ... ". We have done limited experiments due to [rate limit](https://docs.anthropic.com/claude/reference/rate-limits) issues, but we welcome PRs with alternative choices.
[^5]:We believe these evals are saturated for our newer models, but are reporting them for completeness.
[^6]:For newer models (anything on or after o1) we evaluate on [MATH-500](https://github.com/openai/prm800k/tree/main/prm800k/math_splits), which is a newer, IID version of MATH.
[^7]:o-series models do not support using a system prompt.
[^8]:Includes an answer regex tweak for GPQA benchmark.
[^9]:The default reasoning level for o3-mini is "medium".
[^10]:These results are with no tools enabled for o3 or o4-mini

## Legal Stuff
By contributing to evals, you are agreeing to make your evaluation logic and data under the same MIT license as this repository. You must have adequate rights to upload any data used in an eval. OpenAI reserves the right to use this data in future service improvements to our product. Contributions to OpenAI evals will be subject to our usual Usage Policies: https://platform.openai.com/docs/usage-policies.
