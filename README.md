# AgReason: A Toolkit for Agricultural Reasoning

AgReason is a comprehensive Python-based toolkit designed for generating, evaluating, and refining agricultural domain-specific questions and answers using Large Language Models (LLMs). It also includes capabilities for fine-tuning custom LLMs for enhanced agricultural reasoning.

## Core Components

The toolkit is organized into several key modules:

*   **Query Generation (`query_generation/`)**:
    *   Scripts for generating new agricultural questions.
    *   Involves an LLM-based multi-step process to paraphrase and filter questions for quality, factual correctness, and agronomic relevance using prompts defined in `filteration_prompt.py`.
    *   Leverages asynchronous API calls to services like Together AI.

*   **Response Generation (`response_generation/`, `answer-creation-grok/`)**:
    *   Core module for generating answers/responses to agricultural questions using various LLM providers.
    *   The `response_generation/` scripts handle providers like OpenAI, Together, Claude, and Gemini.
    *   The `answer-creation-grok/` module provides specific tools for generating responses and creating datasets using Grok models, including its vision capabilities (e.g., `grok-2-vision-latest`). This includes scripts for parallel API requests, data cleaning, and dataset construction.
    *   Both support asynchronous batch processing of questions and rate limiting.
    *   Manages API keys through a central `keys.json` file.
    *   Custom system prompts for models (e.g., via `system_prompt.json` in `response_generation/`) can be used.

*   **LLM-as-Judge: Evaluation & Scoring (`llm_judge_agthoughts/`, `llm-as-judge/`)**:
    *   **Automated Evaluation with Detailed Rubrics (`llm_judge_agthoughts/`)**: Implements a sophisticated pipeline for in-depth evaluation of model-generated answers.
        *   Uses LLMs (e.g., GPT-4.1) with detailed, agriculture-specific rubrics and prompts (defined in `llm_judge_prompt_big_ds.py` and `llm_prompts.py`) to assess responses on dimensions like accuracy, relevance, feasibility, consistency, and completeness.
        *   Can integrate with human annotation workflows (e.g., from Label Studio) to leverage human feedback and assist in refining metadata associated with the answers.
    *   **Comparative Scoring & Visualization (`llm-as-judge/`)**: Focuses on comparing candidate LLM responses against expert-evaluated ground truth answers.
        *   Facilitates the generation of scores indicating how well candidate responses align with the ground truth.
        *   Includes or supports capabilities for visualizing these comparison scores to assess model performance. (Note: The `llm-as-judge.ipynb` also contains utilities for processing and preparing datasets for evaluation, which can involve reformatting or light refinement of provided answers to fit specific evaluation templates or models).

*   **Model Distillation & Fine-tuning (`distillation/`)**:
    *   Provides scripts for Supervised Fine-Tuning (SFT) of LLMs using the Hugging Face ecosystem (`transformers`, `peft`, `trl`).
    *   Supports LoRA for parameter-efficient fine-tuning.
    *   Formats custom question-answer datasets for training models capable of agricultural reasoning, potentially including chain-of-thought (indicated by `<think>` tokens).

## Key Technologies

*   **Python 3**: Core programming language.
*   **Asynchronous Programming**: `asyncio` and `aiohttp` for efficient handling of API calls.
*   **LLM APIs**: Integration with various providers:
    *   OpenAI (GPT models)
    *   Together AI
    *   Anthropic (Claude models)
    *   Google (Gemini models)
    *   Grok
*   **Hugging Face Suite**:
    *   `transformers` for model loading and tokenization.
    *   `peft` for Low-Rank Adaptation (LoRA).
    *   `trl` for Supervised Fine-Tuning (SFT).
*   **Pandas**: For data manipulation.
*   **Jupyter Notebooks**: For experimentation and specific workflows like dataset creation and answer refinement.

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd AgReason-main
    ```

2.  **Create a Python Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\\\\Scripts\\\\activate
    ```

3.  **Install Dependencies**:
    *A `requirements.txt` file should be present in the root of `AgReason-main/`. If not, it needs to be generated based on the imports in the project.*
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Keys (`keys.json`)**:
    Most scripts interacting with LLM APIs require API keys. Create a file named `keys.json` in the root of the `AgReason-main/` directory (or in specific subdirectories like `response_generation/` if scripts expect it there – check individual script requirements).
    **This file should be added to your `.gitignore` to avoid committing sensitive keys.**

    Example `keys.json` structure:
    ```json
    {
      "gpt": "YOUR_OPENAI_API_KEY",
      "together": "YOUR_TOGETHER_AI_API_KEY",
      "claude": "YOUR_ANTHROPIC_API_KEY",
      "gemini": "YOUR_GOOGLE_GEMINI_API_KEY",
      "grok": "YOUR_GROK_API_KEY"
    }
    ```
    Fill in the keys for the services you intend to use.

5.  **System Prompts (`system_prompt.json`)**:
    The `response_generation/main.py` script can use model-specific system prompts from a `system_prompt.json` file located in its directory. Create this file if you need custom system prompts.
    Example `system_prompt.json` for `response_generation/`:
    ```json
    {
      "gpt-4o-2024-08-06": "You are a helpful assistant specializing in agriculture.",
      "deepseek-ai/DeepSeek-R1": "You are an expert agronomist providing detailed advice."
    }
    ```

## General Workflow Outline

While specific usage depends on the module, a general workflow could be:

1.  **Generate & Filter Queries**: Use scripts in `query_generation/` to create a high-quality set of agricultural questions.
2.  **Generate Responses**: Use `response_generation/main.py` (for providers like OpenAI, Together, Claude, Gemini) and tools in `answer-creation-grok/` (for Grok models) to get answers for these questions.
3.  **Evaluate & Score Responses**:
    *   Use the tools in `llm_judge_agthoughts/` for detailed, rubric-based evaluation of generated responses.
    *   Utilize `llm-as-judge/` to compare candidate LLM responses against expert-evaluated ground truth, generate scores, and visualize performance.
4.  **Fine-tune Custom Models**: Use the `distillation/distil.py` script with your curated question-answer pairs to fine-tune smaller, specialized LLMs for agricultural tasks.

## Directory Structure

*   `AgReason-main/`
    *   `answer-creation-grok/`: Scripts and notebooks for generating responses and datasets using Grok models (part of Response Generation capabilities).
    *   `distillation/`: Scripts for fine-tuning language models.
    *   `llm-as-judge/`: Notebooks and scripts for comparing LLM responses to ground truth, scoring, visualization, and preparing data for such evaluations.
    *   `llm_judge_agthoughts/`: Python scripts and prompts for detailed LLM-based evaluation of agronomic answers, often involving human annotation data.
    *   `query_generation/`: Scripts and prompts for creating and filtering agricultural questions.
    *   `response_generation/`: Scripts for obtaining responses from various LLM APIs (e.g., OpenAI, Together, Claude, Gemini).
    *   `.gitignore`: Specifies intentionally untracked files (ensure `keys.json` is listed).
    *   `README.md`: This file.
    *   `requirements.txt`

## Contributing

Contributions are welcome. Please follow standard fork, branch, and pull request workflows. For issues or feature requests, please use the GitHub issue tracker.

## Citation
