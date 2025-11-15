# Copilot Instructions

## Architecture Overview

This repository contains `playscenario-prompt-lib`, a Python library for managing AI-driven prompt logic. The architecture separates the core prompt generation from the evaluation harness.

-   **`prompts/`**: This is the core, production-grade Python package. It is designed to be "dumb" with minimal dependencies (`pydantic`, `jinja2`). It builds prompts and defines data schemas.
-   **`evaluations/`**: This is a developer-only evaluation harness that consumes the `prompts/` package to test it against live AI models.

The main design principle is that the `prompts` package should be completely independent of the evaluation code and should not have any dependencies on AI models or APIs.

## Developer Workflow

### Setting up the Environment

To set up the development environment, create a virtual environment, install the dependencies from `requirements.txt` and `dev-requirements.txt`, and then install the library in editable mode:

```bash
python -m pip install -e .
```

### Running Evaluations

The primary workflow is running evaluations using `evaluations/evaluate.py`. This script runs test cases defined in `evaluations/test_cases/` against live AI models.

-   **Run a single test case:**
    ```bash
    python evaluations/evaluate.py evaluations/test_cases/create_grumpy_pirate.yaml
    ```
-   **Run all test cases:**
    ```bash
    python evaluations/evaluate.py all
    ```

You can add the `--report` flag to generate a markdown report of the evaluation.

## Key Files and Directories

-   **`prompts/agents/`**: Contains the logic for different "agents". Each agent has a `prompt_factory.py` that assembles prompts from `jinja2` templates.
-   **`prompts/schemas.py`**: Defines the Pydantic data models used for prompt inputs and outputs.
-   **`evaluations/evaluate.py`**: The main entry point for running evaluations.
-   **`evaluations/test_cases/`**: Contains YAML files that define the test cases for the evaluations.
-   **`config/agents.yaml` and `config/models.yaml`**: Configuration files for the agents and the AI models they use.

## Conventions

-   **Prompt Factories**: Each agent in `prompts/agents/` has a `prompt_factory.py`. This file is responsible for creating the prompt for that agent, often using `jinja2` templates.
-   **Jinja2 Templates**: Prompt templates are written using `jinja2` and have a `.j2` extension. They are located in the agent's directory.
-   **Pydantic Schemas**: All data structures are defined as Pydantic models in `prompts/schemas.py`. This allows for strong typing and validation.
-   **YAML Test Cases**: Test cases for evaluations are defined in YAML files in `evaluations/test_cases/`. These files specify the agent to test, the inputs to use, and the expected outputs.
