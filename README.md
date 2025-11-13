# PlayScenario Prompt Library (`playscenario-prompt-lib`)

## Overview

This repository contains `playscenario-prompt-lib`, a standalone, testable, and production-grade Python library for managing all AI-driven prompt logic for the PlayScenarioAI platform.

The core architecture is based on a strict separation of concerns:
- **`prompts/`**: A "dumb" core Python package that only builds prompts and defines data schemas, with minimal dependencies (`pydantic`, `jinja2`).
- **`evaluations/`**: A "smart" developer-only evaluation harness that consumes the `prompts/` package to test it against live AI models.

## Project Setup

To set up the development environment, follow these steps:

1.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r dev-requirements.txt
    ```

3.  **Install the Library in Editable Mode:**
    This step makes the `prompts` package importable by the evaluation harness.
    ```bash
    pip install -e .
    ```

4.  **Set Up Environment Variables:**
    Copy the `.env.example` file to `.env` and add your API keys.
    ```bash
    cp .env.example .env
    ```

## Running Evaluations

The evaluation harness (`evaluations/evaluate.py`) is a command-line tool used to run test cases against live AI models.

**Usage:**
```bash
python3 evaluations/evaluate.py <path_to_test_case.yaml>
```

**Example:**
```bash
python3 evaluations/evaluate.py evaluations/test_cases/create_grumpy_pirate.yaml
```
