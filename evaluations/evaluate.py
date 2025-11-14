import typer
import yaml
import os
import json
import re
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from datetime import datetime

# Import SDKs
import google.generativeai as genai
from openai import OpenAI

# --- LLM Client Abstraction ---

class BaseLlmClient(ABC):
    """Abstract base class for LLM API clients."""
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a response from the LLM.

        Args:
            system_prompt: The system prompt.
            user_prompt: The user prompt.

        Returns:
            The LLM's response as a string.
        """
        pass

class GoogleClient(BaseLlmClient):
    """Client for Google's Generative AI models."""
    def __init__(self, api_key: str, model_name: str, generation_config: Dict[str, Any]):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.generation_config = generation_config

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            system_instruction=system_prompt
        )
        response = model.generate_content(user_prompt)
        return response.text

class OpenAICompatibleClient(BaseLlmClient):
    """Client for OpenAI and other compatible APIs."""
    def __init__(self, api_key: str, model_name: str, generation_config: Dict[str, Any], base_url: Optional[str] = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.generation_config = generation_config

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.generation_config
        )
        return response.choices[0].message.content

# --- Factory to Get the Right Client ---

class LlmClientFactory:
    """Factory to instantiate the correct LLM client based on configuration."""
    def __init__(self, models_config: Dict[str, Any]):
        self.models_config = models_config

    def get_client(self, model_key: str) -> BaseLlmClient:
        model_info = self.models_config.get(model_key)
        if not model_info:
            raise ValueError(f"Model '{model_key}' not found in config.")

        provider = model_info.get("provider")
        api_key_env = model_info.get("api_key_env")
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(f"API key env var '{api_key_env}' not set.")

        model_name = model_info.get("model_name")
        generation_config = model_info.get("generation_config", {})

        if provider == "google":
            return GoogleClient(api_key, model_name, generation_config)
        elif provider in ["openai", "mistral", "cerebras"]:
            base_url = model_info.get("base_url")
            return OpenAICompatibleClient(api_key, model_name, generation_config, base_url)
        else:
            raise NotImplementedError(f"Provider '{provider}' is not supported.")

def strip_markdown(text: str) -> str:
    """Strips markdown code blocks from a string."""
    return re.sub(r"```json\n(.*?)\n```", r"\1", text, flags=re.DOTALL)


def generate_report(
    test_case_name: str,
    model_id: str,
    llm_output: str,
    assertion_results: list[str]
) -> str:
    """Generates a markdown report of the evaluation."""
    report = f"# Evaluation Report for {test_case_name}\n\n"
    report += f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**Model ID**: {model_id}\n\n"

    report += "## LLM Output\n"
    report += "```json\n"
    report += llm_output
    report += "\n```\n\n"

    report += "## Assertions\n"
    for result in assertion_results:
        report += f"- {result}\n"

    return report


def run_evaluation(
    test_case_path: str,
    report: bool,
    models_config_path: str,
    agents_config_path: str,
):
    """
    Runs a single evaluation for a given test case file.
    """
    typer.echo(f"Running evaluation for: {test_case_path}")

    # 1. Load Configurations
    try:
        with open(models_config_path, 'r') as f:
            models_config = yaml.safe_load(f)
        with open(agents_config_path, 'r') as f:
            agents_config = yaml.safe_load(f)
        with open(test_case_path, 'r') as f:
            test_case = yaml.safe_load(f)
    except FileNotFoundError as e:
        typer.secho(f"Error: Config or test file not found - {e}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # 2. Get Test Case Details
    agent_name = test_case.get("agent")
    factory_method_name = test_case.get("factory_method")
    model_key = test_case.get("model")
    inputs = test_case.get("inputs")

    # This maps the friendly name (e.g., 'mistral_large') to the model details
    model_details = models_config.get("models", {}).get(model_key)
    if not model_details:
        typer.secho(f"Error: Model key '{model_key}' not found in {models_config_path}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(f"Agent: {agent_name}, Method: {factory_method_name}, Model: {model_key}")

    # 3. Use the prompts library to build the prompt (dynamic import)
    try:
        # Dynamically import the factory class
        factory_module = __import__(f"prompts.agents.{agent_name}.prompt_factory", fromlist=[f"{agent_name}Factory"])
        FactoryClass = getattr(factory_module, f"{''.join(word.capitalize() for word in agent_name.split('_'))}PromptFactory")

        # Instantiate the factory and the input schema
        factory_instance = FactoryClass()
        if factory_method_name == "build_prompt_create":
            SchemaClass = factory_instance.build_prompt_create.__annotations__['request']
        elif factory_method_name == "build_prompt_edit":
            SchemaClass = factory_instance.build_prompt_edit.__annotations__['request']
        else:
            raise ValueError(f"Unsupported factory method: {factory_method_name}")

        request_obj = SchemaClass(**inputs)

        # Call the specified factory method
        prompt_builder_method = getattr(factory_instance, factory_method_name)
        prompts = prompt_builder_method(request_obj)

        typer.secho("--- System Prompt ---", fg=typer.colors.BLUE)
        typer.echo(prompts.get("system"))
        typer.secho("--- User Prompt ---", fg=typer.colors.BLUE)
        typer.echo(prompts.get("user"))

    except (ImportError, AttributeError, TypeError) as e:
        typer.secho(f"Error building prompt: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # 4. Use LlmClientFactory to call the model
    typer.secho("\n--- LLM API Call ---", fg=typer.colors.YELLOW)
    llm_factory = LlmClientFactory(models_config.get("models"))
    response_text = ""
    try:
        llm_client = llm_factory.get_client(model_key)
        response_text = llm_client.generate(
            system_prompt=prompts.get("system"),
            user_prompt=prompts.get("user")
        )
        typer.secho("API call successful.", fg=typer.colors.GREEN)
        typer.echo(response_text)
    except (ValueError, NotImplementedError, Exception) as e:
        typer.secho(f"Error during LLM API call: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # 5. Run assertions
    typer.secho("\n--- Assertions ---", fg=typer.colors.CYAN)
    assertions = test_case.get("assertions", [])
    if not assertions:
        typer.echo("No assertions defined.")

    all_assertions_passed = True
    assertion_results = []
    for assertion in assertions:
        assertion_type = assertion.get("type")
        typer.echo(f"Running assertion: {assertion_type}")

        # This is a basic implementation. A real harness would be more robust.
        if assertion_type == "is_valid_pydantic_schema":
            try:
                # Dynamically get the schema class from prompts.schemas
                SchemaToValidate = __import__("prompts.schemas", fromlist=[assertion.get("schema")]).__dict__[assertion.get("schema")]
                SchemaToValidate.model_validate_json(strip_markdown(response_text))
                result = f"[PASS] Response validates against {assertion.get('schema')}"
                typer.secho(f"  {result}", fg=typer.colors.GREEN)
                assertion_results.append(result)
            except Exception as e:
                result = f"[FAIL] Pydantic validation failed: {e}"
                typer.secho(f"  {result}", fg=typer.colors.RED)
                assertion_results.append(result)
                all_assertions_passed = False

        elif assertion_type == "field_contains":
            try:
                response_json = json.loads(strip_markdown(response_text))
                field = assertion.get("field")
                expected_values = assertion.get("expected", [])
                field_value = response_json.get(field, "")

                if all(val in field_value for val in expected_values):
                    result = f"Field '{field}' contains expected values."
                    typer.secho(f"  [PASS] {result}", fg=typer.colors.GREEN)
                    assertion_results.append(f"[PASS] {result}")
                else:
                    result = f"Field '{field}' did not contain all expected values."
                    typer.secho(f"  [FAIL] {result}", fg=typer.colors.RED)
                    assertion_results.append(f"[FAIL] {result}")
                    all_assertions_passed = False
            except json.JSONDecodeError:
                result = "Could not parse LLM response as JSON."
                typer.secho(f"  [FAIL] {result}", fg=typer.colors.RED)
                assertion_results.append(f"[FAIL] {result}")
                all_assertions_passed = False

        elif assertion_type == "field_not_contains":
            try:
                response_json = json.loads(strip_markdown(response_text))
                field = assertion.get("field")
                unexpected_values = assertion.get("expected", [])
                field_value = response_json.get(field, "")

                if not any(val in field_value for val in unexpected_values):
                    result = f"Field '{field}' does not contain unexpected values."
                    typer.secho(f"  [PASS] {result}", fg=typer.colors.GREEN)
                    assertion_results.append(f"[PASS] {result}")
                else:
                    result = f"Field '{field}' contained unexpected values."
                    typer.secho(f"  [FAIL] {result}", fg=typer.colors.RED)
                    assertion_results.append(f"[FAIL] {result}")
                    all_assertions_passed = False
            except json.JSONDecodeError:
                result = "Could not parse LLM response as JSON."
                typer.secho(f"  [FAIL] {result}", fg=typer.colors.RED)
                assertion_results.append(f"[FAIL] {result}")
                all_assertions_passed = False

        elif assertion_type == "ai_critique":
            try:
                prompt_path = assertion.get("prompt_path")
                thresholds = assertion.get("thresholds", {})
                critique_model_key = assertion.get("critique_model", "gemini_flash_strict") # Default model

                # 1. Load the critique prompt from the specified file
                with open(prompt_path, 'r') as f:
                    critique_prompt_template = f.read()

                # 2. Get the critique client
                critique_client = llm_factory.get_client(critique_model_key)

                # 3. Inject the original LLM's response into the critique prompt
                # The character data is the 'final_character' part of the response
                main_model_output_json = json.loads(strip_markdown(response_text))
                character_data_json = json.dumps(main_model_output_json.get("final_character", {}), indent=2)
                
                user_prompt_for_critique = critique_prompt_template.replace(
                    "{{ character_data }}", character_data_json
                )

                # 4. Generate the scored critique
                critique_response_text = critique_client.generate(
                    system_prompt="You are an AI evaluator. Follow the user's instructions precisely.",
                    user_prompt=user_prompt_for_critique
                )
                typer.echo(f"  Critique Response:\n{critique_response_text}")

                # 5. Validate the critique response against the schema
                from prompts.schemas import CritiqueScoreSchema
                critique_scores = CritiqueScoreSchema.model_validate_json(strip_markdown(critique_response_text))
                typer.secho("  [PASS] AI critique response is valid JSON and matches CritiqueScoreSchema.", fg=typer.colors.GREEN)

                # 6. Check scores against thresholds
                scores_passed = True
                for metric, threshold in thresholds.items():
                    score = getattr(critique_scores, metric, None)
                    if score is None:
                        result = f"Metric '{metric}' not found in critique response."
                        typer.secho(f"  [FAIL] {result}", fg=typer.colors.RED)
                        assertion_results.append(f"[FAIL] {result}")
                        scores_passed = False
                        continue
                    if score >= threshold:
                        result = f"Metric '{metric}': {score} >= {threshold}"
                        typer.secho(f"  [PASS] {result}", fg=typer.colors.GREEN)
                        assertion_results.append(f"[PASS] {result}")
                    else:
                        result = f"Metric '{metric}': {score} < {threshold}"
                        typer.secho(f"  [FAIL] {result}", fg=typer.colors.RED)
                        assertion_results.append(f"[FAIL] {result}")
                        scores_passed = False
                
                if not scores_passed:
                    all_assertions_passed = False

            except FileNotFoundError:
                result = f"Critique prompt file not found at '{prompt_path}'."
                typer.secho(f"  [FAIL] {result}", fg=typer.colors.RED)
                assertion_results.append(f"[FAIL] {result}")
                all_assertions_passed = False
            except (json.JSONDecodeError, Exception) as e:
                result = f"AI critique failed with an error: {e}"
                typer.secho(f"  [FAIL] {result}", fg=typer.colors.RED)
                assertion_results.append(f"[FAIL] {result}")
                all_assertions_passed = False

    typer.secho("\n--- Evaluation Summary ---", fg=typer.colors.BRIGHT_BLUE)
    if all_assertions_passed:
        typer.secho("All assertions passed!", fg=typer.colors.BRIGHT_GREEN)
    else:
        typer.secho("Some assertions failed.", fg=typer.colors.BRIGHT_RED)

    if report:
        os.makedirs("reports", exist_ok=True)
        test_case_name = os.path.splitext(os.path.basename(test_case_path))[0]
        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        report_filename = f"reports/{test_case_name}_{timestamp}.md"

        report_content = generate_report(
            test_case_name=test_case_name,
            model_id=model_key,
            llm_output=response_text,
            assertion_results=assertion_results,
        )

        with open(report_filename, "w") as f:
            f.write(report_content)

        typer.secho(f"\nReport generated: {report_filename}", fg=typer.colors.BLUE)


def main(
    test_case_path: str = typer.Argument(..., help="Path to the .yaml test case file, or 'all' to run all tests."),
    report: bool = typer.Option(False, "--report", help="Generate a markdown report."),
    models_config_path: str = typer.Option("config/models.yaml", help="Path to the models config file."),
    agents_config_path: str = typer.Option("config/agents.yaml", help="Path to the agents config file."),
):
    """
    A CLI tool to run evaluations on prompts using live AI models.
    """
    load_dotenv()

    if test_case_path.lower() == "all":
        typer.echo("Running all test cases...")
        test_case_dir = "evaluations/test_cases"
        try:
            test_case_files = [os.path.join(test_case_dir, f) for f in os.listdir(test_case_dir) if f.endswith(".yaml")]
            if not test_case_files:
                typer.secho(f"No .yaml files found in {test_case_dir}", fg=typer.colors.YELLOW)
                return
        except FileNotFoundError:
            typer.secho(f"Error: Test case directory not found at '{test_case_dir}'", fg=typer.colors.RED)
            raise typer.Exit(1)

        for test_file in test_case_files:
            run_evaluation(test_file, report, models_config_path, agents_config_path)
            typer.echo("-" * 40) # Separator
    else:
        run_evaluation(test_case_path, report, models_config_path, agents_config_path)


if __name__ == "__main__":
    typer.run(main)
