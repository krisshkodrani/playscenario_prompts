# prompts/agents/scenario_helper/prompt_factory.py
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Optional
from pathlib import Path
from prompts.schemas import (
    ScenarioSchema,
    ScenarioCreationRequest,
    ChainOfThoughtScenarioSchema,
    ScenarioEditRequest
)

class ScenarioHelperPromptFactory:

    def __init__(self, template_dir: Optional[Path] = None):
        base_dir = template_dir or Path("prompts/")
        self._env = Environment(
            loader=FileSystemLoader(base_dir),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._env.filters['tojson_pretty'] = lambda x: json.dumps(x, indent=2)

        # --- Load "Create" Templates ---
        self._system_create_template = self._env.get_template("agents/scenario_helper/_system.j2")
        self._user_create_template = self._env.get_template("agents/scenario_helper/_user.j2")

        # --- Load "Edit" Templates ---
        self._system_edit_template = self._env.get_template("agents/scenario_helper/_system_edit.j2")
        self._user_edit_template = self._env.get_template("agents/scenario_helper/_user_edit.j2")

        # Cache the CoT schema (re-used for both create and edit)
        self._cached_schema = ChainOfThoughtScenarioSchema.model_json_schema()

    # --- CREATE METHODS ---

    def get_system_prompt_create(self) -> str:
        """
        Renders the "create" system prompt, injecting the Pydantic
        JSON schema into it.
        """
        return self._system_create_template.render(
            output_schema=json.dumps(self._cached_schema, indent=2)
        )

    def build_prompt_create(self, request: ScenarioCreationRequest) -> dict:
        """
        Builds the "create" user-facing prompt from a strongly-typed
        ScenarioCreationRequest object.
        """
        system_prompt = self.get_system_prompt_create()
        user_prompt = self._user_create_template.render(**request.model_dump()).strip() + "\n"
        return {"system": system_prompt, "user": user_prompt}

    # --- EDIT METHODS ---

    def get_system_prompt_edit(self) -> str:
        """
        Renders the "edit" system prompt, injecting the Pydantic
        JSON schema into it.
        """
        return self._system_edit_template.render(
            output_schema=json.dumps(self._cached_schema, indent=2)
        )

    def build_prompt_edit(self, request: ScenarioEditRequest) -> dict:
        """
        Builds the "edit" user-facing prompt from a strongly-typed
        ScenarioEditRequest object.
        """
        system_prompt = self.get_system_prompt_edit()
        user_prompt = self._user_edit_template.render(**request.model_dump()).strip() + "\n"
        return {"system": system_prompt, "user": user_prompt}
