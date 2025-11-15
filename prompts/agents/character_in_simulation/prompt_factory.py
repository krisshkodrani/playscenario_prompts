# prompts/agents/character_in_simulation/prompt_factory.py
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Optional
from pathlib import Path
from prompts.schemas import (
    CharacterInSimulationInput,
    CharacterInSimulationOutput,
)

class CharacterInSimulationPromptFactory:

    def __init__(self, template_dir: Optional[Path] = None):
        base_dir = template_dir or Path("prompts/")
        self._env = Environment(
            loader=FileSystemLoader(base_dir),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._env.filters['tojson_pretty'] = lambda x: json.dumps(x, indent=2)

        self._system_template = self._env.get_template("agents/character_in_simulation/_system.j2")
        self._user_template = self._env.get_template("agents/character_in_simulation/_user.j2")

        self._cached_schema = CharacterInSimulationOutput.model_json_schema()

    def get_system_prompt(self) -> str:
        """
        Renders the system prompt, injecting the Pydantic
        JSON schema into it.
        """
        return self._system_template.render(
            output_schema=json.dumps(self._cached_schema, indent=2)
        )

    def build_prompt(self, request: CharacterInSimulationInput) -> dict:
        """
        Builds the user-facing prompt from a strongly-typed
        CharacterInSimulationInput object.
        """
        system_prompt = self.get_system_prompt()
        user_prompt = self._user_template.render(**request.model_dump()).strip() + "\\n"
        return {"system": system_prompt, "user": user_prompt}
