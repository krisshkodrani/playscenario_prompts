# prompts/agents/moderator/prompt_factory.py
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Optional
from pathlib import Path
from prompts.schemas import (
    ScenarioModeratorInput,
    ScenarioModeratorOutput,
)

class ModeratorPromptFactory:

    def __init__(self, template_dir: Optional[Path] = None):
        base_dir = template_dir or Path("prompts/")
        self._env = Environment(
            loader=FileSystemLoader(base_dir),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._env.filters['tojson_pretty'] = lambda x: json.dumps(x, indent=2)

        self._system_template = self._env.get_template("agents/moderator/_system.j2")
        self._user_template = self._env.get_template("agents/moderator/_user.j2")

        self._cached_schema = ScenarioModeratorOutput.model_json_schema()

    def get_system_prompt(self, request: ScenarioModeratorInput) -> str:
        """
        Renders the system prompt, injecting the Pydantic
        JSON schema and scenario data into it.
        """
        return self._system_template.render(
            output_schema=json.dumps(self._cached_schema, indent=2),
            **request.model_dump()
        )

    def build_prompt(self, request: ScenarioModeratorInput) -> dict:
        """
        Builds the user-facing prompt from a strongly-typed
        ScenarioModeratorInput object.
        """
        system_prompt = self.get_system_prompt(request)
        user_prompt = self._user_template.render(**request.model_dump()).strip() + "\n"
        return {"system": system_prompt, "user": user_prompt}
