# prompts/agents/scenario_feedback/prompt_factory.py
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Optional
from pathlib import Path
from prompts.schemas import (
    ScenarioFeedbackRequest,
    ScenarioFeedbackSchema,
)

class ScenarioFeedbackPromptFactory:

    def __init__(self, template_dir: Optional[Path] = None):
        base_dir = template_dir or Path("prompts/")
        self._env = Environment(
            loader=FileSystemLoader(base_dir),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters["tojson"] = lambda x: json.dumps(x)

        self._system_template = self._env.get_template("agents/scenario_feedback/_system.j2")
        self._cached_schema = ScenarioFeedbackSchema.model_json_schema()

    def build_prompt(self, request: ScenarioFeedbackRequest) -> dict:
        """
        Builds the user-facing prompt from a strongly-typed
        ScenarioFeedbackRequest object.
        """
        system_prompt = self._system_template.render(
            **request.model_dump(),
            schema_json=json.dumps(self._cached_schema, indent=2)
        ).strip() + "\n"
        return {"system": system_prompt, "user": "Produce ONLY the JSON now."}
