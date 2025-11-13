from jinja2 import Environment, FileSystemLoader
import json

from prompts.schemas import ChainOfThoughtCharacterSchema, CharacterCreationRequest

class CharacterHelperPromptFactory:
    """
    Builds prompts for the Character Helper agent.
    """
    def __init__(self, template_path: str = "prompts"):
        self.env = Environment(loader=FileSystemLoader(template_path))

    def build_prompt_create(self, request: CharacterCreationRequest) -> dict:
        """
        Builds the system and user prompts for creating a new character.

        Args:
            request: A Pydantic object containing the user's notes.

        Returns:
            A dictionary containing the system and user prompts.
        """
        system_template = self.env.get_template("agents/character_helper/_system.j2")
        user_template = self.env.get_template("agents/character_helper/_user.j2")

        # Generate the JSON schema from the Pydantic model
        output_schema = json.dumps(ChainOfThoughtCharacterSchema.model_json_schema(), indent=2)

        system_prompt = system_template.render(output_schema=output_schema)
        user_prompt = user_template.render(request=request)

        return {
            "system": system_prompt,
            "user": user_prompt
        }
