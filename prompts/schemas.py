from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# --- Core Output Schemas ---

class CharacterSchema(BaseModel):
    """Defines the expected JSON output from the AI for a character."""
    name: str = Field(description="Character name")
    role: str = Field(description="Character role or profession")
    personality: str = Field(description="Detailed personality description")
    expertise_keywords: List[str] = Field(description="List of character expertise keywords")
    background: str = Field(description="Character background and history")
    appearance: str = Field(description="Physical appearance description")
    goals: str = Field(description="Character goals and motivations")
    fears: str = Field(description="Character fears and vulnerabilities")
    notable_quotes: str = Field(description="Example quotes or phrases the character might say")

class ObjectiveSchema(BaseModel):
    """A single objective within a scenario."""
    id: int = Field(description="Objective ID")
    description: str = Field(description="Objective description")
    priority: Literal["critical", "important", "optional"] = Field(description="Objective priority level")

class ScenarioCharacterSchema(BaseModel):
    """Character data structure for scenarios."""
    name: str = Field(description="Character name")
    role: str = Field(default="Character", description="Character role or profession")
    personality: str = Field(description="Detailed personality description")
    expertise_keywords: List[str] = Field(description="List of character expertise keywords")
    avatar_color: str = Field(description="Tailwind CSS background color class (e.g., 'bg-blue-500')")

class ScenarioSchema(BaseModel):
    """Defines the expected JSON output from the AI for a scenario."""
    title: str = Field(description="Scenario title")
    description: str = Field(description="Detailed scenario description")
    category: str = Field(description="Scenario category")
    difficulty: Literal["Beginner", "Intermediate", "Advanced", "Expert"] = Field(description="Scenario difficulty level")
    estimated_duration: int = Field(description="Estimated play duration in minutes")
    objectives: List[ObjectiveSchema] = Field(description="List of scenario objectives")
    win_conditions: str = Field(description="Conditions for scenario success")
    lose_conditions: str = Field(description="Conditions for scenario failure")
    max_turns: int = Field(description="Maximum number of turns/rounds")
    scenario_opening_message: str = Field(description="Opening scene description")
    characters: List[ScenarioCharacterSchema] = Field(description="Characters involved in the scenario")
    tags: List[str] = Field(description="Tags for categorization and search")
    is_public: bool = Field(default=True, description="Whether scenario is publicly visible")


# --- Input Schemas ---

class CharacterCreationRequest(BaseModel):
    """Strongly-typed input for creating a character."""
    role_notes: Optional[str] = Field(default=None, description="User's high-level notes about the character's role or profession")
    personality_notes: Optional[str] = Field(default=None, description="User's high-level notes about the character's personality")
    background_notes: Optional[str] = Field(default=None, description="User's high-level notes about the character's backstory")


# --- Router Schemas ---

class IntentRouterSchema(BaseModel):
    """
    Defines the output for the 'Intent Router' agent.
    This schema tells the application which agent and factory method to call next.
    """
    agent: Literal[
        "character_helper",
        "scenario_helper",
        "moderator",
        "unknown"
    ] = Field(description="The agent to route the request to.")

    factory_method: Literal[
        "build_prompt_create",
        "build_prompt_edit",
        "build_prompt_moderate",
        "not_applicable"
    ] = Field(description="The specific factory method to call on the agent.")

    # Using a generic dict for arguments to remain flexible.
    # The application will be responsible for casting this to the correct Pydantic Input model.
    arguments: dict = Field(description="The arguments to pass to the factory method, conforming to the relevant Input Schema.")
