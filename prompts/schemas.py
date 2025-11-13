from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# --- Core Output Schemas ---

class CharacterSchema(BaseModel):
    """
    The complete data schema for a single character, based on the v2.0 prompt.
    This is the single source of truth for character data.
    """
    name: str = Field(..., description="A realistic, culturally appropriate name for the character.")
    role: str = Field(..., description="The character's specific profession, function, or archetype in the scenario.")
    appearance: str = Field(..., description="Distinctive physical description including features, clothing style, build, and other characteristics that make them memorable.")
    personality: str = Field(..., description="Rich, multi-dimensional personality (min 100 words), including communication style, decision-making, stress response, and flaws.")
    expertise_keywords: List[str] = Field(..., min_length=3, max_length=8, description="A list of 3-8 specific skills and knowledge areas.")
    background: str = Field(..., description="Compelling personal history and background (min 75 words) that explains their expertise, perspective, and motivations.")
    goals: str = Field(..., description="What the character wants to achieve in scenarios—their primary motivations, aspirations, and driving forces.")
    fears: str = Field(..., description="What the character worries about, wants to avoid, or finds challenging. Includes professional concerns and personal vulnerabilities.")
    notable_quotes: str = Field(..., description="2-3 example phrases that capture their voice, perspective, and communication style, separated by ' | '.")

class CharacterInternalThoughtProcess(BaseModel):
    """AI's internal monologue to plan a high-quality character."""
    analysis: str = Field(..., description="My analysis of the user's notes. What is the core concept?")
    role_idea: str = Field(..., description="My plan for the 'role' based on the guidelines.")
    personality_idea: str = Field(..., description="My plan for the 'personality', ensuring it has 5-star quality and internal conflict.")
    background_idea: str = Field(..., description="My plan for the 'background', linking it to skills, goals, and the 5-star rubric.")
    expertise_idea: str = Field(..., description="My plan for the 'expertise_keywords', ensuring they are specific and not generic.")
    quote_idea: str = Field(..., description="A draft idea for a 'notable_quote' that captures the voice.")

class ChainOfThoughtCharacterSchema(BaseModel):
    """
    The new top-level schema that forces Chain-of-Thought.
    The AI fills 'internal_thought_process' first, then 'final_character'.
    """
    internal_thought_process: CharacterInternalThoughtProcess = Field(..., description="Your structured plan to build the character, following all guidelines.")
    final_character: CharacterSchema = Field(..., description="The final, complete character JSON, built from your plan.")

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
    """
    Defines the strongly-typed inputs for *creating* a character,
    replacing the generic 'user_request'.
    """
    role_notes: Optional[str] = Field(None, description="Guidance on the character's role or profession.")
    personality_notes: Optional[str] = Field(None, description="Guidance on personality, demeanor, and internal conflicts.")
    background_notes: Optional[str] = Field(None, description="Guidance on history, key life events, or motivations.")
    expertise_notes: Optional[str] = Field(None, description="Guidance on skills, knowledge, or expertise.")
    goal_notes: Optional[str] = Field(None, description="Guidance on the character's primary objectives.")
    other_notes: Optional[str] = Field(None, description="Any other miscellaneous notes or creative direction.")


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
