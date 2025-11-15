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
    role: str = Field(description="Their specific function or position in this scenario")
    personality: str = Field(description="Detailed personality description including communication style, decision-making approach, interpersonal style, and behavioral patterns. Should be substantial enough to drive interesting interactions and conflicts.")
    expertise_keywords: List[str] = Field(description="A list of specific skills and knowledge areas.")
    background: str = Field(description="Compelling backstory that explains their expertise, perspective, and stake in this scenario.")
    appearance: str = Field(description="Physical description with distinctive characteristics that make them memorable.")
    goals: str = Field(description="What they personally want to achieve in this scenario - their specific motivations.")
    fears: str = Field(description="What they worry about or want to avoid in this specific situation.")
    notable_quotes: str = Field(description="Example statement they might make | Another quote if multiple")
    is_player_character: bool = Field(description="Boolean indicating player vs AI control")

class ScenarioSchema(BaseModel):
    """Defines the expected JSON output from the AI for a scenario."""
    title: str = Field(description="Compelling Scenario Title That Captures the Core Challenge")
    description: str = Field(description="Engaging 2-3 sentence description that hooks the reader and explains the central conflict or opportunity.")
    category: str = Field(description="Primary category from: Business, Science, Politics, Crisis Management, Technology, Social Issues, Healthcare, Education, Environment, International Relations")
    difficulty: Literal["Beginner", "Intermediate", "Advanced", "Expert"] = Field(description="Scenario difficulty level")
    estimated_duration: int = Field(description="Estimated play duration in minutes")
    objectives: List[ObjectiveSchema] = Field(min_length=3, max_length=6, description="List of scenario objectives")
    win_conditions: str = Field(description="Clear, specific success criteria that feel challenging but achievable within the scenario constraints")
    lose_conditions: str = Field(description="Meaningful failure conditions that create stakes and tension without being punitive")
    max_turns: int = Field(description="Maximum number of turns/rounds")
    initial_scene_prompt: str = Field(description="Rich, immersive opening that immediately places participants in the situation, establishes the setting, creates urgency, and provides just enough context to begin decision-making. Should be 3-5 sentences that make participants feel present and engaged.")
    characters: List[ScenarioCharacterSchema] = Field(min_length=2, max_length=6, description="Characters involved in the scenario")
    tags: List[str] = Field(description="Tags for categorization and search")
    is_public: bool = Field(default=True, description="Whether scenario is publicly visible")

class ScenarioInternalThoughtProcess(BaseModel):
    """AI's internal monologue to plan a high-quality scenario."""
    analysis: str = Field(..., description="My analysis of the user's request. What is the core concept of the scenario?")
    title_idea: str = Field(..., description="My plan for the 'title' based on the guidelines.")
    description_idea: str = Field(..., description="My plan for the 'description', ensuring it is engaging.")
    objectives_idea: str = Field(..., description="My plan for the 'objectives', ensuring they are specific and measurable.")
    characters_idea: str = Field(..., description="My plan for the 'characters', ensuring they have diverse roles and perspectives.")
    initial_scene_prompt_idea: str = Field(..., description="A draft idea for the 'initial_scene_prompt' that is immersive.")

class ChainOfThoughtScenarioSchema(BaseModel):
    """
    The new top-level schema that forces Chain-of-Thought for scenarios.
    The AI fills 'internal_thought_process' first, then 'final_scenario'.
    """
    internal_thought_process: ScenarioInternalThoughtProcess = Field(..., description="Your structured plan to build the scenario, following all guidelines.")
    final_scenario: ScenarioSchema = Field(..., description="The final, complete scenario JSON, built from your plan.")

# --- Input Schemas ---

class CritiqueScoreSchema(BaseModel):
    """
    Defines the structured output for a scored AI critique.
    The schema ensures that the critique model returns a consistent,
    machine-readable JSON object with specific metrics.
    """
    creativity: int = Field(..., description="Score (0-10) for how creative and imaginative the character concept is.")
    depth: int = Field(..., description="Score (0-10) for how well-developed and multi-dimensional the character's personality and background are.")
    originality: int = Field(..., description="Score (0-10) for how original and non-cliché the character is.")

class CharacterCreationRequest(BaseModel):
    """
    Defines the strongly-typed input for *creating* a character.
    The user provides a single, natural language request.
    """
    user_request: str = Field(..., description="A natural language description of the character to be created.")


class ScenarioCreationRequest(BaseModel):
    """
    Defines the strongly-typed input for *creating* a scenario.
    The user provides a single, natural language request.
    """
    user_request: str = Field(..., description="A natural language description of the scenario to be created.")


class ScenarioEditRequest(BaseModel):
    """
    Defines the strongly-typed inputs for *editing* a scenario.
    It provides the user's instruction AND the full current scenario.
    """
    edit_request: str = Field(..., description="The user's specific natural language instruction for what to change.")
    current_scenario: ScenarioSchema = Field(..., description="The full, current JSON object of the scenario to be edited.")


class CharacterEditRequest(BaseModel):
    """
    Defines the strongly-typed inputs for *editing* a character.
    It provides the user's instruction AND the full current character.
    """
    edit_request: str = Field(..., description="The user's specific natural language instruction for what to change.")
    current_character: CharacterSchema = Field(..., description="The full, current JSON object of the character to be edited.")


class UserProfileSchema(BaseModel):
    """A minimal user profile."""
    name: str = Field(description="User's name")
    age: int = Field(description="User's age")


class TranscriptItemSchema(BaseModel):
    """A single entry in a transcript."""
    character: str = Field(description="The character speaking")
    line: str = Field(description="The spoken line")


class PerformanceMetricsSchema(BaseModel):
    """A set of performance metrics."""
    relationship_change: int = Field(description="The change in relationship score")


class ScenarioFeedbackRequest(BaseModel):
    """
    Defines the strongly-typed input for requesting scenario feedback.
    """
    scenario_title: str
    scenario_id: str
    instance_id: str
    user_id: str
    prompt_version: str
    user_profile: UserProfileSchema
    transcript: List[TranscriptItemSchema]
    performance_metrics: PerformanceMetricsSchema
    detail_level: str


class AchievementSchema(BaseModel):
    """An achievement of the user."""
    description: str = Field(description="A description of the achievement")


class SuggestionSchema(BaseModel):
    """A suggestion for the user."""
    description: str = Field(description="A description of the suggestion")


class RelationshipSchema(BaseModel):
    """A relationship analysis."""
    character: str = Field(description="The character in the relationship")
    analysis: str = Field(description="An analysis of the relationship")


class ScenarioFeedbackSchema(BaseModel):
    """
    Defines the expected JSON output from the AI for scenario feedback.
    """
    achievements: List[AchievementSchema]
    suggestions: List[SuggestionSchema]
    relationships: List[RelationshipSchema]
    motivational_summary: str


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

class CharacterCritiqueScoreSchema(BaseModel):
    """
    Defines the structured output for a scored AI critique for characters.
    """
    creativity: int = Field(..., description="Score (0-10) for how creative and imaginative the character concept is.")
    depth: int = Field(..., description="Score (0-10) for how well-developed and multi-dimensional the character's personality and background are.")
    originality: int = Field(..., description="Score (0-10) for how original and non-cliché the character is.")

class ScenarioCritiqueScoreSchema(BaseModel):
    """
    Defines the structured output for a scored AI critique for scenarios.
    """
    creativity: int = Field(..., description="Score (0-10) for how creative and imaginative the scenario concept is.")
    originality: int = Field(..., description="Score (0-10) for how original and non-cliché the plot and setting are.")
    engagement: int = Field(..., description="Score (0-10) for how engaging and compelling the scenario's hooks and conflicts are.")


# --- Character in Simulation Schemas ---

class InternalState(BaseModel):
    emotion: str
    thoughts: str
    objective_impact: str

class Metrics(BaseModel):
    authenticity: int
    relevance: int
    engagement: int
    consistency: int

class Flags(BaseModel):
    requires_other_character: bool
    advances_objective: bool
    reveals_information: bool

class CharacterInSimulationOutput(BaseModel):
    character_name: str
    response_type: str
    content: str
    internal_state: InternalState
    suggested_follow_ups: List[str]
    metrics: Metrics
    flags: Flags

class ScenarioObjective(BaseModel):
    description: str
    importance: str

class ObjectiveProgress(BaseModel):
    completion_percentage: int
    status: str
    progress_notes: str

class ConversationMessage(BaseModel):
    speaker: str
    content: str

class CharacterExample(BaseModel):
    situation: str
    style: str
    sample: str


class CharacterInSimulationInput(BaseModel):
    character_name: str
    simulation_name: str
    character_role: str
    character_expertise: str
    character_personality: str
    character_behaviors: List[str]
    selected_strategy: str
    emergency_keywords: List[str]
    scenario_description: str
    current_scene: str
    scenario_objectives: Optional[List[ScenarioObjective]] = None
    objectives_progress: Optional[dict[str, ObjectiveProgress]] = None
    conversation_history: List[ConversationMessage]
    command_type: str
    user_input: str
    character_examples: Optional[List[CharacterExample]] = None
