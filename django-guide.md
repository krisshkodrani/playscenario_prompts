# Django Integration Guide for PlayScenario Prompt Library

This guide provides instructions on how to integrate and use the `playscenario-prompt-lib` within a Django project.

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Core Concepts](#core-concepts)
- [Helper Utility](#helper-utility)
- [Django View Examples](#django-view-examples)
- [Saving to a Django Model](#saving-to-a-django-model)

## Installation & Setup

To use the `playscenario-prompt-lib` in your Django project, you'll need to install it via pip.

1.  **Install the library:**

    You can install the library directly from a Git repository if it's not on PyPI. Add the following line to your `requirements.txt` file:

    ```
    # requirements.txt
    ...
    # playscenario-prompt-lib @ git+https/path/to/your/git/repo.git
    ```

    For local development, you can use an editable install. From the root of this library's repository, run:

    ```bash
    pip install -e .
    ```

    Then, from your Django project's directory, install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

It's crucial to handle API keys securely in a Django project. The recommended approach is to use environment variables and load them in your `settings.py` file.

1.  **Store API Keys in `.env`:**

    In your Django project's root directory, create a `.env` file (if you don't have one already) and add your API keys:

    ```
    # .env
    OPENAI_API_KEY="your_openai_api_key_here"
    GOOGLE_API_KEY="your_google_api_key_here"
    # Add other keys as needed
    ```

    Ensure that `.env` is listed in your `.gitignore` file to avoid committing secrets to version control.

2.  **Load Environment Variables in `settings.py`:**

    Use a library like `python-decouple` or `django-environ` to load the environment variables. Here's an example using `decouple`:

    ```python
    # settings.py
    from decouple import config

    # ... other settings

    OPENAI_API_KEY = config('OPENAI_API_KEY')
    GOOGLE_API_KEY = config('GOOGLE_API_KEY', default='') # Use default if optional
    ```

3.  **Pass Configuration to the Library:**

    When you instantiate the prompt factory or any other class from the library, you'll pass the API keys from your settings. The exact mechanism will be shown in the [Helper Utility](#helper-utility) and [Django View Examples](#django-view-examples) sections.

## Core Concepts

This library is designed with a clear separation of concerns. For a Django developer, the key components to understand are:

-   **Schemas (`prompts.schemas`)**: These are Pydantic models that define the data structures for both the inputs your application will provide (e.g., `ScenarioCreationRequest`) and the structured data you expect back from the AI model (e.g., `ScenarioSchema`). You'll use these to validate data and ensure type safety.

-   **Prompt Factories (`prompts.agents.*.prompt_factory`)**: These are the workhorses of the library. A class like `ScenarioHelperPromptFactory` is responsible for taking your input data (as a schema object), rendering it into the correct Jinja2 templates, and producing the final system and user prompts to be sent to an AI model.

-   **Agents**: This is a conceptual grouping of related prompt factories. For example, everything related to creating and editing scenarios is handled by the `scenario_helper` agent.

Your Django application will typically:
1.  Instantiate a `PromptFactory` for the desired task.
2.  Create a request schema object with data from a user (e.g., from an HTML form).
3.  Call a `build_prompt_*` method on the factory to get the prompts.
4.  Send the prompts to an AI model using a library like `openai` or `google-generativeai`.
5.  Receive the JSON response from the AI and parse it using the corresponding output schema.

## Helper Utility

To simplify the process of using the prompt library within your Django views, you can create a helper utility. This class will encapsulate the logic for initializing the prompt factory and interacting with the AI model.

Create a new file in one of your Django apps, for example, `your_app/prompt_utils.py`:

```python
# your_app/prompt_utils.py

import os
import json
import openai
from django.conf import settings
from prompts.agents.scenario_helper.prompt_factory import ScenarioHelperPromptFactory
from prompts.schemas import ScenarioCreationRequest, ChainOfThoughtScenarioSchema

class PromptService:
    def __init__(self):
        # Initialize the prompt factory
        self.scenario_factory = ScenarioHelperPromptFactory()

        # It's good practice to initialize the OpenAI client once
        # and reuse it.
        if not hasattr(settings, 'OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in Django settings")

        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate_scenario(self, user_request_text: str):
        """
        Generates a scenario based on a user's text request.

        Args:
            user_request_text: The natural language request from the user.

        Returns:
            A Pydantic ScenarioSchema object or None if generation fails.
        """
        # 1. Create a request object
        request = ScenarioCreationRequest(user_request=user_request_text)

        # 2. Build the prompt
        prompt = self.scenario_factory.build_prompt_create(request)

        try:
            # 3. Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Or your preferred model
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                response_format={"type": "json_object"},
            )

            # 4. Parse the response
            response_data = json.loads(response.choices[0].message.content)

            # The library uses a ChainOfThought schema, so we access the final scenario
            structured_response = ChainOfThoughtScenarioSchema(**response_data)
            return structured_response.final_scenario

        except (openai.APIError, json.JSONDecodeError, KeyError) as e:
            # Handle potential errors from the API or JSON parsing
            print(f"Error generating scenario: {e}")
            return None

```

This `PromptService` class can now be easily used in your Django views, as shown in the next section.

## Django View Examples

Here's how you can use the `PromptService` in a Django view. This example assumes you have a simple form where a user can input a request for a scenario.

1.  **Create a Form (optional but recommended):**

    ```python
    # your_app/forms.py
    from django import forms

    class ScenarioRequestForm(forms.Form):
        user_request = forms.CharField(
            widget=forms.Textarea(attrs={'rows': 4}),
            label="Describe the scenario you want to create:"
        )
    ```

2.  **Create the View:**

    This view will handle both displaying the form (`GET` request) and processing the form submission (`POST` request).

    ```python
    # your_app/views.py
    from django.shortcuts import render
    from .forms import ScenarioRequestForm
    from .prompt_utils import PromptService

    def create_scenario_view(request):
        form = ScenarioRequestForm()
        scenario_data = None

        if request.method == 'POST':
            form = ScenarioRequestForm(request.POST)
            if form.is_valid():
                user_request = form.cleaned_data['user_request']

                # Use the prompt service to generate the scenario
                prompt_service = PromptService()
                scenario = prompt_service.generate_scenario(user_request)

                if scenario:
                    # You can pass the Pydantic model directly to the template
                    scenario_data = scenario

        return render(request, 'your_app/create_scenario.html', {
            'form': form,
            'scenario': scenario_data
        })
    ```

3.  **Create the Template:**

    Create a template to render the form and display the generated scenario data.

    ```html
    <!-- your_app/templates/your_app/create_scenario.html -->
    <!DOCTYPE html>
    <html>
    <head>
        <title>Create a Scenario</title>
    </head>
    <body>
        <h1>Create a New Scenario</h1>
        <form method="post">
            {% csrf_token %}
            {{ form.as_p }}
            <button type="submit">Generate Scenario</button>
        </form>

        {% if scenario %}
            <hr>
            <h2>Generated Scenario</h2>
            <h3>{{ scenario.title }}</h3>
            <p><strong>Description:</strong> {{ scenario.description }}</p>
            <p><strong>Category:</strong> {{ scenario.category }}</p>
            <p><strong>Difficulty:</strong> {{ scenario.difficulty }}</p>

            <h4>Objectives</h4>
            <ul>
                {% for obj in scenario.objectives %}
                    <li>{{ obj.description }} ({{ obj.priority }})</li>
                {% endfor %}
            </ul>

            <h4>Characters</h4>
            <ul>
                {% for char in scenario.characters %}
                    <li><strong>{{ char.name }}</strong> - {{ char.role }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    </body>
    </html>
    ```

4.  **Wire up the URL:**

    ```python
    # your_app/urls.py
    from django.urls import path
    from .views import create_scenario_view

    urlpatterns = [
        path('create-scenario/', create_scenario_view, name='create_scenario'),
    ]
    ```

## Saving to a Django Model

After generating a scenario, you'll likely want to save it to your database. Here's how you can define a Django model and update your view to save the data.

1.  **Define the Django Models:**

    In your `models.py`, you can create models that mirror the structure of the `ScenarioSchema`.

    ```python
    # your_app/models.py
    from django.db import models

    class Scenario(models.Model):
        title = models.CharField(max_length=255)
        description = models.TextField()
        category = models.CharField(max_length=100)
        difficulty = models.CharField(max_length=50)
        # Add other fields from ScenarioSchema as needed

        def __str__(self):
            return self.title

    class Objective(models.Model):
        scenario = models.ForeignKey(Scenario, related_name='objectives', on_delete=models.CASCADE)
        description = models.TextField()
        priority = models.CharField(max_length=20)

    class Character(models.Model):
        scenario = models.ForeignKey(Scenario, related_name='characters', on_delete=models.CASCADE)
        name = models.CharField(max_length=255)
        role = models.CharField(max_length=255)
        # Add other fields from ScenarioCharacterSchema as needed
    ```

    *Note: Remember to run `python manage.py makemigrations` and `python manage.py migrate` after defining your models.*

2.  **Update the View to Save the Scenario:**

    Modify the `create_scenario_view` to save the generated scenario to the database.

    ```python
    # your_app/views.py
    from django.shortcuts import render, redirect
    from .forms import ScenarioRequestForm
    from .prompt_utils import PromptService
    from .models import Scenario, Objective, Character # Import your models

    def create_scenario_view(request):
        form = ScenarioRequestForm()

        if request.method == 'POST':
            form = ScenarioRequestForm(request.POST)
            if form.is_valid():
                user_request = form.cleaned_data['user_request']

                prompt_service = PromptService()
                scenario_schema = prompt_service.generate_scenario(user_request)

                if scenario_schema:
                    # Create the Scenario instance
                    scenario_instance = Scenario.objects.create(
                        title=scenario_schema.title,
                        description=scenario_schema.description,
                        category=scenario_schema.category,
                        difficulty=scenario_schema.difficulty,
                        # ... map other fields
                    )

                    # Create related Objective and Character objects
                    for obj_data in scenario_schema.objectives:
                        Objective.objects.create(
                            scenario=scenario_instance,
                            description=obj_data.description,
                            priority=obj_data.priority
                        )

                    for char_data in scenario_schema.characters:
                        Character.objects.create(
                            scenario=scenario_instance,
                            name=char_data.name,
                            role=char_data.role,
                            # ... map other fields
                        )

                    # Redirect to a success page or the scenario detail page
                    # return redirect('scenario_detail', pk=scenario_instance.pk)

        return render(request, 'your_app/create_scenario.html', {'form': form})
    ```

This example demonstrates the full lifecycle: getting a request from a user, using the prompt library to generate data, and saving that data to the Django database.
