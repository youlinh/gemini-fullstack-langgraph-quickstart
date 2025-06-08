import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="deepseek-chat",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="deepseek-chat",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="deepseek-chat",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Create a dictionary with values found, letting Pydantic handle defaults for missing ones.
        final_values = {}
        for field_name in cls.model_fields.keys():
            # Prioritize environment variables
            env_value = os.environ.get(field_name.upper())
            if env_value is not None:
                final_values[field_name] = env_value
                continue # Environment variable takes precedence

            # Then, check configurable, with special handling for reflection_model
            config_value = None
            if field_name == "reflection_model":
                # If 'reasoning_model' is provided in configurable, use it for 'reflection_model'
                config_value = configurable.get("reasoning_model", configurable.get(field_name))
            else:
                config_value = configurable.get(field_name)

            if config_value is not None:
                final_values[field_name] = config_value
            # If neither env_var nor config_value is found, Pydantic will use the default field value

        return cls(**final_values)
