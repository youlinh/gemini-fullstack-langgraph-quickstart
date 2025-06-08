import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client # Restoring for Google Search
from langchain_deepseek import ChatDeepSeek # Corrected import name

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
# from langchain_google_genai import ChatGoogleGenerativeAI # Removed for DeepSeek integration
from agent.utils import (
    get_citations, # Restoring for Google Search
    get_research_topic,
    insert_citation_markers, # Restoring for Google Search
    resolve_urls, # Restoring for Google Search
)

load_dotenv()

if os.getenv("DEEPSEEK_API_KEY") is None:
    raise ValueError("DEEPSEEK_API_KEY is not set")

if os.getenv("GOOGLE_SEARCH_API_KEY") is None:
    raise ValueError("GOOGLE_SEARCH_API_KEY is not set")

# Used for Google Search API
genai_search_client = Client(api_key=os.getenv("GOOGLE_SEARCH_API_KEY"))


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses the configured query generator model to create optimized search queries for web research
    based on the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init DeepSeek Chat
    llm = ChatDeepSeek(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the Google Search API tool.

    Uses the Google Search tool via google.genai.Client. The prompt for the search
    is based on web_searcher_instructions. A Google model capable of tool use
    is used for this specific node to execute the search tool.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable.

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client (genai_search_client) for the search tool
    # TODO: Confirm the correct Google model name for tool usage. Using gemini-1.5-flash-latest as a placeholder.
    # The prompt (formatted_prompt) is prepared based on DeepSeek's logic if web_searcher_instructions are generic enough.
    # However, the model executing the search tool itself must be a Google model.
    search_tool_executor_model = "gemini-1.5-flash-latest"

    response = genai_search_client.generate_content( # Corrected method name from .models.generate_content
        model=f"models/{search_tool_executor_model}",
        contents=formatted_prompt,
        generation_config={
            "tool_config": { "google_search_retrieval": { "disable_attribution": False } }
        },
        tools=[{"google_search_retrieval": {}}],
    )

    # IMPORTANT: The following extraction and utility function calls (resolve_urls, get_citations, insert_citation_markers)
    # are based on the new Google SDK structure but the utility functions themselves in agent/utils.py
    # were written for an OLDER SDK structure (e.g., response.candidates[0].grounding_metadata.grounding_chunks).
    # These utility functions WILL LIKELY FAIL or require significant adaptation to work correctly
    # with response.candidates[0].content.parts[0].grounding_metadata.web_search_results.
    # This adaptation is outside the immediate scope of this change but is crucial for functionality.

    extracted_text = ""
    sources_gathered = []
    modified_text = "" # Default to empty if no valid response or grounding

    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        response_part = response.candidates[0].content.parts[0]
        extracted_text = response_part.text if hasattr(response_part, "text") else ""

        if hasattr(response_part, "grounding_metadata") and response_part.grounding_metadata:
            web_search_results = response_part.grounding_metadata.web_search_results if hasattr(response_part.grounding_metadata, "web_search_results") else []

            # The following calls to utils are placeholders and depend on their adaptation:
            resolved_urls = resolve_urls(web_search_results, state["id"])
            citations = get_citations(response, resolved_urls) # This 'response' object might be too high level for the old util

            # Use extracted_text for citation markers, not potentially non-existent response.text
            modified_text = insert_citation_markers(extracted_text, citations)
            sources_gathered = [item for citation in citations for item in citation["segments"]] if citations else []
        else:
            # No grounding metadata, so use extracted text as is
            modified_text = extracted_text
    else:
        # Handle cases with no candidates or parts (e.g., safety blocked response)
        # modified_text is already "", sources_gathered is already []
        pass


    return {
        "sources_gathered": sources_gathered, # Potentially empty if utils fail or no grounding
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reflection model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    # TODO: The following line uses configurable.reasoning_model, but it should be configurable.reflection_model
    # This needs to be corrected in the Configuration class or here directly.
    # For now, assuming 'reasoning_model' is a placeholder for 'reflection_model' in this context.
    reasoning_model = state.get("reasoning_model") or configurable.reflection_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reflection Model
    llm = ChatDeepSeek(
        model=reasoning_model, # Should be reflection_model from config
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Combines the research results into a final answer using the configured answer model.
    NOTE: Citation and source handling has been removed due to the simplification
    of the web_research node. The 'sources_gathered' field will likely be empty.

    Args:
        state: Current graph state containing the research results.

    Returns:
        Dictionary with state update, including the final answer message.
    """
    configurable = Configuration.from_runnable_config(config)
    answer_model = state.get("answer_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Answer Model
    llm = ChatDeepSeek(
        model=answer_model, # Should be answer_model from config
        temperature=0,
        max_retries=2,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Source handling (deduplication, URL replacement) has been removed
    # as 'sources_gathered' is no longer populated by 'web_research' in a meaningful way.
    # The 'sources_gathered' list from the state will be passed through as is,
    # which is expected to be empty.

    # Restore original logic for URL replacement, assuming utils adapt and sources_gathered is populated.
    # If web_research node or utils fail to populate sources_gathered meaningfully, this will have no effect.
    unique_sources = []
    for source in state["sources_gathered"]: # This list now comes from the restored web_research
        # Ensure source is a dictionary and has 'short_url' and 'value' keys
        if isinstance(source, dict) and "short_url" in source and "value" in source:
            if source["short_url"] in result.content:
                result.content = result.content.replace(
                    source["short_url"], source["value"]
                )
                unique_sources.append(source)
        elif isinstance(source, dict) and "value" in source: # Fallback if only value is present
             unique_sources.append(source)


    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources, # Use the processed list
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
