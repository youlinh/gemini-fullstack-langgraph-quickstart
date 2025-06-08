from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Creates a map of long URLs from Google Search results (via google.genai.Client)
    to a shorter, identified URL. Ensures each original URL gets a consistent
    shortened form while maintaining uniqueness.

    The input `urls_to_resolve` is expected to be a list of objects/dictionaries,
    each having a `web.uri` attribute (e.g., from web_search_results in grounding metadata).
    # TODO: Verify the exact structure of `urls_to_resolve` (expected to be grounding_chunks or similar)
    # from the current Google API response when using the 'google_search' tool.
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/" # Google-specific prefix
    # Assuming `site` objects in `urls_to_resolve` have `web.uri` or similar path to the URL.
    # This line might need adjustment based on the actual structure of `urls_to_resolve`.
    urls = [site.get("uri") if isinstance(site, dict) else getattr(getattr(site, "web", object()), "uri", None) for site in urls_to_resolve]
    urls = [url for url in urls if url] # Filter out None values

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices
    derived from Google Search results.
    # TODO: This function's logic depends on get_citations. Review if get_citations output format changes.

    Args:
        text (str): The original text string.
        citations_list (list): A list of citation dictionaries from `get_citations`.
                               Each dictionary should contain 'start_index', 'end_index',
                               and 'segments' (which provide marker info).
                               Indices are assumed to be for the original text.

    Returns:
        str: The text with citation markers inserted.
    """
    # Sort citations by end_index in descending order.
    # If end_index is the same, secondary sort by start_index descending.
    # This ensures that insertions at the end of the string don't affect
    # the indices of earlier parts of the string that still need to be processed.
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        # These indices refer to positions in the *original* text,
        # but since we iterate from the end, they remain valid for insertion
        # relative to the parts of the string already processed.
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        # Insert the citation marker at the original end_idx position
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text


def get_citations(response, resolved_urls_map):
    """
    Extracts and formats citation information from a Google Search API response
    (via google.genai.Client, specifically using the 'google_search' tool).

    This function processes grounding metadata from the response to construct a list of citation objects.
    Each citation object includes start/end indices of the text segment and formatted markdown links.
    # TODO: Verify path to grounding_supports/grounding_chunks and their structure (web_uri, title)
    # from current Google API response. The current logic iterates grounding_supports and then grounding_chunks,
    # this needs to match the actual response from the 'google_search' tool.

    Args:
        response: The response object from `genai_search_client.generate_content()`,
                  expected to have grounding metadata (e.g., `response.candidates[0].content.parts[0].grounding_metadata`).
                  It also relies on `resolved_urls_map` (from `resolve_urls`)
                  to map chunk URIs to shorter, resolved URLs.

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "start_index" (int): The starting character index of the cited
                                     segment in the original text. Defaults to 0
                                     if not specified.
              - "end_index" (int): The character index immediately after the
                                   end of the cited segment (exclusive).
              - "segments" (list[str]): A list of individual markdown-formatted
                                        links for each grounding chunk.
              - "segment_string" (str): A concatenated string of all markdown-
                                        formatted links for the citation.
              Returns an empty list if no valid candidates or grounding supports
              are found, or if essential data is missing.
    """
    citations = []

    # Ensure response and necessary nested structures are present
    if not response or not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
        return citations

    response_part = response.candidates[0].content.parts[0]
    if not hasattr(response_part, "grounding_metadata") or not response_part.grounding_metadata:
        return citations

    # The following access to `grounding_supports` and `grounding_chunks` is based on an older structure.
    # This needs to be verified against the `web_search_results` or similar in the current API.
    # For instance, `web_search_results` is a list of search results, each might have its own title, URI, and snippets.
    # The concept of 'grounding_supports' tied to specific text segments in the LLM *response*
    # might be different when using `google_search_retrieval` if it primarily returns search snippets
    # rather than a modified LLM response with inline citations.

    # Placeholder: If grounding_supports is still the relevant field
    grounding_supports = response_part.grounding_metadata.grounding_supports if hasattr(response_part.grounding_metadata, "grounding_supports") else []
    grounding_chunks = response_part.grounding_metadata.grounding_chunks if hasattr(response_part.grounding_metadata, "grounding_chunks") else []


    for support in grounding_supports: # This loop structure might be entirely wrong for google_search_retrieval
        citation = {}

        if not hasattr(support, "segment") or support.segment is None:
            continue

        start_index = support.segment.start_index if support.segment.start_index is not None else 0
        if support.segment.end_index is None:
            continue

        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index

        citation["segments"] = []
        if hasattr(support, "grounding_chunk_indices") and support.grounding_chunk_indices:
            for ind in support.grounding_chunk_indices:
                try:
                    # This assumes `grounding_chunks` is a flat list accessible by index,
                    # and that each chunk has `.web.uri` and `.web.title`.
                    if ind < len(grounding_chunks):
                        chunk = grounding_chunks[ind]
                        resolved_url = resolved_urls_map.get(getattr(getattr(chunk, "web", object()), "uri", None), None)
                        title = getattr(getattr(chunk, "web", object()), "title", "")
                        label = title.split(".")[0] if title else "Source" # Simplified label
                        if resolved_url:
                            citation["segments"].append(
                                {
                                    "label": label,
                                    "short_url": resolved_url,
                                    "value": getattr(getattr(chunk, "web", object()), "uri", None),
                                }
                            )
                except (AttributeError, NameError): # Removed IndexError as we check length
                    pass # Skip problematic chunks
        if citation["segments"]: # Only add citation if it has valid segments
            citations.append(citation)
    return citations
