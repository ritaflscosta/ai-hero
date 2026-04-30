"""Utilities for text processing and markdown manipulation."""

import re
from typing import List, Dict, Any, Callable


def sliding_window(seq: List[str], size: int, step: int) -> List[Dict[str, Any]]:
    """
    Create a sliding window over a sequence.
    
    Args:
        seq: The sequence to chunk.
        size: The size of each chunk.
        step: The step size between chunks.
    
    Returns:
        List of dictionaries with 'start' position and 'chunk' content.
    
    Raises:
        ValueError: If size or step is not positive.
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i : i + size]
        result.append({"start": i, "chunk": chunk})
        if i + size >= n:
            break

    return result


def split_markdown_by_level(text: str, level: int = 2) -> List[str]:
    """
    Split markdown text by a specific header level.
    
    Args:
        text: Markdown text as a string.
        level: Header level to split on (default: 2 for "##").
    
    Returns:
        List of sections as strings, each starting with the header.
    """
    # This regex matches markdown headers
    # For level 2, it matches lines starting with "## "
    header_pattern = r"^(#{" + str(level) + r"} )(.+)$"
    pattern = re.compile(header_pattern, re.MULTILINE)

    # Split and keep the headers
    parts = pattern.split(text)

    sections = []
    for i in range(1, len(parts), 3):
        # We step by 3 because regex.split() with
        # capturing groups returns:
        # [before_match, group1, group2, after_match, ...]
        # here group1 is "## ", group2 is the header text
        header = parts[i] + parts[i + 1]  # "## " + "Title"
        header = header.strip()

        # Get the content after this header
        content = ""
        if i + 2 < len(parts):
            content = parts[i + 2].strip()

        if content:
            section = f"{header}\n\n{content}"
        else:
            section = header
        sections.append(section)

    return sections


def intelligent_chunking(
    text: str, llm_function: Callable[[str], str], prompt_template: str
) -> List[str]:
    """
    Use LLM to intelligently split text into logical sections.
    
    Args:
        text: The text to chunk.
        llm_function: Function that takes a prompt and returns LLM response.
        prompt_template: Template string with {document} placeholder for the text.
    
    Returns:
        List of text sections split by the LLM.
    """
    prompt = prompt_template.format(document=text)
    response = llm_function(prompt)
    sections = response.split("---")
    sections = [s.strip() for s in sections if s.strip()]
    return sections
