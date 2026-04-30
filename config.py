"""Configuration for the application."""

from typing import List, Dict

# Repository configurations to fetch and process
REPOSITORIES: List[Dict[str, str]] = [
    #{"owner": "developmentseed", "name": "lonboard"},
    {"owner": "nimaous", "name": "Ad-Tech-For-ML-Practitioners-"},
]

# Logging configuration
LOG_LEVEL = "INFO"

# Document processing configuration
# Split strategy: "markdown", "sliding_window", or "intelligent"
SPLIT_STRATEGY = "markdown"

# For "markdown" strategy: header level to split on
MARKDOWN_HEADER_LEVEL = 2

# For "sliding_window" strategy: chunk size and step
SLIDING_WINDOW_SIZE = 500
SLIDING_WINDOW_STEP = 250

# For "intelligent" strategy: LLM-based chunking prompt template
INTELLIGENT_CHUNKING_PROMPT = """
Split the provided document into logical sections
that make sense for a Q&A system.

Each section should be self-contained and cover
a specific topic or concept.

<DOCUMENT>
{document}
</DOCUMENT>

Use this format:

## Section Name

Section content with all relevant details

---

## Another Section Name

Another section content

---
""".strip()
