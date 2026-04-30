"""Main application for fetching and processing repository data with zLLM integration."""

import argparse
import logging
from typing import Optional

from call_zllm_api import initialize_zllm, llm
from config import (
    REPOSITORIES,
    LOG_LEVEL,
    SPLIT_STRATEGY,
    MARKDOWN_HEADER_LEVEL,
    SLIDING_WINDOW_SIZE,
    SLIDING_WINDOW_STEP,
    INTELLIGENT_CHUNKING_PROMPT,
)
from data_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_repositories(
    processor: DocumentProcessor,
    strategy: str = SPLIT_STRATEGY,
    **strategy_kwargs
) -> None:
    """
    Process all configured repositories: fetch, split into sections, and display stats.

    Args:
        processor: DocumentProcessor instance for fetching and processing documents.
        strategy: Splitting strategy ("markdown", "sliding_window", or "intelligent").
        **strategy_kwargs: Additional kwargs for the splitting strategy.
    """
    all_sections = []

    for repo_config in REPOSITORIES:
        repo_owner = repo_config["owner"]
        repo_name = repo_config["name"]

        try:
            logger.info(f"Processing {repo_owner}/{repo_name}...")
            documents = processor.fetch_repo_data(repo_owner, repo_name)

            sections = processor.split_documents(
                documents, strategy=strategy, **strategy_kwargs
            )
            all_sections.extend(sections)

            logger.info(
                f"✓ {repo_owner}/{repo_name}: {len(documents)} docs → {len(sections)} sections"
            )

        except Exception as e:
            logger.error(f"✗ Failed to process {repo_owner}/{repo_name}: {e}")
            continue

    logger.info(f"\nTotal sections processed: {len(all_sections)}")

    if all_sections:
        example_section = all_sections[3].get("section", "")
        logger.info(f"Example section preview:\n{example_section[:500]}...")


def demo_zllm_api() -> None:
    """Demonstrate zLLM API functionality with a sample prompt."""
    try:
        logger.info("Calling zLLM API with a sample prompt...")
        prompt = "What is the capital of France?"
        response = llm(prompt)
        logger.info(f"Response: {response}")
    except Exception as e:
        logger.error(f"Failed to call zLLM API: {e}")


def main(token: Optional[str] = None, strategy: Optional[str] = None, **strategy_kwargs) -> None:
    """
    Main application entry point.

    Args:
        token: Optional zToken for zLLM authentication.
        strategy: Optional splitting strategy override ("markdown", "sliding_window", or "intelligent").
        **strategy_kwargs: Additional parameters for the splitting strategy.
    """
    logger.info("Starting application...")

    # Use provided strategy or fall back to config
    splitting_strategy = strategy or SPLIT_STRATEGY

    # Prepare strategy kwargs based on strategy type
    if splitting_strategy == "markdown":
        split_kwargs = {"header_level": MARKDOWN_HEADER_LEVEL}
    elif splitting_strategy == "sliding_window":
        split_kwargs = {
            "chunk_size": SLIDING_WINDOW_SIZE,
            "chunk_step": SLIDING_WINDOW_STEP,
        }
    else:  # intelligent
        split_kwargs = {"prompt_template": INTELLIGENT_CHUNKING_PROMPT}

    # Override with any provided kwargs
    split_kwargs.update(strategy_kwargs)

    logger.info(f"Using splitting strategy: '{splitting_strategy}'")
    if splitting_strategy == "markdown":
        logger.info(f"  Header level: {split_kwargs['header_level']}")
    elif splitting_strategy == "sliding_window":
        logger.info(
            f"  Chunk size: {split_kwargs['chunk_size']}, Step: {split_kwargs['chunk_step']}"
        )
    else:
        logger.info(f"  Using LLM-based intelligent chunking")

    # Initialize zLLM if token is provided
    if token:
        try:
            initialize_zllm(token)
            logger.info("zLLM initialized with provided token")
            demo_zllm_api()
        except Exception as e:
            logger.error(f"Failed to initialize zLLM: {e}")
            if splitting_strategy == "intelligent":
                logger.error(
                    "Cannot proceed with 'intelligent' strategy without zLLM. "
                    "Please provide a valid token."
                )
                return
    else:
        if splitting_strategy == "intelligent":
            logger.error(
                "No zToken provided. 'intelligent' strategy requires zLLM. "
                "Please provide a token via --token or ZTOKEN environment variable."
            )
            return
        logger.info("No zToken provided; zLLM functionality will be skipped")

    # Process repositories
    logger.info("\nFetching and processing repositories...")
    processor = DocumentProcessor(llm_function=llm if token else None)
    process_repositories(processor, strategy=splitting_strategy, **split_kwargs)

    logger.info("Application completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process GitHub repository data")
    parser.add_argument(
        "--token",
        type=str,
        help="zToken for zLLM authentication (or set ZTOKEN environment variable)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["markdown", "sliding_window", "intelligent"],
        help="Document splitting strategy (default: from config)",
    )
    parser.add_argument(
        "--header-level",
        type=int,
        help="Header level for markdown strategy (default: from config)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size for sliding_window strategy (default: from config)",
    )
    parser.add_argument(
        "--chunk-step",
        type=int,
        help="Step size for sliding_window strategy (default: from config)",
    )

    args = parser.parse_args()

    # Build strategy kwargs from CLI args
    strategy_kwargs = {}
    if args.header_level:
        strategy_kwargs["header_level"] = args.header_level
    if args.chunk_size:
        strategy_kwargs["chunk_size"] = args.chunk_size
    if args.chunk_step:
        strategy_kwargs["chunk_step"] = args.chunk_step

    main(token=args.token, strategy=args.strategy, **strategy_kwargs)