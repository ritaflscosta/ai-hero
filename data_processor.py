"""Data processing for fetching and transforming repository data."""

import io
import logging
import zipfile
from typing import List, Dict, Any, Callable

import frontmatter
import requests

from text_utils import split_markdown_by_level, sliding_window, intelligent_chunking

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles fetching and processing documents from GitHub repositories."""

    BASE_URL = "https://codeload.github.com"
    MARKDOWN_EXTENSIONS = (".md", ".mdx")

    def __init__(self, llm_function: Callable[[str], str] = None):
        """
        Initialize DocumentProcessor.

        Args:
            llm_function: Optional LLM function for intelligent chunking.
        """
        self.llm_function = llm_function

    def fetch_repo_data(self, repo_owner: str, repo_name: str) -> List[Dict[str, Any]]:
        """
        Download and parse all markdown files from a GitHub repository.

        Args:
            repo_owner: GitHub username or organization.
            repo_name: Repository name.

        Returns:
            List of dictionaries containing file content and metadata.

        Raises:
            Exception: If the repository download fails.
        """
        url = f"{self.BASE_URL}/{repo_owner}/{repo_name}/zip/refs/heads/main"

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to download repository {repo_owner}/{repo_name}: {e}")
            raise

        repository_data = []
        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))

            for file_info in zf.infolist():
                filename = file_info.filename
                filename_lower = filename.lower()

                if not filename_lower.endswith(self.MARKDOWN_EXTENSIONS):
                    continue

                try:
                    with zf.open(file_info) as f_in:
                        content = f_in.read().decode("utf-8", errors="ignore")
                        post = frontmatter.loads(content)
                        data = post.to_dict()
                        data["filename"] = filename
                        repository_data.append(data)
                except Exception as e:
                    logger.warning(f"Error processing {filename}: {e}")
                    continue

            zf.close()
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file for {repo_owner}/{repo_name}: {e}")
            raise

        logger.info(f"Loaded {len(repository_data)} documents from {repo_owner}/{repo_name}")
        return repository_data

    def split_documents(
        self,
        documents: List[Dict[str, Any]],
        strategy: str = "markdown",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Split documents into sections using specified strategy.

        Args:
            documents: List of document dictionaries with 'content' key.
            strategy: Splitting strategy - "markdown", "sliding_window", or "intelligent".
            **kwargs: Strategy-specific parameters:
                - For "markdown": header_level (default: 2)
                - For "sliding_window": chunk_size (default: 500), chunk_step (default: 250)
                - For "intelligent": prompt_template (required)

        Returns:
            List of document sections with metadata preserved.

        Raises:
            ValueError: If invalid strategy is specified or required params are missing.
        """
        if strategy not in ("markdown", "sliding_window", "intelligent"):
            raise ValueError(
                f"Invalid strategy: {strategy}. Must be 'markdown', 'sliding_window', or 'intelligent'"
            )

        if strategy == "intelligent" and self.llm_function is None:
            raise ValueError("LLM function required for 'intelligent' strategy")

        sections = []

        for doc in documents:
            doc_copy = doc.copy()
            doc_content = doc_copy.pop("content", "")

            if not doc_content:
                logger.warning(f"Document {doc.get('filename')} has no content")
                continue

            if strategy == "markdown":
                section_list = self._split_by_markdown(doc_content, **kwargs)
            elif strategy == "sliding_window":
                section_list = self._split_by_sliding_window(doc_content, **kwargs)
            else:  # intelligent
                section_list = self._split_by_intelligent(doc_content, **kwargs)

            for section in section_list:
                section_doc = doc_copy.copy()
                section_doc["section"] = section
                sections.append(section_doc)

        logger.info(
            f"Split {len(documents)} documents into {len(sections)} sections using '{strategy}' strategy"
        )
        return sections

    @staticmethod
    def _split_by_markdown(content: str, header_level: int = 2) -> List[str]:
        """
        Split content by markdown header level.

        Args:
            content: Markdown content as string.
            header_level: Header level to split on (default: 2).

        Returns:
            List of sections.
        """
        return split_markdown_by_level(content, level=header_level)

    @staticmethod
    def _split_by_sliding_window(
        content: str, chunk_size: int = 500, chunk_step: int = 250
    ) -> List[str]:
        """
        Split content using overlapping sliding window chunks.

        Args:
            content: Content as string.
            chunk_size: Size of each chunk in characters (default: 500).
            chunk_step: Step size between chunks in characters (default: 250).

        Returns:
            List of overlapping text chunks.
        """
        chunks = sliding_window(content, size=chunk_size, step=chunk_step)
        return [chunk_dict["chunk"] for chunk_dict in chunks]

    def _split_by_intelligent(self, content: str, prompt_template: str = None) -> List[str]:
        """
        Split content using LLM-based intelligent chunking.

        Args:
            content: Content as string.
            prompt_template: Prompt template with {document} placeholder.

        Returns:
            List of intelligently split sections.

        Raises:
            ValueError: If prompt_template is not provided.
        """
        if prompt_template is None:
            raise ValueError("prompt_template required for intelligent chunking")

        return intelligent_chunking(content, self.llm_function, prompt_template)
