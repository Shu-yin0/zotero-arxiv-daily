from .base import BaseRetriever, register_retriever
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from dataclasses import dataclass
from tempfile import TemporaryDirectory
import feedparser
import multiprocessing
import os
from queue import Empty
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


@dataclass
class _RssAuthor:
    name: str


@dataclass
class _RssArxivResult:
    title: str
    authors: list[_RssAuthor]
    summary: str
    pdf_url: str
    entry_id: str
    paper_id: str

    def source_url(self) -> str:
        return f"https://arxiv.org/e-print/{self.paper_id}"


def _rss_entry_to_result(entry: Any) -> _RssArxivResult:
    paper_id = entry.id.removeprefix("oai:arXiv.org:")
    summary = entry.get("summary", "")
    if "Abstract:" in summary:
        summary = summary.split("Abstract:", 1)[1].strip()

    creator = entry.get("dc_creator") or entry.get("creator") or entry.get("author") or ""
    author_names = [name.strip() for name in creator.split(",") if name.strip()]
    if not author_names:
        author_names = [author.get("name", "").strip() for author in entry.get("authors", [])]
        author_names = [name for name in author_names if name]

    entry_id = next(
        (
            link.get("href")
            for link in entry.get("links", [])
            if link.get("rel") == "alternate" and link.get("href")
        ),
        f"https://arxiv.org/abs/{paper_id}",
    )
    return _RssArxivResult(
        title=entry.title,
        authors=[_RssAuthor(name) for name in author_names],
        summary=summary,
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        entry_id=entry_id,
        paper_id=paper_id,
    )


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # The RSS feed already contains title, authors, abstract, and paper ID.
        # Using it directly avoids the heavily rate-limited export.arxiv.org API.
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        feed_title = feed.get("feed", {}).get("title", "")
        if "Feed error for query" in feed_title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        if not feed.entries:
            raise RuntimeError(f"arXiv RSS returned no entries for query: {query}")

        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        entries = [
            entry
            for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            entries = entries[:10]

        return [_rss_entry_to_result(entry) for entry in entries]

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
