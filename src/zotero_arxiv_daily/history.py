import json
import re
from pathlib import Path

from loguru import logger

from .protocol import Paper


_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


def paper_key(paper: Paper) -> str:
    """Return a stable key, ignoring arXiv version-only URL changes."""
    match = _ARXIV_URL_RE.search(paper.url)
    if match:
        arxiv_id = match.group(1).removesuffix(".pdf")
        arxiv_id = _ARXIV_VERSION_RE.sub("", arxiv_id)
        return f"arxiv:{arxiv_id.lower()}"
    return f"{paper.source.lower()}:{paper.url.strip()}"


class PaperHistory:
    def __init__(self, path: str | Path, max_entries: int = 5000):
        self.path = Path(path)
        self.max_entries = max_entries
        self.keys = self._load()
        self._key_set = set(self.keys)

    def _load(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            keys = data.get("seen", []) if isinstance(data, dict) else data
            if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
                raise ValueError("history must contain a list of string keys")
            return keys[-self.max_entries:]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not read paper history from {self.path}: {exc}")
            return []

    def filter_unseen(self, papers: list[Paper]) -> list[Paper]:
        unseen = []
        current_keys = set()
        for paper in papers:
            key = paper_key(paper)
            if key in self._key_set or key in current_keys:
                continue
            current_keys.add(key)
            unseen.append(paper)
        return unseen

    def record(self, papers: list[Paper]) -> None:
        new_keys = list(dict.fromkeys(paper_key(paper) for paper in papers))
        if not new_keys:
            return

        new_key_set = set(new_keys)
        self.keys = [key for key in self.keys if key not in new_key_set]
        self.keys.extend(new_keys)
        self.keys = self.keys[-self.max_entries:]
        self._key_set = set(self.keys)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "seen": self.keys}
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
