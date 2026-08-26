import json

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.history import PaperHistory, paper_key


def test_paper_key_normalizes_arxiv_version_and_pdf_url():
    abstract_url = make_sample_paper(url="http://arxiv.org/abs/2608.21362v1")
    pdf_url = make_sample_paper(url="https://arxiv.org/pdf/2608.21362v3.pdf")
    assert paper_key(abstract_url) == "arxiv:2608.21362"
    assert paper_key(pdf_url) == "arxiv:2608.21362"


def test_history_filters_seen_and_same_run_duplicates(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"version": 1, "seen": ["arxiv:2608.00001"]}))
    history = PaperHistory(path)
    papers = [
        make_sample_paper(url="https://arxiv.org/abs/2608.00001v2"),
        make_sample_paper(url="https://arxiv.org/abs/2608.00002v1"),
        make_sample_paper(url="https://arxiv.org/abs/2608.00002v3"),
    ]

    unseen = history.filter_unseen(papers)

    assert [paper_key(paper) for paper in unseen] == ["arxiv:2608.00002"]


def test_history_records_atomically_and_honors_limit(tmp_path):
    path = tmp_path / "nested" / "history.json"
    history = PaperHistory(path, max_entries=2)
    history.record([
        make_sample_paper(url="https://arxiv.org/abs/2608.00001v1"),
        make_sample_paper(url="https://arxiv.org/abs/2608.00002v1"),
        make_sample_paper(url="https://arxiv.org/abs/2608.00003v1"),
    ])

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "version": 1,
        "seen": ["arxiv:2608.00002", "arxiv:2608.00003"],
    }
    assert not path.with_suffix(".json.tmp").exists()


def test_history_recovers_from_invalid_file(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("not-json", encoding="utf-8")
    history = PaperHistory(path)

    assert history.keys == []
    assert len(history.filter_unseen([make_sample_paper()])) == 1
