"""The Python the READMEs show, run as written.

Each test takes a fenced block from a README and executes it in a directory holding the files the block names:
`model.safetensors`, the checkpoint the suites share, and `cat.png`, a small valid PNG. A Quickstart runs
unchanged; a showcase snippet is a fragment, so its test supplies the `import uform` the page made above. On the
stub library every caption ties, so the tests check that the block runs and prints, not which caption wins.

File: test/python/quickstart.py
Author: Ash Vardanian
Date: November 16, 2026
"""

from pathlib import Path

import pytest
from encode import png

ROOT = Path(__file__).resolve().parents[2]


def block(readme: str, section: str, position: int = 0) -> str:
    """The `position`-th ```python block of `## section` in `readme`, relative to the repository root."""
    text = (ROOT / readme).read_text(encoding="utf-8")
    body = text.split(f"\n## {section}\n", 1)[1].split("\n## ", 1)[0]
    return body.split("```python\n")[position + 1].split("\n```", 1)[0]


@pytest.fixture
def workspace(tmp_path: Path, checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory holding `model.safetensors` and `cat.png`, made the working directory."""
    (tmp_path / "model.safetensors").write_bytes(checkpoint.read_bytes())
    (tmp_path / "cat.png").write_bytes(png(32))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("readme", ["README.md", "python/README.md"])
def test_quickstart(readme: str, workspace: Path, capsys: pytest.CaptureFixture[str]):
    """The Quickstart encodes two captions and a photo and names the caption closest to the photo."""
    exec(compile(block(readme, "Quickstart"), readme, "exec"), {})
    printed = capsys.readouterr().out
    assert printed.endswith("describes the photo best\n")
    assert printed.startswith(("'a cat on a mat'", "'a sunrise over the bay'"))


def test_models(workspace: Path, capsys: pytest.CaptureFixture[str]):
    """The Models snippet loads a Float8 model and prints its description."""
    exec(compile("import uform\n" + block("python/README.md", "Models"), "python/README.md", "exec"), {})
    assert len(capsys.readouterr().out.split()) == 4
