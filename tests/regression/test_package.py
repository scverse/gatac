"""What ``import gatac`` does to the process: logging and progress bars."""
import logging
import subprocess
import sys

import gatac
from gatac._progress import show_progress, tqdm

LOG = "import logging, os, sys\n{config}\nimport gatac\nlogging.getLogger('gatac.pp').info('hello')\n"


def _stderr(config: str) -> str:
    """stderr of a fresh process that imports gatac after ``config`` and logs one INFO message."""
    code = LOG.format(config=config)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True).stderr


def test_info_shown_once_and_through_the_applications_logging():
    # not configured: gatac's own handler; configured: the application's only
    assert _stderr("").count("gatac.pp - INFO - hello") == 1
    out = _stderr("logging.basicConfig(format='APP %(name)s %(message)s')")
    assert out.count("hello") == 1 and "APP gatac.pp hello" in out


def test_caplog_sees_gatac_messages(caplog):
    logging.getLogger("gatac.io").info("captured")
    assert "captured" in caplog.text


def test_import_leaves_tqdm_of_other_libraries_alone(monkeypatch):
    code = "import os, gatac; print(os.environ.get('TQDM_DISABLE'))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "None"
    # gatac's own bars: off when stderr is no terminal (here, captured), or with TQDM_DISABLE
    assert not show_progress() and tqdm(range(3)).disable
    monkeypatch.setenv("TQDM_DISABLE", "1")
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    assert not show_progress()
    assert gatac.__version__
