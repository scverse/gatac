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
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stderr


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


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_submodules_are_imported_on_first_use():
    out = _run(
        "import sys, gatac\n"
        "assert 'gatac.pp' not in sys.modules and 'cudf' not in sys.modules\n"
        "from gatac import io\n"
        "assert gatac.pp.combine and 'gatac.pp' in sys.modules and io is gatac.io\n"
        "assert {'io', 'pp', 'gpu'} <= set(dir(gatac))"
    )
    assert out.returncode == 0, out.stderr


NO_RAPIDS = "import sys\nsys.modules['cudf'] = None  # as if not installed\n"


def test_without_rapids_the_cli_parses_and_the_modules_say_what_to_install():
    out = _run(NO_RAPIDS + "import gatac.cli\nsys.argv = ['gatac', '--help']\ngatac.cli.main()")
    assert out.returncode == 0 and "convert" in out.stdout, out.stderr
    for code in ("import gatac; gatac.pp", "import gatac.io", "from gatac import gpu"):
        out = _run(NO_RAPIDS + code)
        assert out.returncode == 1 and "pip install 'gatac[cuda12]'" in out.stderr, (code, out.stderr)
