"""main.py must load .env into the real process environment on import.

Settings() reads HARNESS_-prefixed keys straight from the file itself, but
_build_callbacks() reads LANGFUSE_* directly via os.environ.get() — those
keys only become visible if something loads the file into os.environ first.
Docker's env_file: only does this for the containerized api service; `make
api` (bare uvicorn) has no other mechanism.

python-dotenv's load_dotenv() walks upward from the calling file's location
— but only when Python can see a real __main__.__file__ (a script run as
`python foo.py`, matching how `uvicorn` actually launches main.py). Under
`python -c "..."` there is no such file, so it silently falls back to
cwd-based search instead — a difference that would make a `-c`-based test
pass or fail for the wrong reason. These tests drive a real script file
through a throwaway fake package tree (mirroring src/harness/api/main.py's
depth under the repo root) so the exercised code path matches production,
fully isolated from this machine's real .env (gitignored, absent on a fresh
clone or in CI).
"""
import os
import subprocess
import sys
from pathlib import Path


def _build_fake_package(tmp_path: Path) -> str:
    """<tmp_path>/src/pkg/mod.py calls load_dotenv(), mirroring main.py's
    depth under the repo root (src/harness/api/main.py)."""
    pkg_dir = tmp_path / "src" / "pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "mod.py").write_text(
        "from dotenv import load_dotenv\nload_dotenv(override=False)\n"
    )
    return str(tmp_path / "src")


def _run_import(tmp_path: Path, src_path: str, extra_env: dict) -> subprocess.CompletedProcess:
    # A real script file, not -c: matches how uvicorn actually launches
    # main.py (proper __main__.__file__), so find_dotenv() takes the same
    # frame-walking path production does, not the cwd-fallback for -c/REPL.
    runner = tmp_path / "run.py"
    runner.write_text(
        "import pkg.mod\nimport os\nprint(os.environ.get('LANGFUSE_HOST', ''))\n"
    )
    # Strip LANGFUSE_HOST inherited from the parent pytest process — earlier
    # tests importing pymilvus (which calls load_dotenv() itself) can leave
    # the real repo's .env values sitting in this process's os.environ, and
    # override=False would then preserve that stale value over the fake
    # .env this test just wrote.
    clean_env = {k: v for k, v in os.environ.items() if k != "LANGFUSE_HOST"}
    return subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True,
        text=True,
        env={**clean_env, "PYTHONPATH": src_path, **extra_env},
        timeout=30,
    )


def test_import_loads_dotenv_from_the_package_root(tmp_path):
    src_path = _build_fake_package(tmp_path)
    (tmp_path / ".env").write_text("LANGFUSE_HOST=http://from-dotenv:3000\n")

    result = _run_import(tmp_path, src_path, {})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "http://from-dotenv:3000"


def test_real_shell_export_beats_dotenv_file(tmp_path):
    src_path = _build_fake_package(tmp_path)
    (tmp_path / ".env").write_text("LANGFUSE_HOST=http://from-dotenv:3000\n")

    result = _run_import(tmp_path, src_path, {"LANGFUSE_HOST": "http://from-shell:3000"})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "http://from-shell:3000"


def test_main_py_actually_calls_load_dotenv():
    """Cheap regression guard: the real main.py must not lose this call."""
    main_py = Path(__file__).parents[2] / "src" / "harness" / "api" / "main.py"
    source = main_py.read_text()
    assert "load_dotenv" in source
