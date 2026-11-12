"""
Run the GPU test workflow on a Modal GPU, against the working tree as it is.

The workflow is .github/workflows/test-gpu.yml, and uncommitted changes are
included.

    modal run ci/modal_gpu.py                                  # the workflow's stable job, as CI runs it
    modal run ci/modal_gpu.py --pytest-args "tests/api/test_pp.py -x"   # appended to its pytest command
    modal run ci/modal_gpu.py --gpu L4 --cpu 8 --memory 32768  # another machine than the CI runner
    GATAC_ENV=hatch-test.dev-13 modal run ci/modal_gpu.py      # another hatch-test environment
    modal run ci/modal_gpu.py::build                           # build the image only, on CPU
    modal run ci/modal_gpu.py --startup-deadline 600           # allow a slower start
    python ci/modal_gpu.py                                     # print what was read from the workflow

`modal run` needs pyyaml next to modal, e.g. `uvx --with pyyaml modal run ci/modal_gpu.py`,
and hatch, run with uvx unless it is on the PATH.

From the workflow's `test-stable-dev` job, this takes
- the machine: the runner label of its get-envs job, looked up in .cirun.yml
  (g4dn.xlarge: a T4, 4 cores, 16 GiB, the last as a memory limit, so a test
  that needs more fails here as it would in CI);
- timeout-minutes and its env;
- its `run:` steps, run in order with `bash -eo pipefail` as the non-root
  user `runner`, in /home/runner/work/gatac, as on a GitHub runner. A
  step whose `if` has always() or !cancelled() still runs after a failure.
  `hatch run "${ENV_NAME}:<script>"` runs the environment's script directly:
  the image's Python is the environment.

From the hatch-test environment (GATAC_ENV, by default hatch-test.stable-13,
the job CI requires), as `hatch env show --json` resolves it: its Python, its
CUDA extra, its dependencies, dependency groups, env-vars and scripts.

and stands in for the rest:
- `uses:` steps are not run. The working tree is mounted (checkout), the
  image has the Python (setup-uv), the JIT caches persist in the Modal Volume
  gatac-gpu-tests-jit-cache (actions/cache; `modal volume delete` it to start
  over), and nothing is uploaded (codecov).
- The step that creates the hatch environment is not run. Its dependencies
  are one image layer, installed by `uv pip install` (CI resolves them fresh
  too: uv.lock is not committed), and rebuilt only when one of those lists
  changes. gatac is installed at container start, without dependencies, so
  editing it or the tests rebuilds nothing.
- Steps that call git are not run: .git is not mounted. The one that
  restores mtimes for Numba's cache gets the working tree's own instead.
- Steps that set up the runner ($GITHUB_PATH, $GITHUB_ENV) are not run:
  the image has its own CUDA.

A container that crashes while it starts (before the function runs) is
rescheduled by Modal again and again; ``retries`` does not cover that. So
the function reports that it started, and the run is cancelled if no
container does so within ``--startup-deadline`` seconds.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github/workflows/test-gpu.yml"
JOB = "test-stable-dev"
ENV = "hatch-test.stable-13"  # the hatch-test environment of the job CI requires
SRC = "/root/src"  # the working tree, mounted read-only
USER = "runner"  # the steps run as this user, as on a GitHub runner, not as root
WORK = f"/home/{USER}/work/gatac"  # its writable copy, where the steps run: the runner's github.workspace
CACHE = "/cache"  # the Volume behind actions/cache, one archive per environment
IGNORE = ["**/__pycache__", "**/*.pyc"]

# The base image of each extra; its CUDA must not be newer than Modal's driver
BASE_IMAGES = {
    "rapids-cu12": "nvidia/cuda:12.8.1-runtime-ubuntu22.04",
    "rapids-cu13": "nvidia/cuda:13.0.2-runtime-ubuntu22.04",
}
# EC2 GPU family -> (Modal GPU, GiB per vCPU); an xlarge has 4 vCPUs, each size up doubles them
EC2_GPUS = {"g4dn": ("T4", 4), "g5": ("A10G", 4), "g6": ("L4", 4), "g6e": ("L40S", 8)}


def expand(text, values: dict[str, str]) -> str:
    """Replace each ${{ name }} by values[name]; any other expression is an error."""

    def value(match):
        if match[1] not in values:
            raise ValueError(f"{WORKFLOW.name}: no value for ${{{{ {match[1]} }}}} outside GitHub")
        return values[match[1]]

    return re.sub(r"\$\{\{\s*([\w.-]+)\s*\}\}", value, str(text))


def hatch_env(name: str) -> dict:
    """The hatch environment, its matrix and overrides resolved by `hatch env show --json`."""
    hatch = ["hatch"] if shutil.which("hatch") else ["uvx", "hatch"]
    out = subprocess.run([*hatch, "env", "show", "--json"], cwd=ROOT, check=True, capture_output=True, text=True)
    envs = json.loads(out.stdout)
    if name not in envs:
        tests = ", ".join(e for e in envs if e.startswith("hatch-test"))
        sys.exit(f"No hatch environment {name!r}; the test environments: {tests}")
    return envs[name]


def dependencies(env: dict) -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())

    def group(name):
        for dep in project["dependency-groups"][name]:
            yield from group(dep["include-group"]) if isinstance(dep, dict) else [dep]

    return (
        project["project"]["dependencies"]
        + [d for e in env.get("features", []) for d in project["project"]["optional-dependencies"][e]]
        + [d for g in env.get("dependency-groups", []) for d in group(g)]
        + env.get("dependencies", [])
        + env.get("extra-dependencies", [])
        + project["build-system"]["requires"]
    )


def read_workflow(env_name: str | None = None) -> dict:
    import yaml  # only needed locally

    env_name = env_name or ENV
    env = hatch_env(env_name)
    cuda = next((f for f in env.get("features", []) if f in BASE_IMAGES), None)
    if cuda is None:
        sys.exit(f"{env_name}: none of the CUDA extras {list(BASE_IMAGES)} among its features")
    scripts = {k: " && ".join(v) if isinstance(v, list) else v for k, v in env.get("scripts", {}).items()}
    env_vars = env.get("env-vars", {})

    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    job = jobs[JOB]
    # The job's matrix is the JSON that get-envs prints, the runner label in it
    get_envs = "\n".join(step.get("run", "") for step in jobs["get-envs"]["steps"])
    label = re.search(r'runner:\s*"([^"]+)"', get_envs)[1]
    values = {
        "matrix.env.name": env_name,
        "matrix.env.python": env["python"],
        "matrix.env.runner": label,
        "github.workspace": WORK,
    }

    cirun = yaml.safe_load((ROOT / ".cirun.yml").read_text())
    runner = next(r for r in cirun["runners"] if label in r["labels"])
    family, size = runner["instance_type"].split(".")
    gpu, gib_per_cpu = EC2_GPUS[family]
    cpu = 4 * int(size.removesuffix("xlarge") or 1)

    steps, skipped = [], []
    for step in job["steps"]:
        name = step.get("name") or step.get("uses") or step["run"].strip().splitlines()[0]
        if "uses" in step:
            skipped.append((name, "an action"))
            continue
        script = expand(step["run"], values)
        if re.search(r"\bhatch\b.*\benv create\b", script):
            skipped.append((name, "its dependencies are in the image"))
            continue
        if re.search(r"(^|[\s;&|(])git\s", script):
            skipped.append((name, "calls git, and .git is not mounted"))
            continue
        if re.search(r"\$\{?GITHUB_(PATH|ENV)\b", script):
            skipped.append((name, "sets up the runner"))
            continue
        script = re.sub(r'\bhatch run "?\$\{?ENV_NAME\}?:([\w.-]+)"?', lambda m: scripts.get(m[1], m[1]), script)
        if re.search(r"\bhatch\b", script):
            sys.exit(f'{WORKFLOW.name}: step {name!r} calls hatch other than as `hatch run "${{ENV_NAME}}:<script>"`')
        condition = str(step.get("if", ""))
        steps.append(
            {
                "name": name,
                "run": script,
                "always": "always()" in condition or "!cancelled()" in condition,
                "env": {k: expand(v, values) for k, v in step.get("env", {}).items()},
            }
        )

    deps = dependencies(env)
    return {
        "env_name": env_name,
        "cuda": cuda,
        "python": env["python"],
        "base_image": BASE_IMAGES[cuda],
        "dependencies": deps,
        "git": any("git+" in d for d in deps),
        # uv reads these when it installs (rapids_prerelease: the nightly index, prereleases)
        "uv_env": {k: v for k, v in env_vars.items() if k.startswith("UV_")},
        "gpu": gpu,
        "cpu": cpu,
        "memory": cpu * gib_per_cpu * 1024,
        "timeout": 60 * int(job.get("timeout-minutes", 360)),
        "env": {k: expand(v, values) for k, v in job.get("env", {}).items()} | env_vars,
        "steps": steps,
        "skipped": skipped,
    }


def with_pytest_args(steps: list[dict], args: str) -> list[dict]:
    """Append args to the pytest command of the steps."""
    pytest = [s for s in steps if re.search(r"\bpytest\b", s["run"])]
    if len(pytest) != 1:
        sys.exit(f"--pytest-args: expected one pytest step in {WORKFLOW.name}, found {len(pytest)}")
    step = pytest[0]
    run = re.sub(r"^(.*\bpytest\b.*)$", lambda m: f"{m[1]} {args}", step["run"], count=1, flags=re.MULTILINE)
    return [s | {"run": run} if s is step else s for s in steps]


def describe(ci: dict) -> None:
    print(
        f"{WORKFLOW.relative_to(ROOT)}, job {JOB!r}, {ci['env_name']} ({ci['cuda']}, Python {ci['python']}): "
        f"{ci['gpu']}, {ci['cpu']} cores, {ci['memory']} MiB, {ci['timeout'] // 60} min"
    )
    for name, why in ci["skipped"]:
        print(f"  skip  {name} ({why})")
    for step in ci["steps"]:
        print(f"  run   {step['name']}{' (even after a failure)' if step['always'] else ''}")


# Modal re-imports this module inside the container, where the script sits at
# /root/modal_gpu.py and ROOT is "/": the workflow is read and the image defined locally only.
CI = read_workflow(os.environ.get("GATAC_ENV")) if modal.is_local() else None
image = (
    (
        modal.Image.from_registry(CI["base_image"], add_python=CI["python"])
        # for the dependencies installed from git (the dev environment's anndata and scanpy)
        .apt_install(*["git"] * CI["git"])
        .uv_pip_install(*CI["dependencies"], env=CI["uv_env"] or None)
        # hatchling's editable build needs it, and --no-build-isolation does not
        # install it; a layer of its own, so the one above stays cached
        .uv_pip_install("editables")
        .run_commands(f"useradd --create-home --uid 1001 {USER}")  # the uid of GitHub's runner user
        .add_local_file(ROOT / "pyproject.toml", f"{SRC}/pyproject.toml")
        # the package metadata names these, and hatchling checks they exist
        .add_local_file(ROOT / "README.md", f"{SRC}/README.md")
        .add_local_file(ROOT / "LICENSE", f"{SRC}/LICENSE")
        .add_local_dir(ROOT / "src", f"{SRC}/src", ignore=IGNORE)
        .add_local_dir(ROOT / "tests", f"{SRC}/tests", ignore=IGNORE)
    )
    if CI
    else modal.Image.debian_slim()
)

app = modal.App("gatac-gpu-tests", image=image)
jit_cache = modal.Volume.from_name("gatac-gpu-tests-jit-cache", create_if_missing=True)


# The machine and timeout are the workflow's, set with with_options in main
@app.function(gpu="T4", volumes={CACHE: jit_cache})
def job(
    steps: list[dict], env: dict[str, str], mtimes: dict[str, float], cache_key: str, started: modal.Queue | None = None
) -> int:
    import shutil
    import subprocess

    if started is not None:
        started.put(True)
    subprocess.run(
        "echo \"$(nproc) cores visible: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)\"", shell=True, check=False
    )

    # Checkout: a writable copy, with the local mtimes on which Numba's cache is keyed
    shutil.copytree(SRC, WORK)
    for path, mtime in mtimes.items():
        os.utime(f"{WORK}/{path}", (mtime, mtime))
    # One archive, as actions/cache: the Volume reads each file at a few per
    # second, and the caches are over a thousand small files
    cache = Path(CACHE, f"{cache_key}.tar")
    if cache.is_file():
        subprocess.run(["tar", "-xf", cache, "-C", WORK], check=True)
    # A file, as on a runner: a non-root user cannot reopen root's stdout as /dev/stdout
    summary = Path(f"/home/{USER}/work/_temp/step_summary.md")
    summary.parent.mkdir(parents=True)
    summary.touch()

    home = f"/home/{USER}"
    env = {
        **os.environ,
        **env,
        "CI": "true",
        "GITHUB_WORKSPACE": WORK,
        "GITHUB_STEP_SUMMARY": str(summary),
        "HOME": home,
        "USER": USER,
        "LOGNAME": USER,
        "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    env.pop("PYTHONPATH", None)  # Modal's runtime, in /pkg, which only root can read
    # Editable, as hatch installs it: gatac is imported from WORK, where
    # Numba's cache and coverage expect it. .git is not mounted, so hatch-vcs
    # gets the version from the environment. As root, into the image's Python.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "-e", WORK],
        env=env | {"SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0+modal"},
        check=True,
    )
    subprocess.run(["chown", "-R", f"{USER}:{USER}", f"{home}/work"], check=True)

    rc = 0
    for step in steps:
        if rc and not step["always"]:
            print(f"\n=== {step['name']}: skipped after the failure", flush=True)
            continue
        print(f"\n=== {step['name']}", flush=True)
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
            cwd=WORK,
            env=env | step["env"],
            user=USER,
            group=USER,
            extra_groups=[],
            check=False,
        )
        if result.returncode:
            print(f"=== {step['name']}: failed with exit code {result.returncode}", flush=True)
            rc = rc or result.returncode

    if text := summary.read_text():
        print("\n=== Step summary", text, sep="\n", flush=True)

    if Path(WORK, ".cache").is_dir():
        subprocess.run(["tar", "-cf", "/tmp/cache.tar", "-C", WORK, ".cache"], check=True)
        shutil.copyfile("/tmp/cache.tar", cache)
        jit_cache.commit()
    return rc


@app.function()
def build():
    """Nothing to run: invoking it builds and caches the image without a GPU."""


@app.local_entrypoint()
def main(pytest_args: str = "", gpu: str = "", cpu: float = 0, memory: int = 0, startup_deadline: int = 300):
    """gpu, cpu and memory (MiB) default to the workflow's runner."""
    import queue

    ci = CI | {"gpu": gpu or CI["gpu"], "cpu": cpu or CI["cpu"], "memory": memory or CI["memory"]}
    describe(ci)
    steps = with_pytest_args(ci["steps"], pytest_args) if pytest_args else ci["steps"]
    mtimes = {str(p.relative_to(ROOT)): p.stat().st_mtime for p in (ROOT / "src").rglob("*.py")}

    with modal.Queue.ephemeral() as started:
        call = job.with_options(
            gpu=ci["gpu"],
            cpu=ci["cpu"],
            memory=(ci["memory"], ci["memory"]),
            timeout=ci["timeout"],
        ).spawn(steps, ci["env"], mtimes, ci["env_name"], started)
        try:
            started.get(timeout=startup_deadline)
        except queue.Empty:
            call.cancel(terminate_containers=True)
            sys.exit(
                f"No container started within {startup_deadline} s, so the run was "
                "cancelled; the logs above show why (raise --startup-deadline if "
                "a GPU was only slow to come up)."
            )
        rc = call.get()
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    describe(CI)
    print("\nenv:", *(f"  {k}={v}" for k, v in CI["env"].items()), sep="\n")
    for step in CI["steps"]:
        print(f"\n--- {step['name']}\n{step['run'].rstrip()}")
    print(
        f"\nimage: {CI['base_image']} + Python {CI['python']}, {len(CI['dependencies'])} requirements:",
        *CI["dependencies"],
    )
