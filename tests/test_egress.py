"""Egress allowlist tests for the sealed runner.

The compartment (the sandbox container created by ``runInSandbox`` in
``src/runner.js``) is sealed: it is created with ``NetworkMode: 'none'``,
so code inside it has no network at all. The only egress in the whole
pipeline happens on the host side and is limited to the endpoints in the
runner's own config:

* the bridge it polls            -- ``BRIDGE_BASE``
* the forge it ships claims to   -- ``FORGE_BASE``
* the git remote it clones/pushes -- the bounty ``repo_url`` / ``GIT_PUSH_URL``
* the LLM provider, if configured -- ``LLM_BASE_URL`` (empty by default)

These tests pin that policy down in two layers:

1. Static checks (run anywhere, no Docker needed): the container spec in
   ``src/runner.js`` really is network-sealed, every URL literal in
   ``src/`` belongs to the allowlist derived from that same config, and
   the in-compartment script uses no network tools.

2. Live probes (only inside the compartment): blocked domains must fail
   fast, and -- because the compartment is sealed rather than filtered --
   even the allowlisted hosts must be unreachable from inside. Egress is
   a host-side step; the compartment itself reaches nothing.

Run the static checks on any dev machine:

    python3 -m pytest tests/test_egress.py

Run everything, live probes included, inside the sealed sandbox image
(stdlib only, so plain python3 works -- no pip install possible offline):

    docker build -f Dockerfile.sandbox -t widespread-sandbox:latest .
    docker run --rm --network none --read-only --cap-drop ALL \\
        -v "$PWD:/work:rw" -w /work -e RUN_EGRESS_LIVE=1 \\
        widespread-sandbox:latest python3 tests/test_egress.py
"""

import os
import re
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_JS = REPO_ROOT / "src" / "runner.js"
SRC_DIR = REPO_ROOT / "src"
DOCKERFILE = REPO_ROOT / "Dockerfile.sandbox"
RUN_TASK = REPO_ROOT / "scripts" / "run-task.sh"

# Example domains that are on no allowlist. If any of these connects from
# inside the compartment, egress is leaking.
BLOCKED_HOSTS = ["example.com", "example.org", "example.net"]

# A connection to a blocked host must fail within this bound (seconds).
# With --network none the kernel refuses immediately; the bound exists so
# a filtered-but-slow setup (DNS hang, DROP firewall) also fails the test.
FAIL_FAST_BOUND = 3.0

LIVE = os.environ.get("RUN_EGRESS_LIVE") == "1"


# --- allowlist, derived from the repo's own config -------------------------


def _cfg_default(source: str, env_name: str) -> str:
    """Default value of a CFG entry: ``process.env.X || 'default'``."""
    m = re.search(
        r"process\.env\." + env_name + r"\s*\|\|\s*'([^']*)'", source
    )
    return m.group(1) if m else ""


def _host_of(url_or_host: str) -> str:
    if not url_or_host:
        return ""
    if "://" not in url_or_host:
        # scp-style git remote: git@github.com:owner/repo.git
        m = re.match(r"[^@]+@([^:]+):", url_or_host)
        return m.group(1) if m else url_or_host.split("/")[0]
    return urlparse(url_or_host).hostname or ""


def _git_remote_host() -> str:
    """Host of this checkout's origin remote (the git remote in config)."""
    try:
        url = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        url = ""
    return _host_of(url)


def host_egress_allowlist() -> set:
    """Every host the pipeline may reach, taken from runner config/env.

    Mirrors src/runner.js CFG: env vars win, code defaults are the
    fallback -- the same precedence the runner itself uses.
    """
    source = RUNNER_JS.read_text(encoding="utf-8")
    hosts = set()
    for env_name in ("BRIDGE_BASE", "FORGE_BASE", "LLM_BASE_URL"):
        url = os.environ.get(env_name) or _cfg_default(source, env_name)
        host = _host_of(url)
        if host:
            hosts.add(host)
    push_host = _host_of(os.environ.get("GIT_PUSH_URL", ""))
    if push_host:
        hosts.add(push_host)
    git_host = _git_remote_host()
    if git_host:
        hosts.add(git_host)
    return hosts


def git_remote_host() -> str:
    return _host_of(os.environ.get("GIT_PUSH_URL", "")) or _git_remote_host()


def llm_provider_host() -> str:
    source = RUNNER_JS.read_text(encoding="utf-8")
    url = os.environ.get("LLM_BASE_URL") or _cfg_default(source, "LLM_BASE_URL")
    return _host_of(url)


# --- static checks (run anywhere) ------------------------------------------


def test_sandbox_container_is_network_sealed():
    """The compartment is created with NetworkMode 'none', nothing else."""
    source = RUNNER_JS.read_text(encoding="utf-8")
    m = re.search(r"createContainer\(\{(.*?)\n  \}\);", source, re.S)
    assert m, "createContainer call not found in src/runner.js"
    spec = m.group(1)
    assert "NetworkMode: 'none'" in spec
    for leaky in ("PortBindings", "ExposedPorts", "ExtraHosts", "'bridge'", "'host'"):
        assert leaky not in spec, f"container spec opens egress via {leaky}"
    assert "ReadonlyRootfs: true" in spec
    assert "'ALL'" in spec  # CapDrop: ['ALL']


def test_dockerfile_opens_no_ports():
    assert "EXPOSE" not in DOCKERFILE.read_text(encoding="utf-8")


def test_only_allowlisted_url_literals_in_src():
    """Every URL literal under src/ must be an allowlisted host."""
    allowed = host_egress_allowlist()
    assert allowed, "allowlist derived from config is empty"
    offenders = []
    for path in SRC_DIR.rglob("*.js"):
        for url in re.findall(r"https?://[^\s'\"`\)]+", path.read_text(encoding="utf-8")):
            if _host_of(url) not in allowed:
                offenders.append(f"{path.name}: {url}")
    assert not offenders, f"non-allowlisted egress endpoints in src/: {offenders}"


def test_allowlist_defaults_are_the_configured_endpoints():
    """Pin the defaults from runner.js CFG so a silent change fails here."""
    source = RUNNER_JS.read_text(encoding="utf-8")
    assert _host_of(_cfg_default(source, "BRIDGE_BASE")) == (
        "slopcash-bridge.hardwoodstablecoin.workers.dev"
    )
    assert _host_of(_cfg_default(source, "FORGE_BASE")) == "code.widespread.fyi"
    # No LLM provider is configured by default: no LLM egress unless the
    # operator sets LLM_BASE_URL.
    assert _cfg_default(source, "LLM_BASE_URL") == ""


def test_blocked_hosts_are_not_allowlisted():
    allowed = host_egress_allowlist()
    for host in BLOCKED_HOSTS:
        assert host not in allowed


def test_in_compartment_script_uses_no_network_tools():
    script = RUN_TASK.read_text(encoding="utf-8")
    for tool in ("curl", "wget", " nc ", "ncat", "ping", "ssh ", "scp "):
        assert tool not in script, f"run-task.sh shells out to {tool.strip()}"


# --- live probes (inside the compartment only) ------------------------------


def _probe_fails_fast(host: str, port: int = 443) -> None:
    """Connecting must raise, and must do so within FAIL_FAST_BOUND."""
    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=FAIL_FAST_BOUND):
            pass
    except OSError:
        elapsed = time.monotonic() - started
        assert elapsed <= FAIL_FAST_BOUND, f"{host}: failed but too slowly ({elapsed:.1f}s)"
        return
    raise AssertionError(f"{host}:{port} was reachable -- egress is not sealed")


def _skip_unless_live():
    if not LIVE:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            import pytest

            pytest.skip("live probes only run inside the compartment (RUN_EGRESS_LIVE=1)")
        return "skip"
    return None


def test_blocked_domains_fail_fast():
    if _skip_unless_live() == "skip":
        return
    for host in BLOCKED_HOSTS:
        _probe_fails_fast(host)


def test_compartment_itself_has_zero_egress():
    """Even allowlisted hosts are unreachable from inside the compartment.

    The git remote and the LLM provider are host-side steps in this
    design; the compartment must not reach them (or anything) directly.
    """
    if _skip_unless_live() == "skip":
        return
    targets = {git_remote_host(), llm_provider_host()} - {""}
    for host in sorted(targets):
        _probe_fails_fast(host)


if __name__ == "__main__":
    # Stdlib-only runner so the file can execute inside the sealed image,
    # where pip (and the network it would need) is unavailable.
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except BaseException as exc:  # noqa: BLE001 - report and continue
                if type(exc).__name__ == "Skipped":
                    print(f"SKIP {name}")
                else:
                    failures += 1
                    print(f"ERROR {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    raise SystemExit(1 if failures else 0)
