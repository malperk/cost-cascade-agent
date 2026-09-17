"""Git mechanics for the self-improvement runner (plan sections 1, 7, 8).

Every function here runs on the host, invoked only by the trusted runner -
never inside a container. The invariant checks are the actual mechanical
gate: a CODE hypothesis is trusted only as far as `verify_code_invariants`
can independently confirm from the trusted repo's own git state, never
from anything the authoring container claims about itself.
"""

import hashlib
import subprocess
from pathlib import Path

CODE_WHITELIST_PREFIX = "agent/"
CODE_WHITELIST_SUFFIX = ".py"
MAX_CHANGED_FILES = 2
MAX_CHANGED_LINES = 80


class InvariantViolation(Exception):
    """Raised when the trusted repo's actual git state doesn't match what a
    CODE or MODEL hypothesis is required to look like - always a hard FAIL,
    never something the caller should try to work around."""


def _run(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def capture_base_sha(repo_path: Path) -> str:
    return _run(["git", "rev-parse", "main"], repo_path)


def working_tree_clean(repo_path: Path) -> bool:
    return _run(["git", "status", "--porcelain"], repo_path) == ""


def shallow_clone(repo_path: Path, dest: Path) -> None:
    """--depth 1 --no-hardlinks: the authoring container's context, at base_sha, with no
    project history to go looking through for something committed before this loop existed."""
    subprocess.run(
        ["git", "clone", "--depth", "1", "--no-hardlinks", f"file://{repo_path}", str(dest)],
        check=True,
        capture_output=True,
    )
    configure_clone_identity(dest)


def configure_clone_identity(clone_path: Path) -> None:
    """A fresh clone has no author identity - HOME isn't mounted into the authoring container,
    so there's no ~/.gitconfig to inherit one from, and `git commit` fails outright ("Please
    tell me who you are") without this. Set once by the runner right after cloning, not left
    for the agent to remember as an extra step."""
    subprocess.run(
        ["git", "config", "user.email", "self-improvement-loop@local"], cwd=clone_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "self-improvement-loop"], cwd=clone_path, check=True)


def git_archive_export(repo_path: Path, sha: str, dest: Path) -> None:
    """An exact, reproducible source tree for an eval container - not "the repo's agent/
    folder": everything at that SHA (requirements.txt, prompts/, agent/config.py, fixtures),
    mounted read-only, so "baseline = exactly base_sha" is a guarantee, not an assumption."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", sha], cwd=repo_path, capture_output=True, check=True
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)


def fetch_candidate_branch(repo_path: Path, clone_path: Path, branch_name: str) -> str:
    """Pulls the authoring container's one candidate commit out of its disposable clone and
    into the trusted repo, under a new branch - the only thing that crosses that boundary."""
    subprocess.run(
        ["git", "fetch", str(clone_path), f"HEAD:refs/heads/{branch_name}"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    return _run(["git", "rev-parse", branch_name], repo_path)


def verify_code_invariants(clone_path: Path, base_sha: str) -> None:
    """Checked in the CLONE before fetching anything back: HEAD is exactly one non-merge
    commit ahead of base_sha, and the clone's own working tree is clean (the agent must have
    committed everything, not left changes uncommitted for git status --porcelain to hide)."""
    if not working_tree_clean(clone_path):
        raise InvariantViolation("CODE: authoring container left uncommitted changes")

    clone_head = _run(["git", "rev-parse", "HEAD"], clone_path)
    if clone_head == base_sha:
        raise InvariantViolation("CODE: authoring container made no commit at all")

    commit_count = int(_run(["git", "rev-list", "--count", f"{base_sha}..HEAD"], clone_path))
    if commit_count != 1:
        raise InvariantViolation(f"CODE: expected exactly 1 commit ahead of base_sha, found {commit_count}")

    merge_commits = _run(["git", "log", "--merges", f"{base_sha}..HEAD", "--oneline"], clone_path)
    if merge_commits:
        raise InvariantViolation("CODE: candidate history contains a merge commit")


def verify_model_invariants(clone_path: Path, base_sha: str) -> None:
    """For a MODEL hypothesis the authoring container is never expected to touch the clone at
    all - its only output is /cycle-output/hypothesis.json, outside the git tree entirely. Any
    commit or uncommitted change here is an integrity FAIL, not silently ignored."""
    if not working_tree_clean(clone_path):
        raise InvariantViolation("MODEL: authoring container left changes in the clone")
    clone_head = _run(["git", "rev-parse", "HEAD"], clone_path)
    if clone_head != base_sha:
        raise InvariantViolation("MODEL: clone HEAD moved away from base_sha (a commit was made)")


def check_whitelist_and_budget(repo_path: Path, base_sha: str, candidate_sha: str) -> dict:
    """Runs in the TRUSTED repo against the fetched candidate commit - numstat-based (not
    --stat) so binary files are caught (numstat prints "-" for added/deleted on a binary
    file, which int() will reject), and only agent/**/*.py may appear in the diff at all."""
    numstat_output = _run(["git", "diff", "--numstat", base_sha, candidate_sha], repo_path)
    if not numstat_output:
        raise InvariantViolation("CODE: candidate diff is empty")

    changed_files = []
    total_lines = 0
    for line in numstat_output.splitlines():
        added, deleted, path = line.split("\t")
        if added == "-" or deleted == "-":
            raise InvariantViolation(f"CODE: binary file in diff: {path}")
        if not (path.startswith(CODE_WHITELIST_PREFIX) and path.endswith(CODE_WHITELIST_SUFFIX)):
            raise InvariantViolation(f"CODE: candidate touched a non-whitelisted path: {path}")
        changed_files.append(path)
        total_lines += int(added) + int(deleted)

    if len(changed_files) > MAX_CHANGED_FILES:
        raise InvariantViolation(f"CODE: touched {len(changed_files)} files, budget is {MAX_CHANGED_FILES}")
    if total_lines > MAX_CHANGED_LINES:
        raise InvariantViolation(f"CODE: {total_lines} changed lines, budget is {MAX_CHANGED_LINES}")

    name_status = _run(["git", "diff", "--name-status", base_sha, candidate_sha], repo_path)
    for line in name_status.splitlines():
        if line.startswith("R"):
            raise InvariantViolation(f"CODE: candidate diff includes a rename: {line}")

    return {"changed_files": changed_files, "total_lines": total_lines}


def code_fingerprint(repo_path: Path, base_sha: str, candidate_sha: str, metric: str) -> str:
    """Runner-computed, never taken from anything the agent asserts about itself (an
    agent-supplied change_id would be trivially evadable by renaming). Derived from the
    actual validated patch content, so this only catches exact/whitespace-trivial repeats -
    a stated scope limit, not an oversold guarantee."""
    normalized_diff = _run(
        ["git", "diff", "--ignore-all-space", base_sha, candidate_sha, "--", "agent/"], repo_path
    )
    payload = f"code|{metric}|{normalized_diff}".encode()
    return hashlib.sha256(payload).hexdigest()


def model_fingerprint(tier: str, metric: str, candidate_model: str, candidate_provider: str) -> str:
    payload = f"model|{tier}|{metric}|{candidate_model}|{candidate_provider}".encode()
    return hashlib.sha256(payload).hexdigest()


def merge_candidate_ff_only(repo_path: Path, branch_name: str, base_sha: str) -> str:
    """Verifies main hasn't drifted since the candidate was validated, fast-forwards it onto
    the candidate commit, and confirms HEAD actually landed where expected."""
    current_main = _run(["git", "rev-parse", "main"], repo_path)
    if current_main != base_sha:
        raise InvariantViolation(f"main moved from {base_sha} to {current_main} during the cycle")

    subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "merge", "--ff-only", branch_name], cwd=repo_path, check=True, capture_output=True
    )
    new_head = _run(["git", "rev-parse", "HEAD"], repo_path)
    expected_head = _run(["git", "rev-parse", branch_name], repo_path)
    if new_head != expected_head:
        raise InvariantViolation(f"merge did not land on the expected candidate commit ({new_head} != {expected_head})")
    return new_head


def commit_metadata_on_main(repo_path: Path, paths: list[str], message: str) -> None:
    """Runner-only: commits the ledger entry and cycle report on main after every cycle
    (pass or fail), so main's working tree is never left dirty going into the next cycle."""
    subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "add", *paths], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo_path, check=True, capture_output=True)
