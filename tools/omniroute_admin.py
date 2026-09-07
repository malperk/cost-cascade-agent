"""OmniRoute combo administration - trusted-runner-only, never imported by
anything that runs inside an authoring/eval container and never given to
the LLM agent. Uses the raw REST API directly (the `omniroute` CLI's path-
param handling has known bugs for these same endpoints - see project
history), with the real admin key read from the environment.

A MODEL hypothesis's authoring container never calls any of this itself -
it only writes a proposal (candidate_model/candidate_provider/tier) to
hypothesis.json; the trusted runner is the only thing that ever creates,
promotes, or deletes a combo.
"""

import os

import httpx

from agent.config import OMNIROUTE_API_KEY, OMNIROUTE_BASE_URL

_ADMIN_ROOT = OMNIROUTE_BASE_URL[: -len("/v1")] if OMNIROUTE_BASE_URL.endswith("/v1") else OMNIROUTE_BASE_URL
_HEADERS = {"Authorization": f"Bearer {OMNIROUTE_API_KEY}"}


def _client() -> httpx.Client:
    return httpx.Client(base_url=_ADMIN_ROOT, headers=_HEADERS, timeout=30)


def list_combos() -> list[dict]:
    with _client() as c:
        resp = c.get("/api/combos")
        resp.raise_for_status()
        return resp.json()["combos"]


def get_combo(name: str) -> dict | None:
    for combo in list_combos():
        if combo["name"] == name:
            return combo
    return None


def create_combo(name: str, models: list[dict], strategy: str = "priority") -> dict:
    body = {"name": name, "models": models, "strategy": strategy, "config": {}}
    with _client() as c:
        resp = c.post("/api/combos", json=body)
        resp.raise_for_status()
        return resp.json()


def delete_combo_by_id(combo_id: str) -> None:
    with _client() as c:
        resp = c.delete(f"/api/combos/{combo_id}")
        resp.raise_for_status()


def delete_combo(name: str) -> bool:
    """Returns True if a combo with this name existed and was deleted."""
    existing = get_combo(name)
    if existing is None:
        return False
    delete_combo_by_id(existing["id"])
    return True


def snapshot_all(names: list[str]) -> dict[str, dict | None]:
    """Full-fleet snapshot for the pre/post-authoring integrity check - captures None for a
    name that doesn't currently exist, so a combo appearing OR disappearing both count as drift."""
    current = {c["name"]: c for c in list_combos()}
    return {name: current.get(name) for name in names}


def promote_combo(name: str, models: list[dict], strategy: str = "priority") -> dict:
    """Delete-then-recreate the production-named combo with a new definition - this project's
    established pattern (the OmniRoute API has no reliable upsert-by-name), used only after
    every eval gate has already passed."""
    delete_combo(name)
    return create_combo(name, models, strategy)


def combos_equal(a: dict | None, b: dict | None) -> bool:
    """Definition equality for drift/integrity checks - ignores volatile fields
    (timestamps, version counters) that change on every write without the actual
    routing definition changing."""
    if a is None or b is None:
        return a is None and b is None
    keys = ("name", "models", "strategy", "config")
    return all(a.get(k) == b.get(k) for k in keys)
