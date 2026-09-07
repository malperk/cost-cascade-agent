"""Transport abstraction for the self-improvement loop's authoring/eval
containers (plan sections 3/4): the rest of the runner talks only to this
interface, never branches on tier by name itself.

Tier 1 (host broker process + a bind-mounted Unix domain socket per role,
--network none on the container side) is implemented here and confirmed
working on this machine by the tier-1 spike (host<->container UDS bind
mount, httpx UDS transport, and full network isolation all verified before
any of this was written). Tiers 2 (broker-in-a-container + Docker named
volumes) and 3 (--internal network + pf) are documented fallbacks in the
plan for a host where tier 1's known Docker-Desktop-on-macOS risk actually
manifests - they are not implemented here since tier 1 passed; implement
them against this same interface (recovery_state/start_broker/
prepare_channels/teardown, plus a recover_and_teardown(state) staticmethod)
before ever changing CONFIGURED_TIER.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CONFIGURED_TIER = 1

ROLES = ("authoring", "baseline", "candidate")


class Tier1Transport:
    tier = 1

    def __init__(self, cycle_id: str, runtime_dir: Path):
        self.cycle_id = cycle_id
        self.runtime_dir = Path(runtime_dir)
        # short root (see plan section 3: AF_UNIX paths are capped at 104/108 chars) -
        # never nest sockets under the project's own (potentially long) absolute path.
        self.socket_dir = Path(f"/tmp/si/{cycle_id}")
        self.sockets = {role: self.socket_dir / f"{role}.sock" for role in ROLES}
        self.process: subprocess.Popen | None = None
        self.config_path = self.runtime_dir / "broker_config.json"
        self.log_path = self.runtime_dir / "broker.log"

    def recovery_state(self) -> dict:
        return {
            "tier": 1,
            "broker_pid": self.process.pid if self.process else None,
            "broker_container": None,
            "socket_paths": [str(p) for p in self.sockets.values()],
            "socket_volumes": None,
            "docker_network": None,
            "firewall_handle": None,
        }

    def start_broker(self, budgets: dict) -> None:
        self.socket_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "sockets": {role: str(path) for role, path in self.sockets.items()},
            "budgets": budgets,
            "log_path": str(self.log_path),
        }
        self.config_path.write_text(json.dumps(config))

        env = os.environ.copy()  # broker needs OMNIROUTE_API_KEY/BASE_URL from here - the
        # authoring/eval containers it serves never get these, only this subprocess does.
        stdout_path = self.runtime_dir / "broker.stdout"
        self.process = subprocess.Popen(
            # sys.executable, not a bare "python3" - must be the same interpreter/venv the
            # runner itself is running under, so the broker's dependencies are actually there.
            [sys.executable, "-m", "tools.omniroute_broker", "--config", str(self.config_path)],
            env=env,
            stdout=open(stdout_path, "w"),
            stderr=subprocess.STDOUT,
        )
        for _ in range(50):
            if all(p.exists() for p in self.sockets.values()):
                return
            if self.process.poll() is not None:
                raise RuntimeError(f"broker exited early - see {stdout_path}")
            time.sleep(0.1)
        raise RuntimeError(f"broker did not create its sockets in time - see {stdout_path}")

    def prepare_channels(self) -> dict:
        return {role: str(path) for role, path in self.sockets.items()}

    def teardown(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        for path in self.sockets.values():
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def recover_and_teardown(state: dict) -> None:
        pid = state.get("broker_pid")
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pid = None
            if pid:
                for _ in range(20):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.2)
                else:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        for path in state.get("socket_paths") or []:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass


def setup_transport(cycle_id: str, runtime_dir: Path):
    if CONFIGURED_TIER == 1:
        return Tier1Transport(cycle_id, runtime_dir)
    raise NotImplementedError(f"transport tier {CONFIGURED_TIER} not implemented - see plan section 3")


def recover_and_teardown_for_tier(state: dict) -> None:
    tier = state.get("tier")
    if tier == 1:
        Tier1Transport.recover_and_teardown(state)
    else:
        raise NotImplementedError(f"crash recovery for transport tier {tier} not implemented")
