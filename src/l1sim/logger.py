"""Node event logs and deterministic result output."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from l1sim.encoding import encode

_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_CORE_FIELDS = {"time", "node", "event", "height"}


class EventLogger:
    """Append-only JSONL event logger for one node."""

    def __init__(
        self,
        node_id: str,
        scenario_name: str,
        log_root: str | Path = "logs/events",
        time_provider: Callable[[], Any] | None = None,
    ) -> None:
        if not isinstance(node_id, str) or not _IDENTIFIER.fullmatch(node_id):
            raise ValueError("node_id must be a valid identifier")
        if not isinstance(scenario_name, str) or not _IDENTIFIER.fullmatch(scenario_name):
            raise ValueError("scenario_name must be a valid identifier")
        if time_provider is not None and not callable(time_provider):
            raise TypeError("time_provider must be callable")
        self.node_id = node_id
        self.scenario_name = scenario_name
        self.path = Path(log_root) / scenario_name / f"{node_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not time_provider:
            self._simulated_time = 0
            self.time_provider = self._next_simulated_time
        else:
            self.time_provider = time_provider
        self.path.unlink(missing_ok=True)

    def _next_simulated_time(self) -> int:
        value = self._simulated_time
        self._simulated_time += 1
        return value

    def event(self, event: str, height: int = 0, **fields: Any) -> None:
        if not isinstance(event, str) or not event:
            raise ValueError("event must be a non-empty string")
        if type(height) is not int or height < 0:
            raise ValueError("height must be a non-negative integer")
        overwritten = _CORE_FIELDS.intersection(fields)
        if overwritten:
            names = ", ".join(sorted(overwritten))
            raise ValueError(f"event fields cannot override core fields: {names}")
        record: dict[str, Any] = {
            "time": self.time_provider(),
            "node": self.node_id,
            "event": event,
            "height": height,
        }
        record.update(fields)
        line = encode(record).decode("utf-8")
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")


class ResultWriter:
    """Write byte-stable JSON files under logs/results."""

    def __init__(
        self,
        scenario_name: str,
        results_root: str | Path = "logs/results",
    ) -> None:
        if not isinstance(scenario_name, str) or not _IDENTIFIER.fullmatch(scenario_name):
            raise ValueError("scenario_name must be a valid identifier")
        self.scenario_name = scenario_name
        self.path = Path(results_root) / scenario_name
        self.path.mkdir(parents=True, exist_ok=True)

    def write_node(self, node_id: str, result: Mapping[str, Any]) -> Path:
        if not isinstance(node_id, str) or not _IDENTIFIER.fullmatch(node_id):
            raise ValueError("node_id must be a valid identifier")
        return self._write(f"{node_id}.json", result)

    def write_verdict(self, verdict: Mapping[str, Any]) -> Path:
        return self._write("verdict.json", verdict)

    def write_json(self, name: str, data: Mapping[str, Any]) -> Path:
        if not _safe_filename(name):
            raise ValueError("result name must be a safe filename")
        return self._write(name, data)

    def clear(self) -> None:
        """Remove JSON result files before starting a fresh scenario run."""
        for path in self.path.glob("*.json"):
            if path.is_file():
                path.unlink()

    def _write(self, name: str, data: Mapping[str, Any]) -> Path:
        if not _safe_filename(name):
            raise ValueError("result name must be a safe filename")
        output = self.path / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_canonical_json(data))
        return output


class ResultsCollector:
    """Read node result files and construct a small deterministic summary."""

    def __init__(self, scenario_name: str, results_root: str | Path = "logs/results") -> None:
        if not isinstance(scenario_name, str) or not _IDENTIFIER.fullmatch(scenario_name):
            raise ValueError("scenario_name must be a valid identifier")
        self.path = Path(results_root) / scenario_name

    def node_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(path.stem for path in self.path.glob("*.json") if path.name != "verdict.json")
        )

    def load_node(self, node_id: str) -> dict[str, Any]:
        if not isinstance(node_id, str) or not _IDENTIFIER.fullmatch(node_id):
            raise ValueError("node_id must be a valid identifier")
        path = self.path / f"{node_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"node result not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"result must be an object: {path}")
        return value

    def load_nodes(self, node_ids: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        selected = node_ids if node_ids is not None else self.node_ids()
        return {node_id: self.load_node(node_id) for node_id in selected}


def _canonical_json(data: Mapping[str, Any]) -> bytes:
    if not isinstance(data, Mapping):
        raise TypeError("result data must be a mapping")
    return encode(dict(data))


def _safe_filename(name: str) -> bool:
    if not isinstance(name, str) or not name:
        return False
    if name in {".", ".."} or "/" in name or "\\" in name:
        return False
    return Path(name).name == name and not Path(name).is_absolute()
