"""Configuration loading and validation for the Lab01 simulator."""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from l1sim.models import CHAIN_ID

_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


class ConfigError(ValueError):
    """Raised when a genesis or scenario file is invalid."""


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Network timing and fault-injection settings."""

    min_delay_ms: int = 10
    max_delay_ms: int = 200
    drop_rate: float = 0.0
    duplicate_rate: float = 0.0
    gst_height: int = 0
    delta_ms: int = 500
    target_degree: int = 4


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    """Consensus timeout settings."""

    timeout_propose_ms: int = 1000
    timeout_prevote_ms: int = 1000
    timeout_precommit_ms: int = 1000
    timeout_delta_ms: int = 300


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    """Peer discovery multicast settings."""

    multicast_group: str = "239.1.1.1"
    multicast_port: int = 50000
    beacon_interval_ms: int = 1000
    target_degree: int = 4


@dataclass(frozen=True, slots=True)
class GenesisConfig:
    """Validated genesis configuration."""

    chain_id: str
    seed: int
    validators: tuple[str, ...]
    public_keys: dict[str, str]


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Validated scenario configuration."""

    name: str
    chain_id: str
    seed: int
    num_validators: int
    max_height: int
    validators: tuple[str, ...]
    byzantine: tuple[str, ...]
    warmup_ms: int
    max_txs_per_block: int
    network: NetworkConfig
    consensus: ConsensusConfig
    discovery: DiscoveryConfig
    transactions: tuple[dict[str, Any], ...]


def _read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {file_path}: {exc}") from exc

    if not isinstance(value, dict):
        raise ConfigError(f"config must be a JSON object: {file_path}")
    return value


def _required_int(mapping: dict[str, Any], name: str, minimum: int = 0) -> int:
    value = mapping.get(name)
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_int(
    mapping: dict[str, Any],
    name: str,
    default: int,
    minimum: int = 0,
) -> int:
    value = mapping.get(name, default)
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _rate(mapping: dict[str, Any], name: str) -> float:
    value = mapping.get(name, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number between 0 and 1")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be between 0 and 1")
    return value


def _non_empty_string(mapping: dict[str, Any], name: str, default: str) -> str:
    value = mapping.get(name, default)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _port(mapping: dict[str, Any], name: str, default: int) -> int:
    value = mapping.get(name, default)
    if type(value) is not int or not 1 <= value <= 65535:
        raise ConfigError(f"{name} must be an integer between 1 and 65535")
    return value


def load_genesis(path: str | Path = "config/genesis.json") -> GenesisConfig:
    """Load and validate a genesis configuration file."""

    raw = _read_json(path)
    chain_id = raw.get("chain_id")
    if chain_id != CHAIN_ID:
        raise ConfigError(f"genesis chain_id must be {CHAIN_ID!r}")

    seed = _required_int(raw, "seed")
    validator_values = raw.get("validators")
    if not isinstance(validator_values, list) or not validator_values:
        raise ConfigError("validators must be a non-empty list")

    validators: list[str] = []
    public_keys: dict[str, str] = {}
    for item in validator_values:
        if isinstance(item, dict):
            node_id = item.get("id")
            if not isinstance(node_id, str) or not _IDENTIFIER.fullmatch(node_id):
                raise ConfigError(f"invalid validator id: {node_id!r}")
            validators.append(node_id)
            public_key = item.get("public_key")
            if not isinstance(public_key, str):
                raise ConfigError(f"missing public_key for validator {node_id}")
            if len(public_key) != 64 or any(ch not in "0123456789abcdef" for ch in public_key):
                raise ConfigError(f"invalid public_key for validator {node_id}")
            public_keys[node_id] = public_key
        else:
            raise ConfigError("each validator must be an {id, public_key} object")

    if len(set(validators)) != len(validators):
        raise ConfigError("validator ids must be unique")
    if len(validators) < 8 or (len(validators) - 1) % 3 != 0:
        raise ConfigError("validators must follow n = 3f + 1 and contain at least 8 nodes")

    from l1sim.crypto import keypair_from_seed

    for node_id, public_key in public_keys.items():
        derived = keypair_from_seed(seed, node_id).public_bytes.hex()
        if public_key != derived:
            raise ConfigError(f"public_key for validator {node_id} does not match seed-derived key")

    return GenesisConfig(chain_id, seed, tuple(validators), public_keys)


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario configuration file."""

    raw = _read_json(path)
    name = raw.get("name") or Path(path).stem
    if not isinstance(name, str) or not name:
        raise ConfigError("scenario name must be a non-empty string")
    chain_id = raw.get("chain_id", CHAIN_ID)
    if chain_id != CHAIN_ID:
        raise ConfigError(f"scenario chain_id must be {CHAIN_ID!r}")

    seed = _required_int(raw, "seed")
    num_validators = _required_int(raw, "num_validators", 1)
    if num_validators < 8 or (num_validators - 1) % 3 != 0:
        raise ConfigError("num_validators must follow n = 3f + 1 and be at least 8")
    max_height = _required_int(raw, "max_height", 1)

    validators = raw.get("validators", [f"V{i}" for i in range(num_validators)])
    if not isinstance(validators, list) or any(not isinstance(item, str) for item in validators):
        raise ConfigError("validators must be a list of ids")
    validators = tuple(validators)
    if len(validators) != num_validators or len(set(validators)) != len(validators):
        raise ConfigError("validator ids must be unique and match num_validators")
    if any(not _IDENTIFIER.fullmatch(validator) for validator in validators):
        raise ConfigError("validator ids must be valid identifiers")

    byzantine = raw.get("byzantine", [])
    if not isinstance(byzantine, list) or any(not isinstance(item, str) for item in byzantine):
        raise ConfigError("byzantine must be a list of validator ids")
    byzantine = tuple(byzantine)
    if len(set(byzantine)) != len(byzantine) or any(item not in validators for item in byzantine):
        raise ConfigError("byzantine ids must be unique validator ids")

    expected_faulty = (num_validators - 1) // 3
    if len(byzantine) > expected_faulty:
        raise ConfigError(
            f"scenario has {len(byzantine)} Byzantine nodes; at most {expected_faulty} are allowed"
        )

    network_raw = raw.get("network", {})
    if not isinstance(network_raw, dict):
        raise ConfigError("network must be an object")
    network = NetworkConfig(
        min_delay_ms=_optional_int(network_raw, "min_delay_ms", 10),
        max_delay_ms=_optional_int(network_raw, "max_delay_ms", 200),
        drop_rate=_rate(network_raw, "drop_rate"),
        duplicate_rate=_rate(network_raw, "duplicate_rate"),
        gst_height=_optional_int(network_raw, "gst_height", 0),
        delta_ms=_optional_int(network_raw, "delta_ms", 500, 1),
        target_degree=_optional_int(network_raw, "target_degree", 4, 1),
    )
    if network.min_delay_ms > network.max_delay_ms:
        raise ConfigError("min_delay_ms cannot exceed max_delay_ms")
    if network.target_degree >= num_validators:
        raise ConfigError("network target_degree must be lower than num_validators")

    discovery_raw = raw.get("discovery", {})
    if not isinstance(discovery_raw, dict):
        raise ConfigError("discovery must be an object")
    discovery = DiscoveryConfig(
        multicast_group=_non_empty_string(discovery_raw, "multicast_group", "239.1.1.1"),
        multicast_port=_port(discovery_raw, "multicast_port", 50000),
        beacon_interval_ms=_optional_int(discovery_raw, "beacon_interval_ms", 1000, 1),
        target_degree=_optional_int(
            discovery_raw,
            "target_degree",
            network.target_degree,
            1,
        ),
    )
    try:
        multicast_address = ipaddress.IPv4Address(discovery.multicast_group)
        if not multicast_address.is_multicast:
            raise ConfigError("multicast_group must be a multicast IPv4 address")
    except ipaddress.AddressValueError as exc:
        raise ConfigError("multicast_group must be a valid multicast IPv4 address") from exc
    if discovery.target_degree >= num_validators:
        raise ConfigError("discovery target_degree must be lower than num_validators")

    consensus_raw = raw.get("consensus", {})
    if not isinstance(consensus_raw, dict):
        raise ConfigError("consensus must be an object")
    consensus = ConsensusConfig(
        timeout_propose_ms=_optional_int(consensus_raw, "timeout_propose_ms", 1000, 1),
        timeout_prevote_ms=_optional_int(consensus_raw, "timeout_prevote_ms", 1000, 1),
        timeout_precommit_ms=_optional_int(consensus_raw, "timeout_precommit_ms", 1000, 1),
        timeout_delta_ms=_optional_int(consensus_raw, "timeout_delta_ms", 300, 1),
    )

    warmup_ms = _optional_int(raw, "warmup_ms", 1000)
    max_txs_per_block = _optional_int(raw, "max_txs_per_block", 4, 1)

    transactions = raw.get("transactions", raw.get("tx", []))
    if not isinstance(transactions, list):
        raise ConfigError("transactions must be a list")
    for index, tx in enumerate(transactions):
        if not isinstance(tx, dict):
            raise ConfigError(f"transactions[{index}] must be an object")
        missing = {"sender", "key", "value", "nonce"} - set(tx)
        if missing:
            raise ConfigError(f"transactions[{index}] missing fields: {', '.join(sorted(missing))}")
        if not isinstance(tx["sender"], str) or not isinstance(tx["key"], str):
            raise ConfigError(f"transactions[{index}] sender/key must be strings")
        if not _IDENTIFIER.fullmatch(tx["sender"]):
            raise ConfigError(f"transactions[{index}] sender is not a valid identifier")
        if not isinstance(tx["value"], str):
            raise ConfigError(f"transactions[{index}] value must be a string")
        if type(tx["nonce"]) is not int or tx["nonce"] < 0:
            raise ConfigError(f"transactions[{index}] nonce must be a non-negative integer")
        if not tx["key"].startswith(f"{tx['sender']}/"):
            raise ConfigError(f"transactions[{index}] key must start with sender/")

    return ScenarioConfig(
        name=name,
        chain_id=chain_id,
        seed=seed,
        num_validators=num_validators,
        max_height=max_height,
        validators=validators,
        byzantine=byzantine,
        warmup_ms=warmup_ms,
        max_txs_per_block=max_txs_per_block,
        network=network,
        consensus=consensus,
        discovery=discovery,
        transactions=tuple(transactions),
    )


def load_config(
    scenario: str | Path,
    genesis: str | Path = "config/genesis.json",
) -> tuple[GenesisConfig, ScenarioConfig]:
    """Load and validate a genesis/scenario pair."""
    genesis_config = load_genesis(genesis)
    scenario_config = load_scenario(scenario)
    if tuple(scenario_config.validators) != genesis_config.validators:
        raise ConfigError("scenario validators must match genesis validator ids")
    if scenario_config.seed != genesis_config.seed:
        raise ConfigError("scenario seed must match genesis seed")
    if scenario_config.num_validators != len(genesis_config.validators):
        raise ConfigError("scenario num_validators must match genesis validator count")
    if scenario_config.chain_id != genesis_config.chain_id:
        raise ConfigError("scenario chain_id must match genesis chain_id")
    return genesis_config, scenario_config
