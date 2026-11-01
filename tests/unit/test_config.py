from pathlib import Path

import pytest

from l1sim.config import ConfigError, load_config, load_genesis, load_scenario

ROOT = Path(__file__).parents[2]


def test_load_genesis_and_t1_scenario():
    genesis = load_genesis(ROOT / "config/genesis.json")
    scenario = load_scenario(ROOT / "config/t1_normal.json")

    assert genesis.chain_id == "lab01"
    assert len(genesis.validators) == 10
    assert scenario.name == "t1_normal"
    assert len(scenario.transactions) == 20
    assert scenario.consensus.timeout_delta_ms == 300


def test_load_default_scenario():
    scenario = load_scenario(ROOT / "config/default.json")

    assert scenario.chain_id == "lab01"
    assert scenario.num_validators == 10
    assert scenario.consensus.timeout_delta_ms == 300


def test_config_rejects_wrong_chain_id(tmp_path):
    path = tmp_path / "genesis.json"
    path.write_text('{"chain_id":"wrong","seed":42,"validators":["V0"]}', encoding="utf-8")
    with pytest.raises(ConfigError, match="chain_id"):
        load_genesis(path)


def test_genesis_requires_public_keys(tmp_path):
    path = tmp_path / "genesis.json"
    path.write_text(
        '{"chain_id":"lab01","seed":42,"validators":[{"id":"V0"},{"id":"V1"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="public_key"):
        load_genesis(path)


def test_genesis_rejects_empty_validator_id(tmp_path):
    path = tmp_path / "genesis.json"
    path.write_text(
        '{"chain_id":"lab01","seed":42,"validators":[{"id":"","public_key":"'
        + ("00" * 32)
        + '"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="validator"):
        load_genesis(path)


def test_scenario_rejects_wrong_chain_id(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(
        '{"name":"bad","chain_id":"wrong","seed":42,"num_validators":10,'
        '"validators":["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9"],'
        '"max_height":1}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="chain_id"):
        load_scenario(path)


def test_scenario_rejects_unreachable_target_degree(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(
        '{"name":"bad","seed":42,"num_validators":10,'
        '"validators":["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9"],'
        '"max_height":1,"network":{"target_degree":10}}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="target_degree"):
        load_scenario(path)


def test_scenario_rejects_invalid_sender(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(
        '{"name":"bad","seed":42,"num_validators":10,'
        '"validators":["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9"],'
        '"max_height":1,"transactions":[{"sender":"","key":"Alice/account","value":"x","nonce":0}]}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="sender"):
        load_scenario(path)


def test_load_config_rejects_mismatched_pair(tmp_path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        '{"name":"bad","chain_id":"lab01","seed":43,"num_validators":10,'
        '"validators":["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9"],'
        '"max_height":1}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="seed"):
        load_config(scenario, ROOT / "config/genesis.json")


def test_scenario_accepts_discovery_config(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(
        '{"name":"discovery","seed":42,"num_validators":10,'
        '"validators":["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9"],'
        '"max_height":1,"discovery":{"multicast_group":"239.1.1.1",'
        '"multicast_port":50000,"beacon_interval_ms":250,"target_degree":4}}',
        encoding="utf-8",
    )

    scenario = load_scenario(path)

    assert scenario.discovery.multicast_group == "239.1.1.1"
    assert scenario.discovery.multicast_port == 50000
    assert scenario.discovery.beacon_interval_ms == 250
    assert scenario.discovery.target_degree == 4


def test_scenario_rejects_invalid_discovery_group(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(
        '{"name":"bad","seed":42,"num_validators":10,'
        '"validators":["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9"],'
        '"max_height":1,"discovery":{"multicast_group":"127.0.0.1"}}',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="multicast"):
        load_scenario(path)
