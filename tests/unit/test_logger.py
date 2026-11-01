import pytest

from l1sim.logger import EventLogger, ResultsCollector, ResultWriter


def test_result_writer_is_canonical(tmp_path):
    writer = ResultWriter("t1_normal", tmp_path)
    output = writer.write_node("V0", {"b": 2, "a": 1})

    assert output.read_bytes() == b'{"a":1,"b":2}'
    output_again = writer.write_node("V0", {"a": 1, "b": 2})
    assert output_again.read_bytes() == b'{"a":1,"b":2}'


def test_event_logger_writes_one_json_record(tmp_path):
    logger = EventLogger("V0", "t1_normal", tmp_path)
    logger.event("proposal", height=1, proposer="V0")

    line = logger.path.read_text(encoding="utf-8").strip()
    assert '"event":"proposal"' in line
    assert '"height":1' in line
    assert '"node":"V0"' in line


def test_event_logger_resets_and_uses_simulated_time(tmp_path):
    logger = EventLogger("V0", "t1_normal", tmp_path)
    logger.event("first")
    logger.event("second")

    repeated = EventLogger("V0", "t1_normal", tmp_path)
    repeated.event("first")
    lines = repeated.path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 1
    assert '"time":0' in lines[0]


def test_event_logger_rejects_reserved_field_overwrite(tmp_path):
    logger = EventLogger("V0", "t1_normal", tmp_path)
    with pytest.raises(ValueError, match="core fields"):
        logger.event("proposal", height=1, node="V9")


def test_event_logger_custom_time_provider(tmp_path):
    logger = EventLogger(
        "V0",
        "t1_normal",
        tmp_path,
        time_provider=lambda: 123,
    )

    logger.event("proposal", height=2)

    line = logger.path.read_text(encoding="utf-8").strip()
    assert '"time":123' in line


def test_result_writer_rejects_path_traversal(tmp_path):
    writer = ResultWriter("t1_normal", tmp_path)
    with pytest.raises(ValueError, match="safe filename"):
        writer.write_json("../verdict.json", {"ok": True})


def test_result_writer_clear_removes_only_json_results(tmp_path):
    writer = ResultWriter("t1_normal", tmp_path)
    writer.write_node("V0", {"ok": True})
    keep = writer.path / "notes.txt"
    keep.write_text("keep", encoding="utf-8")

    writer.clear()

    assert not (writer.path / "V0.json").exists()
    assert keep.exists()


def test_results_collector_excludes_verdict_and_loads_nodes(tmp_path):
    writer = ResultWriter("t1_normal", tmp_path)
    writer.write_node("V0", {"height": 1})
    writer.write_node("V1", {"height": 2})
    writer.write_verdict({"passed": True})
    collector = ResultsCollector("t1_normal", tmp_path)

    assert collector.node_ids() == ("V0", "V1")
    assert collector.load_nodes() == {
        "V0": {"height": 1},
        "V1": {"height": 2},
    }


def test_results_collector_rejects_missing_node(tmp_path):
    collector = ResultsCollector("t1_normal", tmp_path)

    with pytest.raises(FileNotFoundError, match="node result not found"):
        collector.load_node("V0")
