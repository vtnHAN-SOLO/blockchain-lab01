import json
from pathlib import Path

import pytest

from l1sim.consensus.engine import ConsensusEngine, Step, TimeoutConfig, proposer_for
from l1sim.crypto import verify
from l1sim.encoding import block_hash
from l1sim.execution import Mempool, State
from l1sim.logger import EventLogger
from l1sim.models import GENESIS_HASH, Domain, Kind, Message, Phase
from tests.unit.consensus_harness import TIMEOUTS, Network, sample_txs

CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.json"
CONSENSUS = {
    "timeout_propose_ms": 1000,
    "timeout_prevote_ms": 800,
    "timeout_precommit_ms": 600,
    "timeout_delta_ms": 300,
}


def test_proposer_rotates_with_height_and_round_over_sorted_validators():
    validators = ["V3", "V1", "V0", "V2"]

    assert proposer_for(validators, 1, 0) == "V1"
    assert proposer_for(validators, 1, 1) == "V2"
    assert proposer_for(validators, 2, 0) == "V2"
    assert proposer_for(validators, 3, 1) == "V0"
    assert proposer_for({"V1": None, "V0": None}, 1, 0) == "V1"


def test_proposer_requires_validators():
    with pytest.raises(ValueError):
        proposer_for([], 1, 0)


def test_timeouts_grow_with_round():
    timeouts = TimeoutConfig.from_config(CONSENSUS)

    assert timeouts.duration(Step.PROPOSE, 0) == 1000
    assert timeouts.duration(Step.PREVOTE, 0) == 800
    assert timeouts.duration(Step.PRECOMMIT, 2) == 1200
    assert timeouts.duration(Step.PROPOSE, 1) > timeouts.duration(Step.PROPOSE, 0)


def test_timeout_config_has_no_defaults():
    missing = {key: value for key, value in CONSENSUS.items() if key != "timeout_delta_ms"}

    with pytest.raises(KeyError):
        TimeoutConfig.from_config(missing)


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_timeout_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        TimeoutConfig.from_config({**CONSENSUS, "timeout_delta_ms": bad})


def test_default_config_provides_every_timeout():
    consensus = json.loads(CONFIG.read_text())["consensus"]

    assert TimeoutConfig.from_config(consensus).delta_ms == 300


# --------------------------------------------------------------------------- engine


def _assert_no_double_sign(net):
    """Rule 1-2: every node signed at most one value per (height, round, phase)."""
    for node in net.ids:
        seen = {}
        for vote in net.votes_sent(node):
            slot = (vote.height, vote.round, vote.phase)
            assert seen.setdefault(slot, vote.block_hash) == vote.block_hash, (node, slot)


def _assert_certificates_valid(net, node):
    keys = {voter: pair.public_key for voter, pair in net.keys.items()}
    quorum = net.engines[node].votes.quorum
    for block, certificate in net.finalized[node]:
        assert certificate.block_hash == block_hash(block.header)
        assert certificate.height == block.header.height
        assert len(certificate.precommits) >= quorum
        for vote in certificate.precommits:
            assert verify(keys[vote.voter], Domain.VOTE, vote, vote.signature)


@pytest.mark.parametrize("n", [4, 10])
def test_happy_path_all_nodes_finalize_the_same_blocks(n):
    net = Network(n, txs=sample_txs())
    net.start()

    assert net.run_until_height(3)

    chains = {tuple(net.hashes(node)[:3]) for node in net.ids}
    assert len(chains) == 1
    states = {net.engines[node].state.state_hash() for node in net.ids}
    assert len(states) == 1
    first_block = net.finalized["V0"][0][0]
    assert len(first_block.txs) == len(sample_txs())
    assert first_block.header.proposer == proposer_for(net.ids, 1, 0)
    assert all(block.header.round == 0 for block, _ in net.finalized["V0"])
    for node in net.ids:
        _assert_certificates_valid(net, node)
        assert sorted(net.engines[node].finalized)[:3] == [1, 2, 3]
    _assert_no_double_sign(net)


def test_same_inputs_produce_the_same_event_log():
    runs = []
    for _ in range(2):
        net = Network(4, txs=sample_txs())
        net.start()
        net.run_until_height(3)
        runs.append((net.events, net.hashes("V0")))

    assert runs[0] == runs[1]


def test_silent_proposer_times_out_and_next_round_finalizes():
    silent = proposer_for([f"V{i}" for i in range(4)], 1, 0)

    def drop(src, dst, message):
        return src == silent and message.kind is Kind.BODY and message.height == 1

    net = Network(4, txs=sample_txs(), drop=drop)
    net.start()

    assert net.run_until_height(3)

    for node in net.ids:
        block, certificate = net.finalized[node][0]
        assert block.header.round == 1
        assert block.header.proposer == proposer_for(net.ids, 1, 1) != silent
        assert certificate.precommits[0].round == 1
    assert len({tuple(net.hashes(node)[:3]) for node in net.ids}) == 1

    honest = next(node for node in net.ids if node != silent)
    assert "timeout" in net.event_names(honest)
    assert any(
        (vote.height, vote.round, vote.phase, vote.block_hash) == (1, 0, Phase.PREVOTE, None)
        for vote in net.votes_sent(honest)
    )
    propose_timeouts = [
        delay
        for delay, height, round, step in net.scheduled[honest]
        if height == 1 and step is Step.PROPOSE
    ]
    assert propose_timeouts[:2] == [1000, 1300]
    _assert_no_double_sign(net)


def test_late_proposal_after_nil_prevote_does_not_cause_second_prevote():
    net = Network(4, drop=lambda src, dst, message: True)
    net.start(only=["V0"])
    engine = net.engines["V0"]
    proposer = proposer_for(net.ids, 1, 0)
    block = net.make_block(proposer, 1, 0, parent_hash=GENESIS_HASH)

    assert net.clock.run(lambda: engine.step is Step.PREVOTE)
    net.deliver("V0", Message(Kind.BODY, proposer, 1, block))

    prevotes = [vote for vote in net.votes_sent("V0") if vote.phase is Phase.PREVOTE]
    assert [(vote.round, vote.block_hash) for vote in prevotes] == [(0, None)]
    _assert_no_double_sign(net)


def test_conflicting_proposals_get_only_one_prevote():
    net = Network(4, drop=lambda src, dst, message: True)
    net.start(only=["V0"])
    proposer = proposer_for(net.ids, 1, 0)
    first = net.make_block(proposer, 1, 0, sample_txs()[:1], parent_hash=GENESIS_HASH)
    second = net.make_block(proposer, 1, 0, sample_txs()[:2], parent_hash=GENESIS_HASH)

    net.deliver("V0", Message(Kind.BODY, proposer, 1, first))
    net.deliver("V0", Message(Kind.BODY, proposer, 1, second))

    assert [vote.block_hash for vote in net.votes_sent("V0")] == [block_hash(first.header)]
    assert "proposal_conflict" in net.event_names("V0")


def test_forged_proposal_cannot_take_the_real_proposers_slot():
    net = Network(4, drop=lambda src, dst, message: True)
    net.start(only=["V0"])
    proposer = proposer_for(net.ids, 1, 0)
    forger = next(node for node in net.ids if node not in (proposer, "V0"))
    forged = net.make_block(proposer, 1, 0, parent_hash=GENESIS_HASH, signer=forger)
    real = net.make_block(proposer, 1, 0, sample_txs()[:1], parent_hash=GENESIS_HASH)

    net.deliver("V0", Message(Kind.BODY, proposer, 1, forged))
    net.deliver("V0", Message(Kind.BODY, proposer, 1, real))

    assert [vote.block_hash for vote in net.votes_sent("V0")] == [block_hash(real.header)]
    assert ("proposal_rejected", "bad_signature") in [
        (event, fields.get("reason")) for event, fields in net.events["V0"]
    ]


def test_invalid_block_gets_nil_prevote():
    net = Network(4, drop=lambda src, dst, message: True)
    net.start(only=["V0"])
    proposer = proposer_for(net.ids, 1, 0)
    bad = net.make_block(proposer, 1, 0, parent_hash=GENESIS_HASH, state_hash="f" * 64)

    net.deliver("V0", Message(Kind.BODY, proposer, 1, bad))

    assert [(vote.phase, vote.block_hash) for vote in net.votes_sent("V0")] == [
        (Phase.PREVOTE, None)
    ]
    assert ("invalid_block", "bad_state_hash") in [
        (event, fields.get("reason")) for event, fields in net.events["V0"]
    ]


def test_node_missing_prevotes_still_finalizes_from_precommits():
    def drop(src, dst, message):
        return dst == "V3" and message.kind is Kind.VOTE and message.payload.phase is Phase.PREVOTE

    net = Network(4, txs=sample_txs(), drop=drop)
    net.start()

    assert net.run_until_height(3)

    assert net.hashes("V3")[:3] == net.hashes("V0")[:3]
    block_precommits = [
        vote
        for vote in net.votes_sent("V3")
        if vote.phase is Phase.PRECOMMIT and vote.block_hash is not None
    ]
    assert block_precommits == []
    _assert_no_double_sign(net)


def test_precommit_follows_the_polka_not_the_own_prevote():
    net = Network(4, drop=lambda src, dst, message: True)
    net.start(only=["V0"])
    proposer = proposer_for(net.ids, 1, 0)
    bad = net.make_block(proposer, 1, 0, parent_hash=GENESIS_HASH, state_hash="f" * 64)
    good = net.make_block(proposer, 1, 0, parent_hash=GENESIS_HASH)

    net.deliver("V0", Message(Kind.BODY, proposer, 1, bad))
    for voter in ("V1", "V2", "V3"):
        vote = net.make_vote(voter, Phase.PREVOTE, 1, 0, block_hash(good.header))
        net.deliver("V0", Message(Kind.VOTE, voter, 1, vote))

    # V0 prevoted NIL and does not have a valid copy of the polka block, so it may not
    # precommit it; it waits for the prevote timeout and precommits NIL.
    assert net.clock.run(lambda: net.engines["V0"].step is Step.PRECOMMIT)
    assert [(vote.phase, vote.block_hash) for vote in net.votes_sent("V0")] == [
        (Phase.PREVOTE, None),
        (Phase.PRECOMMIT, None),
    ]


def test_nil_prevote_then_polka_for_a_known_valid_block_precommits_that_block():
    net = Network(4, drop=lambda src, dst, message: True)
    net.start(only=["V0"])
    engine = net.engines["V0"]
    proposer = proposer_for(net.ids, 1, 0)
    block = net.make_block(proposer, 1, 0, parent_hash=GENESIS_HASH)
    digest = block_hash(block.header)

    assert net.clock.run(lambda: engine.step is Step.PREVOTE)
    net.deliver("V0", Message(Kind.BODY, proposer, 1, block))
    for voter in ("V1", "V2", "V3"):
        vote = net.make_vote(voter, Phase.PREVOTE, 1, 0, digest)
        net.deliver("V0", Message(Kind.VOTE, voter, 1, vote))

    assert [(vote.phase, vote.block_hash) for vote in net.votes_sent("V0")] == [
        (Phase.PREVOTE, None),
        (Phase.PRECOMMIT, digest),
    ]


def test_commit_from_an_older_round_is_finalized_in_a_later_round():
    net = Network(4, drop=lambda src, dst, message: True)
    net.start(only=["V0"])
    engine = net.engines["V0"]
    proposer = proposer_for(net.ids, 1, 0)
    block = net.make_block(proposer, 1, 0, parent_hash=GENESIS_HASH)
    digest = block_hash(block.header)

    net.deliver("V0", Message(Kind.BODY, proposer, 1, block))
    assert net.clock.run(lambda: engine.round == 2)

    for voter in ("V1", "V2", "V3"):
        vote = net.make_vote(voter, Phase.PRECOMMIT, 1, 0, digest)
        net.deliver("V0", Message(Kind.VOTE, voter, 1, vote))

    assert net.hashes("V0") == [digest]
    assert net.finalized["V0"][0][1].precommits[0].round == 0
    assert engine.height == 2 and engine.round == 0


def test_future_height_messages_are_buffered_and_old_heights_ignored():
    reference = Network(4, txs=sample_txs())
    reference.start()
    assert reference.run_until_height(2)
    (block1, cert1), (block2, cert2) = reference.finalized["V0"][:2]

    net = Network(4, txs=sample_txs(), drop=lambda src, dst, message: True)
    net.start(only=["V0"])
    engine = net.engines["V0"]

    def send_height(block, certificate):
        net.deliver("V0", Message(Kind.BODY, block.header.proposer, block.header.height, block))
        for vote in certificate.precommits:
            net.deliver("V0", Message(Kind.VOTE, vote.voter, vote.height, vote))

    send_height(block2, cert2)
    assert engine.height == 1 and net.finalized["V0"] == []

    send_height(block1, cert1)
    assert net.hashes("V0") == reference.hashes("V0")[:2]
    assert engine.height == 3

    old = cert1.precommits[0]
    net.deliver("V0", Message(Kind.VOTE, old.voter, old.height, old))
    assert net.event_names("V0")[-1] == "ignored_old_height"


def test_engine_rejects_node_outside_validator_set():
    net = Network(4)

    with pytest.raises(ValueError):
        ConsensusEngine(
            node_id="V9",
            keypair=net.keys["V0"],
            validator_keys={node: pair.public_key for node, pair in net.keys.items()},
            timeouts=TIMEOUTS,
            state=State(),
            mempool=Mempool(),
            max_txs_per_block=1,
            broadcast=lambda message: None,
            schedule_timeout=lambda *args: None,
            cancel_timeout=lambda handle: None,
            on_finalize=lambda block, certificate: None,
            logger=lambda event, **fields: None,
        )


def test_engine_never_sends_node_to_the_logger():
    net = Network(4, txs=sample_txs())
    net.start()

    assert net.run_until_height(2)
    for node, events in net.events.items():
        assert events, node
        assert all("node" not in fields for _, fields in events)


def test_engine_logs_through_event_logger(tmp_path):
    net = Network(4)
    event_logger = EventLogger("V1", "engine_logger", tmp_path)
    engine = ConsensusEngine(
        node_id="V1",
        keypair=net.keys["V1"],
        validator_keys={node: pair.public_key for node, pair in net.keys.items()},
        timeouts=TIMEOUTS,
        state=State(),
        mempool=Mempool(),
        max_txs_per_block=1,
        broadcast=lambda message: None,
        schedule_timeout=lambda *args: None,
        cancel_timeout=lambda handle: None,
        on_finalize=lambda block, certificate: None,
        logger=event_logger.event,
    )

    engine.start()

    records = [json.loads(line) for line in event_logger.path.read_text().splitlines()]
    assert [record["event"] for record in records][:3] == ["new_round", "propose", "prevote"]
    assert all(record["node"] == "V1" for record in records)
    assert all(record["height"] == 1 and record["round"] == 0 for record in records)
