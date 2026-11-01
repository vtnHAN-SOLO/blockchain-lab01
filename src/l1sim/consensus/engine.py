"""Tendermint-style consensus engine: rounds, steps, timeouts and proposer rotation.

The engine never touches the network, the clock or ``asyncio``. Everything that
leaves the engine goes through callbacks injected by ``node.py`` (or by the
test harness), which keeps the state machine deterministic and unit-testable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from l1sim.consensus.vote_set import ADDED, DUPLICATE, EQUIVOCATION, VoteSet
from l1sim.crypto import KeyPair, sign, verify
from l1sim.encoding import block_hash
from l1sim.execution import Mempool, State, apply_block, compute_tx_root, validate_block
from l1sim.models import (
    GENESIS_HASH,
    Block,
    BlockHeader,
    CommitCertificate,
    Domain,
    Kind,
    Message,
    Phase,
    Vote,
)

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class Step(str, Enum):
    """The three steps of a round, in the order a round goes through them.

    ``PROPOSE``: waiting for the round's proposer to send a block.
    ``PREVOTE``: this node has prevoted and waits for a PREVOTE quorum (polka).
    ``PRECOMMIT``: this node has precommitted and waits for a PRECOMMIT quorum.

    The value is also what the engine passes to ``schedule_timeout`` and
    receives back in ``on_timeout``, so a timeout knows which step it ends.
    """

    PROPOSE = "PROPOSE"
    PREVOTE = "PREVOTE"
    PRECOMMIT = "PRECOMMIT"


@dataclass(frozen=True)
class TimeoutConfig:
    """Timeout durations for each step, growing linearly with the round number.

    A step in round r waits ``base + r * delta_ms`` milliseconds, where base is
    ``propose_ms``, ``prevote_ms`` or ``precommit_ms``. Growing timeouts are what
    gives liveness under partial synchrony: before GST messages may be slower
    than any fixed timeout, but after GST message delay is bounded, so some
    round eventually has timeouts long enough for every honest message to
    arrive (Concept Guide, "Timeout"; test T6).

    The values always come from the scenario config (see :meth:`from_config`);
    the engine has no built-in defaults.

    Attributes:
        propose_ms: Base time to wait for the proposal before prevoting NIL.
        prevote_ms: Base time to wait for a polka before precommitting NIL.
        precommit_ms: Base time to wait for a commit before starting the next
            round.
        delta_ms: Extra time added to every step for each round after round 0.
    """

    propose_ms: int
    prevote_ms: int
    precommit_ms: int
    delta_ms: int

    def __post_init__(self) -> None:
        """Reject values that would make timeouts negative or non-integer.

        Raises:
            ValueError: If any field is not an ``int`` (``bool`` is refused too)
                or is negative. Floats are refused because they are not allowed
                in anything that may end up in canonical encoding or logs.
        """
        for name in ("propose_ms", "prevote_ms", "precommit_ms", "delta_ms"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @classmethod
    def from_config(cls, consensus: Mapping[str, object]) -> TimeoutConfig:
        """Build the timeouts from the ``"consensus"`` block of a scenario config.

        Expected keys (all in milliseconds):
        ``timeout_propose_ms``, ``timeout_prevote_ms``, ``timeout_precommit_ms``
        and ``timeout_delta_ms``. Extra keys are ignored.

        Args:
            consensus: The ``"consensus"`` mapping, for example
                ``json.load(f)["consensus"]``.

        Returns:
            A validated :class:`TimeoutConfig`.

        Raises:
            KeyError: If a key is missing. There is deliberately no default, so a
                config that forgets ``timeout_delta_ms`` fails loudly instead of
                silently running with fixed timeouts.
            ValueError: If a value is not a non-negative integer.

        Example:
            >>> TimeoutConfig.from_config({"timeout_propose_ms": 1000,
            ...     "timeout_prevote_ms": 1000, "timeout_precommit_ms": 1000,
            ...     "timeout_delta_ms": 300}).duration(Step.PROPOSE, 2)
            1600
        """
        return cls(
            propose_ms=consensus["timeout_propose_ms"],
            prevote_ms=consensus["timeout_prevote_ms"],
            precommit_ms=consensus["timeout_precommit_ms"],
            delta_ms=consensus["timeout_delta_ms"],
        )

    def duration(self, step: Step, round: int) -> int:
        """Return how long ``step`` may last in ``round``, in milliseconds.

        Args:
            step: The step being timed.
            round: The round number (0 for the first round of a height).

        Returns:
            ``base(step) + round * delta_ms``.

        Raises:
            ValueError: If ``round`` is negative.
        """
        if type(round) is not int or round < 0:
            raise ValueError("round must be a non-negative integer")
        base = {
            Step.PROPOSE: self.propose_ms,
            Step.PREVOTE: self.prevote_ms,
            Step.PRECOMMIT: self.precommit_ms,
        }[Step(step)]
        return base + round * self.delta_ms


def proposer_for(validators: Sequence[str] | Mapping[str, object], height: int, round: int) -> str:
    """Return the validator allowed to propose a block at ``(height, round)``.

    The rule is ``sorted(validators)[(height + round) % n]``. Every node computes
    it locally from the fixed validator set, so all honest nodes agree on the
    proposer without exchanging messages. Sorting makes the result independent
    of the order in which validators appear in the config. Adding the round
    rotates the proposer when a round fails, so a silent proposer only costs one
    round (test T6).

    Args:
        validators: Validator ids, either as a sequence or as the keys of a
            mapping such as ``validator_keys``.
        height: Block height.
        round: Round number at that height.

    Returns:
        The id of the proposer.

    Raises:
        ValueError: If the validator set is empty.

    Example:
        >>> proposer_for(["V2", "V0", "V1", "V3"], height=1, round=0)
        'V1'
        >>> proposer_for(["V2", "V0", "V1", "V3"], height=1, round=1)
        'V2'
    """
    ordered = sorted(validators)
    if not ordered:
        raise ValueError("validator set must not be empty")
    return ordered[(height + round) % len(ordered)]


class ConsensusEngine:
    """Round-based BFT state machine for one validator (Tendermint style, no locking yet).

    One engine runs inside each node. For every height it goes through rounds,
    and every round goes through three steps:

    1. **Propose**: the round's proposer (:func:`proposer_for`) builds a block
       from the mempool, signs the header with ``Domain.HEADER`` and broadcasts
       it. Everyone else waits ``timeout_propose``.
    2. **Prevote**: a node prevotes the block if it is valid (rule 3, via
       :func:`l1sim.execution.validate_block`), otherwise NIL. If no proposal
       arrives before the propose timeout, it prevotes NIL.
    3. **Precommit**: when the node sees 2f+1 PREVOTEs for block B in its
       current round (a polka) *and* it has B and B is valid, it precommits B.
       The decision follows the polka the node sees, not its own prevote. On a
       NIL polka, or when the prevote timeout fires first, it precommits NIL.

    **Finalize**: as soon as 2f+1 PRECOMMITs for the same block exist in *any*
    round of the current height and the node has that block, the block is
    final, whatever round or step the node is in. The engine applies the block
    to its state, stores the commit certificate, calls ``on_finalize`` and
    starts round 0 of the next height. If the precommit timeout fires without a
    commit, the engine starts the next round with the next proposer and longer
    timeouts.

    **Safety rules implemented here**

    - Rule 1–2: at most one PREVOTE and one PRECOMMIT per (height, round);
      see :meth:`_sign_vote` and :attr:`signed_votes`.
    - Rule 3: only valid blocks are prevoted or precommitted; see
      :meth:`_is_valid`.
    - Rule 6: votes are only ever signed for the engine's own current height and
      round, with ``Domain.VOTE``.
    - Rule 7 and equivocation: delegated to :class:`VoteSet`.
    - Rules 4–5 (locking) are **not** implemented yet. :attr:`locked_block`,
      :attr:`locked_round` and the single decision point
      :meth:`_decide_prevote` are where PR 2 adds them.

    **Isolation**

    The engine imports no network, ``asyncio`` or clock code and uses no
    randomness. It only talks to the outside through the callbacks given to
    the constructor, so the same inputs in the same order always produce the
    same outputs. ``node.py`` wires the callbacks to gossip and to
    ``loop.call_later``; the unit tests wire them to an in-memory network and a
    fake clock.

    **Buffering**

    Votes for future rounds or heights are stored in the :class:`VoteSet` right
    away (it is keyed by height). Proposals for future rounds or heights are
    stored in :attr:`proposals` and validated when the engine reaches them.
    Messages for heights that are already finalized are ignored.

    Attributes:
        node_id: This validator's id.
        height: Height currently being decided. Starts at ``start_height``.
        round: Current round at :attr:`height`.
        step: Current :class:`Step` in :attr:`round`.
        locked_block: Placeholder for PR 2 (rules 4–5). Always ``None`` here.
        locked_round: Placeholder for PR 2. ``-1`` means "not locked".
        signed_votes: Every vote this node signed at the current height, keyed
            by ``(height, round, phase)``. Used to enforce rule 1–2.
        votes: The :class:`VoteSet` holding all received and own votes.
        proposals: First valid-looking proposal (correct proposer, correct
            header signature) received for each ``(height, round)``.
        finalized: ``height -> (block, commit certificate)`` for every block this
            engine finalized, kept for block sync and results.
        state: Application state after the last finalized block.
        parent_hash: Hash of the last finalized block (``GENESIS_HASH`` before
            the first one).
    """

    def __init__(
        self,
        *,
        node_id: str,
        keypair: KeyPair,
        validator_keys: Mapping[str, Ed25519PublicKey],
        timeouts: TimeoutConfig,
        state: State,
        mempool: Mempool,
        max_txs_per_block: int,
        broadcast: Callable[[Message], None],
        schedule_timeout: Callable[[int, int, int, Step], Any],
        cancel_timeout: Callable[[Any], None],
        on_finalize: Callable[[Block, CommitCertificate], None],
        logger: Callable[..., None],
        user_keys: Mapping[str, Ed25519PublicKey] | None = None,
        start_height: int = 1,
        parent_hash: str = GENESIS_HASH,
    ) -> None:
        """Create an engine that is ready but idle; call :meth:`start` to begin.

        Args:
            node_id: Id of this validator; must be a key of ``validator_keys``.
            keypair: This validator's key pair, used to sign headers and votes.
            validator_keys: The fixed validator set, ``id -> public key``. It
                decides the proposer order, the quorum size and which votes
                are accepted.
            timeouts: Step timeouts, normally ``TimeoutConfig.from_config(...)``.
            state: State after the last finalized block (the genesis state for
                a fresh chain). The engine replaces it with a new object after
                each finalized block; it never mutates the given object.
            mempool: Pending transactions. The proposer takes transactions from
                it with ``Mempool.take``; finalized transactions are removed with
                ``Mempool.remove_committed``.
            max_txs_per_block: Upper bound passed to ``Mempool.take``.
            broadcast: ``broadcast(message)`` sends a :class:`Message` to every
                other validator. It must not deliver the message back to this
                engine; the engine handles its own proposal and votes locally.
            schedule_timeout: ``schedule_timeout(delay_ms, height, round, step)``
                must call :meth:`on_timeout` with the same ``height``, ``round``
                and ``step`` after ``delay_ms`` milliseconds and return a handle.
            cancel_timeout: ``cancel_timeout(handle)`` cancels a pending timeout.
                Cancelling a handle that already fired must be harmless.
            on_finalize: ``on_finalize(block, certificate)`` is called once per
                finalized block, in height order, after the engine's own state
                has been updated.
            logger: ``logger(event, **fields)`` records one event. The engine
                always passes ``height`` and ``round`` but never ``node``: the
                logger is created per node and writes the node id itself, so
                ``EventLogger(node_id, ...).event`` can be passed directly. The
                logger adds timestamps if it wants them.
            user_keys: Optional ``sender -> public key`` map forwarded to
                ``validate_block`` to check transaction signatures. ``None``
                means node.py already checked them before the mempool.
            start_height: First height to decide (1 for a fresh chain).
            parent_hash: Hash of the block at ``start_height - 1``.

        Raises:
            ValueError: If ``node_id`` is not a validator, if
                ``max_txs_per_block`` is negative, or if the validator set is
                empty.
        """
        if node_id not in validator_keys:
            raise ValueError(f"{node_id!r} is not in the validator set")
        if type(max_txs_per_block) is not int or max_txs_per_block < 0:
            raise ValueError("max_txs_per_block must be a non-negative integer")

        self.node_id = node_id
        self._keypair = keypair
        self._validator_keys = dict(validator_keys)
        self._validators = sorted(self._validator_keys)
        self._timeouts = timeouts
        self._mempool = mempool
        self._max_txs = max_txs_per_block
        self._broadcast = broadcast
        self._schedule_timeout = schedule_timeout
        self._cancel_timeout = cancel_timeout
        self._on_finalize = on_finalize
        self._logger = logger
        self._user_keys = None if user_keys is None else dict(user_keys)

        self.state = state
        self.parent_hash = parent_hash
        self.height = start_height
        self.round = 0
        self.step = Step.PROPOSE

        # Rules 4-5 (PR 2): set on a polka, consulted only in _decide_prevote.
        self.locked_block: Block | None = None
        self.locked_round = -1

        self.signed_votes: dict[tuple[int, int, Phase], Vote] = {}
        self.votes = VoteSet(self._validator_keys)
        self.proposals: dict[tuple[int, int], Block] = {}
        self.finalized: dict[int, tuple[Block, CommitCertificate]] = {}

        self._blocks: dict[str, Block] = {}
        self._validity: dict[str, bool] = {}
        self._waiting_logged: set[str] = set()
        self._timeout_handle: Any = None

    # ------------------------------------------------------------------ inputs

    def start(self) -> None:
        """Begin consensus at :attr:`height` by starting round 0.

        Call it once, after every callback is ready to be used, because the
        proposer of round 0 broadcasts its proposal immediately.
        """
        self.start_round(0)

    def start_round(self, round: int) -> None:
        """Enter ``round`` at the current height and run every rule that now applies.

        Resets the step to PROPOSE, schedules the propose timeout and, if this
        node is the round's proposer, builds and broadcasts a proposal. Votes
        and proposals already buffered for this round are processed right away,
        so a node that arrives late in a round catches up immediately.

        Args:
            round: The round to enter. Must not be lower than the current round
                (rounds only move forward within a height).

        Raises:
            ValueError: If ``round`` is lower than :attr:`round`.
        """
        self._enter_round(round)
        self._progress()

    def on_proposal(self, block: Block) -> None:
        """Handle a proposed block received from the network.

        The block is accepted into :attr:`proposals` only if all of these hold:

        - its height is not already finalized (older heights are ignored and
          logged as ``ignored_old_height``);
        - ``header.proposer`` is the proposer for ``(header.height,
          header.round)``;
        - the header signature verifies with that proposer's key under
          ``Domain.HEADER``. This is checked here, before storing, so a forged
          proposal can never occupy the slot of the real one;
        - it is the first proposal for that ``(height, round)``. A second,
          different proposal for the same slot (a Byzantine proposer) is logged
          as ``proposal_conflict`` and dropped; the same proposal again is
          silently ignored.

        Full validation (rule 3) is deferred until the engine needs the block at
        its current height, because only then is the parent state known.

        Args:
            block: The proposed block (header and transactions).
        """
        header = block.header
        if header.height < self.height:
            self._log("ignored_old_height", kind="proposal", msg_height=header.height)
            return

        expected = proposer_for(self._validators, header.height, header.round)
        if header.proposer != expected:
            self._log("proposal_rejected", reason="bad_proposer", proposer=header.proposer)
            return
        if not verify(self._validator_keys[expected], Domain.HEADER, header, header.signature):
            self._log("proposal_rejected", reason="bad_signature", proposer=header.proposer)
            return

        if not self._store_proposal(block):
            return
        if header.height == self.height:
            self._progress()

    def on_vote(self, vote: Vote) -> None:
        """Handle a PREVOTE or PRECOMMIT received from the network.

        Votes for finalized heights are ignored. Every other vote goes to
        :meth:`VoteSet.add`, which checks the signature, deduplicates (rule 7)
        and detects equivocation. Votes for future rounds and heights are kept
        there until the engine reaches them. When a vote for the current height
        is added, the engine re-checks every rule, because that vote may complete
        a polka or a commit.

        Logged events: ``vote_rejected`` (bad signature, wrong domain or unknown
        voter), ``duplicate_ignored`` and ``equivocation_detected``.

        Args:
            vote: The received vote.
        """
        if vote.height < self.height:
            self._log("ignored_old_height", kind="vote", msg_height=vote.height)
            return

        result = self.votes.add(vote)
        fields = {
            "voter": vote.voter,
            "phase": vote.phase.value,
            "vote_height": vote.height,
            "vote_round": vote.round,
            "block_hash": vote.block_hash,
        }
        if result == DUPLICATE:
            self._log("duplicate_ignored", **fields)
        elif result == EQUIVOCATION:
            self._log("equivocation_detected", **fields)
        elif result != ADDED:
            self._log("vote_rejected", **fields)
        elif vote.height == self.height:
            self._progress()

    def on_timeout(self, height: int, round: int, step: Step) -> None:
        """Handle a timeout scheduled earlier through ``schedule_timeout``.

        A timeout only counts if the engine is still at the same height, round
        and step it was scheduled for; otherwise it is stale (the engine already
        moved on) and is ignored. Effects:

        - PROPOSE timeout: prevote NIL (no proposal in time).
        - PREVOTE timeout: precommit NIL (no polka in time, or polka for a block
          this node does not have).
        - PRECOMMIT timeout: start round ``round + 1`` with the next proposer and
          longer timeouts (test T6).

        Args:
            height: Height the timeout was scheduled for.
            round: Round the timeout was scheduled for.
            step: Step the timeout was scheduled for.
        """
        step = Step(step)
        if (height, round, step) != (self.height, self.round, self.step):
            return

        self._timeout_handle = None
        self._log("timeout", step=step.value)
        if step is Step.PROPOSE:
            self._prevote(self._decide_prevote(None))
        elif step is Step.PREVOTE:
            self._precommit(None)
        else:
            self._enter_round(round + 1)
        self._progress()

    # --------------------------------------------------------------- decisions

    def _decide_prevote(self, block: Block | None) -> str | None:
        """Decide what to prevote for the current round's proposal (the only place).

        Today this is rule 3 only: prevote the block if it is valid, otherwise
        NIL. Keeping the decision in one function is deliberate: PR 2 adds
        locking (rules 4–5) here, e.g. "if :attr:`locked_block` is set, prevote
        it unless there is a newer polka for the proposal", without touching any
        other code path.

        Args:
            block: The proposal for ``(height, round)``, or ``None`` when the
                propose timeout fired without one.

        Returns:
            The block hash to prevote, or ``None`` for NIL.
        """
        if block is None or not self._is_valid(block):
            return None
        return block_hash(block.header)

    def _is_valid(self, block: Block) -> bool:
        """Return whether ``block`` is valid at the current height (rule 3).

        Checks the height (``validate_block`` does not) and then calls
        :func:`l1sim.execution.validate_block` with the current parent hash,
        state and expected proposer for the block's round. That covers the
        proposer, the parent link, the header signature, ``tx_root``, every
        transaction (ownership, nonce, signatures when ``user_keys`` is given)
        and the post-execution ``state_hash``. The result is cached per block
        hash for the current height, and an invalid block is logged once as
        ``invalid_block`` with the reason.

        Args:
            block: A block stored in :attr:`proposals`.

        Returns:
            ``True`` if the block may be voted for and finalized at this height.
        """
        digest = block_hash(block.header)
        if digest in self._validity:
            return self._validity[digest]

        header = block.header
        if header.height != self.height:
            ok, reason = False, "bad_height"
        else:
            ok, reason = validate_block(
                block,
                self.parent_hash,
                self.state,
                proposer_for(self._validators, header.height, header.round),
                self._validator_keys,
                self._user_keys,
            )
        if not ok:
            self._log("invalid_block", block_hash=digest, reason=reason)
        self._validity[digest] = ok
        return ok

    def _progress(self) -> None:
        """Apply every rule whose condition now holds, until nothing changes.

        Called at the end of every public input (proposal, vote, timeout, new
        round). Each pass tries, in order:

        1. finalize (2f+1 PRECOMMITs for a block we have, in any round);
        2. in PROPOSE: prevote once the round's proposal is known;
        3. in PREVOTE: precommit B on a polka for B when we have B and it is
           valid; precommit NIL on a NIL polka.

        Signing a vote adds it to our own :class:`VoteSet`, which can complete a
        quorum, so the loop repeats until a pass changes nothing. Internal steps
        never call this method themselves, which keeps the control flow flat and
        avoids re-entrant calls.
        """
        while True:
            if self._try_finalize():
                continue

            height, round = self.height, self.round
            if self.step is Step.PROPOSE:
                block = self.proposals.get((height, round))
                if block is not None:
                    self._prevote(self._decide_prevote(block))
                    continue

            if self.step is Step.PREVOTE:
                polka = self.votes.has_quorum(height, round, Phase.PREVOTE)
                block = self._blocks.get(polka) if polka else None
                if block is not None and self._is_valid(block):
                    self._precommit(polka)
                    continue
                if self.votes.has_nil_quorum(height, round, Phase.PREVOTE):
                    self._precommit(None)
                    continue
            return

    def _try_finalize(self) -> bool:
        """Finalize the current height if any round has a PRECOMMIT quorum for a known block.

        Every round of the current height is checked (in ascending order), not
        only the current one, so a node that already moved to a later round
        still finalizes when late precommits for an earlier round arrive. The
        node's own lock, round and step do not matter. If the quorum is for a
        block this node never received, it waits (logged once as
        ``commit_waiting_for_block``); fetching that block is block sync's job
        (PR 2). A committed block that fails local validation is never applied.

        Returns:
            ``True`` if a block was finalized (the engine is now at the next
            height), ``False`` otherwise.
        """
        for round in self.votes.rounds(self.height):
            digest = self.votes.has_quorum(self.height, round, Phase.PRECOMMIT)
            if digest is None:
                continue
            block = self._blocks.get(digest)
            if block is None:
                if digest not in self._waiting_logged:
                    self._waiting_logged.add(digest)
                    self._log("commit_waiting_for_block", block_hash=digest, commit_round=round)
                continue
            if self._is_valid(block):
                self._finalize(block, self.votes.certificate(self.height, round, digest))
                return True
        return False

    # ----------------------------------------------------------------- actions

    def _enter_round(self, round: int) -> None:
        """Switch to ``round`` at the current height without running the rules.

        Sets the step to PROPOSE, logs ``new_round``, schedules the propose
        timeout and, if this node is the proposer, proposes. Used by
        :meth:`start_round`, by the precommit timeout and after finalizing.

        Args:
            round: The round to enter.

        Raises:
            ValueError: If ``round`` is lower than :attr:`round`.
        """
        if round < self.round:
            raise ValueError("rounds only move forward within a height")
        self.round = round
        self.step = Step.PROPOSE
        proposer = proposer_for(self._validators, self.height, round)
        self._log("new_round", proposer=proposer)
        self._schedule(Step.PROPOSE)
        if proposer == self.node_id:
            self._propose()

    def _propose(self) -> None:
        """Build, sign, store and broadcast this node's proposal for the current round.

        The block contains up to ``max_txs_per_block`` transactions from
        ``Mempool.take`` (valid on the current state, sorted by sender and
        nonce). The header commits to ``compute_tx_root(txs)`` and to the
        ``state_hash`` obtained with ``apply_block``, so other nodes can re-run
        the transactions and compare. The header is signed with
        ``Domain.HEADER``.

        PR 2: when locked, the proposer should re-propose the locked block
        instead of building a new one.
        """
        txs = self._mempool.take(self.state, self._max_txs)
        header = BlockHeader(
            height=self.height,
            round=self.round,
            parent_hash=self.parent_hash,
            tx_root=compute_tx_root(txs),
            state_hash=apply_block(self.state, txs).state_hash(),
            proposer=self.node_id,
        )
        header = replace(header, signature=sign(self._keypair.private_key, Domain.HEADER, header))
        block = Block(header, tuple(txs))

        self._store_proposal(block)
        self._log("propose", block_hash=block_hash(header), txs=len(block.txs))
        # TODO(node.py): the lab spec requires header-before-body. node.py will split this
        # into HEADER (gossiped) -> BODY_REQUEST -> BODY (sent only to the requester).
        self._broadcast(Message(Kind.BODY, self.node_id, self.height, block))

    def _store_proposal(self, block: Block) -> bool:
        """Remember ``block`` as the proposal for its ``(height, round)`` if the slot is free.

        Also indexes the block by hash so a polka or commit for it can find it.

        Args:
            block: A proposal whose proposer and signature were already checked.

        Returns:
            ``True`` if stored, ``False`` if the slot already holds a proposal
            (the same one, or a conflicting one, which is logged).
        """
        slot = (block.header.height, block.header.round)
        digest = block_hash(block.header)
        existing = self.proposals.get(slot)
        if existing is not None:
            if block_hash(existing.header) != digest:
                self._log("proposal_conflict", block_hash=digest, proposer=block.header.proposer)
            return False
        self.proposals[slot] = block
        self._blocks[digest] = block
        return True

    def _prevote(self, block_hash: str | None) -> None:
        """Sign and send this round's PREVOTE, then move to the PREVOTE step.

        Args:
            block_hash: The block to prevote, or ``None`` for NIL.
        """
        self._sign_vote(Phase.PREVOTE, block_hash)
        self.step = Step.PREVOTE
        self._schedule(Step.PREVOTE)

    def _precommit(self, block_hash: str | None) -> None:
        """Sign and send this round's PRECOMMIT, then move to the PRECOMMIT step.

        PR 2: precommitting a block on a polka is where the lock is taken
        (``locked_block = B``, ``locked_round = round``).

        Args:
            block_hash: The block to precommit, or ``None`` for NIL.
        """
        self._sign_vote(Phase.PRECOMMIT, block_hash)
        self.step = Step.PRECOMMIT
        self._schedule(Step.PRECOMMIT)

    def _sign_vote(self, phase: Phase, block_hash: str | None) -> Vote | None:
        """Sign, record, count and broadcast a vote for the current height and round.

        Rule 1–2: if this node already signed a vote for
        ``(height, round, phase)``, nothing is signed, even if ``block_hash``
        differs (that refusal is logged as ``refuse_double_sign``). This is what
        guarantees an honest node never equivocates, whatever order inputs
        arrive in.

        Rule 6: the vote always uses the engine's own :attr:`height` and
        :attr:`round` and is signed with ``Domain.VOTE``, so it cannot be signed
        for the wrong height, round, phase or domain.

        The vote is added to our own :class:`VoteSet` before broadcasting,
        because ``broadcast`` does not deliver messages back to the sender.

        Args:
            phase: ``Phase.PREVOTE`` or ``Phase.PRECOMMIT``.
            block_hash: The block hash, or ``None`` for NIL.

        Returns:
            The signed vote, or ``None`` if rule 1–2 refused to sign.
        """
        slot = (self.height, self.round, phase)
        previous = self.signed_votes.get(slot)
        if previous is not None:
            if previous.block_hash != block_hash:
                self._log("refuse_double_sign", phase=phase.value, block_hash=block_hash)
            return None

        vote = Vote(phase, self.height, self.round, block_hash, self.node_id)
        vote = replace(vote, signature=sign(self._keypair.private_key, Domain.VOTE, vote))
        self.signed_votes[slot] = vote
        self.votes.add(vote)
        self._log(phase.value.lower(), block_hash=block_hash)
        self._broadcast(Message(Kind.VOTE, self.node_id, self.height, vote))
        return vote

    def _finalize(self, block: Block, certificate: CommitCertificate) -> None:
        """Commit ``block`` at the current height and move to round 0 of the next height.

        Steps, in order: cancel the pending timeout; apply the block's
        transactions to a copy of the state; remove them from the mempool; store
        ``(block, certificate)`` in :attr:`finalized`; log ``finalize``; call
        ``on_finalize``; then reset the per-height data (lock, signed votes,
        caches, old votes and proposals) and enter round 0 of ``height + 1``.
        Votes and proposals already buffered for the new height are kept.

        Args:
            block: The block with a PRECOMMIT quorum; already validated.
            certificate: The PRECOMMIT quorum proving it.
        """
        self._cancel_pending()
        header = block.header
        digest = block_hash(header)
        self.state = apply_block(self.state, block.txs)
        self._mempool.remove_committed(block.txs)
        self.finalized[header.height] = (block, certificate)
        self._log(
            "finalize",
            block_hash=digest,
            commit_round=certificate.precommits[0].round,
            proposer=header.proposer,
            state_hash=self.state.state_hash(),
            txs=len(block.txs),
        )
        self._on_finalize(block, certificate)

        self.parent_hash = digest
        self.height = header.height + 1
        self.round = 0
        self.locked_block = None
        self.locked_round = -1
        self.signed_votes.clear()
        self._validity.clear()
        self._waiting_logged.clear()
        self.votes.prune_below(self.height)
        self.proposals = {
            slot: proposal for slot, proposal in self.proposals.items() if slot[0] >= self.height
        }
        self._blocks = {
            digest: proposal
            for digest, proposal in self._blocks.items()
            if proposal.header.height >= self.height
        }
        self._enter_round(0)

    # ----------------------------------------------------------------- helpers

    def _schedule(self, step: Step) -> None:
        """Replace the pending timeout with one for ``step`` in the current round.

        Only one timeout is pending at a time: entering a new step cancels the
        previous one. The duration is ``TimeoutConfig.duration(step, round)``.

        Args:
            step: The step that just started.
        """
        self._cancel_pending()
        delay = self._timeouts.duration(step, self.round)
        self._timeout_handle = self._schedule_timeout(delay, self.height, self.round, step)

    def _cancel_pending(self) -> None:
        """Cancel the pending timeout, if any, through the ``cancel_timeout`` callback."""
        if self._timeout_handle is not None:
            self._cancel_timeout(self._timeout_handle)
            self._timeout_handle = None

    def _log(self, event: str, **fields: Any) -> None:
        """Send one event to the injected logger with height and round attached.

        The node id is not sent: the logger belongs to one node and adds it
        itself. ``EventLogger.event`` rejects a ``node`` field as a reserved key.

        Args:
            event: Event type, for example ``"prevote"`` or ``"finalize"``.
            **fields: Extra event fields. They must not be named ``node``,
                ``height``, ``round``, ``time`` or ``event``; message heights use
                names such as ``msg_height`` or ``vote_height`` instead.
        """
        self._logger(event, height=self.height, round=self.round, **fields)
