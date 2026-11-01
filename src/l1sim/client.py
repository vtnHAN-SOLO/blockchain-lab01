"""Scenario transaction construction and submission helpers."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from l1sim.config import ConfigError, ScenarioConfig, load_scenario
from l1sim.crypto import keypair_from_seed, sign
from l1sim.encoding import encode
from l1sim.models import Domain, Tx

TxHandler = Callable[[Tx], None]


class TxClient:
    """Build and submit signed transactions.

    The handler is injected by the node/network integration layer.  This keeps
    client signing deterministic and independently testable before networking is
    implemented.
    """

    def __init__(
        self,
        seed: int,
        handler: TxHandler,
        private_keys: Mapping[str, Ed25519PrivateKey] | None = None,
    ) -> None:
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self.seed = seed
        self.handler = handler
        self.private_keys = dict(private_keys or {})
        for sender, private_key in self.private_keys.items():
            if not isinstance(sender, str) or not sender:
                raise TypeError("private key sender must be a non-empty string")
            if not isinstance(private_key, Ed25519PrivateKey):
                raise TypeError("private_keys values must be Ed25519PrivateKey instances")

    def make_tx(self, sender: str, key: str, value: str, nonce: int) -> Tx:
        private_key = self.private_keys.get(sender)
        return make_signed_tx(
            self.seed,
            sender,
            key,
            value,
            nonce,
            private_key=private_key,
        )

    def submit(self, tx: Tx) -> None:
        if not isinstance(tx, Tx):
            raise TypeError("tx must be a Tx instance")
        self.handler(tx)

    def submit_scenario(self, scenario: ScenarioConfig) -> tuple[Tx, ...]:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario must be a ScenarioConfig instance")
        submitted: list[Tx] = []
        for raw_tx in scenario.transactions:
            tx = self.make_tx(
                sender=raw_tx["sender"],
                key=raw_tx["key"],
                value=raw_tx["value"],
                nonce=raw_tx["nonce"],
            )
            self.submit(tx)
            submitted.append(tx)
        return tuple(submitted)


def make_signed_tx(
    seed: int,
    sender: str,
    key: str,
    value: str,
    nonce: int,
    private_key: Ed25519PrivateKey | None = None,
) -> Tx:
    """Convenience function for constructing one signed transaction."""
    if private_key is not None and not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key must be an Ed25519PrivateKey instance")
    unsigned = Tx(sender=sender, key=key, value=value, nonce=nonce)
    signing_key = private_key or keypair_from_seed(seed, sender).private_key
    signature = sign(signing_key, Domain.TX, unsigned)
    return Tx(sender=sender, key=key, value=value, nonce=nonce, signature=signature)


def build_scenario_transactions(scenario: ScenarioConfig) -> tuple[Tx, ...]:
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig instance")
    return tuple(
        make_signed_tx(
            scenario.seed,
            raw_tx["sender"],
            raw_tx["key"],
            raw_tx["value"],
            raw_tx["nonce"],
        )
        for raw_tx in scenario.transactions
    )


def send_scenario(
    scenario: ScenarioConfig,
    handler: TxHandler,
    private_keys: Mapping[str, Ed25519PrivateKey] | None = None,
) -> tuple[Tx, ...]:
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig instance")
    client = TxClient(scenario.seed, handler, private_keys=private_keys)
    return client.submit_scenario(scenario)


def _print_transactions(scenario_path: str) -> int:
    scenario = load_scenario(scenario_path)
    for tx in build_scenario_transactions(scenario):
        encoded = encode(tx) + b"\n"
        output = getattr(sys.stdout, "buffer", None)
        if output is not None:
            output.write(encoded)
            output.flush()
        else:
            sys.stdout.write(encoded.decode("utf-8"))
            sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build transactions from a Lab01 scenario")
    parser.add_argument("--scenario", required=True, help="scenario JSON file")
    args = parser.parse_args(argv)
    try:
        return _print_transactions(args.scenario)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
