"""Market: Collateralized Bilateral Contract Market (Mechanism I) for VAMP.

Contract type chi = (target_formula, deadline_T, loss_l):
    Long payoff:  +(1-l) if resolved by T, else -l
    Short payoff: -(1-l) if resolved by T, else +l

Portfolio: (cash, held_positions, posted_offers)
    Worst-case balance: cash - sum max_liability(position) >= 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np


@dataclass
class ContractType:
    """Contract specification chi = (target, deadline, loss)."""
    target: int         # target formula index
    deadline: int       # T: settlement deadline (timestep)
    loss: float         # l in [0,1]: loss parameter


@dataclass
class Position:
    """A held position in a contract."""
    contract: ContractType
    side: str           # 'long' or 'short'
    settled: bool = False
    pnl: float = 0.0   # realized P&L after settlement


@dataclass
class Offer:
    """A posted offer in the market."""
    offer_id: int
    contract: ContractType
    side: str           # 'long' or 'short' — what the POSTER holds
    price: float        # price the acceptor pays
    poster: int         # agent_id of poster


class BilateralContractMarket:
    """Collateralized bilateral contract market."""

    def __init__(self, max_offers: int, max_own_offers: int):
        self.max_offers = max_offers
        self.max_own_offers = max_own_offers

        # Agent state
        self.cash: Dict[int, float] = {}
        self.positions: Dict[int, List[Position]] = {}
        self.offers: Dict[int, Offer] = {}  # offer_id -> Offer
        self._next_offer_id = 0

    def init_agent(self, agent_id: int, initial_cash: float) -> None:
        """Initialize agent portfolio."""
        self.cash[agent_id] = initial_cash
        self.positions[agent_id] = []

    def reset(self, n_agents: int, initial_cash: float) -> None:
        """Reset market state."""
        self.cash.clear()
        self.positions.clear()
        self.offers.clear()
        self._next_offer_id = 0
        for i in range(n_agents):
            self.init_agent(i, initial_cash)

    def _max_liability(self, position: Position) -> float:
        """Maximum possible loss from a position."""
        if position.settled:
            return 0.0
        if position.side == 'long':
            return position.contract.loss  # worst case: not resolved
        else:
            return 1.0 - position.contract.loss  # worst case: resolved

    def worst_case_balance(self, agent_id: int) -> float:
        """Worst-case balance = cash - sum of max liabilities."""
        total_liability = sum(
            self._max_liability(p) for p in self.positions[agent_id] if not p.settled
        )
        return self.cash[agent_id] - total_liability

    def get_cash(self, agent_id: int) -> float:
        return self.cash[agent_id]

    def _agent_offer_count(self, agent_id: int) -> int:
        return sum(1 for o in self.offers.values() if o.poster == agent_id)

    def create_and_post(
        self,
        agent_id: int,
        target: int,
        deadline: int,
        loss: float,
        side: str,
        price: float,
    ) -> Optional[int]:
        """Create a contract and post one side as an offer.

        The poster takes the specified side. The offer is for the counter-side.
        Returns offer_id or None if constraints violated.
        """
        if len(self.offers) >= self.max_offers:
            return None
        if self._agent_offer_count(agent_id) >= self.max_own_offers:
            return None

        contract = ContractType(target=target, deadline=deadline, loss=loss)
        poster_pos = Position(contract=contract, side=side)

        # Check collateral: poster must afford the position
        test_liability = self._max_liability(poster_pos)
        if self.worst_case_balance(agent_id) < test_liability:
            return None

        # Add position to poster
        self.positions[agent_id].append(poster_pos)

        # Create offer for counter-side
        counter_side = 'short' if side == 'long' else 'long'
        offer_id = self._next_offer_id
        self._next_offer_id += 1
        self.offers[offer_id] = Offer(
            offer_id=offer_id,
            contract=contract,
            side=counter_side,
            price=price,
            poster=agent_id,
        )
        return offer_id

    def accept_offer(self, agent_id: int, offer_id: int) -> bool:
        """Accept an existing offer. Returns True on success."""
        if offer_id not in self.offers:
            return False
        offer = self.offers[offer_id]
        if offer.poster == agent_id:
            return False  # can't accept own offer

        acceptor_pos = Position(contract=offer.contract, side=offer.side)

        # Check collateral for acceptor
        test_liability = self._max_liability(acceptor_pos) + offer.price
        if self.worst_case_balance(agent_id) < test_liability:
            return False

        # Transfer
        self.positions[agent_id].append(acceptor_pos)
        self.cash[agent_id] -= offer.price
        self.cash[offer.poster] += offer.price

        del self.offers[offer_id]
        return True

    def cancel_offer(self, agent_id: int, offer_id: int) -> bool:
        """Cancel own offer. Returns True on success."""
        if offer_id not in self.offers:
            return False
        offer = self.offers[offer_id]
        if offer.poster != agent_id:
            return False

        # Remove the poster's position that was created with this offer
        # Find the matching unsettled position
        counter_side = 'short' if offer.side == 'long' else 'long'
        for i, pos in enumerate(self.positions[agent_id]):
            if (not pos.settled
                and pos.contract.target == offer.contract.target
                and pos.contract.deadline == offer.contract.deadline
                and abs(pos.contract.loss - offer.contract.loss) < 1e-9
                and pos.side == counter_side):
                self.positions[agent_id].pop(i)
                break

        del self.offers[offer_id]
        return True

    def settle(self, timestep: int, resolved_set: Set[int]) -> None:
        """Settle all contracts whose deadline has passed or target is resolved."""
        for agent_id in self.positions:
            for pos in self.positions[agent_id]:
                if pos.settled:
                    continue
                c = pos.contract
                if timestep < c.deadline:
                    continue

                is_resolved = c.target in resolved_set
                if pos.side == 'long':
                    pos.pnl = (1.0 - c.loss) if is_resolved else (-c.loss)
                else:
                    pos.pnl = (-(1.0 - c.loss)) if is_resolved else c.loss

                self.cash[agent_id] += pos.pnl
                pos.settled = True

    def get_active_offers(self) -> Dict[int, Offer]:
        """Return all active offers."""
        return dict(self.offers)

    def get_offer_ids_sorted(self) -> List[int]:
        """Return sorted list of active offer IDs."""
        return sorted(self.offers.keys())
