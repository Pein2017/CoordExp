"""Tensor view for token-wise loss computation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from src.common.errors import LossContractError
from src.losses.vocab import TokenVocabularyGroups
from src.supervision import TokenAtom, TokenSequence


@dataclass(frozen=True)
class LossContext:
    logits: torch.Tensor
    token_sequence: TokenSequence
    vocab_groups: TokenVocabularyGroups
    logits_position_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor):
            raise LossContractError(
                "LossContext logits must be a tensor",
                code="loss.logits_type",
                context={"value_type": type(self.logits).__name__},
            )
        if self.logits.ndim != 3 or int(self.logits.shape[0]) != 1:
            raise LossContractError(
                "LossContext expects logits with shape [1, pack_length, vocab_size]",
                code="loss.logits_shape",
                context={"shape": [int(item) for item in self.logits.shape]},
            )
        physical_positions = tuple(atom.causal_logits_position for atom in self.atoms)
        row_positions = _resolve_logits_row_positions(
            logits_length=int(self.logits.shape[1]),
            pack_length=self.token_sequence.pack_length,
            physical_positions=physical_positions,
            logits_position_ids=self.logits_position_ids,
            device=self.logits.device,
        )
        if int(self.logits.shape[2]) != self.vocab_groups.vocab_size:
            raise LossContractError(
                "LossContext logits vocab size must match token vocabulary groups",
                code="loss.logits_vocab_size",
                context={
                    "logits_vocab_size": int(self.logits.shape[2]),
                    "groups_vocab_size": self.vocab_groups.vocab_size,
                },
            )
        for atom in self.token_sequence.atoms:
            self.vocab_groups.validate_atom(atom)
        object.__setattr__(
            self,
            "logits_positions",
            row_positions,
        )
        object.__setattr__(
            self,
            "physical_logits_positions",
            torch.tensor(physical_positions, dtype=torch.long, device=self.logits.device),
        )
        object.__setattr__(
            self,
            "target_ids",
            torch.tensor(
                [atom.token_id for atom in self.atoms],
                dtype=torch.long,
                device=self.logits.device,
            ),
        )
        object.__setattr__(
            self,
            "segment_indices",
            torch.tensor(
                [atom.segment_index for atom in self.atoms],
                dtype=torch.long,
                device=self.logits.device,
            ),
        )
        object.__setattr__(
            self,
            "token_types",
            tuple(atom.token_type for atom in self.atoms),
        )

    @property
    def atoms(self) -> tuple[TokenAtom, ...]:
        return self.token_sequence.atoms

    def select_logits_fp32(
        self,
        *,
        token_types: Iterable[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[TokenAtom, ...]]:
        indices, atoms = self._selected_atom_indices(token_types=token_types)
        positions = self.logits_positions.index_select(0, indices)
        targets = self.target_ids.index_select(0, indices)
        return self.logits[0].index_select(0, positions).float(), targets, atoms

    def _selected_atom_indices(
        self,
        *,
        token_types: Iterable[str] | None,
    ) -> tuple[torch.Tensor, tuple[TokenAtom, ...]]:
        if token_types is None:
            return (
                torch.arange(len(self.atoms), dtype=torch.long, device=self.logits.device),
                self.atoms,
            )
        allowed = frozenset(str(token_type) for token_type in token_types)
        for token_type in sorted(allowed):
            self.vocab_groups.allowed_ids(token_type)
        selected_python_indices = tuple(
            index
            for index, atom in enumerate(self.atoms)
            if atom.token_type in allowed
        )
        return (
            torch.tensor(selected_python_indices, dtype=torch.long, device=self.logits.device),
            tuple(self.atoms[index] for index in selected_python_indices),
        )


def _resolve_logits_row_positions(
    *,
    logits_length: int,
    pack_length: int,
    physical_positions: tuple[int, ...],
    logits_position_ids: tuple[int, ...] | None,
    device: torch.device,
) -> torch.Tensor:
    if logits_position_ids is None:
        if logits_length != pack_length:
            raise LossContractError(
                "LossContext logits length must match TokenSequence pack length",
                code="loss.logits_pack_length",
                context={"logits_length": logits_length, "pack_length": pack_length},
            )
        return torch.tensor(physical_positions, dtype=torch.long, device=device)

    kept_positions = tuple(int(position) for position in logits_position_ids)
    if len(kept_positions) != logits_length:
        raise LossContractError(
            "compact LossContext logits length must match logits_position_ids",
            code="loss.compact_logits_position_shape",
            context={
                "logits_length": logits_length,
                "logits_position_id_count": len(kept_positions),
            },
        )
    if len(set(kept_positions)) != len(kept_positions):
        raise LossContractError(
            "compact LossContext logits_position_ids must be unique",
            code="loss.compact_logits_position_duplicate",
            context={"logits_position_ids": list(kept_positions)},
        )
    if any(position < 0 or position >= pack_length for position in kept_positions):
        raise LossContractError(
            "compact LossContext logits_position_ids must stay inside pack length",
            code="loss.compact_logits_position_bounds",
            context={"logits_position_ids": list(kept_positions), "pack_length": pack_length},
        )
    row_by_physical_position = {
        physical_position: row_index
        for row_index, physical_position in enumerate(kept_positions)
    }
    missing = [
        position
        for position in physical_positions
        if position not in row_by_physical_position
    ]
    if missing:
        raise LossContractError(
            "compact LossContext logits_position_ids must cover every supervised atom",
            code="loss.compact_logits_position_missing",
            context={
                "missing_physical_logits_positions": missing,
                "logits_position_ids": list(kept_positions),
            },
        )
    return torch.tensor(
        [row_by_physical_position[position] for position in physical_positions],
        dtype=torch.long,
        device=device,
    )


__all__ = ["LossContext"]
