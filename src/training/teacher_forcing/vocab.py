from __future__ import annotations

from dataclasses import dataclass

from .roles import TokenRole


@dataclass(frozen=True)
class RoleVocab:
    schema_token_ids: frozenset[int]
    text_token_ids: frozenset[int]
    coord_token_ids: frozenset[int]
    stop_token_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_token_ids", frozenset(self.schema_token_ids))
        object.__setattr__(self, "text_token_ids", frozenset(self.text_token_ids))
        object.__setattr__(self, "coord_token_ids", frozenset(self.coord_token_ids))

        role_sets = (
            ("schema_token_ids", self.schema_token_ids),
            ("text_token_ids", self.text_token_ids),
            ("coord_token_ids", self.coord_token_ids),
        )
        for field_name, token_ids in role_sets:
            if self.stop_token_id in token_ids:
                raise ValueError(f"stop_token_id must not appear in {field_name}")

        for left_index, (left_name, left_ids) in enumerate(role_sets):
            for right_name, right_ids in role_sets[left_index + 1 :]:
                if left_ids & right_ids:
                    raise ValueError(
                        f"role vocab token ids must be disjoint: {left_name} and {right_name} overlap"
                    )

    @property
    def stop_token_ids(self) -> frozenset[int]:
        return frozenset({self.stop_token_id})

    def token_ids_for_role(self, role: TokenRole) -> frozenset[int]:
        if role is TokenRole.SCHEMA:
            return self.schema_token_ids
        if role is TokenRole.TEXT:
            return self.text_token_ids
        if role is TokenRole.COORD:
            return self.coord_token_ids
        if role is TokenRole.STOP:
            return self.stop_token_ids
        raise ValueError(f"unsupported token role: {role!r}")

    def token_ids_for_roles(self, roles: frozenset[TokenRole]) -> frozenset[int]:
        token_ids: set[int] = set()
        for role in roles:
            token_ids.update(self.token_ids_for_role(role))
        return frozenset(token_ids)
