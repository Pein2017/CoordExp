from src.data_collators.token_types import TokenType, _dumps_with_types


def _char_types_from_spans(length: int, spans: list[tuple[int, int, int]]) -> list[int]:
    # Mirror the module's internal helper (kept local to avoid importing private helpers).
    arr = [TokenType.FORMAT] * length
    for start, end, typ in spans:
        for i in range(start, min(end, length)):
            arr[i] = typ
    return arr


def test_coord_token_literals_are_bare_and_marked_as_coord() -> None:
    payload = {
        "objects": [
            {
                "desc": "x",
                "bbox_2d": [
                    "<|coord_12|>",
                    "<|coord_34|>",
                    "<|coord_56|>",
                    "<|coord_78|>",
                ],
            }
        ]
    }
    text, spans = _dumps_with_types(payload)
    # CoordJSON literals are bare (no surrounding quotes).
    needle = "<|coord_12|>"
    start = text.find(needle)
    assert start >= 0, f"coord token literal not found in JSON text: {text!r}"
    assert f'"{needle}"' not in text
    end = start + len(needle) - 1

    char_types = _char_types_from_spans(len(text), spans)

    assert text[start] == "<"
    assert char_types[start] == TokenType.COORD
    assert text[end] == ">"
    assert char_types[end] == TokenType.COORD
