from src.data_collators.token_types import TokenType, _dumps_with_types


def _char_types_from_spans(length: int, spans: list[tuple[int, int, int]]) -> list[int]:
    # Mirror the module's internal helper (kept local to avoid importing private helpers).
    arr = [TokenType.FORMAT] * length
    for start, end, typ in spans:
        for i in range(start, min(end, length)):
            arr[i] = typ
    return arr


def test_coord_token_string_marks_quotes_as_format() -> None:
    payload = {
        "object_1": {
            "desc": "x",
            "bbox_2d": ["<|coord_12|>", "<|coord_34|>", "<|coord_56|>", "<|coord_78|>"],
        }
    }
    text, spans = _dumps_with_types(payload)
    # Find the first coord token string in the serialized JSON.
    needle = '"<|coord_12|>"'
    start = text.find(needle)
    assert start >= 0, f"coord token string not found in JSON text: {text!r}"
    end = start + len(needle) - 1  # index of the closing quote

    char_types = _char_types_from_spans(len(text), spans)

    # Quotes are FORMAT; inner substring is COORD.
    assert text[start] == "\""
    assert char_types[start] == TokenType.FORMAT
    assert char_types[end] == TokenType.FORMAT

    assert text[start + 1] == "<"
    assert char_types[start + 1] == TokenType.COORD
    assert text[end - 1] == ">"
    assert char_types[end - 1] == TokenType.COORD
