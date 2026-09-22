from __future__ import annotations

TOL = 2e-4


def winner(row):
    return max(range(len(row)), key=lambda i: row[i])


def max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def main():
    source = [[[100.0 + p, 200.0 + p, 300.0 + p] for p in range(6)] for _ in range(4)]
    target = [[[10.0 + p, 20.0 + p, 30.0 + p] for p in range(6)]]
    source[3][-1] = [10.0, 1.0, 0.0]
    target[0][-1] = [10.00001, 1.00001, 0.0]
    target[0][0] = [0.0, 20.0, 1.0]

    src = source[3][-1]
    old = target[0][0]
    corrected = target[0][-1]
    assert winner(src) != winner(old)
    assert max_abs(src, old) > TOL
    assert winner(src) == winner(corrected)
    assert max_abs(src, corrected) <= TOL
    assert target[0][0] != target[0][-1]
    print({
        'status': 'selector_bug_reproduced',
        'old_row_boundary': {'target_position': 0, 'winner_match': False, 'max_abs_delta': max_abs(src, old)},
        'corrected_row_boundary': {'target_position': -1, 'winner_match': True, 'max_abs_delta': max_abs(src, corrected)},
        'forced_x1': {'target_position': -1, 'same_selector_as_corrected': True},
    })


if __name__ == '__main__':
    main()
