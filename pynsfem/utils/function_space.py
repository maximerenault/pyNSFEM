from itertools import combinations, product


def power_tuples_sum(n: int, power: int):
    """
    Generates all n-tuples of non-negative integers that sum to the given power.
    Based on the combinatorial "stars and bars" problem.
    """
    for c in combinations(range(power + n - 1), n - 1):
        yield tuple(b - a - 1 for a, b in zip((-1,) + c, c + (power + n - 1,)))

def power_tuples_max(n: int, power: int):
    """
    Generates all n-tuples of non-negative integers that have a maximum value of the given power.
    """
    for tup in product(range(power + 1), repeat=n):
        if max(tup) == power:
            yield tup
