# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Functions for checking and computing prime numbers."""

__all__ = ["is_prime", "next_prime"]


def _legendre(a: int, m: int) -> int:
    """Legendre symbol (a|m).

    Returns:
        (int): If a is a non-residue, m-1 instead of -1
    """
    return pow(a, (m - 1) >> 1, m)


def _is_sprp(n: int, b: int = 2) -> bool:
    """Strong probable prime."""
    d = n - 1
    s = 0
    while d & 1 == 0:
        s += 1
        d >>= 1

    x = pow(b, d, n)
    if x == 1 or x == n - 1:
        return True

    for r in range(1, s):
        x = (x * x) % n
        if x == 1:
            return False
        elif x == n - 1:
            return True
    return False


def _is_lucas_prp(n: int, D: int) -> bool:
    """Lucas probable prime."""
    Q = (1 - D) >> 2

    # n+1 = 2**r*s where s is odd
    s = n + 1
    r = 0
    while s & 1 == 0:
        r += 1
        s >>= 1

    # calculate the bit reversal of (odd) s
    # e.g. 19 (10011) <=> 25 (11001)
    t = 0
    while s > 0:
        if s & 1:
            t += 1
            s -= 1
        else:
            t <<= 1
            s >>= 1

    # use the same bit reversal process to calculate the s-th Lucas number
    # keep track of q = Q**n as we go
    U = 0
    V = 2
    q = 1
    # mod_inv(2, n)
    inv_2 = (n + 1) >> 1
    while t > 0:
        if t & 1 == 1:
            # U, V of n+1
            U, V = ((U + V) * inv_2) % n, ((D * U + V) * inv_2) % n
            q = (q * Q) % n
            t -= 1
        else:
            # U, V of n*2
            U, V = (U * V) % n, (V * V - 2 * q) % n
            q = (q * q) % n
            t >>= 1

    # double s until we have the 2**r*sth Lucas number
    while r > 0:
        U, V = (U * V) % n, (V * V - 2 * q) % n
        q = (q * q) % n
        r -= 1

    # primality check
    # if n is prime, n divides the n+1st Lucas number, given the assumptions
    return U == 0


# primes less than 212
SMALL_PRIMES = set(
    [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
        211,
    ]
)

# pre-calced sieve of eratosthenes for n = 2, 3, 5, 7
SIEVE_INDICES = [
    1,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    121,
    127,
    131,
    137,
    139,
    143,
    149,
    151,
    157,
    163,
    167,
    169,
    173,
    179,
    181,
    187,
    191,
    193,
    197,
    199,
    209,
]

# distances between sieve values
SIEVE_OFFSETS = [
    10,
    2,
    4,
    2,
    4,
    6,
    2,
    6,
    4,
    2,
    4,
    6,
    6,
    2,
    6,
    4,
    2,
    6,
    4,
    6,
    8,
    4,
    2,
    4,
    2,
    4,
    8,
    6,
    4,
    6,
    2,
    4,
    6,
    2,
    6,
    6,
    4,
    2,
    4,
    6,
    2,
    6,
    4,
    2,
    4,
    2,
    10,
    2,
]


def is_prime(n: int) -> bool:
    """An almost certain primality check."""
    if n < 212:
        return n in SMALL_PRIMES

    for p in SMALL_PRIMES:
        if n % p == 0:
            return False

    # if n is a 32-bit integer (max. int is 2147483647), perform full trial division
    if n <= 2147483647:
        i = 211
        while i * i < n:
            for o in SIEVE_OFFSETS:
                i += o
                if n % i == 0:
                    return False
        return True

    # Baillie-PSW
    # this is technically a probabalistic test, but there are no known pseudoprimes
    if not _is_sprp(n):
        return False
    a = 5
    s = 2
    while _legendre(a, n) != n - 1:
        s = -s
        a = s - a
    return _is_lucas_prp(n, a)


# next prime strictly larger than n
def next_prime(n: int) -> int:
    """Next prime strictly larger than n."""
    if n < 2:
        return 2
    # first odd larger than n
    n = (n + 1) | 1
    if n < 212:
        while True:
            if n in SMALL_PRIMES:
                return n
            n += 2

    # find our position in the sieve rotation via binary search
    x = int(n % 210)
    s = 0
    e = 47
    m = 24
    while m != e:
        if SIEVE_INDICES[m] < x:
            s = m
            m = (s + e + 1) >> 1
        else:
            e = m
            m = (s + e) >> 1

    i = int(n + (SIEVE_INDICES[m] - x))
    # adjust offsets
    offs = SIEVE_OFFSETS[m:] + SIEVE_OFFSETS[:m]
    while True:
        for o in offs:
            if is_prime(i):
                return i
            i += o
