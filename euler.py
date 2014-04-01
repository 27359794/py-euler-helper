"""
use me with python3 only :)
"""

import random

def primes_to(n):
    """List of sorted primes in [2,n].

    >>> primes_to(10)
    [2, 3, 5, 7]
    >>> primes_to(11)
    [2, 3, 5, 7, 11]

    """
    isPrime = [True for i in range(0, n+1)]
    for i in range(2, int(n**0.5)+1):
        if not isPrime[i]:
            continue
        for j in range(i*i, n+1, i):
            isPrime[j] = False
    return [x for x in range(2, len(isPrime)) if isPrime[x]]


def is_prime(n):
    """Miller-Rabin primality test.

    >>> is_prime(97)
    True
    >>> is_prime(96)
    False
    """
    def _miller_rabin_pass(a, s, d, n):
        a_to_power = pow(a, d, n)
        if a_to_power == 1:
            return True
        for _ in range(s-1):
            if a_to_power == n - 1:
                return True
            a_to_power = (a_to_power * a_to_power) % n
        return a_to_power == n - 1

    if n == 1:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d >>= 1
        s += 1

    for _ in range(20):
        a = 0
        while a == 0:
            a = random.randrange(n)
        if not _miller_rabin_pass(a, s, d, n):
            return False
    return True


def matrix_exp_mod(mat, exp, mod=float('inf')):
    """
    Matrix exponentiation (modular optional).
    M is a numpy matrix.
    """
    if exp == 1:
        return mat % mod
    elif exp % 2 != 0:
        return (mat * matrix_exp_mod(mat, exp-1, mod)) % mod
    else:  # exp even
        half = matrix_exp_mod(mat, exp//2, mod)
        return (half * half) % mod


def factorise(n):
    """
    Set of factors of n, in O(sqrt n).

    >>> list(sorted(factorise(100)))
    [1, 2, 4, 5, 10, 20, 25, 50, 100]

    """
    factors = set()
    for i in range(1, int(n**0.5 + 1)):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return factors


_PRIMES = None

def init_primes(n):
    """
    Initialise prime array (used by module's prime-related functions) to
    primes up to n.
    """
    global _PRIMES
    _PRIMES = primes_to(n)


def prime_factorise(n):
    """
    Sorted sequence of prime factors with repetitions. Call init_primes(n)
    first, where n >= the largest prime in a number you're prime-factorising.

    >>> init_primes(100)
    >>> prime_factorise(300)
    [2, 2, 3, 5, 5]
    >>> prime_factorise(97)
    [97]

    """
    global _PRIMES
    if _PRIMES is None:
        raise RuntimeError("must call euler.init_primes(n) first.")

    if is_prime(n):
        return [n] # special-cased for speed

    pfs = []
    cur = n
    pp = 0

    while cur > 1:
        if cur % _PRIMES[pp] == 0:
            pfs.append(_PRIMES[pp])
            cur //= _PRIMES[pp]
            if is_prime(cur):
                return pfs + [cur]
        else:
            pp += 1
    return pfs


def memoize(f):
    """
    Memoization decorator.

    >>> @memoize
    ... def fib(n):
    ...     return 1 if n < 2 else fib(n-1) + fib(n-2)
    ...
    >>> fib(100)
    573147844013817084101

    """
    cache = {}
    def helper(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return helper


def gcd(a, b):
    """
    >>> gcd(9, 3)
    3
    >>> gcd(12, 6)
    6
    >>> gcd(6, 12)
    6
    >>> gcd(72, 81)
    9

    """
    if b > a:
        return gcd(b, a)
    elif b == 0:
        return a
    else:
        return gcd(b, a%b)


def to_base(num, b, numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    """
    >>> to_base(int('deadbeef', 16), 16)
    'deadbeef'
    >>> to_base(1234, 10)
    '1234'

    """
    return ((num == 0) and numerals[0]) or (to_base(num // b, b,
            numerals).lstrip(numerals[0]) + numerals[num % b])


def partitions_of(n):
    """
    Sorted list of sorted tuples, each tuple being a partition of n.

    >>> partitions_of(5)
    [(1, 1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 3), (1, 2, 2), (1, 4), (2, 3), (5,)]

    """
    parts = []

    @memoize
    def gen(cur, remaining, cur_num):
        if remaining == 0:
            parts.append(cur)
        else:
            if cur_num <= remaining:
                gen(cur + (cur_num,), remaining - cur_num, cur_num)
                gen(cur, remaining, cur_num + 1)

    gen((), n, 1)
    return parts


import doctest
doctest.testmod()
