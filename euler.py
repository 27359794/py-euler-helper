"""
Helper functions for solving Project Euler problems.
Author: Dan Goldbach
Python 3 only.
"""

import random
import itertools

def primes_to(n):
    """
    List of sorted primes in [2,n].

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
    """
    Miller-Rabin primality test.

    >>> is_prime(97)
    True
    >>> is_prime(96)
    False
    """
    if n <= 1:
        return False

    def _miller_rabin_pass(a, s, d, n):
        a_to_power = pow(a, d, n)
        if a_to_power == 1:
            return True
        for _ in range(s-1):
            if a_to_power == n - 1:
                return True
            a_to_power = (a_to_power * a_to_power) % n
        return a_to_power == n - 1

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
    """Matrix exponentiation (modular optional). M is a numpy matrix."""
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
    _check_primes_initialised()

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

    This was partially made redundant by Python 3's functools.lru_cache.
    Eventually I'll transition over to lru_cache but this is useful for now. In
    particular, memoize doesn't require that arguments be hashable. It's also
    more streamlined.

    >>> @memoize
    ... def fib(n):
    ...     return 1 if n < 2 else fib(n-1) + fib(n-2)
    ...
    >>> fib(100)  # Without memoization, this would take yonks.
    573147844013817084101
    """
    cache = {}
    def helper(*args):
        key = f.__name__ + repr(args)
        if key not in cache:
            cache[key] = f(*args)
        return cache[key]
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
    assert a >= 0 and b >= 0

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
    assert b >= 1

    if num == 0:
        return numerals[0]
    else:
        return (to_base(num // b, b, numerals).lstrip(numerals[0])
                + numerals[num % b])


def partitions_of(n):
    """
    Sorted list of sorted tuples, each tuple being a partition of n.

    >>> partitions_of(4)
    [(1, 1, 1, 1), (1, 1, 2), (1, 3), (2, 2), (4,)]
    >>> partitions_of(5)
    [(1, 1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 3), (1, 2, 2), (1, 4), (2, 3), (5,)]
    """
    assert n >= 0

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


@memoize
def binom(n, k):
    """
    NOTE: these doctests don't run as expected because of weird interplay
    between the @memoize decorator and doctests. If you change this function,
    please test it manually. The doctests are left as examples.
    >>> binom(20, 0)
    1
    >>> binom(20, 20)
    1
    >>> binom(5, 2)
    10
    >>> binom(100, 99)
    100
    """
    assert n >= 0 and k >= 0
    assert k <= n

    if k == 0 or k == n:
        return 1
    else:
        return binom(n-1, k) + binom(n-1, k-1)

def popcount(n):
    """
    Number of 1's in the binary representation of n. Also known as Hamming
    weight.

    >>> popcount(0)
    0
    >>> popcount(1)
    1
    >>> popcount(16)
    1
    >>> popcount(int('10110100111101', 2))
    9
    """
    i = 1
    ans = 0
    while i <= n:
        if n & i:
            ans += 1
        i *= 2
    return ans

def digit_sum(n):
    """
    >>> digit_sum(0)
    0
    >>> digit_sum(1)
    1
    >>> digit_sum(10)
    1
    >>> digit_sum(123456)
    21
    """
    assert n >= 0, "digit sum only defined for non-negative numbers"
    return sum(int(d) for d in str(n))

def all_combinations(iterable):
    """
    Iterator for every tuple combination of items in iterable, of all sizes, in
    size order then lexicographic order.

    >>> for comb in all_combinations([1, 2, 3]):
    ...     print(comb)
    ()
    (1,)
    (2,)
    (3,)
    (1, 2)
    (1, 3)
    (2, 3)
    (1, 2, 3)
    """
    for size in range(0, len(iterable) + 1):
        # TODO: change this pair of lines to a `yield from` following release of 
        #       pypy 3.4, when `yield from` was introduced.
        for c in itertools.combinations(iterable, size):
            yield c

def prod(xs):
    """
    >>> prod([5, 10])
    50
    >>> prod([3, 4, 5, 6])
    360
    >>> prod([0])
    0
    >>> prod([])
    1
    """
    t = 1
    for x in xs:
        t *= x
    return t

def totient(n):
    """
    Euler's totient of n: number of positive integers less than and coprime to
    n.

    Requires primes initialised up to sqrt(n).

    >>> init_primes(10)
    >>> totient(1)
    1
    >>> totient(9)
    6
    >>> totient(36)
    12
    """
    _check_primes_initialised()

    total = 1
    for prime in _PRIMES:
        if prime > n**0.5:
            break
        if n % prime == 0:
            total *= 1 - 1/float(prime)
    return round(n * total)

def _check_primes_initialised():
    """Helper function for prime-utilising functions."""
    if _PRIMES is None:
        raise RuntimeError("must call euler.init_primes(n) first.")

import doctest
doctest.testmod()
