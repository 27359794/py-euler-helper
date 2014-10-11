"""
Helper functions for solving Project Euler problems.
Author: Dan Goldbach

Python 3 only.

This library is intentionally over-zealous when it comes to asserting
preconditions and invariants. Consider running the interpreter with the -O flag
to strip out asserts.

"""

import collections
import itertools
import random

# Global vars consulted by many of the prime-related functions in the module.
# Initialised by init_primes(n).
_PRIMES = None
_PRIME_LIMIT = None

def primes_to(n):
    """
    List of sorted primes in [2,n] in O(n log n).

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

def prime_factorise_range(n):
    """
    Dictionary of {k: sorted prime factors of k with repetitions} for k in 2..n,
    in O(n log n) with overhead.

    Prefer prime_factorise(n) if you only need factors of one n.

    Requires init_primes(>=n) first.

    >>> init_primes(20)
    >>> pfs = prime_factorise_range(11)
    >>> print([pfs[i] for i in range(2, 12)])
    [[2], [3], [2, 2], [5], [3, 2], [7], [2, 2, 2], [3, 3], [5, 2], [11]]
    """
    _check_primes_initialised_to(n)

    pfs = collections.defaultdict(list,
            {p: [p] for p in itertools.takewhile(lambda m: m <= n, _PRIMES)})

    for x in range(2, n + 1):
        for m in range(1, x + 1):
            if x * m > n:
                break
            if pfs[x * m]:
                continue
            pfs[x * m] = pfs[x] + pfs[m]

    return pfs

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

def init_primes(n):
    """
    Initialise prime array (used by module's prime-related functions) to primes
    not greater than n.
    """
    global _PRIMES, _PRIME_LIMIT
    _PRIME_LIMIT = n
    _PRIMES = primes_to(n)

def prime_factorise(n):
    """
    Sorted sequence of prime factors with repetitions, in O(sqrt n) with
    overhead. Call init_primes(n) first, where n >= the largest prime in a
    number you're prime-factorising.  Prefer prime_factorise_range(n) if you
    require prime factors of many numbers, because it runs in O(n log n).

    >>> init_primes(100)
    >>> prime_factorise(300)
    [2, 2, 3, 5, 5]
    >>> prime_factorise(97)
    [97]
    """
    _check_primes_initialised_to(n**0.5 + 1)

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

def factors_from_prime_factors(prime_factor_list):
    """
    Compute factors of a number given its prime factorisation. Efficiently deals 
    with duplicate prime factors, unlike algorithms that naively take 
    combinations of elements from the prime factor list.

    >>> sorted(factors_from_prime_factors([2]))  # n = 2
    [1, 2]
    >>> sorted(factors_from_prime_factors([2, 2, 3, 7]))  # n = 84
    [1, 2, 3, 4, 6, 7, 12, 14, 21, 28, 42, 84]
    >>> len(factors_from_prime_factors([2]*200))  # n = 2^200
    201
    """
    pf_count = sorted(collections.Counter(prime_factor_list).items())
    factors = []

    def _gen(pi, n):
        if pi == len(pf_count):
            factors.append(n)
        else:
            for i in range(pf_count[pi][1] + 1):
                _gen(pi+1, n * pf_count[pi][0]**i)

    _gen(0, 1)
    return factors


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
    Greatest common denominator in O(log max(a,b))

    >>> gcd(9, 3)
    3
    >>> gcd(12, 6)
    6
    >>> gcd(6, 12)
    6
    >>> gcd(72, 81)
    9
    >>> gcd(29, 97)
    1
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
    Binomial coefficient C(n, k).

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
    Number of 1's in the binary representation of n. 

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
    n. O(sqrt n).

    >>> init_primes(50000)
    >>> totient(1)
    1
    >>> totient(9)
    6
    >>> totient(14)
    6
    >>> totient(36)
    12
    >>> totient(23485)
    14400
    """
    total = 1
    for prime in set(prime_factorise(n)):
        total *= 1 - 1/prime
    return round(n * total)

################################################################################
# Computational Geometry
################################################################################

def point_line_cmp(query, p1, p2):
    """
    Determine the position of (x,y) point query relative to the line passing
    through p1, p2 in that direction.

     0 if query, p1, p2 collinear
    -1 if query is to the left of p1->p2.
    +1 if query is to the right of p1->p2.

    >>> point_line_cmp((3,0), (1,0), (2,0))
    0
    >>> point_line_cmp((-3,1), (1,0), (2,0))
    -1
    >>> point_line_cmp((10,-5), (1,0), (2,0))
    1
    >>> point_line_cmp((-1,1), (-1,-1), (1,1))
    -1
    >>> point_line_cmp((-1,1), (1,1), (-1,-1))
    1
    >>> point_line_cmp((-1,-1), (1,1), (-5,-5))
    0
    >>> point_line_cmp((0,1), (0,-1), (0,0))
    0
    """
    assert p1 != p2, 'p1, p2 must be distinct points to specify a line'
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = (p2[0]-p1[0])*p1[1] - (p2[1]-p1[1])*p1[0]
    res = a*query[0] + b*query[1] + c

    if res == 0:
        return 0
    return -1 if res < 0 else 1


def point_angle_cmp(base, a, b):
    """
    Compare the angle of point a to point b, relative to point base. Can be used
    as a sort comparator, as long as you're careful about transitivity: ensure
    that all the points in the sort list lie to one side of any line drawn
    through base.

     0  if a, b, base are collinear.
    -1  if a is to the left of b.
    +1  if a is to the right of b.

    >>> point_angle_cmp((0,1), (1,0), (0,0))
    -1
    >>> point_angle_cmp((1,-1), (1,1), (0,0))
    1
    >>> point_angle_cmp((0,1), (1,0), (1,1))
    1
    >>> point_angle_cmp((20.5,9), (-30,-2), (25,15))
    -1
    >>> point_angle_cmp((20.5,9), (-30,-2), (25,-15))
    1
    >>> point_angle_cmp((20.5,9), (-30,2), (-25,-15))
    1
    >>> point_angle_cmp((0,0), (1,1), (2,2))
    0
    >>> point_angle_cmp((1,1), (3,1), (5,1))
    0
    >>> point_angle_cmp((-1, -1), (-1, 2), (-1, 20))
    0
    """
    aa = (a[0] - base[0], a[1] - base[1])
    bb = (b[0] - base[0], b[1] - base[1])
    cross = _cross_product(aa[0], aa[1], bb[0], bb[1])

    if cross == 0:
        return 0
    return -1 if cross < 0 else 1

def _cross_product(x0, y0, x1, y1):
    return x0*y1 - x1*y0

################################################################################
# Internal
################################################################################

def _check_primes_initialised_to(n):
    """
    Helper function for prime-utilising functions. Check that the prime list
    has been initialised for primes up to at least n.
    """
    if _PRIMES is None or _PRIME_LIMIT < n:
        raise RuntimeError(
                "must call euler.init_primes(>={:.0f}) first.".format(n))

import doctest
doctest.testmod()
