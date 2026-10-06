# HackWithInfy Question Bank — Easy / Medium / Hard / Complex
Built directly from your uploaded practice PDFs. Solutions marked **✅ Tested** were run against every sample case in the source material and matched exactly. Solutions marked **🔧 Approach + Reference Code** give the correct technique and a working sketch, but you should re-verify them against the exact sample I/O yourself before trusting them blind — a couple of the source problems have ambiguous or inconsistent wording in the original PDF (noted where relevant).

---

# EASY (6 questions)

## E1. Range Assign + Range Sum Queries ✅ Tested
**Problem:** Array `A`, queries: Type 1 `(l,r)` sets `A[i] = (i-l+1)*A[l]` for `i` in `[l,r]`; Type 2 `(l,r)` asks for `sum(A[l..r])`. Sum all Type-2 answers mod `1e9+7`.

**Key insight:** Type 1 overwrites a whole range with an **arithmetic progression** (first term `A[l]`, common difference `A[l]`). That's a segment tree with **lazy propagation**, where the lazy tag stores `(anchor_index, base_value)` instead of a flat "add X" — each node recomputes its own sum using the arithmetic series formula `n*(first+last)/2`.

```python
class SegTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0]*(4*self.n)
        self.lazy = [None]*(4*self.n)      # (anchor_index l0, value)
        self.arr = arr
        self._build(1, 0, self.n-1)

    def _build(self, node, l, r):
        if l == r:
            self.tree[node] = self.arr[l]
            return
        mid = (l+r)//2
        self._build(2*node, l, mid)
        self._build(2*node+1, mid+1, r)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]

    def _apply(self, node, l, r, l0, val):
        first = (l - l0 + 1) * val
        last  = (r - l0 + 1) * val
        cnt = r - l + 1
        self.tree[node] = (first + last) * cnt // 2
        self.lazy[node] = (l0, val)

    def _push_down(self, node, l, r):
        if self.lazy[node] is not None:
            mid = (l+r)//2
            l0, val = self.lazy[node]
            self._apply(2*node, l, mid, l0, val)
            self._apply(2*node+1, mid+1, r, l0, val)
            self.lazy[node] = None

    def update(self, ql, qr, l0, val, node=1, l=0, r=None):
        if r is None: r = self.n-1
        if qr < l or r < ql: return
        if ql <= l and r <= qr:
            self._apply(node, l, r, l0, val); return
        self._push_down(node, l, r)
        mid = (l+r)//2
        self.update(ql, qr, l0, val, 2*node, l, mid)
        self.update(ql, qr, l0, val, 2*node+1, mid+1, r)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]

    def query(self, ql, qr, node=1, l=0, r=None):
        if r is None: r = self.n-1
        if qr < l or r < ql: return 0
        if ql <= l and r <= qr: return self.tree[node]
        self._push_down(node, l, r)
        mid = (l+r)//2
        return self.query(ql, qr, 2*node, l, mid) + self.query(ql, qr, 2*node+1, mid+1, r)

def solve(n, A, queries):
    st = SegTree(A[:])
    total = 0
    MOD = 10**9 + 7
    for t, l, r in queries:
        if t == 1:
            a_l = st.query(l, l)          # current A[l], before applying
            st.update(l, r, l, a_l)
        else:
            total = (total + st.query(l, r)) % MOD
    return total
```
Verified against all 3 sample cases → outputs `60`, `111`, `46` exactly.

**Complexity:** O((n+q) log n).

---

## E2. Max-Sum Subarray with ≤K Distinct Elements ✅ Tested
**Problem:** Find the maximum sum of a contiguous subarray using at most `k` distinct values. Empty subarray allowed (sum 0).

**Key insight:** Two pointers maintain the largest valid window `[left, right]`. Within that window, the best subarray *ending exactly at* `right` is `prefix[right+1] - min(prefix[j])` for `j` in `[left, right]` — track that minimum with a **monotonic deque**.

```python
from collections import deque, defaultdict

def solve(N, k, A):
    freq = defaultdict(int)
    distinct = 0
    left = 0
    prefix = [0]*(N+1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]

    dq = deque([0])   # candidate prefix indices, increasing prefix value
    best = 0

    for right in range(N):
        freq[A[right]] += 1
        if freq[A[right]] == 1:
            distinct += 1
        while distinct > k:
            freq[A[left]] -= 1
            if freq[A[left]] == 0:
                distinct -= 1
            left += 1

        while dq and dq[0] < left:
            dq.popleft()
        while dq and prefix[dq[-1]] >= prefix[right+1]:
            dq.pop()
        dq.append(right+1)

        while dq and dq[0] < left:
            dq.popleft()
        if dq:
            best = max(best, prefix[right+1] - prefix[dq[0]])

    return best
```
Verified against all 3 sample cases → outputs `12`, `0`, `6` exactly.

**Complexity:** O(n).

---

## E3. Minimum Initial Oil to Minimize Disturbances ✅ Tested
**Problem:** Sequence of buy(-1)/sell(+1) operations against a tank of capacity `C`; a disturbance happens if you try to sell a full tank or buy from an empty one (operation is skipped, not undone). Find the minimum starting oil `X` giving the fewest disturbances.

**Key insight:** Disturbance count as a function of `X` is not affected simply by a running-minimum trick here, because clamping means the trajectory itself changes once a disturbance occurs. The clean way: simulate `f(X)` in O(n), and search over `X ∈ [0, C]` — either brute force (fine for small `C`) or **ternary search** assuming the disturbance-count curve is unimodal in `X` (true for the given constraints in practice).

```python
def disturbances(X, C, A):
    level = min(X, C)
    d = 0
    for a in A:
        if a == 1:      # sell
            if level == C: d += 1
            else: level += 1
        else:            # buy
            if level == 0: d += 1
            else: level -= 1
    return d

def solve(N, C, A):
    lo, hi = 0, C
    while hi - lo > 2:
        m1 = lo + (hi-lo)//3
        m2 = hi - (hi-lo)//3
        if disturbances(m1, C, A) <= disturbances(m2, C, A):
            hi = m2
        else:
            lo = m1
    best_d = best_X = None
    for X in range(lo, hi+1):
        d = disturbances(X, C, A)
        if best_d is None or d < best_d or (d == best_d and X < best_X):
            best_d, best_X = d, X
    return best_X
```
Verified against all 3 sample cases → outputs `1`, `2`, `0` exactly.

**Complexity:** O(n log C).

---

## E4. Minimum Moves to Reduce N Soldiers to 1 🔧 Approach + Reference Code
**Problem:** Operations: `-1`, or set to some function of "half"/"two-thirds" of current value (rounding down); minimize moves to reach 1.

**⚠️ Heads-up:** The exact rounding semantics in the source PDF are genuinely ambiguous — the worked examples for "reduce by half" are only consistent with the reading **"new value = floor(current/2)"**, not the more literal "subtract floor(current/2)". The "two-thirds" operation isn't actually exercised in any sample case, so its exact formula can't be pinned down from the material alone. **Test this against the real judge's first couple of cases before trusting it.**

**Pattern (regardless of exact formula):** this is the classic "reduce N to 1 with mixed operations" shape (same family as LeetCode's "Integer Replacement" / "Minimum number of days to eat N oranges"). Use **memoized recursion** — don't do naive `n-1` all the way down; only recurse through `-1` a few steps near small `n`, and let the divide-style operations do the heavy lifting for large `n`.

```python
import sys
from functools import lru_cache
sys.setrecursionlimit(10000)

@lru_cache(maxsize=None)
def f(n):
    if n == 1:
        return 0
    half = n // 2
    third = n // 3
    best = 1 + f(n-1) if n - 1 >= 1 and n <= 6 else float('inf')  # only use -1 near small n
    if half >= 1:
        best = min(best, 1 + f(half) + (n - 2*half))  # pay n%2 extra -1 steps if needed to land exactly on half*2 first, adjust per actual rules
    if third >= 1:
        best = min(best, 1 + f(third) + (n - 3*third))
    return best

# NOTE: adapt the "+ (n - 2*half)" adjustment terms once you confirm the exact
# operation semantics from the real judge — this is the standard template shape,
# the arithmetic detail depends on resolving the ambiguity above.
```
**Complexity (once formula is nailed down):** O(log²n) amortized via memoization — this is the important takeaway regardless of the exact rounding rule.

---

## E5. Multi-Source Grid Invasion (BFS spread) ✅ Tested
*(Already fully solved and verified earlier in this chat — see above. Recap of the trick: seed BFS with every `A` cell simultaneously at distance 0; answer is `max` distance over all `E` cells; return `-1` if any `E` is unreachable.)*

---

## E6. Food Stamps — Diminishing Returns Purchases ✅ Tested
**Problem:** `n` food types, buying type `i` for the `t`-th time gives `v[i] - d[i]*(t-1)` points; total purchases ≤ `M`. Maximize total points.

**Key insight:** Each type's values form a **decreasing arithmetic sequence**. The optimal strategy is always to take the top `K = min(M, total positive-value purchases)` values across *all* sequences combined — and because each sequence is individually sorted descending, "top K globally" is automatically a valid prefix per type. Find that cutoff with **binary search on a threshold value λ** (classic "top-K from multiple arithmetic sequences" trick).

```python
def count_and_sum(lam, V, D):
    total_count = total_sum = 0
    for v, d in zip(V, D):
        if v < lam:
            continue
        t = (v - lam)//d + 1
        total_count += t
        total_sum += t*v - d*t*(t-1)//2
    return total_count, total_sum

def solve(n, m, V, D):
    total_positive, _ = count_and_sum(1, V, D)
    K = min(m, total_positive)
    if K == 0:
        return 0
    lo, hi = -10**9 - 10, 10**9 + 10
    while lo < hi:
        mid = (lo + hi + 1) // 2
        c, _ = count_and_sum(mid, V, D)
        if c >= K: lo = mid
        else: hi = mid - 1
    c, s = count_and_sum(lo, V, D)
    return s - lo * (c - K)
```
Verified against all 3 sample cases → outputs `5`, `12`, `27` exactly.

**Complexity:** O(n log(max value)).

---

# MEDIUM (5 questions)

## M1. Maximum Expert Number (Team Partition by MEX) ✅ Tested
**Problem:** Partition array into contiguous teams; each team scores its **MEX** (smallest non-negative value missing from it); maximize total score.

**Key insight:** `dp[i]` = best score for the first `i` elements. To extend, track `last_seen[v]` = most recent index of value `v`; shrink a window from `i` backward through `0,1,2,...` while each value is still present, updating `dp[i] = max(dp[i], dp[j-1] + k+1)` where `j` is the leftmost boundary needed to guarantee values `0..k` are all inside.

```python
def solve(n, A):
    dp = [0]*(n+1)
    last_seen = {}
    for i in range(1, n+1):
        last_seen[A[i-1]] = i
        best = dp[i-1]
        j = i
        k = 0
        while k in last_seen:
            j = min(j, last_seen[k])
            best = max(best, dp[j-1] + k + 1)
            k += 1
        dp[i] = best
    return dp[n]
```
Verified against all 3 sample cases → outputs `3`, `5`, `10` exactly.

**Complexity:** O(n · maxValue) worst case (here `A[i] ≤ 1000`, so this is fine for `n ≤ 1e5`).

---

## M2. Covered Ranges in a Connected Component (DSU) ✅ Tested
**Problem:** Union-Find with edge-add queries; for a "beauty" query on node `u`, return the number of maximal runs of consecutive integers present in `u`'s connected component.

**Key insight:** Maintain `runs[root]` incrementally on the DSU. When you union two components, their combined run count starts as `runsA + runsB`; then check the (at most 4) boundary positions touched by the new edge — every time you discover `value x` and `value x+1` are now in the same component for the first time, that merges two runs into one, so decrement by 1.

```python
class DSU:
    def __init__(self, n):
        self.p = list(range(n+2)); self.r = [0]*(n+2)
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return ra
        if self.r[ra] < self.r[rb]: ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]: self.r[ra] += 1
        return ra

def solve(n, queries):
    dsu = DSU(n)
    runs = [1]*(n+2)
    merged_boundary = [False]*(n+2)

    def try_merge(x):
        if 1 <= x < n and not merged_boundary[x] and dsu.find(x) == dsu.find(x+1):
            merged_boundary[x] = True
            runs[dsu.find(x)] -= 1

    total = 0
    for t, u, v in queries:
        if t == 1:
            ru, rv = dsu.find(u), dsu.find(v)
            if ru != rv:
                combined = runs[ru] + runs[rv]
                new_root = dsu.union(u, v)
                runs[new_root] = combined
                for x in (u-1, u, v-1, v):
                    try_merge(x)
        else:
            total += runs[dsu.find(u)]
    return total
```
Verified against all 3 sample cases → outputs `1`, `2`, `4` exactly.

**Complexity:** O((n+q) α(n)).

---

## M3. Minimum Jumps on a Circular Chair Ring 🔧 Approach + Reference Code
**Problem:** `N` people in a circle; person on chair `i` can jump `A[i]` seats left or right. Find minimum jumps from chair `X` to chair `Y`.

**Key insight:** This is unweighted-graph shortest path in disguise — build an implicit graph where node `i` connects to `((i + A[i] - 1) % N) + 1` and `((i - A[i] - 1) % N) + 1` (careful with 1-indexing and modular wraparound), then run plain BFS from `X`.

```python
from collections import deque

def solve(N, X, Y, A):
    def right(i):
        return (i - 1 + A[i-1]) % N + 1
    def left(i):
        return (i - 1 - A[i-1]) % N + 1

    if X == Y:
        return 0
    dist = [-1]*(N+1)
    dist[X] = 0
    q = deque([X])
    while q:
        u = q.popleft()
        for v in (right(u), left(u)):
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                if v == Y:
                    return dist[v]
                q.append(v)
    return -1
```
**Complexity:** O(N). *(Approach validated against the problem shape; re-check the exact modular indexing against sample 1/2/3 in your PDF before the exam — off-by-one on circular indexing is the classic bug here.)*

---

## M4. Good Subsequence with GCD Exactly P 🔧 Approach + Reference Code
**Problem:** After each point-update query, check whether **any** non-empty subsequence of length `< n` has GCD exactly `p`.

**Key insight (two-part):**
1. Map every array value to itself if divisible by `p`, else `0` (since `gcd(x, 0) = x`, non-multiples become harmless in a GCD segment tree).
2. You need **two** things from the segment tree at the root: the **full GCD** of the mapped array, and the **best GCD achievable while excluding exactly one element** (this handles the "length must be `< n`" constraint even when *every* element is divisible by `p`). Maintain both bottom-up: each node stores `(full_gcd, best_exclude_one)`, where `best_exclude_one = min(gcd(L.exclude_one, R.full), gcd(L.full, R.exclude_one))`.
3. Answer is YES iff `full_gcd == p` **and** `best_exclude_one == p`.

```python
import math

class ExcludeOneGCDTree:
    def __init__(self, arr, p):
        self.n = len(arr)
        self.p = p
        self.full = [0]*(4*self.n)
        self.exc  = [0]*(4*self.n)     # 0 acts as gcd-identity (empty selection)
        self.arr = [v if v % p == 0 else 0 for v in arr]
        self._build(1, 0, self.n-1)

    def _build(self, node, l, r):
        if l == r:
            self.full[node] = self.arr[l]
            self.exc[node] = 0          # excluding the only leaf => empty
            return
        mid = (l+r)//2
        self._build(2*node, l, mid); self._build(2*node+1, mid+1, r)
        self._pull(node)

    def _pull(self, node):
        L, R = 2*node, 2*node+1
        self.full[node] = math.gcd(self.full[L], self.full[R])
        self.exc[node] = min(
            math.gcd(self.exc[L], self.full[R]),
            math.gcd(self.full[L], self.exc[R])
        )

    def update(self, pos, newval, node=1, l=0, r=None):
        if r is None: r = self.n-1
        if l == r:
            self.arr[pos] = newval if newval % self.p == 0 else 0
            self.full[node] = self.arr[pos]
            return
        mid = (l+r)//2
        if pos <= mid: self.update(pos, newval, 2*node, l, mid)
        else: self.update(pos, newval, 2*node+1, mid+1, r)
        self._pull(node)

    def query_yes(self):
        return self.full[1] == self.p and self.exc[1] == self.p

def solve(n, A, p, queries):   # queries: list of (i, j), i is 1-indexed
    tree = ExcludeOneGCDTree(A, p)
    yes_count = 0
    for i, j in queries:
        tree.update(i-1, j)
        if tree.query_yes():
            yes_count += 1
    return yes_count
```
**Complexity:** O((n+q) log n). *(The `n==1` edge case is handled automatically — a single-element array's `exc` value stays `0`, which can never equal `p ≥ 1`, correctly forcing NO.)*

---

## M5. Amazingness — Partition Maximizing Subset-XOR Sum 🔧 Approach + Reference Code
**Problem:** Partition array into subarrays of length ≥ K; each subarray's "beauty" = max XOR of a subset within it; maximize total beauty.

**Key insight:** `dp[i] = max(dp[i], dp[j] + max_subset_xor(A[j+1..i]))` for valid `j` with `i-j ≥ K`. The inner "max subset XOR of a range" is computed with a **linear basis (XOR basis)**. Because ranges overlap heavily, maintain a rolling XOR basis as you extend `i`, and combine with a monotonic pointer for the `≥K` constraint window.

```python
def max_xor_of_basis(vals):
    basis = [0]*20   # values up to 1e5 < 2^17, 20 is a safe bit-width
    for v in vals:
        x = v
        for b in range(19, -1, -1):
            if not (x >> b) & 1: continue
            if basis[b] == 0:
                basis[b] = x; break
            x ^= basis[b]
    res = 0
    for b in range(19, -1, -1):
        if (res ^ basis[b]) > res:
            res ^= basis[b]
    return res

def solve_bruteish(n, K, A):
    # O(n^2) reference version -- correct but only fast enough for small n.
    # For n up to 1e5 you need an incremental / offline basis-merging structure;
    # start from this to validate correctness on small cases, then optimize.
    dp = [float('-inf')]*(n+1)
    dp[0] = 0
    for i in range(1, n+1):
        for j in range(0, i-K+1):
            beauty = max_xor_of_basis(A[j:i])
            if dp[j] != float('-inf'):
                dp[i] = max(dp[i], dp[j] + beauty)
    return dp[n]
```
**Complexity:** O(n²) reference version (fine to validate logic on the small samples); production version needs a smarter incremental basis approach for `n ≤ 1e5` — flag this one for extra practice time since it's the least "template-shaped" of the mediums.

---

# HARD (5 questions)

## H1. Minimum K for Longest Increasing Path with Window Constraint 🔧 Approach
**Problem:** Directed edge `i → j` exists if `p[i] < p[j]` and `|i-j| ≤ k`. Find minimum `k` such that the longest path has ≥ `m` nodes.

**Approach:** Binary search on `k` (monotonic: larger `k` only adds edges, never removes reachability, so the longest path length is non-decreasing in `k`). For a fixed `k`, the longest path = **longest increasing subsequence restricted to a sliding window of size k** — compute with a Fenwick tree / segment tree over *values* that supports "max DP value among positions in `[i-k, i-1]` with smaller permutation value," refreshed in a sliding fashion.

```python
def longest_path_for_k(n, k, p):
    import bisect
    # BIT over value-space for max query with sliding index window
    from collections import deque
    tree = [0]*(n+2)
    def update(pos, val):
        while pos <= n:
            tree[pos] = max(tree[pos], val); pos += pos & (-pos)
    def query(pos):
        res = 0
        while pos > 0:
            res = max(res, tree[pos]); pos -= pos & (-pos)
        return res

    dp = [1]*n
    window = deque()   # holds indices in current valid window, in order
    for i in range(n):
        while window and window[0] < i - k:
            window.popleft()
        best = query(p[i]-1) if p[i]-1 >= 1 else 0
        dp[i] = best + 1
        update(p[i], dp[i])
        window.append(i)
    return max(dp)

def solve(n, m, p):
    lo, hi = 0, n-1
    while lo < hi:
        mid = (lo+hi)//2
        if longest_path_for_k(n, mid, p) >= m:
            hi = mid
        else:
            lo = mid + 1
    return lo
```
**Note:** the BIT approach above doesn't fully respect the sliding-window eviction (BITs don't support deletion cleanly) — for a fully correct O(n log n) per check you'd want a segment tree over positions with periodic rebuilding, or accept O(n²) per check (fine if `n` is small in the actual hidden tests) while binary-searching `k`. Flag this as one to workshop with a friend or re-derive slowly — it's genuinely one of the harder patterns in your set.

**Complexity target:** O(n log n log n) with a correct sliding-window LIS structure.

---

## H2. Some Help — XP from Next-Multiple Jumps 🔧 Approach
**Problem:** For each soldier `i`, find the first soldier to the right whose power is a multiple of soldier `i`'s power; sum the max bonus in that range across all rounds.

**Approach:**
1. For each distinct power value `v`, precompute positions of soldiers with power `v`, `2v`, `3v`, ... (bounded since `A[i] ≤ 1e5`) using a "next occurrence" structure per value — or simpler: for each `i`, scan multiples of `A[i]` and binary-search the nearest position `> i` holding that multiple, take the minimum such position over all multiples.
2. Once you know the range `[i, R]` for each `i`, answer "max bonus in range" with a **sparse table** (static array, no updates) in O(1) per query after O(n log n) preprocessing.

```python
import math

def build_sparse_table(arr):
    n = len(arr)
    LOG = max(1, n.bit_length())
    st = [arr[:]]
    j = 1
    while (1 << j) <= n:
        prev = st[-1]
        cur = [max(prev[i], prev[i + (1 << (j-1))]) for i in range(n - (1 << j) + 1)]
        st.append(cur)
        j += 1
    return st

def query_max(st, l, r):   # inclusive, 0-indexed
    length = r - l + 1
    k = length.bit_length() - 1
    return max(st[k][l], st[k][r - (1 << k) + 1])

def solve(n, A, Bonus):
    from collections import defaultdict
    pos_by_value = defaultdict(list)   # value -> sorted list of positions (0-indexed)
    for i, v in enumerate(A):
        pos_by_value[v].append(i)

    st = build_sparse_table(Bonus)
    total = 0
    import bisect
    for i in range(n):
        R = None
        v = A[i]
        m = v
        while m <= 10**5:
            lst = pos_by_value.get(m)
            if lst:
                idx = bisect.bisect_right(lst, i)
                if idx < len(lst):
                    cand = lst[idx]
                    if R is None or cand < R:
                        R = cand
            m += v
        if R is not None:
            total += query_max(st, i, R)
    return total
```
**Complexity:** O((max_value/min_value) · log n) amortized for the multiple-scan, which is the classic harmonic-series bound `O(V log V)` overall when done across all `i` — acceptable for `V ≤ 1e5`.

---

## H3. Pairs with Frequency/Distinct-Count Constraint 🔧 Approach
**Problem:** Count pairs `(i,j)`, `i<j`, where `frequency(1,i,A[i]) + frequency(j,N,A[j]) ≤ ⌊distinct(1,i)/2⌋ + ⌊distinct(j,N)/2⌋`.

**Approach:** Precompute, for every index `i`: `freqLeft[i] = frequency(1,i,A[i])` and `distLeft[i] = distinct(1,i)` via a forward pass with a hashmap/count array (both are prefix-computable in O(n)). Symmetric backward pass gives `freqRight[j]` and `distRight[j]`. Once you have `leftScore[i] = ⌊distLeft[i]/2⌋ - freqLeft[i]` and `rightScore[j] = ⌊distRight[j]/2⌋ - freqRight[j]`, the condition becomes `leftScore[i] + rightScore[j] ≥ 0`. Now it's a classic **count pairs with i<j and value[i]+value[j] ≥ 0** problem — sort-and-two-pointer or a Fenwick tree over compressed values, processed left to right.

```python
def solve(N, A):
    from collections import defaultdict
    freqLeft = [0]*N; distLeft = [0]*N
    cnt = defaultdict(int); d = 0
    for i in range(N):
        cnt[A[i]] += 1
        if cnt[A[i]] == 1: d += 1
        freqLeft[i] = cnt[A[i]]
        distLeft[i] = d

    freqRight = [0]*N; distRight = [0]*N
    cnt = defaultdict(int); d = 0
    for i in range(N-1, -1, -1):
        cnt[A[i]] += 1
        if cnt[A[i]] == 1: d += 1
        freqRight[i] = cnt[A[i]]
        distRight[i] = d

    leftScore  = [distLeft[i]//2 - freqLeft[i] for i in range(N)]
    rightScore = [distRight[i]//2 - freqRight[i] for i in range(N)]

    # count pairs i<j with leftScore[i] + rightScore[j] >= 0
    # i.e. leftScore[i] >= -rightScore[j]; use a Fenwick tree over leftScore values
    # inserted incrementally as j moves right, or sort approach below (simpler, O(n log n)):
    import bisect
    sorted_left = []
    ans = 0
    # process j in increasing order? careful: need i<j, so insert leftScore[i] as i grows
    # then for each j query count of leftScore[i] >= -rightScore[j] among inserted i<j
    for j in range(N):
        # first, if j>0, ensure leftScore up to j-1 inserted (do insert BEFORE moving to next j)
        target = -rightScore[j]
        idx = bisect.bisect_left(sorted_left, target)
        ans += len(sorted_left) - idx
        bisect.insort(sorted_left, leftScore[j])   # insert current index's leftScore for future j's
    return ans
```
**Complexity:** O(n log n). *(Double-check the exact off-by-one on when `leftScore[j]` gets inserted relative to the query for `j` — the loop above inserts `leftScore[j]` right after querying with it, meaning by the time you query for a later `j'`, all `i < j'` have been inserted, which is the correct order.)*

---

## H4. Tree Beauty — Perfect-Square-Product Pairs per Subtree ✅ Approach Verified Logically
**Problem:** `beauty(u)` = count of pairs `(i,j)` in `subtree(u)` where `a[i]*a[j]` is a perfect square. Sum `beauty(u)` over all `u`.

**Key insight:** `a[i]*a[j]` is a perfect square **iff** the square-free parts of `a[i]` and `a[j]` are equal. So this becomes: for each node, count pairs with equal square-free part inside its subtree — a textbook **small-to-large merging on trees** problem, but with a twist: a pair with LCA `u` contributes to `beauty(u)` **and every ancestor of `u`**, so its total contribution to the final sum is `(depth(u)+1)`.

```python
import sys
from collections import defaultdict
sys.setrecursionlimit(300000)

def squarefree(x):
    res = 1; d = 2
    while d*d <= x:
        cnt = 0
        while x % d == 0:
            x //= d; cnt += 1
        if cnt % 2 == 1: res *= d
        d += 1
    if x > 1: res *= x
    return res

def solve(n, parent, a, MOD=10**9+7):
    sf = [0] + [squarefree(v) for v in a]     # 1-indexed
    children = defaultdict(list)
    for i in range(2, n+1):
        children[parent[i]].append(i)

    depth = [0]*(n+1)
    ans = 0

    def dfs(u):
        nonlocal ans
        cur = defaultdict(int)
        for c in children[u]:
            depth[c] = depth[u] + 1
            child_map = dfs(c)
            if len(child_map) > len(cur):
                cur, child_map = child_map, cur
            for key, cnt in child_map.items():
                # pairs formed here have LCA = u -> weight (depth[u]+1)
                ans = (ans + cnt * cur[key] * (depth[u] + 1)) % MOD
                cur[key] += cnt
        # now fold in u itself -- pairs with u also have LCA = u
        ans = (ans + cur[sf[u]] * (depth[u] + 1)) % MOD
        cur[sf[u]] += 1
        return cur

    dfs(1)
    return ans
```
**⚠️ Recursion depth:** for `n` up to `1e5`, Python's recursion will crash on a skewed tree. Convert `dfs` to an explicit iterative post-order traversal before submitting — the logic above is correct, but the recursive shell needs replacing under real constraints.

**Complexity:** O(n log²n) (standard small-to-large bound).

---

## H5. Longest Non-Decreasing Subsequence with XOR ≥ M 🔧 Approach
**Problem:** Find the longest non-decreasing subsequence whose XOR is `≥ M`. Constraints are small (`N,M ≤ 1000`), which is the tell.

**Approach:** With `N ≤ 1000`, a DP over `(index, xor value)` is affordable: `dp[i][x]` = does a non-decreasing subsequence ending at `i` with XOR exactly `x` exist, and if so, its max length. Transition from all `j<i` with `A[j] ≤ A[i]`.

```python
def solve(N, M, A):
    # dp[i] = dict: xor_value -> max length of non-decreasing subseq ending at i with that xor
    dp = [dict() for _ in range(N)]
    best = 0
    for i in range(N):
        dp[i][A[i]] = 1
        if A[i] >= M:
            best = max(best, 1)
        for j in range(i):
            if A[j] <= A[i]:
                for xv, ln in dp[j].items():
                    nx = xv ^ A[i]
                    if dp[i].get(nx, 0) < ln + 1:
                        dp[i][nx] = ln + 1
                    if nx >= M:
                        best = max(best, ln + 1)
    return best
```
**Complexity:** O(N² · distinct-xor-states) — with `N ≤ 1000` and values `≤ 1000` (so XOR values bounded ~`2^10`), this comfortably fits within limits. Don't try to over-optimize this one; the small constraints are your signal that an O(N²) or O(N² log N) approach is *intended*.

---

# COMPLEX (2 questions — from your material; expect composite problems in this shape)

## C1. Tree Edge Flipping + Pattern Matching 🔧 Approach Only
**Problem:** Flip a matching of parent-child edges (each flip toggles both endpoints and costs `M`); for each binary-string query, maximize the count of root-to-leaf paths containing that string as a substring, minimizing cost among optimal solutions.

**Approach shape (for orientation, not full code — this genuinely needs 30+ minutes even for a strong competitor):**
1. This is fundamentally a **tree DP where each node's state includes "which flips have been used along the root path so far" reduced to just the current toggled/untoggled parity of the node** (since flips form a matching, a node is affected by at most one incident flip).
2. For the string-matching part, build a **KMP automaton** for the query string and track, per node, the automaton state reachable along each root-to-node path under each flip-parity choice.
3. DP state: `dp[node][parity][automaton_state]` = minimum cost to reach this configuration. Combine with a **greedy/matching argument** for selecting which edges to flip (this is the piece requiring the most care — the "no two flipped edges share a node" constraint is a matching constraint, so consider it as a small tree-DP knapsack: `dp[node][flip_edge_to_parent?]`).
4. Answer per query = min cost over all `dp[leaf][*][accepting_state]` combinations, maximized for path count first, then minimized for cost.

**Test-day strategy for Complex questions like this:** don't try to build the full DP cold. Get a **correct brute force** (small `n`, try all valid matchings by brute enumeration) passing the samples first — that's partial credit locked in — then optimize only the piece you're confident about (usually the tree DP shape) if time remains.

---

## C2. Layer-Split Path Maximization with Penalties 🔧 Approach Only
**Problem:** Path in a graph where layers must be non-decreasing; jumping from layer `x` to `y>x` costs `(y-x)²`; maximize `sum(values) - sum(penalties)`.

**Approach shape:**
1. Sort/group nodes by layer. This is really a **DAG longest-path (max weight) problem** once you note edges only go from lower-or-equal layer to higher-or-equal layer.
2. Define `best[layer]` = the best achievable score for a path ending at a node in that layer. Since the penalty `(y-x)²` only depends on the *layer values*, not the specific node, you can separate "which node to pick" (maximize `V[u]`) from "which layer transition to take" (minimize penalty), computed per layer using a **convex-hull trick / simple DP over at most `K` layers** since `K ≤ 1e5`.
3. `dp[layer] = max over prev_layer ≤ layer of (dp[prev_layer] - (layer-prev_layer)²) + max_value_at(layer)`. This inner expression is a classic **"DP optimization with quadratic cost"** shape — if `K` is small enough, an O(K²) DP works directly; for full-size constraints you'd reach for divide-and-conquer optimization or the convex hull trick since the cost function is convex in the layer gap.

```python
def solve_bruteish(N, layers_data, edges, K):
    # small-K reference DP: O(K^2), correct but not scaled for K=1e5
    from collections import defaultdict
    best_val_at_layer = defaultdict(lambda: float('-inf'))
    for node_layer, val in layers_data:
        best_val_at_layer[node_layer] = max(best_val_at_layer[node_layer], val)

    layers_sorted = sorted(best_val_at_layer.keys())
    dp = {}
    ans = float('-inf')
    for L in layers_sorted:
        best_here = best_val_at_layer[L]     # start fresh path at this layer
        for prevL in layers_sorted:
            if prevL < L and prevL in dp:
                cand = dp[prevL] - (L - prevL)**2 + best_val_at_layer[L]
                best_here = max(best_here, cand)
        dp[L] = best_here
        ans = max(ans, best_here)
    return ans
```
**Complexity:** O(K²) reference (fine for small `K`, matches the provided samples); production version for `K` up to `1e5` needs D&C optimization or CHT — flag as "attempt brute force first, optimize only if time remains," same as C1.

---

# Quick-Reference: Which Pattern Goes With Which Signal

| If you see... | Reach for... |
|---|---|
| "spread simultaneously from multiple sources" | Multi-source BFS |
| "range assign + range sum, arithmetic-looking update" | Segment tree, custom lazy tag |
| "at most K distinct in a window" | Two pointers + monotonic deque |
| "minimum starting value so a running total never dips below 0" | Running-minimum prefix sum |
| "connected component property after each edge add" | DSU with incrementally maintained property |
| "MEX of a partition, maximize/minimize" | DP + last-seen-position pointer |
| "GCD/sum/max of a range, with point updates" | Segment tree (custom merge function) |
| "GCD exactly equal to some target" | Map non-multiples to 0, track full + exclude-one GCD |
| "subset XOR, maximize" | Linear (XOR) basis |
| "subtree pair-counting" | Small-to-large merging on trees, weight by depth if pairs matter to every ancestor |
| "shortest path with jump distances" | Plain BFS on implicit graph |
| "next occurrence of a multiple/related value" | Harmonic-series multiple enumeration + sparse table |
| "small N/M (≤1000) despite fancy wording" | O(N²) DP is *intended* — don't overthink |
| "flip edges / matching + pattern matching" | Tree DP with automaton state, brute force for partial credit |
| "non-decreasing layers with quadratic transition cost" | DP with convex cost — brute O(K²) first, optimize later |

Good luck on the 28th.
