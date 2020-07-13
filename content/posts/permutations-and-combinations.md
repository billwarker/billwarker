---
title: "Permutations and Combinations"
author: "Will Barker"
date: 2020-07-07
tags: ["statistics"]
draft: false
---

# Permutations of a Set

Permutations of a set are particular orderings of its elements. To calculate the number of permutations (i.e. the possible orderings) of a subset $k$ distinct elements from a set of size $n$, the formula is:

$$N(n, k) = \frac{n!}{(n-k)!}$$

$n!$ represents all orderings for every element in the set. $(n-k)!$ represents the orderings of the elements we're not including in the subset of $k$ elements - dividing by this number allows us to factor them out of $n!$ in the numerator and give us the number of permutations.

<!--more--> 


```python
import itertools
import math

S = ['A', 'B', 'C', 'D']

permutations = itertools.permutations(S, 2) # P(4,2)
for p in permutations:
    print(p)
```

    ('A', 'B')
    ('A', 'C')
    ('A', 'D')
    ('B', 'A')
    ('B', 'C')
    ('B', 'D')
    ('C', 'A')
    ('C', 'B')
    ('C', 'D')
    ('D', 'A')
    ('D', 'B')
    ('D', 'C')



```python
def permutations(n, k):
    return int(math.factorial(n)/math.factorial(n-k))

permutations(4, 2)
```




    12



## Combinations of a Set

Combinations don't care about order - they are just the different subsets of elements in the set. To calculate the number of combinations (i.e. subsets) of $k$ distinct elements from a set of size $n$, the formula is:

$$C(n, k) = \frac{n!}{k!(n-k)!}$$

An easy way to understand combinations is in relation to permutations - basically, we are calculating the number of permutations of a subset created by $P(n, k)$ and then just dividing that number by the possible ways to order its elements (since combinations don't care about that).

$$C(n, k) = \frac{P(n, k)}{k!} = \frac{n!}{(n-k)!}*\frac{1}{k!}$$


```python
S = ['A', 'B', 'C', 'D']

combinations = itertools.combinations(S, 2) # P(4,2)
for c in combinations:
    print(c)
```

    ('A', 'B')
    ('A', 'C')
    ('A', 'D')
    ('B', 'C')
    ('B', 'D')
    ('C', 'D')



```python
def combinations(n, k):
    return int(permutations(n, k)/math.factorial(k))

combinations(4, 2)
```




    6


