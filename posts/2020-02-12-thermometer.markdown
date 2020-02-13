---
title: Thermometer continuations
author: Calvin
---

**Note: This blog post is still a rough draft. Read on with caution.**

$$ \text{"Delimited continuations are the mother of all monads!"} $$

**NB:** I read this paper awhile ago, but I only recently decided to write a small post about it. You can find the code for this [here](https://github.com/calwoo/delimited).

The famous [1994](https://dl.acm.org/doi/10.1145/174675.178047) paper of Filinski showed denotationally that any monadic (computational) effect can be replicated via delimited continuations, e.g. exceptions/state/nondeterminism/etc. However, most languages do not expose continuations as a first-class citizen, as doing so usually involves capturing the function call stack as a value to be passed around in the code. 

This post will be a summarized view of the functional pearl of [Koppel et al.](https://arxiv.org/pdf/1710.10385.pdf) "Capturing the future by replaying the past". There they show that in any language with exception and state, you can implement delimited continuations-- the style they call as thermometer continuations. This is a cool idea, and allows us to implement multi-shot delimited continuations (i.e. can be used multiple times) in most languages. For reference, Python's generators are considered one-shot delimited continuations, where the suspended generator after a `yield` statement is the continuation, but you can only run the continuation once when you decide to resume it.

Before we push onwards in the post, we just get an inkling of the general idea. We stated before that a delimited continuation is like an exception handler that can be resumed after execution-- the `try` block acts as a `shift` operator where the continuation parameter is given by `throw`, while the `except` block is the part of the `reset` block outside of a shift which demarcates the continuation captured by the shift block. There's really only one problem-- for `shift/reset`, after applying the continuation, we can resume the computation in the `shift` block! Exceptions don't normally do this.

The key intuition is this: just run it again! Actually, this isn't even the intuition. It's really the *entire* idea.


### nondeterminism
A non-trivial effect that many people see is nondeterminism-- if a function has many choices in its execution, then there may not be a single output to a given input. We would like to capture the outputs of all these choices in a composable fashion. The way we usually do this in functional languages is via a monad. For example, the function in Python

```python
def test_fn():
    if choose([True, False]):
        return choose([1,2])
    else:
        return choose([3,4])
```

becomes monadic in Haskell:

```haskell
test_fn = [True, False] >>= \b ->
                if b
                then [1,2] >>= \r -> return r
                else [3,4] >>= \r -> return r
```

But it would be nice if Python did have a primitive like `choose` so that we could write these functions without monads (which when unwinded come down to continuation-passing style). Let's approach this one step at a time. Suppose we just want to perform nondeterminism with a *single choice* with only *2 options*:

```python
def test_fn0():
    return 3 * choose(5,6)
```

How can we get the list of possible outputs from each possible choice of this function? When the function hits the `choose` clause, there are two possible paths the function could go-- 5 or 6. This has an obvious answer: run it twice, once with each of the choices. In code the `choice` operator becomes

```python
# State-bit to record which branch to take during handling.
first_time = False

def choose(x1, x2):
    if first_time:
        return x1
    else:
        return x2
```

Running the delimited continuation is given by effect handlers, which performs the key idea for thermometer continuations:

```python
def with_nondeterminism(fn):
    results = []
    global first_time
    # Take first branch first
    first_time = True
    results.append(fn())
    # Take second branch
    first_time = False
    results.append(fn())
    return results
```

Running this on examples gives us great results!

```python
> with_nondeterminism(lambda : 3 * choose(5,6))
[15, 18]
```

Let's extend this a bit to deal with `choose` clauses with more than two choices. Since a bool can only express 2 values, we need something else to represent the "state" of our continuation. Naturally we'll store the index (and the total length of the choice list, since we need to know where to stop counting). 

```python
# Global state will be a pair, indicating current id of branch,
# and total number of branches
state = (None, None)

def start_idx(xs):
    return (0, len(xs))

def next_idx(k, length):
    if k + 1 == length:
        return (None, None)
    else:
        return (k + 1, length)

def get(xs, k, length):
    return xs[k]
```

