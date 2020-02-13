---
title: Thermometer continuations, part 1
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

Aside from the slightly-added complexity of having an index as global state, the same idea stays the same: **we replay the function repeatedly, each time using a new value of the choice function**. Now the `choose` operator is a suitably extended version of the 2-choice version above:

```python
def choose(xs):
    global state
    if len(xs) == 0:
        raise ValueError("it's the end")
    else:
        # Grab a value based on the current global state
        if state[0] is None:
            state = start_idx(xs)
            return get(xs, *state)
        else:
            return get(xs, *state)
```

where the effect handler `with_nondeterminism` is updated to run through the index state one-by-one, as opposed to just flipping a boolean state:

```python
def with_nondeterminism(fn):
    global state
    try:
        # Run the function with current state
        results = [fn()]
        if state[0] is None:
            return results
        else:
            if next_idx(*state)[0] is None:
                return results
            else:
                state = next_idx(*state)
                return results + with_nondeterminism(fn)
    except ValueError:
        return []
```

However, we notice that our global state can only deal with a single `choose` operator-- how do we extend this to deal with multiple choose branches as in the original `test_fn` above? If you think about what the function execution would look like with all choices enumerated, this looks like a (stateful) tree. Performing a traversal of this tree while keeping state is effectively what an effect handler for `choose` would be doing! So taking a cue from iterative [traversal](https://stackoverflow.com/questions/1294701/post-order-traversal-of-binary-tree-without-recursion) techniques (see? coding interviews are helpful!), we will keep along our global state two stacks: the **future** and the **past**.

```python
past   = []
future = []
```

The `past` contains choices already made. The `future` contains the known choices *to be* made. The basic idea is that we record in our stacks the path taken by the execution of the program (this is similar to MCMC done in probabilistic programming languages). We then modify a single choice in our path at each new iteration, until we have exhausted all the possible paths possible through the tree. 


```python
def next_path(xs):
    if len(xs) == 0:
        return []
    else:
        i = xs[0]
        if next_idx(*i)[0] is None:
            return next_path(xs[1:])
        else:
            return [next_idx(*i)] + xs[1:]
```

How is this handler supposed to work? When the execution of the handler reaches a call to choose, it reads the choice from the future stack, and pushes the remainder to the past. If the future is unknown, then it means we have reached a choose statement for the first time, at which we pick the first choice and record it in the past stack.

```python
def choose(xs):
    global past, future
    if len(xs) == 0:
        raise ValueError("it's the end")
    else:
        if len(future) == 0:
            # If there is no future, start a new path index and
            # push it into the past.
            i = start_idx(xs)
            past.insert(0, i)
            return get(xs, *i)
        else:
            # Otherwise, read the instruction from the future stack
            # and execute, pushing back into the past.
            i = future.pop(0)
            past.insert(0, i)
            return get(xs, *i)

def with_nondeterminism(fn):
    global past, future
    try:
        results = [fn()]
        next_future = list(reversed(next_path(past)))
        # Reset past/future stacks
        past   = []
        future = next_future
        if len(future) == 0:
            return results
        else:
            return results + with_nondeterminism(fn)
    except ValueError:
        return []
```

Testing this gives us the expected behavior:

```python
> with_nondeterminism(test_fn)
[1, 2, 3, 4]
```


### next steps
And that's pretty much it. Delimited continuations are only a slightly more complicated version of this-- we are traversing paths through execution trees as well, except this time the trees have state in their nodes. In the next post we'll start looking at delimited continuations in the `shift/reset` paradigm and use this to build some cool algebraic effects in Python.
