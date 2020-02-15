---
title: Thermometer continuations, part 2
author: Calvin
---

**Note: This blog post is still a rough draft. Read on with caution.**

In this post we'll describe the construction of the `shift/reset` control operators for implementing delimited continuations, under the style of replay-based thermometer continuations. Recall that the main idea of thermometer continuations is simple: continuations act as "resumable exceptions", so we need to simulate the "resume" part of a continuation, which we can do by the dumbest possible way-- just do the entire computation again! But this time with our already established choices intact until the point where the resumption is called.

We will again build this in stages, following the paper of Koppel et al. But the construction is so simple one has to think if one could have invented it themselves!

Again, the code is on my github [here](https://github.com/calwoo/delimited).


### baby version
Just like for the case of nondeterministic effects, we will first restrict to a version of the `shift/reset` operators that assumes the reset block only contains one shift, which invokes its passed continuation 0 or multiple times. As before we need to some form of global state. What do we need?

The `reset` operator takes in a lambda function with no arguments, returning an expression comprising of (in this case, a single) `shift` operators and other computations. Semantically, when a continuation in `shift` is called, the continuation passed in is the rest of the computation outside of the `shift` operator, delimited by the `reset` block. As such, to capture this continuation we need to "replay" this function again, this time with the `shift` operator replaced by the value called in the `shift` block.

For example, in the expression

```python
reset(lambda: 2 * shift(lambda k: 1 + k(5)))
```

we compute the expression `1 + k(5)`, where the argument `k` is given by the continuation `2 * []` captured by the `reset` statement. Hence once the shift block is hit, we replay the entire computation in `reset`, except this time `shift(...)` is replaced by `5`, so we get `2 * 5`, which is the return value of the `k(5)` in the `shift` block. We continue the computation from there to get `1 + 10` or `11` as the final value.

Hence as a first step, we need to keep around the function inside the `reset` block as a piece of global state, so that we can replay it again when necessary. We also need to keep around a floating `state` global variable, which will be the value replacing the `shift` block during replays of the `reset` block (this what is happening during a single replay):

```python
cur_expr = None
state = None
```

The `reset` operator is fairly simple: we just want to run the function.

```python
def reset(fn):
    global state, cur_expr
    cur_expr = fn
    state = None
    try:
        fn()
    except Done as e:
        return e.value
```

Here, `Done` is an exception wrapping the return value of a (single) `shift` block:

```python
class Done(Exception):
    def __init__(self, value):
        self.value = value
```

Uh, why are we wrapping a return value of a `shift` block in an exception? And why are we throwing an exception in the first place? It's because after the `shift` computation, we just want that as our return value-- we don't want to continue in the `reset` block after a `shift` statement!

Look at our example above-- after the `shift` block gives us 11, we want to abandon the rest and return that as our total return from the `reset` block. If we were to go on afterwards, we would get `2 * 11` or 22. This is why we need both **exceptions** and **state** to build thermometer continuations. To exit out of the `reset` early, we throw an exception.

The replay loop is contained then in the `shift` operator-- we will step through this:

```python
def shift(fn):
    global state

    if state is not None:
        return state
    else:
        def k(x):
            global state
            state = x
            return cur_expr()
        # Recursively call the replay
        result = fn(k)
        raise Done(result)
```

Firstly we see that `shift` relies on a global `state` variable. Recall here that the argument to `shift` is a lambda function `fn = lambda k: (...)` where the argument `k` will be given the value of the captured delimited continuation after the `shift` block, delimited by the `reset`. Hence when running a `shift` block, when we hit an instance of the argument `k` in `fn`, we replay the entire `reset` block where the `shift` block is replaced by the value of the called `k` function. If this is overwhelming, read this paragraph again slowly while writing down how to compute the delimited continuation example above.

The implementation of this (baby) `shift` is clear then: if the state is not `None`, that means we are performing a replay where a value has been passed into the continuation. Then here, `shift` will act as if the entire block is that value, for the purpose of performing the delimited continuation.

Otherwise, we are running the `shift` block for the first time. Here `k` represents the captured continuation: during the replay, we'll replay the entire computation with the state set to the value called in `shift`, so that on the next pass the other if condition will ignore this `shift` block.

```python
        def k(x):
            global state
            state = x
            return cur_expr()
```

We then recursively call the replay and when we hit a result, we create an exception to abort the computation inside the `reset` block so that we don't perform the further computation outside of the `shift` blocks.

This implementation of the `shift/reset` control operators is enough for simple delimited control statements such as

```python
> reset(lambda: 2 * shift(lambda k: 1 + k(5)))
11
> reset(lambda: 1 + shift(lambda k: k(1) * k(2) * k(3)))
24
```

Since we've succeeded in using one `shift` block, let's see how to extend this to more!

### return of the stacks
Suppose we have a delimited control statement with multiple `shift` operators in it:

```python
reset(lambda: 1 + shift(lambda k: k(3)) * shift(lambda m: 1 + m(4)))
```

During computation we hit the first `shift` statement and it calls the captured `k` on 3. The continuation captured by `k` is given by

```
1 + [] * shift(\m -> 1 + m(4))
```

which is itself a call to a `shift` statement, where the continuation captured by `m` is `<return of first shift> * [] + 1`. By the semantics, this continuation is run with the value plugged into `m`, i.e. 4. Here the replay sets the return of the first `shift` to be the value plugged into `k`, i.e. 3, so we have the continuation captured by `m` applied to 4 gives the value 1 + 4 * 3 = 13. This value is then finished up in the `shift` block to give the return value of 14.

Note that in the above we need to keep track of all the return values for each `shift` operator during the replay. This is the origin of the **thermometer**, which will be a stack of values replacing the `shift` operators during a replay. As a start, our global state will consist of the function to be called repeatedly (in the `reset` block):

```python
cur_expr = None
```

and some stacks to form our "thermometer":

```python
past = []
future = []
```

Here, state is represented by a past and future stack. What if our calls to `shift` are nested? Looking at the semantics of a control statement like

```python
1 + reset(lambda: 2 + shift(lambda k: 3 * shift(lambda l: l(k(10)))))
```

we see that we can't just blindly set a `shift` operator to a fixed value, because in a nested call, we will prematurely replace the entire `shift` value with the past state during replay without reaching inwards and evaluating the inside `shift`. Hence we need to keep track of nested calls with an extra stack

```python
nest = []
```

This gives us the full global state necessary to implement delimited continuations. First we show what a thermometer is: it is a data structure that keeps track of the stateful replays of a `reset` operator.

```python
def thermometer(fn, fn_future):
    global past, future, nest, cur_expr
    # Push state of current reset block into nest stack
    nest.append((cur_expr, past.copy(), future.copy()))
    # Set up the thermometer
    past = []
    future = fn_future
    cur_expr = fn
    # Run the computation
    def run():
        try:
            return fn()
        except Done as e:
            return e.value
    result = run()
    # Undo the nesting
    try:
        # Set the thermometer state for recursive return
        prev_expr, prev_past, prev_future = nest.pop()
        cur_expr = prev_expr
        past = prev_past
        future = prev_future
        return result
    except IndexError:
        raise ValueError
```

Once we have the thermometer set up, the `reset` operator is the initial instance of one:

```python
def reset(fn):
    return thermometer(fn, [])
```

As before, in the `run` function, the `shift` operator will return its final value to `reset` via a thrown exception, which will ignore the remainder of the computation outside of those operators. 

The thermometer gives us the ability to replay a `reset` block with the `shift` statements replaced by the evaluated `future` stack. This informs how to build the `shift` operator. Put another way, the thermometer (which is the future) contains the values of all effectful computations that have perspired in the execution of a `shift` operator.

There are two cases in the execution of a `shift` block: the first is if the block is called for the first time. In this case, we'll replay the entire computation with the state set to the value called in `shift`'s captured continuation, so that on the next pass the other case will ignore that block:

```python
        new_future = list(reversed(past))
        our_expr = cur_expr
        def k(v):
            return thermometer(our_expr, new_future + [v])
        past.append(None)
        # Recursively call the replay
        result = fn(k)
        # When we hit a result, create an exception to abort the computation in
        # the reset block so that we don't perform the further computation outside
        # of the shift blocks.
        raise Done(result)
```

The other case is during a replay, in which the `shift` block will have a determined value given by the `future` stack:

```python
        val = future.pop(0)
        past.append(val)
        return val
```

Combined into a single operator, this gives the `shift` function:

```python
def shift(fn):
    global past, future, cur_expr
  
    case = None
    if len(future) == 0:
        case = 1
    else:
        val = future.pop(0)
        if val is None:
            case = 1
        else:
            case = 2
    # Case 1
    if case == 1:
        new_future = list(reversed(past))
        our_expr = cur_expr
        def k(v):
            return thermometer(our_expr, new_future + [v])
        past.append(None)
        # Recursively call the replay
        result = fn(k)
        raise Done(result)
    # Case 2
    elif case == 2:
        past.append(val)
        return val
```

This gives our general delimited control operators `shift/reset`, all in Python! We can test this out and get a bunch of cool results from it.

```python
> reset(lambda: 2 * shift(lambda k: 1 + k(5)))
11
> reset(lambda: 1 + shift(lambda k: k(1) * k(2) * k(3)))
24
> 1 + reset(lambda: 2 + shift(lambda k:
>           3 * shift(lambda l: l(k(10)))))
37
```


### closing
To encapsulate all this control logic into a single object, the `delim` library on my github exposes a single class called `Cont` which gives delimited control to any codebase. An example use is similar to the above:

```python
import delim

C = delim.Cont()
ex = C.reset(lambda: 1 + C.shift(lambda k: k(1) * k(2) * k(3)))
print(ex2) # => 24
```

Delimited control gives us a bunch of algebraic effects and handlers that we would be happy to use in a modern programming language. In the next post of the series, I'll use delimited continuations to implement reverse-mode autodifferentiation, a less well-known algebraic effect in the programming language theory literature. Thanks for reading!