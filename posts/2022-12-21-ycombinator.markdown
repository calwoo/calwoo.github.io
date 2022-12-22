---
title: Recursion and fixed points
author: Calvin
---


I haven't written in awhile, but I have a few posts stored up. But of course, I get distracted a lot, so this is going to be a short post about recursion in the lambda calculus. It also gives me an excuse to just review the lambda calculus in general!

## lambda calculus

The **lambda calculus** is a mathematical theory of computation, in which all computation is done using $\lambda$-functions. A $\lambda$-function is a pretty restrictive form of function-- they are single argument functions that always return a single thing. For example, we can have

```
λx.3x+1  # represents \x -> 3*x + 1
```

where the comment describes a Haskell lambda. 

The single argument isn't too much of a restriction, by the way. We can extend to (finite) multiple arguments by **currying**:

```
λx.λy.3xy+x-1
```

which is equivalently in Python given by

```python
def f(x, y):
    return 3*x*y + x - 1

# same as
def f(x):
    def g(y):
        return 3*x*y + x - 1
    return g
```

As $\lambda$-functions are indeed functions, we can apply them on other expressions like

```
[λx.3x+1] 7
```

which would give 22. This all looks dandy, except that in the lambda calculus, we really only have functions (well, plus variable bindings and evaluations). So an expression like `λx.3x+1` will require us to figure out some things, like what the symbols `3` and `+` mean in this theory of computation.

How would you, for example, model the natural numbers $\mathbf{N}$ in the $\lambda$-calculus? [Alonzo Church](https://en.wikipedia.org/wiki/Alonzo_Church) thought about these *encodings* a lot in the $\lambda$-calculus, and his solution is an elegant one. I don't know of a good way to motivate this definition from first principles-- it's better to just stare at the definition and stew in it.

To start, we let the numeral $1$ to be defined as the $\lambda$-function

```
1 = λf.λx.f(x)
```

In the $\lambda$-calculus, we don't really have much. The only thing we have is functions and function applications. So for counting, we might as well count the number of function applications we have towards a single input. This motivates the other **Church numerals**

```
2 = λf.λx.f(f(x))
3 = λf.λx.f(f(f(x)))
...
n = λf.λx.f(...f(x))  # where function is applied n times
```

We also can define $0$ as `0 = λf.λx.x`, having no function applications. An alternative way to generate the Church numerals is to start with something like the base numeral $0$ given above and applying the **successor function** `succ`, given as

```
succ = λg.λf.λx.f(g(f)(x))
```

The successor function is just function application at it's core. For example, we can see that

```
succ(2) = succ(λa.λy.a(a(y)))
        = λf.λx.f([λa.λy.a(a(y))] f x)
        = λf.λx.f(f(f(x)))
        = 3
```

If we really want to call these things numbers, we better have a way to do stuff with them. One thing you want to do with numbers is add them. How do we define `+`? Since we've reduced the natural numbers to repeated function application, addition is then relatively easy-- just apply the function even more times. For example, if you have 2 and 3, which means you applied a function twice and three times, then applying them in sequence just means you applied it 5 times, which is the sum `2 + 3 == 5`.

Formalizing, we can create the $\lambda$-function

```
add = λn.λm.n(succ)(m)
```

Roughly translated, we apply the successor function $n$ times, starting with $m$. As an example, we can try

```
add(2)(3) = [λn.λm.n(succ)(m)] 2 3
          = 2(succ)(3)
          = [λf.λx.f(f(x))] succ 3
          = succ(succ(3))
          = succ(4)
          = 5
```

where we did the analysis for `succ` before.

Okay, cute! Multiplication is then another easy hop from here-- instead of applying `succ` $n$ times, we apply $add(m)$ (which is again a function).

```
mult = λn.λm.n(add(m))(0)
```

For example, we can compute

```
mult(2)(3) = [λn.λm.n(add(m))(0)] 2 3
           = 2(add(3))(0)
           = [λf.λx.f(f(x))] add(3) 0
           = add(3)(add(3)(0))
           = add(3)(3)
           = 6
```

For sanity's sake, we can explicitly write out

```
mult(x)(0) = [λn.λm.n(add(m))(0)] x 0
           = x(add(0))(0)
           = [λf.λx.f(...f(x))] add(0) 0
           = add(0)(... add(0)(0))
           = add(0)(... 0)
           = 0

mult(0)(x) = [λn.λm.n(add(m))(0)] 0 x
           = 0(add(x))(0)
           = [λf.λy.y] add(x) 0
           = 0
```

Whew! Now we've finally defined the $\lambda$-function `λx.3x+1`. We note that in the above, we have been using [alpha](https://en.wikipedia.org/wiki/Lambda_calculus#%CE%B1-conversion) and [beta](https://en.wikipedia.org/wiki/Lambda_calculus#%CE%B2-reduction_2) reductions liberally. Anyway, the $\lambda$-calculus is incredible expressive, and can provably do everything a Turing machine can do (leading to the [Church-Turing thesis](https://en.wikipedia.org/wiki/Church%E2%80%93Turing_thesis)).

So for example, we can do **recursion**.


## recursion

Let's look at an example of a recursive function, the factorial:

```python
def fact(n: int) -> int:
    if n == 0:
        return 1
    else:
        return n * fact(n - 1)
```

A key feature of this definition is that it refers to the function itself in the definition. This self-reference requires us to give the function a *name*, which isn't something we can do in the standard $\lambda$-calculus. $\lambda$-functions are *anonymous*, hence it's difficult to write a recursive function this way.

However, if we step back and think about our goal, we might have a path forward. We want to construct a function `fact` in the $\lambda$-calculus in such a way that `fact` is a "definitionally a function of itself". That is, there is some other lambda `g`, such that we can express `fact` as

```
fact = g(fact)
```

We see that the `fact` $\lambda$-function is hence a **fixed point** of the $\lambda$-function `g`! In this way, we have converted our question of defining a recursive function (that may require self-reference) into a question of computing the fixed points of $\lambda$-functions.

So now just need a way to compute the fixed points of $\lambda$-functions. Our goal is to divine a higher-order function

```haskell
fix :: (a -> a) -> a
```

such that for any function `f`, we have `fix f = f (fix f)`-- that is, `fix` takes a function and returns a fixed point of the function. Haskell is kinda cheating when we define this, by the way-- you can write

```haskell
fix f = let {x = f x} in x
```

and just let lazy evaluation semantics take hold to make sure you don't cause a horrific stack overflow. 

But in the standard $\lambda$-calculus we can't get off this easily. To begin with, lets try to express the simplest recursive function

```
loop = loop
```

Here, loop is the true self-referential function. Indeed, in Python

```python
def loop(x):
    return loop(x)
```

How would we write this in the $\lambda$-calculus? Brace yourself for some magic:

```
loop = (λx.x x) (λx.x x)
```

If we sit back and expand it out in terms of the function application and $\beta$-reduction rules, we see that indeed this does give `loop = loop` by definition.

From here, it's a little logical leap to get to our implementation of `fix f`: we want to loop, but when we do, we want to apply our function `f`. This is because, in analogy to the fixed point theorems we find in mathematics all the time, we often get a fixed point of a function by infinitely applying the function to an input. In our case it looks like

```
fix f = f (fix f)
      = f (f (fix f))
      = f (f ... f (fix f))
```

So we do the same magic as `loop`, except now we add an `f` when we perform the loop:

```
fix = λf.(λx.f (x x)) (λx.f (x x))
```

This implementation of `fix` is also known as the **Y combinator** (yes, just like [that](https://www.ycombinator.com/) Y-combinator). This is an awesome definition, but we should at least verifies that it gives us a fixed point!

```
fix f = [λg.(λx.g (x x)) (λx.g (x x))] f
      = (λx.f (x x)) (λx.f (x x))
      = f ((λx.f (x x)) (λx.f (x x)))
      = f ([λg.(λx.g (x x)) (λx.g (x x))] f)
      = f (fix f)
```

This is pretty incredible. To turn out `fact` into a $\lambda$-function then, we need to express `fact` then as the fixed point of an associated $\lambda$-function. Taking the $\lambda$

```
λr.λx.if x == 0 then 1 else r(x - 1) * x
```

we have that

```
fact = fix λr.λx.if x == 0 then 1 else r(x - 1) * x
```

In python, we can write the above as

```python
# the Y combinator
fix = lambda f: (lambda x: f(x(x)))(lambda x: f(x(x)))

# the associated recursion lambda for fact
g = lambda r: (lambda x: 1 if x == 0 else r(x - 1) * x)

# the factorial
fact = fix(g)
```

Testing, we see in the interpreter

```
> fact(9)
RecursionError: maximum recursion depth exceeded
```

Whoops! Python eagerly evaluates those function calls, leading to a burst stack. The usual way to get around this is by passing in a *thunk*, giving

```python
fix_thunk = lambda f: \
    (lambda x: f(lambda y: x(x)(y)))\
    (lambda x: f(lambda y: x(x)(y)))

fact_thunk = fix_thunk(g)
```

Then, in fact, we get finally a running computation:

```
> fact_thunk(9)
362880
```
