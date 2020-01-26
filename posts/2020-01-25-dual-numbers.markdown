---
title: Dual numbers in autodifferentiation
author: Calvin
---

So last week I went to the [NYCPython](https://www.meetup.com/nycpython/) lightning talk night at [Metis](https://www.thisismetis.com/) and I had a blast! Thanks to everyone who talked, there was a lot of really cool stuff there (including a static analysis constraint checker for Python utilizing the Z3 solver)! As a result, I'm sorta inspired to give a talk at the next one, but since they're 5 minutes long I'm not sure how I would fit something interesting into that time.

Instead I'll just write a blog post about something I've been learning about in the last year, and that's autodifferentation. It's an old idea, with many different forms but it has been immensely influential for modern machine learning when people started realizing you could use stochastic gradient descent for *almost* anything. But even more recently, there has been a push to make differentiation of programs a "first-class entity" in programming languages, and there has been many ideas as to how to express this as a language primitive. 

For example, wouldn't it be nice if your favorite programming language had built-in support for generating gradients of your code without explicitly coding it in? This is the dream of **differentiable programming**-- you would write a program with free or constrained parameters that lived in various metric spaces (or types), and you would just apply the `grad` primitive and all of a sudden your function would be trained by some kind of gradient descent to give back optimal values. This is vague, but it would akin to the sort of transformation occuring with probabilistic programming languages, in which Bayesian inference is a "first-class entity"-- you only have to define the code, and all stochastic elements are captured by the runtime-system and inference is performed automatically for you without you having to hardcode a custom algorithm yourself. 


### dual numbers
Let $f: X\to Y$ be a map between smooth manifolds (or schemes, or whatever). In differential geometry one learns that there is an induced map on the tangent bundles given by the pushforward

$$ f_* : T_x X\to T_{f(x)} Y $$

In local coordinates, this pushforward is a linear map given precisely by the Jacobian, the matrix of partials $\frac{\partial f^j}{\partial x^i}$. The relevant fact for machine learning is that the pushforward preserves composition-- that is

$$ (g\circ f)_* = g_*\circ f_* : T_x X \to T_{f(x)} Y \to T_{g(f(x))} Z $$

This is the essence behind **forward-mode auto-differentiation**-- we compute these pushforwards and propagate them through the computational graph. However in fields like algebraic geometry, it tends to be more natural to consider the cotangent space $T_x^* X$, which is the linear dual of the tangent spaces $T^*_x X = \text{Lin}(T_x X, \mathbb{R})$. By precomposition, we get the *adjoint map* to the pushforward, 

$$ f^* : T_{f(x)}^* Y \to T_x^* X $$

In this setting, the composition is reversed, and this gives us the **reverse-mode auto-differentiation**. The linear map $f^*$ is given in local coordinates by the adjoint (transpose) of the Jacobian, which explains its form in the backpropagation algorithm.

In this post we'll focus on another way to interpret the forward-mode. First, we interpret the tangent space $T_x X$ as the collection of equivalence classes of functions $k \to X$ passing through $x \in X$ (here $k$ can be taken as $\mathbb{R}$), where functions are equivalent when they have equivalent first-order derivatives. By restricting to functions with Taylor expansions, we can remove the dependence on equivalence classes and get reified functions, this time from an algebraic analogue of first-order Taylor expansions-- the **dual numbers**:

$$ T_x X \simeq \text{Hom}_x(\text{Spec}(k[x] / x^2), X) $$

where $\text{Hom}_x$ denotes maps relative to the point $x$. From this we see that the pushforward $f_*$ takes on a simple form-- it's just postcomposition:

$$ f_* : \text{Hom}_x(\text{Spec}(k[x] / x^2), X) \to \text{Hom}_x(\text{Spec}(k[x] / x^2), Y) $$


### forward-mode autodiff
*Upshot:* We can use this perspective of the pushforward to perform forward-mode auto-differentation. We build a computational graph and push through the inputs a dual number. In the end we output a value that contains the derivative we are looking for.

We'll build an example of this in Python using operator overloading. Python does this by using *magic methods* like `__add__` and `__mul__`. To start, we need a representation of a dual number.

```python
from typing import Union

class Dual:
    def __init__(self, val: float, tangent: float = 0):
        self.val = val
        self.tangent = tangent
```

Algebraically, an element of the dual numbers $k[x]/(x^2)$ is of the form $a + bx$. The algebra of dual numbers is straightforward, and is given by

```python
    def __add__(self, other: Union["Dual", float]) -> "Dual":
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.tangent + other.tangent) 
        else:
            return Dual(self.val + other, self.tangent)

    def __sub__(self, other: Union["Dual", float]) -> "Dual":
        if isinstance(other, Dual):
            return Dual(self.val - other.val, self.tangent - other.tangent)
        else:
            return Dual(self.val - other, self.tangent)

    def __mul__(self, other: Union["Dual", float]) -> "Dual":
        if isinstance(other, Dual):
            return Dual(self.val * other.val, self.val * other.tangent + self.tangent * other.val)
        else:
            return Dual(self.val * other, self.tangent * other)
    
    def __pow__(self, other: float) -> "Dual":
        return Dual(self.val ** other, self.tangent * other * self.val ** (other - 1))
```

Since all arithmetic operations are overloaded, performing forward-mode autodiff is as easy as plugging it into the functions and extracting the tangent component:

```python
> x = Dual(3.0, 1.0)
> f = lambda x: x**2 + x - 3
> f(x).tangent
7.0
```

We can also perform this calculation for multivariate functions:

```python
> x = Dual(3.0, 1.0)
> y = Dual(4.0, 0.0)
> f = lambda x, y: x**2 + x*y - y
> f(x, y).tangent
10.0        # derivative w.r.t x
> x = Dual(3.0, 0.0)
> y = Dual(4.0, 1.0)
> f = lambda x, y: x**2 + x*y - y
> f(x, y).tangent
2.0         # derivative w.r.t y
```

That's cool.


### reverse-mode autodiff
Note that for the function $f:\mathbb{R}^2\to\mathbb{R}$ above, we had to perform the forward-mode procedure twice to compute the directional derivatives. In general, for a function $f:\mathbb{R}^n\to\mathbb{R}$ we require $\mathcal{O}(n)$ forward-passes to compute the complete Jacobian of the function. As these are the kind of functions ubiquitously found in machine learning (e.g. loss functions), we seek more efficient ways to compute this.

Instead, reverse-mode autodifferentiation propagates errors backwards from the output of the computation. To keep track of the gradients, we take any computation and lift it to a computational graph DSL (domain-specific language). 

To start, the computational graph is a collection of nodes:

```python
from typing import List, Tuple, Optional

class Node:
    def __init__(self, value: float):
        self.value = value
        self.children: List[Tuple["Node", float]] = []
        self.grad_value: Optional[float] = None
```

Each node in a graph contains a list of references to each child node-- this is because when computing the gradient value in a node, we compute it as a weighted sum of the propagated gradients from the children. As a convention, each reference in the `children` list is a tuple of the child node and the intermediate gradient $\frac{d\text{child}}{d\text{parent}}$. 

Computations are then build via Python's magic methods (operator overloading also):

```python
    @nodeize
    def __mul__(self, other: "Node") -> "Node":
        new_node = Node(self.value * other.value)
        # attach hooks to dependency nodes
        self.children.append((new_node, other.value))
        other.children.append((new_node, self.value))
        return new_node

    @nodeize
    def __add__(self, other: "Node") -> "Node":
        new_node = Node(self.value + other.value)
        # attach hooks to dependency nodes
        self.children.append((new_node, 1.0))
        other.children.append((new_node, 1.0))
        return new_node

    def __pow__(self, other: float) -> "Node":
        new_node = Node(self.value ** other)
        self.children.append((new_node, other * self.value ** (other - 1)))
        return new_node
```

Here, `nodeize` is a decorator to avoid excessive `isinstance` checks (we don't type-hint it for readability):

```python
def nodeize(fn):
    def _fn(cls, other):
        if isinstance(other, float):
            other = Node(other)
        return fn(cls, other)
    return _fn
```

Finally, gradients are computed by the process above:

```python
    @property
    def grad(self) -> float:
        # accept gradients from all children
        if self.grad_value is None:
            self.grad_value = sum(var.grad * dx for var, dx in self.children)

        return self.grad_value
```

This gives a fully-fledged reverse-mode autodifferentation system.

```python
> x = Node(2.0)
> y = Node(1.0)
> z = x * y ** 2 + x ** 2
> # Set gradient value at output to start reverse-mode
> z.grad_value = 1.0 
> x.grad
5.0
> y.grad
4.0
```


### closing
These are two very basic ways to build an autodiff system. I am not sure how to connect the dual number perspective to reverse-mode autodifferentiation (I would need a dual-number approach to the cotangent space, but making it explicit might be difficult). If anyone knows how, let me know please!

In a later blog post, I'll talk about another way to approach reverse-mode autodifferentiation using something I have been interested in for the past year: effect handlers and delimited continuations.