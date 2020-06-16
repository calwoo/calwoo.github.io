---
title: Probability monads
author: Calvin
---

**Note: This blog post is still a rough draft. Read on with caution.**

I've been a big fan of probabilistic programming for a year now, and being a data scientist, I am always looking for ways to "Bayesify" our models to encode epistemic uncertainty. However, my main motivation a year back for learning about Bayesian statistics (unlike many people) was a functional pearl ["Probabilistic functional programming in Haskell"](http://web.engr.oregonstate.edu/~erwig/papers/PFP_JFP06.pdf) by Erwig-Kollmansberger. 

There they describe a nice way to encode basic probabilistic structures into a functional paradigm-- and it was so naturally done. This led me to try and learn about the Giry monad and really just led me down the rabbit hole of Haskell and functional programming in general. Despite this, I never really wrote down my notes on this, even on paper. So this post will be effectively a dump of notes on the topic of categorical probability theory and the associated probability monads.


### giry monad

Let $\mathcal{C}$ be a base category of "spaces", for example we can take $\mathcal{C}=\text{Set}^{\text{fin}}$ to be finite sets, or $\mathcal{C}=\text{Meas}^{\text{bdd}}$ to be the category of bounded measurable spaces. Then we can try and define probability monads to be endofunctors that encode a notion of "random variable" or "distribution" on objects in $\mathcal{C}$.

As a first attempt, we can define the **distribution monad** on $\text{Set}^{\text{fin}}$ to be defined by the functor that sends a finite set $X$ to finite distributions $p:X\to [0,1]$ (i.e. $p(x)\ge 0$ for all $x\in X$ and $\sum_{x} p(x) = 1$). We can extend this to non-finite sets by restricting ourselves to finitely-*supported* distributions. Turning this mapping into a functor is motivated by measure theory: the pushforward of measures over a map of base spaces $f:X\to Y$ is given by 

$$ f_*p(y) = \sum_{x: f^{-1}(y)} p(x) $$

This indeed gives a monad $\text{Dist}_\text{fin}$: the unit is given by the Dirac measure supported at a point $x$, and the monadic composition is given by a marginalization process. What is interesting is to consider what the composition of Kleisli morphisms $X\to\text{Dist}_\text{fin}Y$ is like. 

This is again a straightforward computation-- a Kleisli morphism $X\to\text{Dist}_\text{fin}Y$ is equivalent to a function $k:X\times Y\to [0,1]$ such that for each $x\in X$, $k(x,-): Y\to [0,1]$ is a finite-supported probability distribution on $Y$. Unwinding the definition of Kleisli composition in terms of the monadic composition above, we see that for morphisms $k:X\to\text{Dist}_\text{fin}Y$ and $h:Y\to\text{Dist}_\text{fin}Z$ we have

$$ (h\circ k)(x, z) = \sum_{y: Y} k(x,y)h(y,z) $$

which are the **Chapman-Kolmogorov** equations. Hence we get Markovian properties from Kleisli composition in our formulation of probability theory. 

We **note** that the above doesn't give us a functor on $\text{Set}^\text{fin}$ as the sets $\text{Dist}_\text{fin}X$ are surely never finite. However, this doesn't stop us from abusing the notation, and having it give us inspiration as to what to do in the case of bounded measurable spaces. 

In this case, let $X$ be a measurable space and let $\text{Prob}(X)$ be the space of probability measures on $X$. To make this a measure space, we need to equip $\text{Prob}(X)$ with a $\sigma$-algebra structure. We do this by borrowing the $\sigma$-algebra structure on $\mathbf{R}$. Let $f\in\text{Meas}(X,[0,1])$ be a measurable function. Then for each, there is an integration function $\epsilon_f: \text{Prob}(X)\to\mathbf{R}$ induced from it:

$$ p \mapsto \int_X f(x) dp(x) $$

We define the $\sigma$-algebra on $\text{Prob}(X)$ to be the smallest such that makes all of the $\epsilon_f$ measurable. Now that we are dealing with real measures, functoriality of $\text{Prob}$ comes from the true pushforward of measures

$$ f_*p(A) = p(f^{-1}(B)) $$

the unit is given by the Dirac distribution (tempered?) and the monadic composition is given analogously as above to the marginalization $b_X:\text{Prob}(\text{Prob}(X)) \to \text{Prob}(X)$

$$ b_X(\mathcal{P})(A) = \int_{p: \text{Prob}(X)} p(A)\cdot d\mathcal{P}(p) $$

This together gives us the **Giry monad** $\text{Prob}$ on the category $\text{Meas}^\text{bdd}$ of bounded measurable spaces.


### monoidal structures

It's worth pointing out that there is a monoidal structure on the probability monads that correspond to the independent joint distribution on product spaces. Let $X, Y$ be base spaces of interest: then for given probability distributions/measures $p, q$ on $X, Y$ respectively, we get the product joint distribution $p\otimes q$ on $X\times Y$. That is, we get a monoidal structure

$$ \otimes: \text{Prob}(X)\times \text{Prob}(Y)\to \text{Prob}(X\times Y) $$

on the (Giry) monads, making them [commutative monads](https://ncatlab.org/nlab/show/commutative+monad). Since the monoidal product is derived from the product on base spaces, we functorially get projection morphisms

$$ \text{Prob}(X) \xleftarrow{\pi_{X*}} \text{Prob}(X\times Y) \xrightarrow{\pi_{Y*}}\text{Prob}(Y) $$

Given a joint distribution $p_{XY}\in\text{Prob}(X\times Y)$, we can compute its pushforward along the projection on a measurable subspace $A$ as

$$ \pi_{X*}p_{XY}(A) = p_{XY}(\pi_X^{-1}(A)) $$

which is the **marginalization** operator. Hence there is an opportunity to reframe many computations in probability theory in the language of pushforwards. 

Another such computation comes from the fact that the strong monoidal property of the monad $\text{Prob}$ gives operations on $\text{Prob}(X)$ derived from operations on $X$. For example, the addition operator $+:\mathbf{R}\times\mathbf{R}\to\mathbf{R}$ leads to an operator on probability distributions over $\mathbf{R}$

$$ \text{Prob}(\mathbf{R})\times\text{Prob}(\mathbf{R})\xrightarrow{\otimes}\text{Prob}(\mathbf{R}\times\mathbf{R})\xrightarrow{+_*}\text{Prob}(\mathbf{R}) $$ 

But computing this, we see this is given by $(p,q)\mapsto (p\otimes q)(+^{-1}(-))$, which on local coordinates is given by the integral

$$ x \mapsto (p\otimes q)(+^{-1}(x)) = \int_{y: \mathbf{R}} p(x-y) q(y) \cdot d\mu_\text{Lebesgue} $$

where $\mu_\text{Lebesgue}$ is the Lebesgue measure on $\mathbf{R}$. Hence the monoidal operation on probability distributions derived from addition of reals is the **convolution** operator. There are likely many other fun examples one can cook up by playing around with pushforwards. 


### a projection formula?

A small aside before moving onto the Kantorovich monad. Let $f:X\to Y$ be a map of measurable spaces, and let $q$ be a probability distribution on $Y$. Suppose that $f$ has bounded fibers, that is for each $y\in Y$, the preimage $f^{-1}(y)$ is measurable of finite measure. Then we can define a **pullback** measure $f^*q$ on $X$ given by

$$ df^*q(x) = \frac{1}{\mu_X(f^{-1}(f(x)))} \cdot dq(f(x)) $$

Let $p, q$ be probability measures on $X, Y$ respectively, and $f:X\to Y$ a measurable function between them. Then we can look at the joint probability distribution

$$ d(f_*p\otimes q)(a,b) = df_*p(a)\otimes dq(b) = dp(f^{-1}(a))\otimes dq(b) $$

On the other hand, the pushforward given by

$$ d((f\times f)_*(p\otimes f^*q))(a, b) = d(p\otimes f^*q)(f^{-1}(a), f^{-1}(b))$$

is exactly the same. Indeed, we see that because

$$ df^*q(f^{-1}(b)) = \sum_{a: f^{-1}(b)} \frac{1}{\mu_X(f^{-1}(b))}\cdot dq(b) = dq(b) $$

we get the identity of measures

$$ f_*p\otimes q = (f\times f)_*(p\otimes f^*q) $$

This is a form of [projection formula](https://stacks.math.columbia.edu/tag/01E6) from algebraic geometry. This is fairly interesting because it suggests some formulation of these probability monads in terms of **probability sheaves** over measurable base spaces.

It is also worth investigating the origin of these projection formulae in terms of adjunctions in a Wirthmuller context, following Fausk-Hu-May's ["Isomorphisms between left and right adjoints"](http://www.tac.mta.ca/tac/volumes/11/4/11-04.pdf).


### kantorovich monad

Let $\mathcal{C} = \text{Met}^\text{cmpl}$ be the category of complete metric spaces and 1-Lipschitz maps. Since we have no implicit boundedness condition on the spaces, we need to impose a boundedness conditions on the space of probability measures we will use as our probability monad.

Which one? The main reason for imposing a boundedness condition in the first place is to ensure that functions $X\to\mathbf{R}$ (which generate the $\sigma$-algebra) are well-defined and measurable for all the measures we could impose on $X$ (assuming the Lebesgue measure $\mu_\text{Lebesgue}$ on $\mathbf{R}$). For $X$ a complete metric space, this doesn't always hold for general probability measures $p$ on $X$, so the most naive condition we could impose on admissible $p$ is just for these maps to be defined.

It turns out that this is equivalent to $p$ having **finite first moment**, that is, the (upper bound) of the [1-Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric) of the probability measure

$$ \int_{x,y:X\times X} \text{d}(x, y)\cdot dp(x) dp(y) < \infty $$

is finite. This is also known as the Kantorovich metric (because of its role in Kantorvich-Rubenstein duality). While finiteness of the first moment gives finiteness of the 1-Wasserstein metric, it isn't by itself the metric. We define the **1-Wasserstein metric** as

$$ \lVert p \rVert_{\text{Wass}, 1} = \inf_{\mu: \Gamma(p,q)}{\int_{x,y:X\times X} \text{d}(x,y)\cdot d\mu(x,y)} $$

where the infimum runs over all joint probability distributions $\mu$ on $X\times X$ with marginals $p,q$.

Let $\text{Prob}_{\text{Wass}, 1}(X)$ be the space of probability measures on $X$ with finite first moment. This is a complete metric space under the 1-Wasserstein metric, and $\text{Prob}_{\text{Wass}, 1}$ provides a probability monad, called the **Kantorovich monad** on complete metric spaces. There is an interesting characterization of this monad in terms of "colimits of finite samples", following [Fritz-Perrone](http://www.tac.mta.ca/tac/volumes/34/7/34-07abs.html). 

The basis for this characterization comes from the following probabilistic fact: under the 1-Wasserstein metric, any probability distribution on a complete metric space with finite first moment can be approximated by finite empirical distributions. To encode this in the monad, we need to represent the space of finite empirical distributions over $X$. This is straightforward-- let $X^{\otimes n}$ be the $n$-fold product of $X$ with the metric given by the normalized sum

$$ d(\{x_i\}_i, \{y_j\}_j) = \frac{1}{n}\sum_{k} d(x_k, y_k) $$

Since distributions are independent of the order of the points being presented, we take the quotient space under the symmetric group action (all permutations) to get $(X^{\otimes n})_{\Sigma_n}$ as our space of finite empirical distributions. Here, the metric is defined as the discrete 1-Wasserstein metric

$$ d(\{x_i\}_i, \{y_j\}_j) = \min_{\sigma\in\Sigma_n}{\frac{1}{n}\sum_{k} d(x_k, y_{\sigma(k)})} $$

Diagonal maps $X\to X^{\otimes k}$ define transition maps $(X^{\otimes n})_{\Sigma_n}\to(X^{\otimes nk})_{\Sigma_{nk}}$ which define a functor 

$$ X^{\text{sym}}: \mathbf{N}_\text{mult}\to \text{Met}^\text{cmpl} $$

from the multiplicative monoid of natural numbers to complete metric spaces. The main theorem of Fritz-Perrone is that the colimit of this functor recovers the Kantorovich space of probability measures

$$ \text{Prob}_{\text{Wass}, 1}(X) \simeq \text{colim}_{n: \mathbf{N}_\text{mult}}{X^\text{sym}} = \text{colim}_n (X^{\otimes n})_{\Sigma_n} $$

But this doesn't just recover the space-- since the monadic actions of each of the $(X^{\otimes n})_{\Sigma_n}$ are compatible, the *monadic structure* on $\text{Prob}_{\text{Wass}, 1}$ itself is recovered. 


### algebras over a probability monad

An algebra for the Kantorovich monad is a complete metric space $X$ equipped with an **expectation map** $\mathbf{E}_{p}[x]:\text{Prob}_{\text{Wass}, 1}(X)\to X$. There is a suggestive interpretation of the higher-order "relations" $\text{Prob}_{\text{Wass}, 1}(\text{Prob}_{\text{Wass}, 1}(X))$. Consider the two maps

$$  p\in\text{Prob}_{\text{Wass}, 1}(X)\xleftarrow{\mu} \text{Prob}_{\text{Wass}, 1}(\text{Prob}_{\text{Wass}, 1}(X)) 
    \xrightarrow{\text{Prob}_{\text{Wass}, 1}(\textbf{E}_p[x])} \text{Prob}_{\text{Wass}, 1}(X) \ni q $$

We call an element $k\in\text{Prob}_{\text{Wass}, 1}(\text{Prob}_{\text{Wass}, 1}(X))$ that maps to $p, q$ a **partial evaluation** of $p, q$. Such an element is ultimately an element of the 1-truncated monadic bar construction $\tau_{\le 1}B\text{Prob}_{\text{Wass}, 1}(X)$. 

It turns out such elements can be defined as **families of conditional expectations**, and the tower law for conditional expectations come from the fact that the expectation map on distributions $p, q$ related by such a family give the same value in $X$. We are hence led to the question: what does the rest of the monadic bar construction give us probabilistically?

And what about cohomology?