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

A small aside before moving onto the Kantorovich monad. 