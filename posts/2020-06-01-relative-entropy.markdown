---
title: Kullback-Leibler divergence, categorically
author: Calvin
---

**Note: This blog post is still a rough draft. Read on with caution.**

In my rabbit-hole readings on entropy and its algebro-topological connections, I stumbled upon an interesting paper by Baez-Fritz ["A Bayesian characterization of relative entropy"](https://arxiv.org/abs/1402.3067). What's particularly funny is that the authors wanted to name this a "categorical characterization", but in fear of scaring off the broader mathematical community from reading this otherwise beautiful paper, they opted to call it a "Bayesian" one. Funny, because I don't yet understand the Bayesian-ness of it.

A plus of this paper is that it gives an introduction to probability theory from a categorical viewpoint, which I want to review first before diving into Kullback-Leibler divergence.


### finite sets

Let $\mathbf{R}^{\ge 0}=[0,\infty)$ be the commutative rig (ring without negatives) of nonzero reals. Let $\text{Mod}^{\text{f.g, free}}_{[0,\infty)}$ be the category of finitely-generated free $\mathbf{R}^{\ge 0}$-modules. As usual, a finitely-generated free $\mathbf{R}^{\ge 0}$-module is isomorphic to a product $[0, \infty)^n$ for some $n\ge 0$, and morphisms are given by matrices over $\mathbf{R}^{\ge 0}$. 

We give the category $\text{Mod}^{\text{f.g, free}}_{[0,\infty)}$ a monoidal structure via the normal $\otimes$-product of free modules. Also since we're dealing with free modules over a commutative rig, we can take matrix transposes of the corresponding morphisms and give $\text{Mod}^{\text{f.g, free}}_{[0,\infty)}$ the structure of a $\dagger$-category (recall that a $\dagger$-category is one with a contravariant endofunctor that sends any morphism $f:a\to b$ to a "transpose" $f^\dagger:b\to a$).

In all, this makes $\text{Mod}^{\text{f.g, free}}_{[0,\infty)}$ into a symmetric monoidal $\dagger$-category. This is the base category where the remaining constructions will come into play. First, we need to get back the category of finite sets $\text{Set}^{\text{fin}}$ from within this category.

But isn't this easy? Each finitely-generated free module is $[0, \infty)^S$ for some finite set $S$. But the trouble is that it isn't obvious how to get that $S$ out-- a priori if given a finitely-generated free module $A$, how can we detect the set $S$? 

The **key** is the familiar(!) fact that **sets have diagonal maps** $S \to S\times S$. This gives rise to a coproduct $\Delta:[0,\infty)^S\to [0,\infty)^S\otimes [0,\infty)^S$ that turns $[0,\infty)^S$ into a **special commutative $\dagger$-Frobenius algebra**. Then the set of grouplike elements $\Delta(x)=x\otimes x$ gives us our set $S$! The upshot is that we get an equivalence of categories 

$$ \text{Set}^{\text{fin}}\simeq\text{ComFrob}^\dagger_{\mathbf{R}^{\ge 0}, \Delta}\hookrightarrow \text{Mod}^{\text{f.g, free}}_{[0,\infty)} $$

We should be careful here. The $\Delta$ in $\text{ComFrob}^\dagger_{\mathbf{R}^{\ge 0}, \Delta}$ denotes that the maps are only morphisms of the underlying coalgebra. If we allow all Frobenius homomorphisms (preserving all structure), the axioms of the special algebra (multiplication-comultiplication interaction) forces the morphisms to be an isomorphism on the finite sets.


### probability, categorically

From above, a map between special commutative $\dagger$-Frobenius algebras that preserves the comultiplication and counit are given by maps of the equivalent finite sets. But note that by tracing the definitions, the counit map $\epsilon: A\to\mathbf{R}^{\ge 0}$ is given by integration with respect to counting measure. So if we relax the restriction that morphisms between algebras preserve the comultiplication map, then on the underlying sets, maps can send grouplike elements to *distributions on the finite set* in such a way that preserves *measure*. We hence call such morphisms **stochastic maps** and define the category of such sets and maps $\text{Stoch}^\text{fin}\simeq \text{ComFrob}^\dagger_{\mathbf{R}^{\ge 0}, \epsilon}$. 

If the $epsilon$ map is an analogue of Lesbegue integration, then a section of this map is a **measure** on the finite set $S$. Hence we define a **finite measure space** as a special commutative $\dagger$-Frobenius algebra $A$ with a map

$$ \mu: \mathbf{R}^{\ge 0} \to A $$

Then such a finite measure space is a finite *probability* space if integrating the measure is 1: $\epsilon\circ\mu = 1$. We denote the category of such finite probability spaces as $\text{Prob}^\text{fin}$, where morphisms are given by the obvious commutative triangles. 

Okay, great! Now what? To be fair, there isn't much you can do with general probability distributions on finite sets. You can do some cute things, like considering the monoidal/composition structures on $\text{Prob}^\text{fin}$ (with parallels to the Giry monad), but it isn't enough to do any kind of *statistics*. Another cute thing is something I thought of while trying to understand this material: given a special commutative $\dagger$-Frobenius algebra $A$, we can fix an isomorphism $A\simeq [0,\infty)^S$. Then for any probability distribution $p(s): [0,\infty)\to [0,\infty)^S$, and family of spaces $p(t|s): [0,\infty)\to [0,\infty)^T$, one can form the composition $p(s,t): [0,\infty)\to [0,\infty)^{S\times T}$. But Bayes theorem then just comes down the fact that $S\times T\simeq T\times S$ and that we can repartition the composition (look at the underlying matrix) to reflect that.

How do we encode statistics into the category $\text{Prob}^\text{fin}$? The slogan is that "statistics is the inverse problem of probability". That is, given a stochastic map $f: A\to B$, statistics is composed of hypotheses: if $f$ describes a sampling procedure, and we observe $y\in Y$, then our hypothesis describes the distribution on $X$ after observing $y$. That is, a **hypothesis** is a stochastic map $s:B\to A$. But we also need to be consistent-- we shouldn't say anything about points of $X$ that don't actually give us anything about our observed result, that is, $s$ is actually a *section* of $f$, $f\circ s = \text{id}_Y$. 

We can wrap such finite probability spaces into a category $\text{Stat}^\text{fin}$ where morphisms are maps of finite probability spaces equipped with a stochastic hypothesis section, as above. 


### relative entropy

Most of the time in $\text{Stat}^\text{fin}$, $s$ is off the mark. But occasionally, our hypothesis about the distribution of $A$ is quite on the nose, and then we would say our hypothesis is **optimal**. Diagrammatically, we would say that $s$ makes the triangle of distributions commute. We can call the category of such maps $\text{FP}$, though I don't really know why that is the case.

Now we get to our Bayesian characterization. Given an object $(f:(A,p)\to (B,q), s: B\to A))$ in $\text{Stat}^\text{fin}$, we know that $s$ furnishes us a probability distribution over $A$ for each measurement value $b\in B$. Hence there is a **prior** we impose on $p$ given on each $a\in A$ by the sum

$$ p^\text{prior}_a = \sum_{b\in B} q_b\cdot s(b)_a $$

This is our "guess" about what the probability distribution over $A$ is given our hypothesis $s$. The true **posterior** would thus be given by the distribution $p$ itself. The discrete $KL$-divergence of these two distributions hence furnish us a functor

$$ \text{KL}: \text{Stat}^\text{fin} \to \mathbf{R}^{\ge 0} $$

What special properties does this functor have? Suppose we restrict to the subcategory $\text{FP}\hookrightarrow\text{Stat}^\text{fin}$. Then by optimality, the prior $p^\text{prior}$ is $p$! Hence the functor $\text{KL}$ *vanishes* on the category $\text{FP}$. The result of Baez-Fritz is that along with some convexity/functoriality conditions, this entirely characterizes (up to a constant) Kullback-Leibler divergence. 


