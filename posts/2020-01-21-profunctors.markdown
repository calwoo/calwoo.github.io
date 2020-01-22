---
title: Profunctors are generalized functors
---

**Note: This blog post is still a rough draft. Read on with caution.**

There's a thread of connections that I've been curiously following for a few weeks now, but I've not been able to grasp it fully. I've heard people say at various Haskell meetups that there are relations between lenses/prisms, profunctors, and traversals. I don't really know what lenses are, and traversals seem too different to me from profunctors.

As far as I can tell, `Traversable` is a type class that endows a data container/functor the ability to "walk along" itself while processing a computational effect (which comes from the fact that `Traversable` instances subclass `Applicative`s). How in the world does this have anything to do with a "pro"functor?

In this post, I'm going to look over what profunctors are in the mathematical sense, and in later posts I'll hopefully try and enlighten myself as to the connections to the functional programming perspective.

### into category theory
I warn that this will not be an introductory post on category theory. Those who are interested in definitions and introductory material should look at the [nLab](https://ncatlab.org/nlab/show/HomePage) and [Riehl's book](https://www.amazon.com/Category-Theory-Context-Aurora-Originals/dp/048680903X/ref=sr_1_1?keywords=categories+in+context&qid=1579664959&sr=8-1).

First lets talk about regular functions between topological spaces $f:X\to Y$. In topology it's sort of an imperative to try to geometricize any existing notion relating other topological notions, and this includes functions themselves (incidentally, it's like the functional programmer's pragma of trying to lift anything to a first-class notion). So we are seeking generalized ways to look at $f$. One obvious way is to identify the function with it's graph $\hat{f}\subset X\times Y$.

One way to generalize functions $f:X\to Y$ then is to treat a "generalized function" as a function over the product space $X\times Y$ (recursive, I know!). Since we don't want to be this recursive, we'll treat this in the sense of algebraic geometry, and say that a generalized function is a section of a vector bundle over the product $X\times Y$. In this language, the original functions we are talking about are sections of the trivial line bundle with support over the graph of these functions. (I won't be pedantic here, you can trace the definitions for yourselves if you're unsatisfied).

The upshot of that in algebraic geometry is that such vector bundles give rise to *bimodules*. Bimodules can be thought of as "variating families of maps"-- as a poignant example, imaging taking a section of a vector bundle over $X\times Y$ (say, $\rho: E \to X\times Y$) and vary a point $x$ in $X$. The preimages $\rho^{-1}({x}\times Y)$ form a varying family of functions, apt for the name "generalized function".

In my opinion, the single most important theorem about bimodules is the [Eilenberg-Watts theorem](https://ncatlab.org/nlab/show/Eilenberg-Watts+theorem). The version we care about describes an equivalence of categories between bimodules and colimit-preserving functors between module categories:

$$ {}_R \text{Mod}_{S} \stackrel{\simeq}{\to} \text{Func}_{coc}(\text{Mod}_R, \text{Mod}_S) $$

From this we start to understand what a profunctor is as a "generalized functor": it should be a "bimodule over enriched categories", or **equivalently** it should be a colimit-preserving functor between "categories containing all infinite constructions" ($\text{Mod}_R$ is a *complete and cocomplete* category).


### bimodules in categories
So lets unwind what these are. Since we are in the setting of enriched category theory, let $V$ be a closed monoidal category (if we're in the world of Haskell, $V$ would be cartesian closed, but in a linear type situation, we take $V$ to be more general). For a category $C$ enriched over $V$, a $C$-*module* can be defined as a functor $\rho: C\to V$. A $C$-$D$ *bimodule* is an enriched functor

$$ C^{op}\otimes D\to V $$

When $V$ is the category of sets (or types Hask), this is a profunctor.

If you're confused by this, keep a concrete example in your head: let $V$ be the category $\text{Vect}^{\text{fin}}$ of finite-dimensional vector spaces over a field $k$, and let $C$ be any $V$-enriched category with one object. Then a $C$-module in the above sense is exactly a $k$-algebra in the classical sense, and a $C$-$D$ bimodule here is a bimodule in the classical sense.

Another example of a profunctor is the hom functor $\text{Hom}_C(-,-): C^{op}\times C\to \text{Set}$. This example makes it clear that a profunctor is a "varying family of functors", where varying an object $c$ of $C$ gives rise to functors $\text{Hom}_C(c,-): C\to\text{Set}$. 

### profunctors are functors between presheaves
How do we get to our Eilenberg-Watts description of profunctors? Our favorite trick-- the Yoneda lemma. What does the Yoneda lemma say? For our purposes, it guarantees us an embedding of any (small) category $C$ into its category of presheaves $\text{PShv}(C) = \text{Func}(C^{op}, \text{Set})$. Hence for any old functor $F:C\to D$, we get a composition

$$ C \stackrel{F}{\longrightarrow} D \stackrel{Yoneda}{\longrightarrow} \text{PShv}(D) $$

which by adjunction gives rise to our first definition of a profunctor. Continuing from here, another guarantee we have here is that as $\text{PShv}(D)$ is cocomplete, by Kan extension we can lift our composition to a functor between the cocompletion of $C$ (which happens to be $\text{PShv(C)}$!) and $\text{PShv}(D)$ that preserves colimits:

$$ \text{PShv}(C) \to \text{PShv}(D) $$

This is the Eilenberg-Watts theorem! And also, this is a strikingly nice categorical description of a profunctor in terms of colimit properties.


### composition of profunctors
If we have functors, we'd like to compose them. If we have profunctors, we'd like to compose them too. How?

Let's fix an example. Let $C$ be a category, and consider the $\text{Hom}$ profunctor. How would we compute the composition $\text{Hom}_C\circ\text{Hom}_C$? Let $a, b$ be two objects of $C$, and consider the value of the composition $(\text{Hom}_C\circ\text{Hom}_C)(a, b)$, which is a set.

Naturally, we'd think of the elements in this set to correspond to compositions of original functions (afterall, that's what $\text{Hom}_C(-,-)$ represents):

$$ a \to ? \to b $$

But which object "$?$" does the map pass through? Just like Feynman's path integrals, we could let the function pass through anything! In essence, we can consider all the possible ways to get from $a \to b$ while passing through another object, which is represented by the coproduct

$$ \coprod_{d\in C} \text{Hom}_C(a,d)\times\text{Hom}_C(d,b) $$

This isn't the true composition however, because there are a lot of equivalent ways to get from $a\to b$ through different objects! In essence, we will try to identify as many "commutative diagrams" as possible to get a lossless compression of the above set. Mathematically, this is a **coend**, which is a kind of Kan extension.

In general, for profunctors $F$ and $G$, the composition is given by the coend:

$$ G \circ F := \int^{d \in D} F(d,-)\otimes G(-,d) $$

where here, the $\otimes$ denotes the monoidal structure of the enrichment category. As above, a coend should be thought of as all the ways to "create a varying family of functors by using the family of functors defined by $F$ and $G$".


### hochschild homology?
Finally, I'd like to look at another way to describe generalized functions. Another description of the graph $\hat{f}$ of a function $f:X\to Y$ is as a **correspondence** or **span**:

$$ X \leftarrow Z \to Y $$

If the category we're working in has (homotopy) pullbacks (a kind of colimit), we can compose such spans using them. In particular, if we're working in a derived/homotopical setting, given a span $X\leftarrow A\to X$, it's self-composition is given by the derived tensor product

$$ X \leftarrow \text{Spec}(A\otimes^{\mathbb{L}}_X A) \to X $$

The object $A\otimes^{\mathbb{L}}_X A$ is known as the **Hochschild homology** of the bimodule $A$ over $X$, and is a central object of study in noncommutative algebraic geometry and algebraic topology. It would be interesting to see how this object makes its way into computer science as these other tidbits of category theory already has.




































