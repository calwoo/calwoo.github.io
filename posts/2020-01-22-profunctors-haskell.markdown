---
title: Optics in Haskell
---

**Note: This blog post is not finished yet. Read on with caution.**

I'll admit that I am slightly skeptical of "category theory" in Haskell. I don't think I've been programming enough to see how many advanced abstractions can turn into usable structures, and profunctors + lenses seem to be one source of this mysticism that I'm struggling to understand. If anyone has a truly intuitive picture as to why this should all fit together, please reach out to me!

Anyway, in this post we'll look at how profunctors play out in Haskell, and how this ends up turning into the lens abstraction that functional programmers know and love. As in all things in category theory, we have to start with the Yoneda lemma.


### the yoneda lemma
Follow almost any categorical thought long enough and you'll find the Yoneda lemma. For a mathematician, the intuitive explanation of Yoneda is this: given any object of a category $C$, we can completely identify the object (up to unique isomorphism) by the entire set of functions mapping into that object. Another way I've heard it is that Yoneda is like a particle accelerator: If you study all the collisions between particles and your target long enough, you'll end up knowing the target particle.

Mathematically, if $C$ is a category, we can identify natural transformations between $\text{Hom}$-functors with homs themselves:

$$ \text{Yoneda: } \text{Nat}(\text{Hom}_C(a,-), \text{Hom}_C(b,-))\simeq \text{Hom}_C(b, a) $$

More generally, for any functor $f:C\to \text{Set}$, we get a Yoneda of the form (but same spirit):

$$ \text{Yoneda: } \text{Nat}(\text{Hom}_C(a,-), f) \simeq f(a) $$

How do we get about using this in Haskell? We unwind the definitions-- the set of natural transformations is a special version of an *end*, a categorical limit construction. To unwind this, you just need to remember that a natural transformation in the above Yoneda is a **collection** of maps $\text{Hom}_C(a,x)\to f(x)$ for all $x$, such that these maps are compatible by natural coherences. In Haskell, this can be written in the form

```haskell
{-# LANGUAGE RankNTypes #-}

yoneda :: Functor f => (forall x. (a -> x) -> f x) -> f a
yoneda gs = gs id
```

**NOTE:** This type signature doesn't enforce the coherences for an end! So this is slightly confusing to me. Unless I am wrong, in which case, please let me know!

Now for a bit of a mind-bender-- $C$ can be any category! So we take the category to the functor category from $C\to\text{Set}$, call it $[C,\text{Set}]$. Then we have

$$ \text{Nat}(\text{Hom}_{[C,\text{Set}]}(f,-), \text{Hom}_{[C,\text{Set}]}(g,-)) \simeq \text{Hom}_{[C,\text{Set}]}(g,f) $$

for functors $f, g$. 

Recall what $\text{Hom}_{[C,\text{Set}]}(-,-)$ is: it's natural transformations $\text{Nat}(-,-)$! Now letting $a, b$ be objects in $C$, and letting $f=\text{Hom}_C(a,-), g=\text{Hom}_C(b,-)$, we have

$$ \text{Nat}_{[C,\text{Set}]}(\text{Nat}(\text{Hom}_C(a,-), -), \text{Nat}(\text{Hom}_C(b,-), -)) \simeq
    \text{Nat}(\text{Hom}_C(b,-), \text{Hom}_C(a,-)) $$

Magic time: apply the original Yoneda lemma for $C$ and we get finally:

$$ \text{Nat}_{[C,\text{Set}]}([-](a), [-](b)) \simeq \text{Hom}_C(a, b) $$

where $[-](x): [C,\text{Set}]\to\text{Set}$ is the application functor $f\mapsto f(x)$. Unwinding this as an end, we remember that the natural transformations $[-](a) \to [-](b)$ are a **collection** of maps from $f(a)\to f(b)$ for **any functor**. In Haskell, this gives the amazing functional identity

```haskell
import Data.Functor.Identity

yoneda' :: (forall f. Functor f => f a -> f b) -> (a -> b)
yoneda' gs = \x -> runIdentity $ gs (Identity x)
```

This is where we get our first glimpse at the philosophy of lenses: If we want to understand an object, it suffices to probe it with as many different "views" as possible, and if all these views coherently agree, we can say we understand the object!


### profunctors and isos


