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
As explained in the [previous post](/posts/2020-01-21-profunctors.html), a profunctor from $C\to D$ is a functor $p: D^{op}\times C \to\text{Set}$. This mix of covariance and contravariance can be described in the `Profunctor` type class

```haskell
class Profunctor p where
    dimap :: (s -> a) -> (b -> t) -> (p a b -> p s t)
```

This class describes the double-lift, since given objects $(a, b)$ and $(s, t)$ of $D^{op}\times C$, a "map" in that category is given by a pair of maps $s\to a$ and $b\to t$. Applying the Yoneda lemma above for profunctors means running the Yoneda twice, so that we can read off as

```haskell
yonedaPro :: (forall p. Profunctor p => p a b -> p s t) -> Iso s t a b
```

where

```haskell
data Iso s t a b = Iso (s -> a) (b -> t)
```

is just a product type for pairs of functions.

**NOTE:** I don't know why they call this `Iso`, since it doesn't really have much to do with isomorphisms. In fact, they are usually NOT isomorphisms. I prefer to call these things by what they really are in mathematics: twisted arrows `TwArr`. But alas, the crowd has spoken.


### profunctor composition
If profunctors are generalized functors, then they should be able to compose. Recall that the composition of two functors is again a functor

```haskell
newtype Compose f g a = Compose { getCompose :: f (g a) }

instance (Functor f, Functor g) => Functor (Compose f g) where
    fmap :: (a -> b) -> Compose f g a -> Compose f g b
    fmap f (Compose x) = Compose $ fmap (fmap f) x
```

But given two profunctors $D^{op}\times C\to\text{Set}, E^{op}\times D\to\text{Set}$, how do we compose them? Lets recall the definition of the composition as a Kan extension:

$$ G \circ F := \int^{d \in D} F(d,-)\otimes G(-,d) $$

In traditional Haskell fashion, we won't enforce the coend gluing laws in the type, but in spirit: unwinding this definition we see that we get a valid constructor of elements of the composition-- we take a "path in F" and a "path in G".

```haskell
data PCompose p q x y where
    ProCompose :: (p d y, q x d) -> PCompose p q x y
```

As a sanity check, lets see what this looks like for functors "completed" to profunctors. What do I mean by this? Recall from last post there was a way to promote an ordinary functor $f:C\to D$ to a profunctor by post-composing with the Yoneda embedding:

$$ f^{\text{pro}}: C \stackrel{F}{\longrightarrow} D \stackrel{Yoneda}{\longrightarrow} \text{PShv}(D) $$

In terms of objects, it sends $c\mapsto\text{Hom}_D(-,f(c))$. This inspires the following promotion operator in Haskell:

```haskell
newtype Promotion f d c = Promotion { runPromoted :: d -> f c }

instance Functor f => Profunctor (Promotion f) where
    dimap :: (s -> a) -> (b -> t) -> (Promotion f a b -> Promotion f s t)
    dimap g h pf = Promotion $ 
        fmap h . runPromoted pf . g
```

Now we wish to study the composition of promoted functors. In one direction we have

```haskell
PCompose (Promotion f) (Promotion g) x y
    == (Promotion f d y, Promotion g x d)
    == (d -> f y, x -> g d)
```

and in the other we have

```haskell
Promotion (Compose g f) x y
    == x -> (Compose g f) y
    == x -> g (f y)
```

But there is a clear isomorphism between `(d -> f y, x -> g d)`$\simeq$`x -> g (f y)` given by

```haskell
compIso :: (Functor f, Functor g) => (d -> f y, x -> g d) -> (x -> g (f y))
compIso (h, k) = fmap h . k

compIsoInv :: (Functor f, Functor g) => (x -> g (f y)) -> (d -> f y, x -> g d)
compIsoInv k = (id, k)
```

And this should make sense! Profunctor composition should be a version of regular composition. 


### profunctor optics, finally
So far, we've been wandering around profunctors and Yoneda, and haven't talked about optics. What's the upshot of using all of this? 

Remember, profunctors are generalizations of functions. In fact, the canonical example of a profunctor is given by $\text{Hom}_C(-,-)$

```haskell
instance Profunctor (->) where
    dimap :: (s -> a) -> (b -> t) -> ((a -> b) -> (s -> t))
    dimap h k = \f -> k . f . h
```

In this way it's clear that profunctors perform some kind of interplay between covariance and contravariant functoriality, which is in essence what an optic is doing. An *optic* is an abstraction for understanding an object and a "focus". Here a canonical example is that of a *lens*, which gives an interface for setters and getters in objects-with-attributes (I won't say classes). 

A trivial example of an optic is one in which the there is no interaction between the focus and the object-- given an object, we can view the focus, but given the (might be polymorphically-changed) focus, we might get a new object that had nothing to do with the original whole! In such generality, all we can say about this optic is that it is composed to two arrows of "opposing direction". This sounds familiar: it's just a twisted arrow! (or as we called it in Haskell, an `Iso`). And above, we've already identified via Yoneda what twisted arrows are in terms of profunctors-- it's just an end

$$ \text{Iso}_{s,t,a,b} = \int_{p\in\text{Prof}} \text{Hom}_{\text{Set}}(p(s,a), p(b,t)) $$

where $\text{Prof}$ is the category of profunctors (we defined composition above). 

In this form, we could experiment by giving our profunctors more structure, dependent on the monoidal properties of the underlying enrichment (which is $\text{Set}$ in this case). We know that $\text{Set}$ is a symmetric monoidal category under the cartesian product $\times$, so restrict ourselves to profunctors that respect the monoidal structure:

```haskell
class Profunctor p => Cartesian p where
    first  :: p a b -> p (a, c) (b, c)
    second :: p a b -> p (c, a) (c, b)
```

What optic does this give us? Computing the end as above, we have

$$ \int_{p\in\text{Prof}^{\text{Cart}}} \text{Hom}_{\text{Set}}(p(s,a), p(b,t)) \simeq 
   \int^{c\in\text{Set}} \text{Hom}_{\text{Set}}(\text{mul}_c(s,a), \text{mul}_c(b,t))
$$

where $\text{mul}_c:C^{op}\times C\to\text{Set}$ is the profunctor given by $(a,b)\mapsto\text{Hom}_{\text{Set}}(a,b\times c)$. By the usual computation, this is given by a pair of functions

$$ \int^{c\in\text{Set}} \text{Hom}_{\text{Set}}(s, a\times c)\times\text{Hom}_{\text{Set}}(b\times c, t) $$

which by the Yoneda lemma gives $\text{Hom}_{\text{Set}}(s,a)\times\text{Hom}_{\text{Set}}(s\times b, t)$. This is precisely a *lens*!

```haskell
data Lens s t a b = Lens { view   :: s -> a
                         , update :: (b, s) -> t 
                         }
```

Hence we get our first nontrivial optics--

$$ \text{Lens}_{s,t,a,b} = \int_{p\in\text{Prof}^{\text{Cart}}} \text{Hom}_{\text{Set}}(p(s,a), p(b,t)) $$

```haskell
type LensP s t a b = forall p. Cartesian p => p a b -> p s t
```


### diversion: the isomorphism in code
I think it's a great exercise to show this equivalence between the lens definitions explicitly. The derivation in Haskell is as follows.

```haskell
comparison :: Lens s t a b -> LensP s t a b
comparison (Lens v u) = dimap dup u . second . lmap v
    where
        lmap :: (a' -> a) -> p a b -> p a' b
        lmap f = dimap f id

        dup  :: a -> (a, a)
        dup x = (x, x)

-- constant functor
type Constant b a = Constant { runConstant :: b }
    deriving Functor

comparisonInv :: LensP s t a b -> Lens s t a b
comparisonInv pfn = Lens view' update'
    where
        view' :: s -> a
        view' s = runConstant . runPromotion (pfn (Promotion Constant))

        update' :: (b, s) -> t
        update' (b, s) = pfn (const b) s
```


### intuitions and final remarks
This gives us a great tool-box for building optics-- enrich our profunctors with special properties and we get different optics. If we let our optics preserve the cocartesian structure of $\text{Set}$, we get *prisms*. 

$$ \text{Prisms}_{s,t,a,b} = \int_{p\in\text{Prof}^{\text{coCart}}} \text{Hom}_{\text{Set}}(p(s,a), p(b,t)) $$

If we let our profunctors be derived from polynomial functors, we get traversals. This is awesome.

I close with my intuition as to why profunctors are needed for this. It comes down to the co-Yoneda lemma, which roughly states that all profunctors are colimits of the promoted functors. In this way, we can just rename profunctors as **colimit-completions of functors**. Under this perspective, as functors are ways to peer into the structure of objects, the completion allows us to "break parts off" of objects-- the remainder part will be the quotient, which is a kind of colimit! The colimit-completion property assures that such an object exists in our set. It would be nice to expand on this intuition a little more in the future, but I'll leave that for a later post.