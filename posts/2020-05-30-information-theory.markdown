---
title: Entropy in information theory
author: Calvin
---

A lot has happened since my last post, especially the coronavirus pandemic sweeping the world. I am also now a senior data scientist/machine learning engineer at Prognos Health, a healthcare-tech startup focused on creating an analytics platform for analyzing clinical lab data at scale.

This post won't have anything to do with that, but I might talk about that in the future. I stumbled upon some really cool stuff linking homology to information theory, and I wanted to spend some blog posts putting down my notes on the papers I'm reading regarding this. A lot of this material will come form the paper of Baudot-Bennequin "The homological nature of entropy". 


### entropy

Let $p$ be a probability measure on a finite set $X$. Shannon defined in 1948 a definition of the entropy for $p$, a measure of the information content:

$$ H(p) = - \sum_{i\in X} p_i\ln{p_i} $$

A few notes: (1) if $p$ is localized on a single element $i$, say $p_i = 1$ then $H(p)=0$, and (2) if $p$ is uniformly-distributed over $X$, then entropy is maximized.

Where does this come from? While there are many derivations coming from physics or whatnot, there is a simple axiomatic characterization of entropy. Let $\text{Prob}^\text{fin}$ be the category of finite sets equipped with discrete probability measures, where a morphism $f:(X, p)\to (Y, q)$ is a measure-preserving function between finite probability spaces. We make the *heuristic* declaration that whatever measure of information we want to assign to the measure $p$ that it decreases across the map $f$-- that is, information is **lost across a channel**.

Hence, we assume there is a function $F:\text{Hom}_{\text{Prob}^\text{fin}}(p, q)\to \mathbf{R}^{\ge 0}$ called the **information loss**, which satisfies the following:

(1) **functoriality** the amount of imformation lost in a composition of channels is the sum of the loss in each:

$$ F(p\to q\to r) = F(p \to q) + F(q \to r) $$

(2) **convexity** effectively this says that for a discrete probability distribution $\xi$ over a finite collection of maps $\{f_i:p\to q\}_{i=1..m}$, we have that $F$ preserves expectations, i.e. $F(\mathbf{E}_\xi(\{f_i:p\to q\}_{i=1..m}) = \mathbf{E}_\xi(F(f_i)))$. In the case where $m=2$, this is the convexity relationship where the choice of channel is given by a Bernoulli coin flip.

(3) **continuity** $F$ is continuous in $f$.

Let us also impose a normalization condition: 

(4) **identity** the loss of the identity channel is 0, $F(\text{id}_X) = 0$.

We claim that this uniquely characterizes Shannon entropy. Let $\zeta_p$ be the unique morphism from any $p$ to the one-point set $*$. From our heuristic we see that the total information loss of this map should be the total information stored in $p$. So define the **entropy** of $p$ to be $F(\zeta_p)$. 

Note that we have for any $f:p \to q$ that $\zeta_p = \zeta_q \circ f$. Hence by functoriality we have 

$$ F(\zeta_p) - F(\zeta_q) = F(f) $$

So we are reduced to computing form of the entropies $F(\zeta_p)$. Let $X_1,..., X_n$ be a collection of finite sets and $q_1,...,q_n$ their probability measures. Let $p$ be a probability measure on the $q$'s and form their expectation measure $z = \mathbf{E}_p[q_k]$ on the disjoint union of the $X_i$'s. Form the map

$$ f = \mathbf{E}_p[\zeta_{q_k}]: \mathbf{E}_p[q_k] \to (\{1,...,n\}, p) $$

From the convexity property we see that $F(f) = F(\mathbf{E}_p[\zeta_{q_k}]) = \mathbf{E}_p[F(\zeta_{q_k})]$, and from the functoriality above we see that $F(f) = F(\zeta_{\mathbf{E}_p[q_k]}) - F(\zeta_p)$. Hence

$$ F(\zeta_{\mathbf{E}_p[q_k]}) - F(\zeta_p) = \mathbf{E}_p[F(\zeta_{q_k})] $$

But this is the [strong additivity condition](https://www.sciencedirect.com/science/article/pii/S0076539208627368) of Shannon entropy, which characterizes entropy up to a constant.


### operads and additivity

Where does the strong additivity condition come from? It is actually fairly fundamental, enough that I would market it as a fundamental theorem of the field (though that term is severely overloaded, and often poorly used, this one included). This was made extremely clear to me after reading the [nLab](https://ncatlab.org/johnbaez/show/Entropy+as+a+functor) post on operadic entropy.

Let $q_1,..., q_n$ be probability distributions on $X_1,..., X_n$ finite sets, and let $p$ be a probability distribution over the $q$'s, as above. We will write the expectation measure $\mathbf{E}_p[q_k]$ with the symbology $p \circ (q_1,..., q_n)$. This is then a probability distribution on the disjoint union $\amalg_k {X_k}$. What is the Shannon entropy of this distribution?

By computation, we get

$$ H(p \circ (q_1,..., q_n)) = -\sum_{ij} p_i q_{ij}\log{p_i q_{ij}} = -\sum_{ij} p_i q_{ij}\log{p_i} -\sum_{ij} p_i q_{ij}\log{q_{ij}} $$

But one term is $H(p)$ and the other is $\mathbf{E}_p[H(q)]$. Following the lead of Leinster, we symbologize the expectation via the suggestive term $p(H(q_1),..., H(q_n))$. Then we get the "algebraic" equation

$$ H(p \circ (q_1,..., q_n)) = H(p) + p(H(q_1),..., H(q_n)) $$

I say "algebraic" because this looks like $H$ is a homomorphism of sort, except for the *extra term* $H(p)$. Leinster in his [note](https://www.maths.ed.ac.uk/~tl/operadic_entropy.pdf) outlines an incredibly slick way to get this out as a part of some operadic technology.

Let $\mathcal{O}$ be a (symmetric) $\Sigma$-operad and consider $\mathcal{O}$-algebras in the category $\text{Cat}$ of (small) categories (call them categorical $\mathcal{O}$-algebras). Recall that an $\mathcal{O}$-algebra $A$ in any category is a collection of maps $\mathcal{O}(k)\otimes A^{\otimes k}\to A$ satisfying the usual commutative, associative, and identity laws. We define a **lax map** between categorical $\mathcal{O}$-algebras $A, B$ to be a functor $A\to B$ with natural transformations given by the commutative diagram

$$
\begin{matrix}
\mathcal{O}(k) \otimes A^{\otimes k} &\to        &\mathcal{O}(k) \otimes B^{\otimes k} \\
\downarrow     &\Leftarrow &\downarrow     \\
A              &\to        &B
\end{matrix}
$$

satisfying obvious axioms. Let $1$ be the terminal category (with its unique $\mathcal{O}$-algebra structure). A **lax point** of the $\mathcal{O}$-algebra $A$ is a lax map $1\to A$. 

Unwinding this definition, we see that a lax point consists of an object $a\in A$ and a collection of maps $h_\theta: \theta(a,...,a)\to a$ for every $\theta\in\mathcal{O}(k)$ for each $k\ge 0$. Composition of operads gives us an equality between the composition

$$ \theta(\theta_1(a,...,a),...,\theta_m(a,...,a)) \xrightarrow{\theta(h_{\theta_1},..., h_{\theta_m})} \theta(a,...,a)
    \xrightarrow{h_\theta} a $$

and the map

$$ (\theta\circ (\theta_1, ..., \theta_m))(a,..., a) \xrightarrow{h_{\theta\circ (\theta_1, ..., \theta_m)}} a $$

where the equivalence of the domains is given by [operadic composition](https://en.wikipedia.org/wiki/Operad). This gives us the tantalizing functional equation

$$ h_{\theta\circ (\theta_1, ..., \theta_m)} = h_{\theta}\circ \theta(h_{\theta_1},..., h_{\theta_m}) $$

This looks like strong additivity! To get us as close as possible, consider the operad $\mathcal{O}$ to be given by the simplex/probability operad $\Delta$. This is the operad of finite discrete probability distributions on finite sets. Let $A$ be the additive monoid of reals $\mathbf{R}$. Then for any $p\in\Delta(k)$, the operadic action is given by $p(a_1,..., a_k) = \sum_i p_i a_i$.

Then for any lax point $\gamma:\Delta \to A$ we get the identity

$$ \gamma(p\circ (q_1, ..., q_m)) = \gamma(p) + p(\gamma(q_1),..., \gamma(q_m)) $$

as above! Finally, we note that $H(p) = -\mathbb{E}_p[\log{p}]$ gives a lax point $H:\Delta\to\mathbf{R}$.


### derivations and partitions

Let $\beta > 0$ be a temperature parameter, and consider the **partition function** 

$$ Z(p; \beta) = \sum_{i \in X} p_i^\beta $$

If we let $p_i = e^{-\beta H_i}$ for $H_i$ the suggestively defined Hamiltonian, we get the usual physicists definition. But we don't really care for that. What matters more to us is that under the definition of probability density composition given by the probability operad, the partition function $Z(-) = Z(-;\beta)$ is an $\Delta$-algebra homomorphism $1\to\mathbf{R}$:

$$ Z(p\circ (q_1, ..., q_m)) = p(Z(q_1),..., Z(q_m)) $$

What does entropy have to do with the partition function?

$$ H(p) = -\left. \frac{d}{d\beta}Z(p; \beta)\right|_{\beta=1} $$

With some calculation, one can show that the derivative of the $\Delta$-algebra homomorphism equation above gives us strong additivity, which is a remarkable fact. 


### conclusion

Next time I'll talk about some things I'm reading about information cohomology. One nice corollary of the machinery there is that the Shannon entropy is given as a 1-cocycle in the information complex, which might tie in nicely with this material above (I'm not sure). My mathematical heart would love there to be some kind of interplay between some kind of de Rham cohomology and the information cohomology, but I'll have to read on to find out more.