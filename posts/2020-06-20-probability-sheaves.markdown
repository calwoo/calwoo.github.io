---
title: Probability sheaves
author: Calvin
---

**Note: This blog post is still a rough draft. Read on with caution.**

Since I'm trying to learn about information cohomology, I would hope there is a sheaf theory underlying everything. There are inklings of one, but I'm not sure how developed it is. In this post I want to write up some notes on probability sheaves and their connection to the monadic viewpoint of probability theory.


### probability sheaves

To talk about (pre)sheaves, we need a convenient category of spaces to do probability theory on. The category we will use is that of **Polish spaces**, that is, the category of topological spaces homeomorphic to separable complete metric spaces. Let $\text{Polish}=\text{Met}^{\text{cmpl, sep}}$ be the category of Polish spaces. Why is this nice? It's easy to turn a Polish space into a measurable one-- just take the $\sigma$-algebra of Borel sets. So implicitly when we talk about a Polish space $X$, we are talking about a Borel-measurable space with a Borel probability measure. *A technical detail*: maps between Polish spaces are measure-preserving, and we identify maps that differ on a measure-zero space. This is ultimately because we will impose the same restriction on random variables, because some important random variable constructions are only well-defined up to measure-zero equivalence.

Why should we consider sheaves when we talk about probability theory? A famous [blog post](https://terrytao.wordpress.com/2010/01/01/254a-notes-0-a-review-of-probability-theory/) of Tao pushes the viewpoint of modern probability theory that rejects mention of the underlying sample space. Indeed, this Grothendieck-inspired view defines probability theory as the study of "concepts that are preserved with respect to extension of the underlying sample space". Indeed, the slogan here is: probability theory only concerns itself with random properties, not with the underlying space they are organized on. This multi-scale viewpoint (extension) fits well with the categorical notion underlying presheaves (restriction).

Anyway, onwards to sheaves. Let $A$ be a Polish space (which we will consider the space of values). Let $\text{RV}(A):\text{Polish}^\text{op}\to\text{Set}$ be the presheaf of random variables with values in $A$, up to measure-zero equivalence:

$$ \text{RV}(A)(\Omega) = \{\text{random variables }X:\Omega\to A\} / \sim_0 $$

where $\sim_0$ is the relation of measure-zero equivalence. 

**Note**: This is not merely the Yoneda embedding! Here, a random variable $X:\Omega\to A$ may not be measure-preserving, but maps between Polish spaces in our category are. 

To form sheaves, we must endow $\text{Polish}$ with a Grothendieck topology. As all maps between Polish spaces are measure-preserving, random variables are essentially "invariant" to pullbacks **and** pushforwards. Indeed, this implies that we can take any map in $\text{Polish}$ to be a covering map. We express this by giving $\text{Polish}$ the **atomic topology**, in which all maps are coverings, and denote this Grothendieck site by $\text{Polish}_\text{atomic}$. As a corollary, we see that all presheaves are sheaves on the random variable topos defined here.

To check this is a valid Grothendieck site, the only property we need to verify is the existence of pullback coverings: for a pair of maps $\Omega_a\to\Omega\leftarrow\Omega_b$ between Polish spaces, we define the **independent pullback** to be the Polish space $\Omega_a\times_\Omega \Omega_b$ given by the standard set-theoretic pullback endowed with the unique probability measure characterized by the algebraic measures

$$ d(\Omega_a\times_\Omega\Omega_b)(x, y) = d\Omega_a(x)\otimes d\Omega_b(y) $$

for each $(x,y)\in\Omega_a\times_\Omega\Omega_b$. Put another way, we see that the probability measure $d(\Omega_a\times_\Omega\Omega_b)$ is a parameterized family of conditionally independent measures varying over $\Omega$.


### cohomology

