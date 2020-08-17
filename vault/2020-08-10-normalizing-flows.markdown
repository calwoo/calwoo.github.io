---
title: Normalizing flows
author: Calvin
---

**Note: This blog post is a collection of notes. There might be discontinuities of narrative flow.**

### introduction

In a nutshell, **normalizing flows** is a technique for building complex probability distributions from simpler ones. Unlike other methods that deal with controlling the dependency structures between random variables (e.g. probabilistic graphical models) to construct interesting probability distributions, normalizing flows seek out bijective diffeomorphisms of the underlying space to transform distributions the other distributions directly. While we lose properties of interpretability of the resulting generative model, we gain a considerable amount of expressiveness, which is valuable when the desired distribution is something that is difficult to model in the first place.

### jacobians and change

The mathematical framework behind normalizing flows is fairly simple. Let $z\sim p(z)$ be a sample from a *base distribution*, and let $T:X_z\to\mathbf{R}^d$ be a diffeomorphism (bijective transformation) from the underlying space of $z$ to some Euclidean space. A diffeomorphism locally distorts the volume of space via the **change of variables** formula

$$ p(z)\cdot |\text{det }{J^{-1}_T(z)}| = p_x(x) $$

where here, $x=T(z)$ is the transformed random variable, and $J_T(z)$ is the Jacobian of the diffeomorphism $T$. To see this is the case (since I always forget the sign on the Jacobian), we note that for a given infinitesimal volume $\text{vol}(A)$ centered around $z$, the volume of $T(A)$ is 

$$ \text{vol}(T(A))\simeq\text{vol}(A)\cdot\det{J_T(z)} $$

Since $T$ is assumed to preserve probability measure, this gives us the assumed signature of the formula above.

So now that you have one transformation, why not chain up a bunch of them? Such composable chains of transformations are called **flows**. Apparently, the choice of word *normalizing* is only to invoke the property of bijectivity (and probability measure preservation) that such diffeomorphisms possess.

### likelihood ascent

**Question:** Given a density we want to estimate $p(x)$, how can we construct a normalizing flow $T$ from a simple distribution $p(z)$ to this one? 

To fix notations, let $p^*(x)$ be the target distribution we want to estimate. We want to approximate $p^*(x)$ as closely as possible using a parameterized distribution $p_x(x;\theta)$, where $\theta$ are the parameters of the diffeomorphism $T=T_\theta$ from a base distribution $p_z(z;\psi)$. Here we have two sets of moving knobs-- we can functionally alter our transformation $T_\theta(z)$ via $\theta$, and we can adjust our base via $\psi$. 

As an objective, we want to find the parameters $\theta, \psi$ that minimizes the KL-divergence

$$ \text{D}_\text{KL}(p^*(x)||p_x(x;\theta)) \propto -\mathbf{E}_{x\sim p^*(x)}\left[\log{p_x(x;\theta})\right] $$

By change of variable this expands to

$$ \propto -\mathbf{E}_{x\sim p^*(x)}\left[\log{p_u(T^{-1}(x;\theta);\psi)} + \log{\det{J_{T^{-1}}(x;\theta)}}\right] $$

This can be given an Monte Carlo approximation by sampling from the density $p*(x)$, and minimizing the above via gradient descent gives a way to construct the normalizing flows. The above procedure is known as the **forward KL-divergence method**.

### reverse KL and duality

In the above we have a target distribution $p_x^*(x)=p^*(x)$ that we are trying to approximate with a parameterized family $p_x(x;\theta)$. Alternatively, as $T_\theta$ is a diffeomorphism, we see that the inverse flow on $p_x^*(x)$ induces a distribution $p_z^*(z;\theta)$. Similar to the method above, we can this time try to minimize the KL-divergence between the base-level distributions $p_z^*(z;\theta)$ and $p_z(z;\psi)$

$$ \text{D}_\text{KL}(p_z^*(z;\theta)||p_z(z;\psi)) = \mathbf{E}_{z\sim p_z^*(z;\theta)}\left[\log{p_z^*(z;\theta)}-\log{p_z(z;\psi)}\right] $$

This is the **reverse KL-divergence method**. Are these two related somehow? I imagine there is a category-theoretic abstraction that allows me to express these statements as decategorified $\text{Hom}$-spaces of some sort. Then it would be clear that there is a **duality** between the KL-divergences given by:

$$ \text{D}_\text{KL}(p^*(x)||p_x(x;\theta)) = \text{D}_\text{KL}(p_z^*(z;\theta)||p_z(z;\psi)) $$

Indeed, writing the pushforward distributions by using $T_{\theta,*}$, we have

$$ \text{D}_\text{KL}(p^*(x)||T_{\theta,*}p_z(z;\psi)) = \text{D}_\text{KL}(T^{-1}_{\theta,*}p^*(x)||p_z(z;\psi)) $$

which can be considered a categorical adjunction, albeit a weak one.

### autoregressive flows



