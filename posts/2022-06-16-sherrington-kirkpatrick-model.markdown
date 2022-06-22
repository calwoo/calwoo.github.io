---
title: Sherrington-Kirkpatrick model and the replica method
author: Calvin
---

**Note: This blog post is still a rough draft. Read on with caution.**

I've recently been in a craze of reading papers in machine learning that uses techniques from statistical mechanics. One of the motivating goals for these approaches is that "classical" theories of generalization in statistical learning, like those based on Vapnik-Chervonenkis dimension arguments, don't adequetely capture what is going on in the deep learning space. Indeed, one of the fundamental issues with these capacity measures is that they capture *worst-case* performance phenomenology, which is sufficiently successful for many classical machine learning methods because there exist good bounds on generalization gaps.

However, a shift in thinking was triggered after Zhang et al's [paper](https://arxiv.org/abs/1611.03530) "Understanding deep learning requires rethinking generalization". In this paper, they show that for deep learning the generalization gap can be arbitrarily large by fitting large neural networks on unstructured random noise. Indeed, the generalization performance of a neural network cannot be attributed to the existence of a bounded generalization gap, as it can have arbitrarily large capacity to memorize a training set. But (large) neural networks **do** generalize, so something outside of these classical capacity measures have to have some say in this.

The statistical mechanical approach is to understand the *typical*-case behavior directly. Much of this involves building small toy models amenable to exact analysis and then studying their statistical properties. I hope to have some blog posts in the future that explains some of these toy models that are relevant to machine learning.


### spin glasses

In this post, I will be going through the solution of a **spin glass** model, the Sherrington-Kirkpatrick model. In physics, a spin glass is a magnetic state of matter characterized by frustrated interactions. Individual spins that make up the magnetic material are in a state of disorder-- both in their individual states and their couplings with other adjacent states. And unlike the energy surfaces that characterize ferromagnetic magnetic materials (such as those modeled by the Ising model), spin glass energy landscapes can have lots of metastable states and saddles. This reminds one of the loss surfaces of complex multilayer neural networks!

Indeed, there is a long literature connecting neural network models with spin glasses. Studying small toy models for spin glasses can give insight into the structure of these loss surfaces (indeed, this has been done in some respect, such as in [this paper](https://arxiv.org/abs/1412.0233) of Choromanska et al).


### sherrington-kirkpatrick

Models in statistical mechanics starts with its Hamiltonian, or energy function. For the Sherrington-Kirkpatrick model, we are working with an infinite-range Hamiltonian over $N$ spins $s_i$ given by

$$ H=-\sum_{i,j} J_{ij}s_i s_j $$

Here, $J_{ij}$ is the coupling interaction between the spins $s_i$ and $s_j$. Here, we take the $s_i$ to be **Ising spins**, that is $s_i\in\{-1, 1\}$. Since we are working with a spin glass, we assume that the coupling strengths $J_{ij}$ undergo *quenched disorder*, which is a fancy way of saying that they are also random, but their fluctuations occur at a time scale far larger than the time scale that the spins themselves undergo thermodynamic fluctuation. In this model, we take

$$ J_{ij}\sim p(J_{ij})=\frac{1}{J}\sqrt{\frac{N}{2\pi}}\exp{\left[-NJ^2_{ij}/2J^2\right]} $$

that is, the couplings are sampled from a normal distribution with zero mean and variance $J^2/N$.

Given a Hamiltonian, we build the partition function (given a sample realization of the interactions)

$$ Z_J = \sum_{\{s_i\}}\exp(-\beta H[s]) $$

and use it to compute the free energy as its logarithm. Since we are looking to understand the typical-case behavior of the spin glass, we will look at the expectation of the free energy over the quenched disorder $\langle\log Z_J\rangle_J$. The reason why this expectation captures the typical behavior of the spin glass is because the free energy is a *self-averaging* quantity. Mathematically, this translate to a high-dimensional concentration of measure, in which the most probable states of the distribution and the expectation coincide.

However, computing this expectation is hard! If the logarithm was outside of the expectation

$$ \log \langle Z_J\rangle_J $$

we'd have a much easier time. This is the **annealed disorder** computation, and is usually a high-temperature approximation to the true quenched computation we want to do.


### replica calculation

The standard tool to get around this is called the **replica trick**. It is based on the identity (given by Taylor expansion):

$$ \langle\log Z_J\rangle_J = \lim_{n\to 0}\frac{1}{n}\log\langle Z^n_J\rangle_J $$

This reduces the hard calculation of computing the quenched average of the free energy into the limit over annealed averages using $n$ *replicas* of the thermodynamic system involved. Note that this method is very nonrigorous-- what does it even mean physically to work with a fraction of a replica? However, the statistical physics literature has used this technique to great success, producing exact results that align with numerical simulations, and so we will use this tool to study the Sherrington-Kirkpatrick model.

It remains then to compute the average of the partition function of $n$ replicas

$$ 
\begin{align}
\langle Z^n\rangle_J 
&= \int{\prod_{ij} p(J_{ij})dJ_{ij}\sum_{\{s^\alpha\}}\exp\left(\beta\sum_{ij}J_{ij}\sum_{\alpha=1}^n s_i^\alpha s_j^\alpha\right)} \\
&= \sum_{\{s^\alpha\}}\prod_{ij}\left\langle\exp\left(\beta J_{ij}\sum_{\alpha=1}^n s_i^\alpha s_j^\alpha\right)\right\rangle_{J_{ij}} \\
&= \sum_{\{s^\alpha\}}\prod_{ij}\exp\left(\frac{(\beta J)^2}{2N}\sum_{\alpha, \beta = 1}^n s_i^\alpha s_j^\alpha s_i^\beta s_j^\beta\right) \\
&= \sum_{\{s^\alpha\}}\exp\left[\frac{(\beta J)^2}{2N}\sum_{ij}\sum_{\alpha, \beta = 1}^n s_i^\alpha s_j^\alpha s_i^\beta s_j^\beta\right]
\end{align}
$$

Going from equations $(2)$ to $(3)$ is really just a Gaussian integral, but the physicists have christened it as a **Hubbard-Stratonovich transform**,

$$ \langle e^{zx} \rangle_z = \exp\left(\frac{1}{2}\sigma^2 x^2\right) $$

where $z\sim N(0,\sigma^2)$. Looking at the above sum, we see that when $\alpha=\beta$, since we are dealing with Ising spins $s_i^\alpha s_j^\alpha s_i^\beta s_j^\beta = 1$, and so $\sum_{ij}\sum_{\alpha=\beta}s_i^\alpha s_j^\alpha s_i^\beta s_j^\beta = nN^2$. Hence, separating out the $\alpha=\beta$ terms we have

$$
\sum_{\{s^\alpha\}}\exp\left[\frac{(\beta J)^2}{2N}\sum_{ij}\sum_{\alpha, \beta = 1}^n s_i^\alpha s_j^\alpha s_i^\beta s_j^\beta\right]
= \exp\left(\frac{1}{4}(\beta J)^2 nN\right)\sum_{\{s^\alpha\}}\exp\left[\frac{(\beta J)^2}{2N}\sum_{\alpha < \beta} \left(\sum_i s_i^\alpha s_i^\beta\right)^2\right]
$$

We see that all our degrees of freedom $s_i$ across replicas are coupled together in a quadratic term in the exponential. Again, we will use a Hubbard-Stratonovich transform to decouple this quadratic term into a linear term, with the *cost* of introducing a Gaussian integral back into the mix. However, this is often fine because the concentration effects of Gaussian random variables will allow us other analytic tricks to compute the resulting integral. Using the transform in the form

$$ e^{\lambda a^2/2} = \sqrt{\frac{\lambda}{2\pi}}\int_{\mathbf{R}} dx \exp\left[-\lambda\frac{x^2}{2} + a\lambda x\right] $$

we can write:

$$
\langle Z^n\rangle_J
= \exp\left(\frac{1}{4}(\beta J)^2 nN\right)\prod_{\alpha < \beta}\left[\sqrt{\frac{N}{2\pi}}(\beta J)\int_{\mathbf{R}} dq_{\alpha\beta}\right]\exp\left(-N\frac{(\beta J)^2}{2}\sum_{\alpha < \beta} q_{\alpha\beta}^2 + N\log \operatorname*{Tr} \exp[-H]\right)
$$

where $H=(-\beta J)^2\sum_{\alpha < \beta} q_{\alpha\beta} s^\alpha s^\beta$ and the trace is over the $n$ Ising spins $s^\alpha$ for each replica. The $q_{\alpha\beta}$ form the components of a matrix, and this will be a central object of study in the remainder of the calculation.

We are now interested in computing the **free energy** (which is really an effective energy function)

$$ f(\beta) = \lim_{n\to 0} \lim_{N\to\infty} \left(-\frac{1}{\beta N n}\log\langle Z_J^n\rangle_J\right) $$

where here we have used the replica trick to rewrite the quenched average in terms of the annealed one. In principle, we should be taking the thermodynamic limit ($N\to\infty$) before the replica limit, but then the calculation we will do becomes impossible. We will instead just assume this is correct and carry on.

In the thermodynamic limit we can treat the integral above as a saddle point calculation, also known as a [method of steepest descent](https://en.wikipedia.org/wiki/Method_of_steepest_descent). In the next section we'll do a brief digression into the simplified steepest descent method we will use, called Laplace's method.


## laplace's method

Suppose our goal is to approximate an integral of the form

$$ \int_a^b e^{Mf(x)} dx $$

where $f:\mathbf{R}\to\mathbf{R}$ is a smooth function and $M$ sufficiently large. The intuition of the method is as follows: as $M$ increases in size, the fluctuations in the function near a global maximum increases while outside of it they fade away. Hence, the function becomes well approximated by a Gaussian centered on the extremum with increasingly larger variance.

We can use this intuition to give a computation of the integral in the asymptotic limit. By Taylor expansion around the extremum $x_0$ of $f$, we can write

$$ f(x) \simeq f(x_0) + f'(x_0)(x-x_0) + \frac{1}{2}f''(x_0)(x-x_0)^2 $$

As we are at an extremum, $f'(x_0)=0$, and so we have approximated our integral as

$$ \int_a^b e^{Mf(x)} dx \simeq e^{Mf(x_0)}\int_a^b \exp\left(-\frac{1}{2}M |f''(x_0)|(x-x_0)^2\right) $$

Note that as $M$ is sufficiently large, the exponential quadratic $e^{-Mz^2}$ decays exponentially fast on the boundary, and so we can further approximate by letting the endpoints blow up towards infinity. Thus the approximating integral is Gaussian, and can be easily computed:

$$ \int_a^b e^{Mf(x)} dx \simeq e^{Mf(x_0)}\sqrt{\frac{2\pi}{M |f''(x_0)|}} $$

as $M\to\infty$. 

Let's apply Laplace's method to our annealed average above. We want to compute the integrals

$$ \prod_{\alpha < \beta}\left[\sqrt{\frac{N}{2\pi}}(\beta J)\int_{\mathbf{R}} dq_{\alpha\beta}\right]\exp\left(-N\frac{(\beta J)^2}{2}\sum_{\alpha < \beta} q_{\alpha\beta}^2 + N\log \operatorname*{Tr} \exp[-H]\right) $$

which hinges on computing integrals of the form

$$ \int_\mathbf{R} dq_{\alpha\beta} \exp\left[-N\left(\frac{(\beta J)^2}{2} q^2_{\alpha\beta} - \log\operatorname*{Tr}\exp[\beta J q_{\alpha\beta}s^\alpha s^\beta]\right)\right] $$

In the thermodynamic limit ($N\to\infty$), we use Laplace's method to asymptotically express this integral as

$$ e^{-Nf(\tilde{q})}\sqrt{\frac{2\pi}{N(\beta J)^2}} $$

where $f(\tilde{q})$ is the function value on the extremum value of $q_{\alpha\beta}$. Bundling up over all the $\alpha,\beta$, we get that

$$ 
\begin{align}
\langle Z^n\rangle_J
&= \exp\left(\frac{1}{4}(\beta J)^2 nN\right)\prod_{\alpha < \beta}\left[\sqrt{\frac{N}{2\pi}}(\beta J)\int_{\mathbf{R}} dq_{\alpha\beta}\right]\exp\left(-N\frac{(\beta J)^2}{2}\sum_{\alpha < \beta} q_{\alpha\beta}^2 + N\log \operatorname*{Tr} \exp[-H]\right) \\
&= \exp\left(-N\operatorname*{extr}_q{\mathcal{S}[q]}\right)
\end{align}
$$

where

$$ \mathcal{S}[q] = -\frac{(\beta J)^2 n}{4}+\frac{(\beta J)^2}{2}\sum_{\alpha < \beta} q_{\alpha\beta}^2 - \log\operatorname*{Tr} \exp[-H] $$

is the *effective action*, and the saddle point extremum is taken over all matrices $q$. We obtain the expression for the free energy

$$ f(\beta) = \lim_{n\to 0}\frac{1}{\beta n}\operatorname*{extr}_q{\mathcal{S}[q]} $$

Why did we say *extremum* as opposed to minimum? The replica method is weird because of the limit $n\to 0$, which doesn't really make sense in an optimization standpoint. Since the size of the matrix $q$ is dependent on number of replicas $n$, what does it mean for the dimensions to go to $0$? To talk about what kind of extremum, we must talk about the eigenvalues of the Hessian, but what does it mean to have $1/3$ eigenvalues? 

Before we get to this, consider the minimum of the effective action. Differentiating with respect to $q_{\alpha\beta}$ and setting to $0$ gives

$$ q_{\alpha\beta} = \frac{\operatorname*{Tr}{s^\alpha s^\beta e^{-H}}}{\operatorname*{Tr}{e^{-H}}} = \langle s^\alpha s^\beta\rangle_H $$

where the average is over the Gibbs distribution described by $H$. This is a *self-consistency equation* for the order parameter *q*. At this point we have to realize that it is futile to perform the minimization of the effective action with a generic $n$-by-$n$ symmetric matrix, and so we are forced to take a specific parameterization of a family of such matrices as an **ansatz**. 


## replica symmetry

What kind of parameterizations are we looking for? Note that the effective energy $\mathcal{S}[q]$ is independent under permutations of the replicas. Consider the toy example of a function of two variables $f(x,y)$ that is symmetric

$$ f(x,y) = f(y,x) $$

and consider an extremum $(x_0, y_0)$ of $f$.

There are two possibilities for this extremum. The first is that $x_0=y_0$. Then clearly the extremum itself is invariant under the action of the exchange group, and so we have found geometrically a single energy valley (extremum pit) of the function.

The other is that $x_0 \neq y_0$. Then necessarily $(x_0, y_0)$ and $(y_0, x_0)$ are both extrema, and so in this situation we have multiple energy valleys for us to fall into.

As a first choice, we pick the easiest ansatz, one in which the matrix $q$ is itself invariant to the permutation group on replicas. This leads to the **replica symmetry solution**

$$ q_{\alpha\beta} = q - q\delta_{\alpha\beta} $$

where off-diagonal elements take a constant value $q$ and otherwise $0$ (I apologize for overloading the symbol $q$ but it's clear which is which). Plugging this ansatz in our effective action yields

$$
\begin{align}
\mathcal{S}[q]
&= -\frac{(\beta J)^2 n}{4}+\frac{(\beta J)^2}{2}\sum_{\alpha < \beta} q_{\alpha\beta}^2 - \log\operatorname*{Tr} \exp[-H] \\
&= -\frac{(\beta J)^2 n}{4}+\frac{(\beta J)^2}{2}\frac{n(n-1)}{2}q^2-\log\sum_{\{s^\alpha\}}\exp\left[-\frac{(\beta J)^2}{2} \left(q\left(\sum_\alpha s^\alpha\right)^2 - qn\right)\right]
\end{align}
$$

For that squared term we use another Hubbard-Stratonovich transform to decouple the replicas, giving us

$$ -\frac{(\beta J)^2 n}{4}+\frac{(\beta J)^2}{2}\frac{n(n-1)}{2}q^2-\log\sum_{\{s^\alpha\}} \exp\left[\frac{(\beta J)^2 qn}{2}\right]\int_\mathbf{R} Dz \exp\left(\beta J\sqrt{q}z\sum_\alpha s^\alpha\right) $$

where $Dz=\frac{dz}{\sqrt{2\pi}}e^{-z^2/2}$ is the standard Gaussian measure. Note that this decouples the sum of replicas into independent components, which we can calculate analytically

$$
\begin{align}
\sum_{\{s^\alpha\}}\exp\left(\beta J\sqrt{q}\sum_\alpha s^\alpha\right)
&= \prod_\alpha\sum_{\{s^\alpha\}} \exp\left(\beta J\sqrt{q}z s^\alpha\right) \\
&= \prod_\alpha 2\cosh(\beta J\sqrt{q}z) \\
&= (2\cosh(\beta J \sqrt{q}z))^n
\end{align}
$$

To make this analytically more tractable, we Taylor expand the power expression as $a^n\simeq 1 + n\log{a}+\mathcal{O}(n^2)$, so that our effective action becomes

$$
\begin{align}
\mathcal{S}[q]
&= -\frac{(\beta J)^2 n}{4}+\frac{(\beta J)^2}{2}\frac{n(n-1)}{2}q-\log\sum_{\{s^\alpha\}} \exp\left[\frac{(\beta J)^2 qn}{2}\right]\int_\mathbf{R} Dz \exp\left(\beta J\sqrt{q}z\sum_\alpha s^\alpha\right) \\
&\simeq -\frac{(\beta J)^2 n}{4}+\frac{(\beta J)^2}{2}\frac{n(n-1)}{2}q - \frac{(\beta J)^2 qn}{2} - \log\left(1+n\int_\mathbf{R}Dz \log(2\cosh(\beta J \sqrt{q}z)\right) \\
&\simeq -\frac{(\beta J)^2 n}{4}+\frac{(\beta J)^2}{2}\frac{n(n-1)}{2}q - \frac{(\beta J)^2 qn}{2}-n\int_\mathbf{R}Dz \log(2\cosh(\beta J \sqrt{q}z))
\end{align}
$$

Dividing out by $\beta n$ and taking the $n\to 0$ limit, we have

$$ \lim_{n\to 0} \frac{1}{\beta n}\mathcal{S}[q] =
-\frac{\beta J^2}{4}(1-q)^2-\frac{1}{\beta}\int_\mathbf{R}Dz \log(2\cosh(\beta J \sqrt{q}z))
$$

The extremum of this object is our free energy, and by a simple computation is attained the self-consistency equation for the parameter $q$

$$ q = \int_\mathbf{R} Dz \tanh^2(\beta J \sqrt{q}z) $$

This always admits a solution given by $q=0$. In this case the free energy is given by

$$ f_{RS}(\beta) = -\frac{\beta J^2}{4} - \frac{1}{\beta}\log 2 $$


## negative entropy

Since the entropy is given by $-\partial f/\partial\beta$, we have that the entropy is negative when

$$ \beta^2 > -\frac{4\log 2}{J^2} $$

In fact, we can try and calculate the entropy in zero temperature analytically. First we derive the low-temperature form of the order parameter $q$. Note that $q\to 1$ as $\beta\to\infty$ (recall that $\beta=1/T$ for temperature $T$). 

Hence we can take as approximation $q\simeq 1-aT$ in the low-temperature regime, for some $a>0$. The self-consistency equation for $q$ gives

$$
\begin{align}
1 - aT
&= \int Dz\tanh^2(\beta J\sqrt{1-aT} z) \\
&= 1 - \int Dz \operatorname{sech}^2(\beta J\sqrt{1-aT} z) \\
&\to 1 - \int Dz \operatorname{sech}^2(\beta J z)
\end{align}
$$

as $q\to 1$, and so as $\beta\to\infty$,

$$
\begin{align}
\int Dz \operatorname{sech}^2(\beta J z)
&= \frac{1}{\beta J}\int Dz\frac{d}{dz}\tanh(\beta J z) \\
&\to \frac{1}{\beta J}\int Dz \frac{d}{dz}(2\theta(z)-1) \\
&= \frac{2}{\beta J}\int Dz \delta(z) \\
&= \sqrt{\frac{2}{\pi}}\frac{T}{J}
\end{align}
$$

where $\theta(z)$ is the Heaviside step function. Thus $a=\sqrt{\frac{2}{\pi}}\frac{1}{J}$. Plugging this into the free energy:

$$ f(\beta) = -\frac{\beta J^2}{4}(1-q)^2-\frac{1}{\beta}\int_\mathbf{R}Dz \log(2\cosh(\beta J \sqrt{q}z)) $$

in the first term we find a contribution of $-T/2\pi$. The second term's integral can be easily written as

$$ 2\int_0^\infty Dz\left\{\beta J \sqrt{q}z + \log(1 + e^{-2\beta J\sqrt{q}z})\right\} $$

Taking a Taylor approximation $\sqrt{q}\simeq 1 - aT/2$, we can evaluate the term to give

$$ \frac{2\beta J(1-aT/2)}{\sqrt{2\pi}} + 2\int_0^\infty Dz e^{2\beta J\sqrt{q}z} $$

where the last integral can be computed analytically. However, we will note that it is $\mathcal{O}(T^2)$. We check this by first seeing that

$$ \lim_{T\to 0+} \int_0^\infty Dz e^{-2\beta J\sqrt{q}z} = \int_0^\infty Dz \lim_{\beta\to\infty} e^{-2\beta J z} = 0 $$

So the integral is at least $\mathcal{O}(T)$. Taking the integral with respect to $T$ of this integral gives

$$ \int_0^\infty Dz e^{2\beta J \sqrt{q}z}\cdot\left[-2J\left(-\frac{1}{T^2}\sqrt{1-aT}-\frac{a}{2T\sqrt{1-aT}}\right)\right] $$

As $T\to 0+$, the exponential dominates and so we note that this expression converges to 0. This proves the claim. Hence this term does not contribute to the entropy.

Combining these contributions we get that in the low-temperature limit, the free energy asymptotically behaves as

$$ f \simeq -\sqrt{\frac{2}{\pi}}J + \frac{T}{2\pi} $$

and so the zero-temperature entropy has the fun value $-1/2\pi$.

This is nonphysical, and corresponds to the fact that the replica symmetry ansatz is wrong! This leads to the startling realization that we have to **break** replica symmetry in order to get a physically-relevant solution.

In the next post, we will describe Parisi's replica symmetry breaking solution to this problem.

