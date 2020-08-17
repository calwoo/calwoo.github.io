---
title: Dirichlet processes
author: Calvin
---

**Note: This blog post is still a rough draft. Read on with caution.**


Suppose for instance that you wanted to perform a clustering of data into 2 partitions, say with a Gaussian mixture. Since we only have two clusters, we can do so with a Bayesian hierarchical model as such:

$$ \begin{align} \pi &\sim \text{Bernoulli}(p_\text{param}) \\
    x_i|\pi &\sim \text{Normal}(\mu_\pi, \sigma^2_\text{fixed}) \end{align} $$

In this model, the *assignment* latent variable $\pi$ is sampled from a Bernoulli distribution (there are only 2 clusters), and the choice of assignment tells us which of the two Gaussians we should sample our datapoint from. The usual Bayesian inference procedures allow us to tune the generative model to more adeptly match a dataset that we pass in. 

However, there is a certain inflexibility in using an unconstrained parameter $p_\text{param}$ in the above model. It is a common modeling choice to impose a distributional restriction on the probability $p$-- in some sense, we want to *parameterize $\text{Bernoulli}(p)$*. The usual choice is to sample $p$ from a **beta** distribution, which turns our model into

$$ \begin{align} p &\sim \text{Beta}(\alpha, \beta) \\
    \pi|p &\sim \text{Bernoulli}(p) \\
    x_i|\pi,p &\sim \text{Normal}(\mu_\pi, \sigma^2_\text{fixed}) \end{align} $$

Why did we choose the **beta** distribution over all the other distributions with support in $[0,1]$? We can go into heuristics like saying it's a maximal entropy distribution or whatnot, but it's better to just say we chose it because it's [conjugate](https://en.wikipedia.org/wiki/Conjugate_prior) to the Bernoulli distribution. 

For the purposes of variational inference, we are often interested in sampling efficiently from the beta distribution. A particularly nice way (that generalizes to the Dirichlet distribution below) is as follows: we sample $v\sim\text{Beta}(\alpha, \beta)$ by first drawing from independent Gamma distributions $x\sim\text{Gamma}(\alpha,1)$ and $y\sim\text{Gamma}(\beta,1)$ and setting $v=x/(x+y)$. 

```python
def beta_sample(a, b):
    x = gamma(a, 1).sample()
    y = gamma(b, 1).sample()
    v = x / (x + y)
    return v
```

The problem with this is that reparameterization gradients for the Gamma distribution can be fairly difficult to deal with, and can involve costly infinite series approximations. Another approach to inference with the beta distribution is to approximate it with the **Kumaraswamy distribution**

$$ \text{Kumaraswamy}(x; \alpha, \beta) = \alpha\beta\cdot x^{\alpha-1}(1-x^\alpha)^{\beta-1} $$

with support on $(0, 1)$, $\alpha,\beta > 0$. Reparameterizing the Kumaraswarmy distribution is simple: it has a tractible inverse CDF, so inverse sampling is easy to do:

$$ \text{cdf}^{-1}_{\alpha,\beta}(u) = (1-u^{\frac{1}{\beta}})^{\frac{1}{\alpha}} $$ 


### number of clusters > 2

Now suppose we have a clustering problem with more than 2 clusters. What changes in our Bayesian model? First off, since our assignment variables aren't binary anymore, but instead sampled from $\{1,...,K\}$, we exchange our Bernoulli distribution for a **categorical** distribution $\text{Categorical}(q_1,...,q_K)$. 

What is $(q_1,...,q_K)$ in this case? In this setting it is a **discrete probability distribution** over the terms $\{1,...,K\}$, so in particular each $q_j\ge 0$ and $\sum_{i=1..K} q_i = 1$. Geometrically, one could describe the distribution $(q_1,...,q_K)$ as a point in the $(K-1)$-simplex

$$ \Delta^{K-1}=\left\{(q_1,...,q_K)\middle| \sum_{i=1}^K q_i = 1, q_j \ge 0 \right\}\subset \mathbf{R}^K $$

This shows that whatever the replacement for a beta distribution should be, it should be a **prior over the simplex**. The distribution we will use is the **Dirichlet distribution**, which turns our model into the multi-cluster hierarchical model

$$ \begin{align} (q_1,...,q_K) &\sim \text{Dirichlet}(\alpha_1,...,\alpha_K) \\
    \pi|q &\sim \text{Categorical}(q_1,...,q_K) \\
    x_i|\pi,q &\sim \text{Normal}(\mu_\pi, \sigma^2_\text{fixed}) \end{align} $$

The Dirichlet process is a continuous distribution supported on the simplex $\Delta^{K-1}$. It has the fairly simple probability density function

$$ \text{Dirichlet}(q;\alpha_1,...,\alpha_K) = \frac{\Gamma(\alpha_0)}{\prod_{j=1}^K \Gamma(\alpha_j)} \prod_{j=1}^K q_j^{\alpha_j-1} $$

where here, $\alpha_0 = \sum_{j=1}^K \alpha_j$. This is a generalization of the beta distribution-- indeed if $K=2$, we have the density becomes

$$ \text{Beta}(p;\alpha,\beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} p^{\alpha-1}(1-p)^{\beta-1} $$

which is the density of the $\text{Beta}(\alpha, \beta)$ distribution. Analogously, the Dirichlet distribution is the conjugate prior to the categorical distribution, which makes inference somewhat easier. 

Now that we have a distribution with a density that allows us to compute log probabilities of, let's turn our attention to **sampling** from the Dirichlet distribution. Interestingly, one of the most computationally efficient ways of sampling is similar to the Gamma-reparameterization above for the beta distribution: to sample $(q_1,...,q_K)\sim\text{Dirichlet}(\alpha_1,...,\alpha_K)$, first sample $z_j\sim\text{Gamma}(\alpha_j,1)$ for $j=1,...,K$ and then set $q_i = z_i \left/ \middle(\sum_{j=1}^K z_j\right)$. 

```python
def dirichlet_sample(alphas):
    zs = []
    for alpha in alphas:
        z = gamma(alpha, 1).sample()
        zs.append(z)

    qs = [z / sum(zs) for z in zs]
    return qs
```

Again, as with the gamma-reparameterization for the beta distribution, the difficulty in reparameterizating path-gradients for the Gamma distribution makes this fairly difficult to use, even though it is mathematically elegant. 


### pots and sticks

Instead, we will turn to other sampling methods. We will describe two methods, both of which are important especially when `n_clusters` $\to\infty$ as we can generalize these to the nonparametric stochastic processes later on.

The first method is known as **Polya's urn**. Remember, our goal is to generate a sample from the Dirichlet distribution, $(q_1,...,q_K)\sim\text{Dirichlet}(\alpha_1,...,\alpha_K)$. Suppose we have an urn with $\alpha_i$ balls of color $i$ in it, for $i=1,...,K$ (suspend your senses for a bit, and believe the numbers have [color](https://en.wikipedia.org/wiki/Synesthesia)). From this point on, we perform a simple action iteratively until we get tired of it: we draw a ball uniformly from the urn, and then return it to the urn with an additional ball of the same color. Then the desired sample is the limit of the histograms of the colored balls as a discrete probability distribution. 

Why would this even work? Restricting to the case $K=2$, intuitively, this is just performing the posterior conjugation for the beta-binomial mixture model. The limiting histogram is then a marginalization over the resulting parameters, which gives the resulting sample. Details can usually be found in many probability textbooks. 

In code, this can be given by

```python
def polya_urn(initial_balls, n_steps=5000):
    # initial_balls = [alpha_1,...,alpha_K]
    K = len(initial_balls)
    urn = array(initial_balls)
    for _ in range(n_steps):
        ps = urn / sum(urn)
        # choose a color ball
        color = random.choice(range(K), p=ps)
        # increase ball count
        urn[color] += 1

    q = urn / sum(urn)
    return q
```

Now let's describe the second of our sampling methods, the **stick-breaking** approach. Intuitively, since we are sampling a sequence of numbers $(q_1,...,q_K)$ such that $\sum_{j=1}^K q_j = 1$, we can think of each $q_i$ as the length of the pieces of a unit-length stick broken in $K$ pieces. The trick is to figure out how to break the stick such that the length of the pieces are a sample from the Dirichlet distribution. 

Inductively, we assume that we can do this for the beta distribution: that is, we can sample from $\text{Beta}(\alpha,\beta)$ with impunity. Then to break a stick into $K$ pieces, we start by sampling $q_1\sim\text{Beta}(\alpha_1,\sum_{j=2}^K \alpha_j)$. This represents the first break in the stick. Next, we simulate the break in the 2nd length of the stick by sampling $u_2\sim\text{Beta}(\alpha_2,\sum_{j=3}^K \alpha_j)$. Scaling this to be in support $[0, 1-q_1]$, we let $q_2=u_1(1-q_1)$.  

Repeating this process until the end, we get a sample $(q_1,...,q_K)\sim\text{Dirichlet}(\alpha_1,...,\alpha_K)$. To see that this works, it's better to look at the code:

```python
def stick_breaking(alphas):
    K = len(alphas)
    us, qs = [], []
    for i in range(K - 1):
        u = beta(alpha[i], sum(alpha[i+1:])).sample()
        if len(us) > 0: 
            q = u * prod(1 - us)
        else:
            q = u
        
        qs.append(q)
        us.append(u)

    # last piece of stick
    qk = 1 - sum(qs)
    qs.append(qk)
    assert sum(qs) == 1
    return qs
```

This is a much better sampling process for 2 reasons: 1) this looks like the gamma-reparametrization above, which can be seen by expanding the beta distributions into gamma samplers, so it has attractive mathematical properties, and 2) although reparameterizing the beta is a pain, we can resort to the approximate posterior trick with Kumaraswarmy distributions to give a reparameterizable sampler. I have not seen this in the literature, but I would be very surprised if it doesn't exist. 


### dirichlet processes

The above assumed that we were performing clustering with a finite, fixed number of clusters $K$. What if we don't want this restriction? The field of **Bayesian nonparametrics** deals with models whose complexity grows as the number of data grows. Indeed, we would like a clustering algorithm that could *learn* the number of clusters needed for efficiently capturing a dataset. 

As with the above, we want a prior over the space of all **infinite** discrete distributions (note that here, we don't restrict our sample space to be countably discrete. Indeed, it can be an arbitrary probability space-- we only merely want to model convergent discrete measures on it). A distribution we will use here is the **Dirichlet process**, which is a distribution over the space of discrete probability measures on a probability space $\Omega$,

$$ \text{Prob}_\text{fin}(\Omega) = \left\{\text{discrete probability measures }\mu=\sum_{i=1}^\infty \omega_i\delta_{x^{(i)}}\text{ on }\Omega\right\} $$

Note as a discrete probability measure, $\sum_{i=1}^\infty \omega_i = 1$. 

Although we are in the business of forming a prior distribution over (discrete) distributions on a possible unbounded space, we do want to impose a weak boundedness condition on the resulting prior. As a consequence, the Dirichlet process is a function of two parameters: an existing *base distribution* $G_0$ over the sample space $\Omega$ and a *scaling parameter* $\alpha$. 

We give a **formal definition** of the Dirichlet process, following [Ferguson](https://projecteuclid.org/euclid.aos/1176342360). A distribution over $\text{Prob}_\text{fin}(\Omega)$ is a Dirichlet process, $\text{DP}(G_0,\alpha)$, if for any finite partition of $\Omega$, given by $A_1\cup\cdot\cdot\cdot\cup A_k = \Omega$ and for samples $G\sim\text{DP}(G_0,\alpha)$, the random vector $(G(A_1),...,G(A_k))$ is distributed according to a Dirichlet distribution

$$  (G(A_1),...,G(A_k))\sim\text{Dirichlet}(\alpha G_0(A_1),...,\alpha G_0(A_k)) $$

How to parse this definition? First off, since $G_0$ is a distribution on $\Omega$, we have $\sum_{i=1}^k G_0(A_i) = 1$, and similarly for $G$, as $G$ is a sample from a distribution over probability distributions. Secondly, even though for any given realization $G$, $(G(A_1),..., G(A_k))$ is a fixed vector, since $G$ is randomly varying (it's sampled from $\text{DP}(G_0,\alpha)$), this makes the vector $(G(A_1),..., G(A_k))$ random as well. Third, this is a *non-constructive* definition, just like the definition of the [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process#Definition), so we will want to give more constructive definitions to work with it. 

**Note:** in the situations before, our Dirichlet distributions acted as a prior over *cluster assignment probabilities*, where the actual cluster assignment is done via sampling from a $\text{Categorical}(q_1,...,q_K)$ distribution. In particular, a finite Dirichlet-Gaussian mixture model separates the assignment of clusters from the question of where the clusters are centered. Sampling from a Dirichlet process give however a *discrete probability measure*,

$$ G = \sum_{i=1}^\infty \omega_i\delta_{x^{(i)}} \sim \text{DP}(G_0, \alpha) $$

which already has encoded the information of the centers of each cluster (as the $x^{(i)}$). A common thing to do is to take this sampled distribution and convolve it with another density function. This allows us to perform density estimation with the Dirichlet process samples.


### $\infty$-sticks and chinese food

In this section we're gonna describe the **Chinese restaurant process**. We'll see that this process gives a way to sample from a Dirichlet process. To start, imagine a Chinese restaurant with (countably) infinite number of tables $i=1,2,...$. This is a very popular restaurant, so a constant infinite stream of customers come in one-by-one and sit at some table. 

How do customers decide which table to sit at? Randomly, of course-- but with rules: 1) the first customer always sits at the first table, $z_1=1$. 2) After $n-1$ customers have been seated, the $n$th customer either sits at a **new table** $z_n$ with probability $\frac{\alpha}{n-1+\alpha}$ or sits at an **occupied table** with probability $\frac{c}{n-1+\alpha}$ where $c$ is the number of people already sitting at that table. Here, $\alpha$ is a fixed scalar parameter. 

The above sampling process is described by the Chinese restaurant process

$$ z_n \sim \text{CRP}(\alpha; z_1,...,z_{n-1}) $$

which, described in code is:

```python
def chinese_restaurant_process(n_customers, alpha):
    zs = []
    # rule 1: first customer always sits at first table
    zs.append(1)

    for i in range(1, n_customers):
        # collect customer assignments into histogram
        hist = collect_into_histogram(n_customers)
        
        cur_n_customers = i + 1
        cur_n_tables = len(zs)
        normalizing_factor = cur_n_customers - 1 + alpha

        hist.append(alpha)
        ps = hist / normalizing_factor

        # rule 2: nth customer sits at new or old table with
        # above determined probabilities
        z = random.choice(range(1, cur_n_tables + 2), p=ps)
        zs.append(z)
    
    return zs
```

Why a Chinese restaurant? No idea. If I had to hazard a guess, it's because the number of customers at a given table can grow unboundedly, which if you've ever walked into a Chinatown *dim sum* restaurant, seems accurate. 

How does this connect with the Dirichlet process? It turns out that the generative story

$$ \begin{align} x^{(1)}, x^{(2)},... &\sim G_0 \\
    z_n &\sim \text{CRP}(\alpha;z_1,...,z_{n-1}) \end{align} $$

is equivalent to the Dirichlet process

$$ \begin{align} G = \sum_{i=1}^\infty \omega_i\delta_{x^{(i)}} &\sim \text{DP}(G_0, \alpha) \\
    x^{(z_n)} &\sim G \end{align} $$

In some sense, the Chinese restaurant process decouples the Dirichlet process into an assignment phase and a density phase. As a consequence, this process doesn't directly construct the discrete distributions $G\sim\text{DP}(G_0,\alpha)$. We can adapt the **stick-breaking** construction for the Dirichlet distribution in previous sections to work for $\infty$-many clusters. 

Unlike the finite stick-breaking construction for the Dirichlet distribution which is an iterative one, the construction for $\infty$-many clusters can be more parallelizable. The construction can actually be described by the following generative model

$$ \begin{align} x^{(1)}, x^{(2)},... &\sim G_0 \\
    v_1, v_2,... &\sim \text{Beta}(1,\alpha) \\
    \omega_j &= v_j\prod_{i=1}^{j-1} (1-v_i) \\
    G &= \sum_{j=1}^\infty \omega_j \delta_{x^{(j)}}
    \end{align} $$

The resulting $G\sim\text{DP}(G_0,\alpha)$ is a realization from the Dirichlet process. 

**Remark:** if in the above generative model we let $0 \le d < 1$ be a *discount* parameter, and instead sample $v_k\sim\text{Beta}(1-d,\alpha+kd)$, we get the **Pitman-Yor process**. If $d=0$, this process degenerates to a Dirichlet process. What is the heuristic difference between these two? It comes down to the number of clusters that arise from the sampling process.

For a Dirichlet process, after sampling $n$ iid samples the number of unique clusters we would have seen is of the order $\mathcal{O}(\alpha\log{n})$. However, for a Pitman-Yor process, we see that this grows as a **power law**, $\mathcal{O}(\alpha n^d)$. 


### stochastic memoization

Above we've described constructions to produce the weights of a discrete distribution that form a realization of the Dirichlet process

$$ G = \sum_{i=1}^\infty \omega_i\delta_{x^{(i)}} \sim \text{DP}(G_0, \alpha) $$

but it has the unwieldy implication that to sample from a Dirichlet process you need to construct the infinite sum all at once. But our code can't take that stress! While many computable generative models will truncate the sum or the stick-breaking process to a finite-memory stage, another way is to stare at the Chinese restaurant process--

$$ z_n \sim \text{CRP}(\alpha;z_1,...,z_{n-1}) $$

Note that this process has two features: 1) it's an iterative, sequential construction and 2) the sampling process changes after each sample is generated. This gives us a clue as to how to structure a generative story for the Dirichlet process-- we do this iteratively, *memoizing* previous realizations and storing it in the distribution's *closure*. This is the **stochastic memoization** process described by [Roy-Mansinghka-Goodman-Tenenbaum](http://danroy.org/papers/RoyManGooTen-ICMLNPB-2008.pdf). Put into place with the Dirichlet process:

```python
class DirichletProcess:
    def __init__(self, G_0, alpha):
        self.G_0 = G_0
        self.alpha = alpha

        self.memo = []
        self.weights = []

    def sample(self):
        remaining_stick = 1 - sum(self.weights)
        # determine which table to sit at
        table_id = random.choice(range(0, len(self.weights) + 1), 
                                 p=self.weights + [remaining_stick])
        
        # if table_id == len(self.weights), we create a new table
        if table_id == len(self.weights):
            new_table_loc = self.G_0.sample()
            self.memo.append(new_table_loc)

            # break a piece off the stick for new weight
            new_piece = beta(1, self.alpha).sample()
            new_weight = new_piece * remaining_stick
            self.weights.append(new_weight)
            return self.memo[-1]
        else:
            # return the memoized table location
            return self.memo[table_id]
```

This is a stochastically memoized version of the Dirichlet process. 


### closing

This post was kinda messy hodgepodge of notes about Bayesian nonparametrics and specifically the Dirichlet process. There is a lot more to write about, such as exchangeability, hierarchical Dirichlet/Pitman-Yor processes, nonparametric LDA, etc. But maybe I'll push that off to a later date.




