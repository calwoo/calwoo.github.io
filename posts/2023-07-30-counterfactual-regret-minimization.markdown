---
title: Counterfactual regret minimization
author: Calvin
---

For the past few days I've been wanting to learn about modern game-playing AIs, and what are the main techniques out there powering them. I remember years ago when I started reading computer science books about heuristic search techniques and programming a Tic-Tac-Toe AI using various tree search algorithms. I was fairly interested in reinforcement learning for awhile, read about MCTS and DeepMind's successes with AlphaGo, but then I got busy with other stuff and ended up not reading much on it afterwards.

Recently I've started to get back into reading about reinforcement learning for *imperfect information games* (games in which each player has private information about the state of the game that they do not share with each other). It turns out that many of these techniques require a fair amount of background in game theory and their solution concepts, so I figured I'd write some notes on the topic.

In this post, which should be the first of at least 2 more, we will give a brief overview of the definitions and concepts from classical game theory that we need, restricting ourselves to two-player zero-sum games. Then we'll introduce the optimization idea of regret minimization, and give an overview of counterfactual regret minimization, an efficient implementation of regret minimization for imperfect information games.

### game theory

In this section we will give an brief introduction to some game-theoretic concepts, rushing to introduce the central concept of a Nash equilibrium of a normal-form game.

First, we should define what we mean by a **normal form game**. We will be considering games where there are multiple *agents* (say, $n$ of them), and where each agent can take a certain action $a_i$ from it's *action space* $A_i$. In these games, all agents perform their action simultaneously, with the actions of the other agents not influencing immediately each other. In this way, the game's outcome is determined by the single *action profile*
$$ a = (a_1, ..., a_n) \in A = A_1 \times ... \times A_n $$
What would cause an agent to take a particular action over others? In this setting we assume each agent is equipped with a *utility function* $u_i: A\rightarrow \mathbb{R}$ that effectively gives the payoff that each action profile gives to the $i$th agent. It is then a fair assumption that the goal of an agent is to maximize their payoff.

It can be convenient (especially in low-dimensions) to represent a normal-form game as a $n$-dimensional matrix of payoffs. For example, a canonical example is the prisoner's dilemma problem

```
           SILENT     CONFESS
        -----------------------
SILENT  | (-1, -1) | (-5, 0)  |
CONFESS | (0, -5)  | (-3, -3) |
        -----------------------
```

Here, the rows denote the action of the first player, while the columns are the action of the second player. For example, if player 1 confesses and player 2 stays silent, the entry $(0, -5)$ means player 1 gets a payoff of 0 and player 2 gets a payoff of -5.

We mention that we are most interested at the moment in **zero-sum games**, which are a particular class of two player normal-form games in which the sum of the payoffs of the agents is always 0, regardless of the action profile. Zero-sum games encapsulates games in which "one player's loss is another's gain".

### pareto optimality

How an agent chooses an action from its action space in a game is called it's **strategy**. It could select a single action and play it, which is called a **pure strategy**. However, it is convenient if an agent can choose among many different actions, where the action to play is sampled from a probability distribution. The choice of a distribution over actions is called a **mixed strategy**.

In a normal-form game, each agent takes a strategy $s_i$, which collectively forms a *strategy profile* $s=(s_1,...,s_n)$. We can extend each agent's utility function to a function on strategy profiles $u_i(s_1,...,s_n)$ by defining it as the expected utility marginalized over the joint distribution of action profiles. 

Since the goal of every agent is to maximize their utility, this induces a strict partial ordering on the set of (mixed) strategy profiles: a profile $s$ **Pareto dominates** another strategy profile $s'$ if for all agents $i$, $u_i(s) \ge u_i(s'$), and there exists an agent $j$ in which $u_j(s) > u_j(s')$. In other words, in a Pareto dominated strategy profile some player can be made better off without making any other player worse off. The optima of this partial ordering are called **Pareto optimal** strategies.

A few observations about Pareto optimality: first, every game has at least one Pareto optimal strategy. In fact, we can always find a Pareto optimal strategy composed of pure strategies for each agent-- for example, for agent $1$, we can consider the set of pure strategy profiles with the highest payoff for them (there may be multiple action profiles with the same payoff). Then for agent $2$, we choose the subset of that set that has the highest payoff for them, etc. The resulting pure strategy profiles are all Pareto optimal. Second, in a zero-sum game, all strategy profiles are Pareto optimal.

### nash equilibrium

For each agent $i$, let $s_i$ denote a (mixed) strategy the agent could take, and $s_{-i}=(s_1,...,s_{i-1}, s_{i+1},...,s_n)$ a strategy profile without agent $i$ considered. That is, we are isolating agent $i$'s strategy from a full strategy profile $s=(s_i, s_{-i})$ for the game.

If agent $i$ knew the strategy profile $s_{-i}$ of the other agents, it could choose the strategy $s^*_i$ that would be benefit itself. That is, for any other strategy $s'_i$,
$$ u_i(s^*_i, s_{-i}) \ge u_i(s'_i, s_{-i}) $$
We call $s^*_i$ the **best response** of agent $i$ to the other's strategy $s_{-i}$.

If the other agents knew of each other's strategies, they would all be continually trying to change their strategies in order to implement the best responses to one another. Iteratively, they may eventually reach a stable state-- they may reach a strategy profile $s=(s_1,...,s_n)$ in which for all agents $i$, $s_i$ is a best response to the remainder $s_{-i}$. This is a **Nash equilibrium**. Note that equilibria may not be unique.

Since we're working with zero-sum games, there are a couple of properties Nash equilibria enjoy that we will use. For one, in zero-sum games the Nash equilibria are *interchangable*: if $s=(s_1, s_2)$ is a Nash equilibrium and $s' = (s'_1, s'_2)$ is another, then $(s_1, s'_2)$ and $(s'_1, s_2)$ are both Nash equilibria. 

**Proof:** Recall in a zero-sum game, we have for any strategy profile $s$, $u_1(s) + u_2(s) = 0$. We can see then that
$$ u_1(s_1, s_2) = -u_2(s_1, s_2) \le -u_2(s_1, s'_2) = u_1(s_1, s'_2) \le u_1(s'_1, s'_2) $$
By symmetry, we get $u_1(s_1, s_2) \ge u_1(s'_1, s'_2)$, and so $u_i(s_1, s_2) = u_i(s'_1, s'_2)$.

Consider the profile $(s_1, s'_2)$, and let $s''_1$ be another strategy that agent $1$ could take. Using the above equality we can see
$$ u_1(s''_1, s'_2) \le u_1(s'_1, s'_2) = -u_2(s'_1, s'_2) $$
since $(s'_1, s'_2)$ is a Nash equilibrium. By the above, $u_2(s'_1, s'_2) = u_2(s_1, s_2)$ so
$$ -u_2(s'_1, s'_2) = -u_2(s_1, s_2) \le -u_2(s_1, s'_2) = u_1(s_1, s'_2)$$
Combining with above, we see that $s_1$ is the best response for agent $1$ for the strategy $s'_2$ of agent $2$. A symmetric argument for a strategy $s''_2$ taken by agent $2$ shows that $(s_1, s'_2)$ is a Nash equilibrium. The case for $(s'_1, s_2)$ follows similarly. $\square$

The second property is one that we proved in the proof above: in a zero-sum game, the expected payoff to each player is the same for every Nash equilibrium.

### computing equilibria

Let's restrict for now to the case of two-player zero-sum games. How would we compute the Nash equilibria of such a game? We first introduce the *maxmin* strategy for an agent.

Suppose we have a conservative agent-- they would want to maximize their expected utility regardless of what the other agents do. In particular, they would seek a strategy that maximizes their expected utility in the *worst case scenario* that the other agents act to minimize their payoff. This is the **maxmin strategy**
$$ \underline{s}_i = \arg\max_{s_i}\min_{s_{-i}}{u_i(s_i, s_{-i})} $$
The payoff from this strategy is the **maxmin value**
$$ \underline{v}_i = \max_{s_i}\min_{s_{-i}}{u_i(s_i, s_{-i})} $$
Analogously, an agent could seek a strategy to maximally punish the payoffs of their opponents, regardless of the damage to themselves. This leads to the **minmax strategy** and value
$$ \overline{v}_i = \min_{s_i}\max_{s_{-i}}{u_i(s_i, s_{-i})} $$
Note that in general, $\underline{v}_i \le \overline{v}_i$. However, when we're in a finite two-player zero-sum game, we can prove something even stronger. Let $(s^*_i, s^*_{-i})$ be a Nash equilibrium and $v^*_i$ be the expected utility of agent $i$ in this equilibrium.

First, we observe that $v^*_i \ge \underline{v_i}$. This is because if $v^*_i < \underline{v}_i$ then agent $i$ could gain greater utility by using their maxmin strategy instead, which is a contradiction to $s^*$ being a Nash equilibrium. In an equilibrium, all agents acts according to their best response to the other's strategies:
$$ v^*_{-i} = \max_{s_{-i}}{u_{-i}(s^*_i, s_{-i})} $$
As we're in a zero-sum game, $v^*_i = -v^*_{-i}$ and $u_i=-u_{-i}$, so
$$ \begin{aligned} v^*_i &= -v^*_{-i} \\ &= -\max_{s_{-i}}{u_{-i}(s^*_i, s_{-i})} \\
&= \min_{s_{-i}}{-u_{-i}(s^*_i, s_{-i})} \\
&= \min_{s_{-i}}{u_i(s^*_i, s_{-i})}
\end{aligned}
$$
By definition then,
$$ \underline{v}_i = \max_{s_i}\min_{s_{-i}}{u_i(s_i, s_{-i})} \ge \min_{s_{-i}}{u_i(s^*_i, s_{-i})} = v^*_i $$
Hence $\underline{v}_i = v^*_i$ and so our maxmin strategy is actually a Nash equilibrium! This is the **minimax theorem** proven by von Neumann in 1928.

In general, computing the Nash equilibria of a general normal-form game is computationally difficult (even from a complexity theory standpoint)! More precisely, a few years ago Babichenko-Rubinstein [show](https://arxiv.org/abs/1608.06580) that there is no guaranteed method for players to find even an approximate Nash equilibrium unless they tell each other almost everything about their preferences.

We instead focus on a different solution type that is more computationally efficient, the **correlated equilibria**, which is a strict generalization of Nash equilibrium. It helps to start with an example:

```
        STOP         GO
     -------------------------
STOP | (0, 0) |    (0, 1)    |
GO   | (1, 0) | (-100, -100) |
     -------------------------
```

In this **traffic light game**, each player has two actions-- stop or go. It's clear that there are two pure-strategy Nash equilibria, the off-diagonals `(GO, STOP)` and `(STOP, GO)`. However, these are not ideal, as only one person benefits in either situation.

Let's try and compute a mixed strategy Nash equilibrium for this game. Let the strategy of the first player be $\sigma=(p, 1-p)$ where $p$ is the probability of `GO`. How do we choose $p$? It should be chosen such that if player 2 plays with a best response, player 1 plays in such a way that player 2 is indifferent between their actions, that is

$$  u_2(\sigma, \text{GO}) = u_2(\sigma, \text{STOP}) $$

If this were to not hold, e.g. $u_2(\sigma, \text{GO}) > u_2(\sigma, \text{STOP})$, then player 2 would have a better time choosing `GO` more often, contradicting that they play with a best response. Computing out the expected utilities, we have

$$ 1\cdot (1-p) - 100p = 0 $$

which implies $p = 1/101$. Hence in a mixed Nash equilibrium, both players go at a traffic stop with absurdly low probability! This is obviously very unideal.

Luckily, this isn't how traffic stops at intersections work in the real world. In this game-theoretic setting, each mixed strategy that goes into a Nash equilibrium gives a probability distribution over actions for each player-- however, each action ends up choosing their action independently, without any communication between them. In real-life, we have *traffic lights*, which act as a signal to both players which suggests their action. This *correlates* the action of each player without fixing them to a single pure strategy.

This gives us the idea of a **correlated equilibrium**: a distribution $\mathcal{D}$ over action profiles $A$ is a correlated equilbrium if for each player $i$ and action $a^*_i$, we have

$$ \mathbf{E}_{a\sim\mathcal{D}}[u_i(a)] \ge \mathbf{E}_{a\sim\mathcal{D}}[u_i(a^*_i, a_{-i})|a_i] $$

that is, after a profile $a$ is drawn, playing $a_i$ is a best response for player $i$ conditioned on seeing $a_i$, given that everyone else will play according to $a$. In the traffic light game, conditioned on seeing `STOP`, a player knows the other player sees `GO` so their best response is to `STOP`, and vice versa.

Correlated equilibria generalize Nash equilibrium-- any Nash equilibrium is a correlated one, in the case that each player's actions are drawn from independent distributions, i.e.

$$ \Pr(a|\sigma) = \prod_{i\in N} \Pr(a_i|\sigma) $$

Despite looking as complex the definition of a Nash equilibrium, correlated equilibria are much more tractable to compute: they can be computed using no-regret learning, i.e. regret minimization. We discuss this next.

### regret minimization

In this section we give the basics of regret minimization, treating it initially from the perspective of online optimization. Let $\mathcal{X}$ be a space of strategies for a given agent. We consider decision processes in which at time $t=1,2,...$ the agent (player) will play an action $x_t\in\mathcal{X}$, receive "feedback" from the environment (the game, other players, etc) and then use it to formulate a response $x_{t+1}\in\mathcal{X}$.

Given $\mathcal{X}$ and a set $\Phi$ of linear transforms $\phi:\mathcal{X}\rightarrow\mathcal{X}$, a **$\Phi$-regret minimizer** for $\mathcal{X}$ is a model for a decision maker that repeatedly interacts with the environment via the API

* `NextStrategy` which outputs a strategy $x_t\in\mathcal{X}$ at decision time $t$.
* `ObserveUtility`($\ell^t$) which updates the decision-making process of the agent, in the form of a linear utility function (or vector) $\ell^t:\mathcal{X}\rightarrow\mathbf{R}$.

The quality metric of our minimizer is given by **cumulative $\Phi$-regret**

$$ R^T_\Phi = \max_{\hat{\phi}\in\Phi}\left\{\sum_{t=1}^T\left(\ell^t(\hat{\phi}(x_t))-\ell^t(x_t)\right)\right\} $$

where the interior term $\ell^t(\hat{\phi}(x_t))-\ell^t(x_t)$ is the *regret* at time $t$ of not changing our strategy by $\hat{\phi}$. The **goal** of a $\Phi$-regret minimizer is to guarantee its $\Phi$-regret grows **asymptotically sublinearly** as $T\rightarrow\infty$.

Intuitively, the class $\Phi$ of functions constraints the kind of "regret" we can feel-- a function $\phi\in\Phi$ tells us how we could have hypothetically swapped out our choice of action for a more optimal one, and the regret measures how much better that choice could have been for our decision. In this way, we want to *minimize* our regret by more often choosing optimal actions.

For an example of a class $\Phi$, we can take $\Phi=\{\text{all linear maps }\mathcal{X}\rightarrow\mathcal{X}\}$. Then the notion of $\Phi$-regret is called **swap regret**. Intuitively, it is the measure of how much a player can improve by switching any action we choose to the best decision possible in hindsight.

**Note:** If we restrict our choice of $\Phi$ to be $\Phi=\{\phi_{a\rightarrow b}\}_{a,b\in\mathcal{X}}$ where

$$  
\begin{equation}
    \phi_{a\rightarrow b}(x) = 
    \begin{cases}
        x & \text{if } x\neq a\\
        b & \text{if } x=a
    \end{cases}
\end{equation}
$$

then we get the closely related concept of **internal regret**.

A very special case of $\Phi$-regret comes when $\Phi$ is the set of constant functions.

**Defn:** (Regret minimizer) An **external regret minimizer** for $\mathcal{X}$ is a $\Phi^{\text{const}}$-regret minimizer for

$$ \Phi^{\text{const}} = \left\{\phi_{\hat{x}}: x\mapsto \hat{x}\right\}_{\hat{x}\in\mathcal{X}} $$

The $\Phi^{\text{const}}$-regret is just called **external regret**

$$ R^T = \max_{\hat{x}\in\mathcal{X}}\left\{\sum_{t=1}^T\left(\ell^t(\hat{x}) - \ell^t(x_t)\right)\right\} $$

Regret minimizers are useful in helping us find best responses in various game-theoretic settings. Suppose we're playing an $n$-player game where players $1,...,n-1$ play stochastically, i.e. at each time $t$, we get a strategy $x^{(i)}_t\in\mathcal{X}^{(i)}$ with 

$$ \mathbf{E}[x^{(i)}_t] = \bar{x}^{(i)}\in\mathcal{X}^{(i)} $$

We let player $n$ picks strategies according to an algorithm that guarantees sublinear external regret (i.e. via a $\Phi^{\text{const}}$-regret minimizer), where the utility function is given by the multilinear payoff functional

$$ \ell^t(x^{(n)}) = u_n(x^{(1)}_t, x^{(2)}_t,..., x^{(n-1)}_t, x^{(n)}) $$

The *claim* is that the average of player $n$'s strategies converges to the best response to $\bar{x}^{(1)},...,\bar{x}^{(n-1)}$:

$$ \frac{1}{T}\sum_{t=1}^T x^{(n)}_t \xrightarrow{T\rightarrow\infty} \argmax_{\hat{x}^{(n)}\in\mathcal{X}^{(n)}}\left\{u_n(\bar{x}^{(1)},...,\bar{x}^{(n-1)}, \hat{x}^{(n)})\right\} $$

To see this, note that by multilinearity,

$$
\begin{align*}
    R^T &= \max_{\hat{x}\in\mathcal{X}^{(n)}}\left\{\sum_{t=1}^T\left(u_n(x^{(1)}_t,...,\hat{x})-u_n(x^{(1)}_t,...,x^{(n)}_t)\right)\right\} \\
        &= \max_{\hat{x}\in\mathcal{X}^{(n)}}\left\{\sum_{t=1}^T u_n(x^{(1)}_t,...,\hat{x}-x^{(n)}_t)\right\}
\end{align*}
$$

and as $\frac{R^T}{T}\to 0$ by sublinearity of regret, this proves that $\frac{1}{T}\sum_{t=1}^T x_t^{(n)}\to\hat{x}$.

Our main use of regret minimization is to compute (correlated) equilibria. Let's restrict to the case of two-person zero-sum games. Given strategies $x\in\mathcal{X}\subset\mathbf{R}^n, y\in\mathcal{Y}\subset\mathbf{R}^m$ and a linear payoff matrix $A\in\operatorname{Mat}_{n,m}$ for player 1, the **utility** of player 1 is given by $x^\top A y$.

We seek a Nash equilibrium, which we have prove is a minimax solution

$$ \max_{x\in\mathcal{X}}\min_{y\in\mathcal{Y}} x^\top A y $$

i.e. we want to use regret minimization to compute a bilinear saddle point.

**Rmk:** In two-player zero-sum games, regret minimization processes will converge to Nash equilibria, since in this situation there are no extra correlated equilibria. This intuitively makes sense since zero-sum games are purely adversarial and there is no "gain" from cooperation in these situations. For some extended thoughts on this, see this [paper](https://www.kellogg.northwestern.edu/research/math/papers/45.pdf) of Rosenthal.

To compute these equilibria, we create a regret minimizer $\mathcal{R}_\mathcal{X}, \mathcal{R}_\mathcal{Y}$ per player with utility functions

$$
\begin{align*}
    \ell^t_\mathcal{X} &: x\mapsto (Ay_t)^\top x \\
    \ell^t_\mathcal{Y} &: y\mapsto -(A^\top x_t)^\top y_t
\end{align*}
$$

where $x_t, y_t$ are strategies generated by the regret minimizers at time $t$. This idea is called **self-play**.

Let $\gamma$ be the **saddle point gap**

$$ 
\begin{align*}
    \gamma(x,y) &= \left(\max_{\hat{x}\in\mathcal{X}} \hat{x}^\top Ay - x^\top Ay\right) + \left(x^\top Ay - \min_{\hat{y}\in\mathcal{Y}} x^\top A\hat{y}\right) \\
    &= \max_{\hat{x}\in\mathcal{X}} \hat{x}^\top Ay - \min_{\hat{y}\in\mathcal{Y}} x^\top A\hat{y}
\end{align*}    
$$

where the left term is the best response payoff to $t$ and right right term is the best response payoff to $x$ (both in the perspective of player 1). If $\gamma(x,y)=0$ then the strategy profile $\sigma=(x,y)$ is a Nash equilibrium.

**Thm:** As $T\to\infty$, the average strategies $\bar{x}=\frac{1}{T}\sum_{t=1}^T x_t$ and $\bar{y}=\frac{1}{T}\sum_{t=1}^T y_t$ approaches a Nash equilibrium.

*Proof:* By definition, we can write

$$
\begin{align*}
    \frac{1}{T}\left(R^T_\mathcal{X} + R^T_\mathcal{Y}\right)
    &= \frac{1}{T}\max_{\hat{x}\in\mathcal{X}}\left\{\sum_{t=1}^T\left(\ell_\mathcal{X}^t(\hat{x})-\ell_\mathcal{X}^t(x_t)\right)\right\}
        + \frac{1}{T}\max_{\hat{y}\in\mathcal{Y}}\left\{\sum_{t=1}^T\left(\ell_\mathcal{Y}^t(\hat{Y})-\ell_\mathcal{Y}^t(y_t)\right)\right\} \\
    &= \frac{1}{T}\max_{\hat{x}\in\mathcal{X}}\left\{\sum_{t=1}^T \ell^t_\mathcal{X}(\hat{x})\right\}
        + \frac{1}{T}\max_{\hat{y}\in\mathcal{Y}}\left\{\sum_{t=1}^T \ell^t_\mathcal{Y}(\hat{y})\right\}
\end{align*}
$$

as $\ell^t_\mathcal{X}(x_t) + \ell_\mathcal{Y}^t(y_t) = (Ay_t)^\top x_t - (A^\top x_t)^\top y_t = 0$. Continuing,

$$
\begin{align*}
    &= \frac{1}{T}\max_{\hat{x}\in\mathcal{X}}\left\{\sum_{t=1}^T \hat{x}^\top Ay_t\right\}
        + \frac{1}{T}\max_{\hat{y}\in\mathcal{Y}}\left\{\sum_{t=1}^T (-x_t^\top A\hat{y} \right\} \\
    &= \max_{\hat{x}\in\mathcal{X}}\hat{x}^\top A\bar{y} - \min_{\hat{y}\in\mathcal{Y}} \bar{x}^\top A\hat{y} \\
    &= \gamma(\bar{x}, \bar{y})
\end{align*}
$$

By sublinearity of the regret minimizers, $\frac{R^T_\mathcal{X}+R^T_\mathcal{Y}}{T}\to 0$. $\square$

### minimax via regret

In this section, we get another glimpse into the power of regret minimizers in optimization problems by reproving the minimax theorem. We first start with the easy part-- note that since

$$ \min_{y\in\mathcal{Y}} x^\top Ay \le x^\top Ay $$

we get automatically that applying $\max_{x\in\mathcal{X}}$ on both sides is also true. Since the right side is then still a function of $y$, we can minimize it and get

$$ \max_{x\in\mathcal{X}}\min_{y\in\mathcal{Y}} x^\top Ay \le \min_{y\in\mathcal{Y}}\max_{x\in\mathcal{X}} x^\top Ay $$

This is **weak duality**.

To prove the minimax theorem, we need to prove the inequality in the other direction. As in the previous regret learning situation, we play a repeated game between a regret minimizer an the environment: $\mathcal{R}_\mathcal{X}$ chooses a strategy $x_t\in\mathcal{X}$ and the environment plays $y_t\in\mathcal{Y}$ such that $y_t$ is a best response:

$$ y_t \in\argmin_{y\in\mathcal{Y}} x_t^\top Ay $$

The utility function observed by $\mathcal{R}_\mathcal{X}$ at each time $t$ is given by

$$ \ell^t_\mathcal{X}: x\mapsto x^\top Ay_t $$

We assume that $\mathcal{R}_\mathcal{X}$ gives sublinear regret in the worst case. Let $\bar{x}^T=\frac{1}{T}\sum_{t=1}^T x_t$ and $\bar{y}^T=\frac{1}{T}\sum_{t=1}^T y_t$ be the average strategies up to time $T$. For all $t$,

$$ \max_{x\in\mathcal{X}}\min_{y\in\mathcal{Y}} x^\top Ay \ge \frac{1}{T}\min_{y\in\mathcal{Y}}\sum_{t=1}^T x_t^\top Ay \text{ as each } \min_{y\in\mathcal{Y}} x^\top_t Ay\le \max_{x\in\mathcal{X}}\min_{y\in\mathcal{Y}} x^\top Ay $$

As each $y_t$ is the best response of the environment to the strategy $x_t$, we have

$$ \min_{y\in\mathcal{Y}} x_t^\top Ay = x^\top_t Ay_t $$

so that

$$ \max_{x\in\mathcal{X}}\min_{y\in\mathcal{Y}} x^\top Ay \ge \frac{1}{T}\min_{y\in\mathcal{Y}}\sum_{t=1}^T x_t^\top Ay \ge \frac{1}{T}\sum_{t=1}^T x^\top_t Ay_t $$

By definition of regret $R^T_\mathcal{X}=\max_{\hat{x}\in\mathcal{X}}\left\{\sum_{t=1}^T\left(\hat{x}^\top Ay_t-x^\top_t Ay_t\right)\right\}$ so

$$
\begin{align*}
    \frac{1}{T}\sum_{t=1}^T x_t^\top Ay_t &\ge \frac{1}{T}\max_{\hat{x}\in\mathcal{X}}\sum_{t=1}^T \hat{x}^\top Ay_t - \frac{R^T_\mathcal{X}}{T} \\
    &\ge \min_{y\in\mathcal{Y}}\max_{x\in\mathcal{X}} x^\top Ay - \frac{R^T_\mathcal{X}}{T}
\end{align*}
$$

By sublinearity, $\frac{R^T_\mathcal{X}}{T}\to 0$, so we see that

$$ \max_{x\in\mathcal{X}}\min_{y\in\mathcal{Y}} x^\top Ay \ge \min_{y\in\mathcal{Y}}\max_{x\in\mathcal{X}} x^\top Ay $$

This proves minimax.

### regret matching

