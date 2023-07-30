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

