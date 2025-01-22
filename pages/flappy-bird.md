---
title: Flappy Bird
---

# Table of Contents
* [Environment Setup](#environment-setup)
* [Experience Replay](#experience-replay)
* [Q-Learning Formula](#q-learning-formula)
* [Calculating Loss](#calculating-loss)
* [The Epsilon Greedy Function](#the-epsilon-greedy-function)

# Overview of Flappy Bird Gymnasium

*Flappy Bird* is a game where the player controls a bird in a side scrolling game. The game presents a series of pipes and the bird has to fly between them without hitting the pipe or the top of the screen. *Flappy Bird Gymnasium* exposes a description of the game in two formats. One format is a lidar (light detection and sensing) representation and the other is an array with twelve elements which represent the following attributes of the game: 

* the last pipe's horizontal position
* the last top pipe's vertical position
* the last bottom pipe's vertical position
* the next pipe's horizontal position
* the next top pipe's vertical position
* the next bottom pipe's vertical position
* the next next pipe's horizontal position
* the next next top pipe's vertical position
* the next next bottom pipe's vertical position
* player's vertical position
* player's vertical velocity
* player's rotation

The actions, or *action space* is defined as having the following two state:

State | State Description
------|-------------------
 0    | Do nothing
 1    | Flap

There are four *rewards* that are assigned, based on the actions of the bird:

Rewards | Description
--------|-----------------------------
 +0.1   | For every frame Flappy Bird stays alive
 +1.0   | Each time Flappy Bird successfully passes a pipe
 -1.0   | When Flappy Bird loses the game
 −0.5   | When Flappy bird loses the game by touching the top of the screen

# Environment Setup

The commands below show the creation of a Python virtual environment and the installation of the flappy-bird-gymnasium, tensorflow and torch libraries.
```
nadim@mypc:~$ python3 -m venv ai_flappy
nadim@mypc:~$ . ai_flappy/bin/activate
(ai_flappy) nadim@mypc:~$ pip install flappy-bird-gymnasium
(ai_flappy) nadim@mypc:~$ pip install tensorflow
(ai_flappy) nadim@mypc:~$ pip install torch --index-url https://download.pytorch.org/whl/cpu
(ai_flappy) nadim@mypc:~$ pip install pyyaml
(ai_flappy) nadim@mypc:~$ 
```

# Experience Replay
*Experience replay* is used in training a neural network. It consists of the following components:

* State - A description of the environment
* Action - The action being performed (e.g. flap or no flap)
* New state - A new description of the environment
* Reward - A positive or negative reward, based on the action in the environment
* Terminated - A flag indicating whether or not the game has ended

These five elements are stored in a Python *dequeu*, which is a fixed length array or a *deque* (pronounced *deck*): When a new element is added to a deque, such that the addition would exceed the size of the deque, then then the *oldest* element of the deque is *popped off*. A deque is a type of FIFO (first in, first out) queue.

# Q-Learning Formula
This is the *Q-Learning Formula* which is used to move the policy network to the target network.

> $$ q[state, action] = q[state, action] + learningRate * (reward * discount * max(q[state_n,:]) -q[state,action]) $$

Where: 

> $$ state_n $$

-is the new state.

So the change is represented by:

> $$ reward * discount * max(q[state_n,:]) -q[state,action] $$

The effect of the change is tempered by the *learning rate* and is typically very small, e.g. 0.01 or 0.001.

The *DQN Target Formula* is defined as follows. 

If new state is terminal i.e. the game is over:

> $$ q[state, action] = reward $$

Otherwise:

> $$ reward + discount * max(q[state_n,:]) $$

# Calculating Loss
We are using the *Mean Square Error* (MSE) function to calculate loss:

> $$ loss = mse(current_q, target_q) $$

-and the *mean square error* function is simply:

> $$ \frac{( C - T )^2} { 2 } $$

# The Epsilon Greedy Function
The *epsilon greedy function* can be described as follows:
```
if rand() < epsilon:
  choose random action
else:
  choose best calculated action

decrease epsilon
```
The *choose best calculated action* is the best Q value.
```

# Conclusion and Lessons Learned

The *Flappy Bird* tutorial helped me gain a better, more theoretical, understandin of Linear Q Networks. I also learned about using yaml
for configuration managment. Yaml is a bit better than using the Python ConfigParser which I used in the my [AI Snake Game](https://ai.osoyalce.com/pages/ai-snake-game.html). It provides support for booleans and nested configuration elements including lists.

I have also learned how to include *Latex* into my GitHub pages website as showcased on this page.

However, the *AI Snake Game* remains my favorite sandbox for ongoing AI Development. It has three input features corresponding to *go straight*, *turn left* and *turn right*, versus two input features that the *Flappy Bird* simulation uses (flap or don't flap). The AI Snake Gamee also has a more varied game state. Specifically, when the snake reaches a length that is more than twice the width of the board (I'm using a 20x20 board), then there is an added challenge. With my current setup, the AI can achieve scores of up to around 50, but not really any higher. At that point in the game, the AI has settled into a strategy of moving the snake around the edge of the screen and then cutting through the middle to get the food. It continues to the other edge and then circles again. While this strategy is good for scores up to 40, it fails to reach scores in the 60s because it ends up hitting itself.

I am exploring strategies to have the AI develop a new, better strategies, but so far nothing has worked. It remains stuck in a *local minimum*.

# Links
* [Implement Deep Q-Learning with PyTorch and Train Flappy Bird! DQN PyTorch](https://www.youtube.com/watch?v=arR7KzlYs4w&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi&index=1&ab_channel=JohnnyCode)
* [Flappy Bird Gymnasium on GitHub](https://github.com/markub3327/flappy-bird-gymnasium)
* [Tutorial Code on GitHub](https://github.com/johnnycode8/dqn_pytorch/blob/main/agent.py)
* [AI Snake Game](https://ai.osoyalce.com/ai/ai-snake-game.html)
* [The Flappy Bird code in my GitHub repository](#https://github.com/NadimGhaznavi/ai/tree/main/flappy_bird)

