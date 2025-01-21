---
title: Flappy Bird
---

# Table of Contents
* [Environment Setup](#environment-setup)
* [Experience Replay](#experience-replay)
* [Q-Learning Formula](#q-learning-formula)
* [Calculating Loss](#calculating-loss)

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

* 0 - do nothing
* 1 - flap

There are four *rewards* that are assigned, based on the actions of the bird:

* +0.1 - every frame it stays alive
* +1.0 - successfully passing a pipe
* -1.0 - dying
* −0.5 - touch the top of the screen

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
```
q[state, action] = q[state, action] + learning_rate * \
                   (reward * discount * max(q[new_state,:]) -q[state,action])
```
So the change is represented by:
```
reward * discount * max(q[new_state,:]) -q[state,action]
```
The effect of the change is tempered by the *learning_rate* and is typically very small, e.g. 0.01 or 0.001.

The *DQN Target Formula* is defined as:
```
q[state, action] = reward if new_state is terminal else reward + discount * max(q[new_state,:])
```

# Calculating Loss
We are using the *Mean Square Error* (MSE) function to calculate loss.
```
loss = mse(current_q, target_q)
```
-where MSE is:

$$\frac{( C - T )^2} { 2 }$$


# Links
* [Implement Deep Q-Learning with PyTorch and Train Flappy Bird! DQN PyTorch](https://www.youtube.com/watch?v=arR7KzlYs4w&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi&index=1&ab_channel=JohnnyCode)
* [Flappy Bird Gymnasium on GitHub](https://github.com/markub3327/flappy-bird-gymnasium)
