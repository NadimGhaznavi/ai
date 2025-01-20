---
title: AI Snake Game
---

# Table of Contents
* [Introduction and Scope](#introduction-and-scope)
* [My Motivation](#my-motivation)
* [Goals](#goals)
* [Technical Components](#technical-components)
* [Environment Setup](#environment-setup)
* [Running the Snake Game](#running-the-snake-game)
* [Running the AI Snake Game](#running-the-ai-snake-game)
* [AI Snake Game Keyboard Shortcuts](#ai-snake-game-keyboard-shortcuts)
* [Codebase Architecture](#codebase-architecture)
* [Command Line Options](#command-line-options)
* [Matplotlib Game Score Plot](#matplotlib-game-score-plot)
* [Limitations and Lessons Learned](#limitations-and-lessons-learned)
* [Links](#Links)
* [Credits and Acknowledgements](#credits-and-acknowledgements)

# Introduction and Scope
This project is based on the classic *Snake Game* where the player uses the arrow keys to control a snake.

![AI Snake Game Screenshot](/assets/images/snake/ai_snake_game.png)

As the snake moves, the goal of the game is to maneuver the snake so that it *eats* the food. Every time the snake eats a block of food it grows one segment. The game ends when the snake hits the edge of the screen or the food. The score corresponds to the number of food chunks the snake has eaten.

The AI Snake Game has an AI controlling the snake. At the beginning of the game the AI is pretty terrible, but after about twenty-five games it has a length of eight and after a couple hundred or more games it can achieve
scores of forty or more!

Please note that I am still actively tinkering with this project, so the command line switches may be slightly different (usually more of them) than documented on this page.

# My Motivation
It is clear to me that the *next new thing* is Artificial Intelligence. It's amazing how quickly it is pervading society! AI assistants, AI generated content writing, images and film. Self driving cars, movie recomendations and market analysis.

As a techie I wanted to learn how to create an AI and this project is my first foray into that arena. I did this project as an educational exercise to teach myself some of the technology that is under the hood.

# Goals
I'm always trying to improve my coding skills. I know that parts of the code are pretty messy and could use a serious cleanup and re-write, but my goals do **not** include writing perfect code.

* Learn more Python
* Learn about PyTorch and ML
* Learn more about GitHub pages 
* See how different neural network shapes affect the AI agent's learning process
* See how different hyper-parameters affect the AI agent's learning process
* See how adding layers *on-the-fly* affects the performance of the AI

If you look at the [command line options](#command-line-options) section you will see that I have implemented switches to do the following:

* Set the number of nodes in the neural network
* Set the number of layers in the neural network
* Set three different layer sizes in the neural network
* Dynamically add a layer when the AI reaches a specific score

# Technical Components
This project is written in Python. It uses the following components:

* [Python](https://python.org) - The Python programming language
* [Git](https://git-scm.com/) - Distributed version control system
* [PyGame](https://www.pygame.org/docs/) - Python library for creating graphical games and user interfaces
* [PyTorch](https://pytorch.org/get-started/locally/) - Backend library for AI development
* [Matplotlib](https://matplotlib.org/) - Visualization library
* [IPython](https://ipython.org/) - Used to connect PyGame and Matplotlib

# Environment Setup
I strongly recommend setting up a *virtual environment* to run and modify the *AI Snake Game*. You will need Python installed in order to do this. By using a virtual environment you won't be altering the overall state of your Python installation. This will avoid screwing up programs and components on your system that use Python.

While you can easily run this on Windows, the instructions I have provided are for Linux as that is my preferred operating system. If you are running on Windows, you might want to follow the links in [technical components](#technical-components) section of this page and use information from those sites on setting up your environment. In particular, the [Pyorch](#https://pytorch.org/get-started/locally/) component.

The command below creates a virtual environment called *ai_dev* in your current working directory.

```
python3 -m venv ai_dev
```

Next, you'll need to copy the code from this GitHub repository to your computer:
```
git clone https://github.com/nadimghaznavi/ai.git
```

This will download my *ai* git repository onto your system in a directory called *ai*.

You will need to install the Python libraries that the code uses. Again, you'll want to do so in your virtual development environment. The first step is to activate the environment:
```
. ai_dev/bin/activate
```
Your prompt will change, indicating that you're now in the virtual environment:

```
nadim@mypc:~$ . ai_dev/bin/activate
(ai_dev) nadim@mypc:~$
```
Once you have activated the virtual environment you will need to install some Python libraries. You can use *pip* to do so.
```
(ai_dev) nadim@mypc:~$ pip install matplotlib
(ai_dev) nadim@mypc:~$ pip install torch --index-url https://download.pytorch.org/whl/cpu
(ai_dev) nadim@mypc:~$ pip install pygame
(ai_dev) nadim@mypc:~$ pip install IPython
```

# Running the Snake Game
To run the original snake game, first activate your virtual environment, then launch the game:
```
nadim@mypc:~$ . ai_dev/bin/activate
(ai_dev) nadim@mypc:~$ cd ai/snake
(ai_dev) nadim@mypc:~/ai/snake$ python SnakeGame.py
```
Then use the arrow keys to control the snake and go for the *food*. Have fun!!!!

# Running the AI Snake Game
To watch the AI play and learn the snake game just activate your virtual
environment, navigate to the snake directory and launch the *asg.py* front end:
```
nadim@mypc:~$ . ai_dev/bin/activate
(ai_dev) nadim@mypc:~$ cd ai/snake
(ai_dev) nadim@mypc:~/ai/snake$ python asg.py
```

# AI Snake Game Keyboard Shortcuts
I've coded in some additional keyboard shortcuts into the AI Snake game:

Key       | Description
----------|-------------
 m        | Print the neural network to console
 a        | Speed the game up
 z        | Slow the game down
 p        | Pause the game
 spacebar | Resume the game
 q        | Quit the game

# Codebase Architecture
The `asg.py` is the main front end to the AI Snake Game. It's the code that you need to execute in order to run a simulation. 

I have implemented a neural network architecture with three blocks; B1, B2, B3. By using the `--b1_nodes`, `--b1_layers`, `--b2_nodes` and so on, you can configure the number of nodes and layers in each block when you start the simulation.  I also implemented `--b1_score`, `--b2_score` and `--b3_score` switches to insert a B1, B2 or B3 layer *on-the-fly* when a particular score is reached.

I have refactored and extended the code a *LOT* from the original version that I started with. Here's a breakdown of the files and directories and what they are.

## AIAgent.py
This file houses the *AI Agent* or the AI player.

## AISnakeGameConfig.py
This file handles reading the AI Snake Game configuration settings from the *AISnakeGame.ini* file.

## AISnakeGame.ini
This file controls a lot of the settings for the AI simulation.

## AISnakeGame.py
This a modified version of the SnakeGame.py. It's been changed so that the AIAgent acts as the player instead of a human being.

## AISnakeGameUtils.py
This file has a few functions where I couldn't find a good place for them. 

## arial.ttf
The actual snake game and AI snake game uses this file to render the scores, moves and times shown at the top of the game screen.

## asg.py
This is the front end to the AI Snake game. It's the file you need to run to launch the AI Snake Game.

## batch_scripts
This directory has a couple of batch scripts I wrote to run the AI Snake Game in batch mode while changing one or more parameters. You can use this to run a bunch of simulations overnight and then look at the highscore files to see how different settings affect the performance of the AI.

I included these as examples. I encourage you to author your own and experiement!

## EpsilonAlgo.py
The epsilon algorithm is a standard algorithm used to inject a decreasing amount of randomness into the initial stages of the game. It implements the *exploration* part of *exploration/exploitation* in machine learning.

## LinearQNet.py
The LinearQNet class houses the PyTorch neural network model. that the AI Snake Game uses. I have extended this class pretty significantly. For example, I've added functions to add in new layers on-the-fly with differing numbers of nodes and load and save functions to take snapshots of the running simulations.

## next_ai_version.txt
This file holds a number that the code uses for the version of the simulation. It is incremented every time you run the `asg.py` front end.

## NuAlgo.py
This is a class I wrote to try and optimize the learning behaviour of the AI Agent. By tweaking this code and the settings in the AISnakeGame.ini I have managed to train the AI to reach a high score of 80!!

## QTrainer.py
This is part of the reinforcement learning that is used in this code to train the AI. It houses the *optimizer* that tweaks the neural network settings as the game runs.

## README.md
A standard GitHub README file. It points at this page.

## reference_sims
A few simulation files I saved, because they performed well. You can copy them into the *sim_data* directory and load them with the `-v` switch to run them.

## sim_data
This directory is automatically created when you run the asg.py script. Here are three example files that were created during a simulation run:

* 409_sim_checkpoint.ptc - A Simulation checkpoint file
* 409_sim_desc.txt - A file that contains some of the simulation settings
* 409_sim_highscore.csv - A CSV file with highscores from the simulation run

## SnakeGameElement.py
This contains some helper classes used by the AISnakeGame.

## SnakeGamePlots.py
This has the `plot()` function that launches the *matplotlib* pop-up window that graphs out the game scores in realtime as you run the simulation.

## SnakeGame.py
The original Snake Game that you can play.

# Command Line Options
I've implemented a lot of options to the `asg.py` front end:

```
usage: asg.py [-h] [-b1n B1_NODES] [-b1l B1_LAYERS] [-b1s B1_SCORE] [-b2n B2_NODES]
              [-b2l B2_LAYERS] [-b2s B2_SCORE] [-b3n B3_NODES] [-b3l B3_LAYERS]
              [-b3s B3_SCORE] [-e EPSILON] [-mg MAX_GAMES] [-ms MAX_SCORE]
              [-msn MAX_SCORE_NUM] [-nls NEW_LAYER_SCORE] [-nbg NU_BAD_GAMES]
              [-nmm NU_MAX_MOVES] [-nps NU_PRINT_STATS] [-ns NU_SCORE] [-nv NU_VALUE]
              [-nvm NU_VALUE_MAX] [-s SPEED] [-sd SIM_DATA_DIR] [-v AI_VERSION]
```

Here's a more detailed description of the options. You can see these by passing a `-h` to the `asg.py` script:
```
AI Snake Game

options:
  -h, --help            show this help message and exit
  -b1n B1_NODES, --b1_nodes B1_NODES
                        Number of nodes in the first block 1 layer.
  -b1l B1_LAYERS, --b1_layers B1_LAYERS
                        Number of hidden block 1 layers.
  -b1s B1_SCORE, --b1_score B1_SCORE
                        Insert a B1 layer when reaching this score.
  -b2n B2_NODES, --b2_nodes B2_NODES
                        Number of nodes in the hidden block 2 layer(s).
  -b2l B2_LAYERS, --b2_layers B2_LAYERS
                        Number of hidden block 2 layers.
  -b2s B2_SCORE, --b2_score B2_SCORE
                        Insert a B2 layer when reaching this score.
  -b3n B3_NODES, --b3_nodes B3_NODES
                        Number of nodes in the block 3 hidden layer(s).
  -b3l B3_LAYERS, --b3_layers B3_LAYERS
                        Number of block 3 hidden layers.
  -b3s B3_SCORE, --b3_score B3_SCORE
                        Insert a B3 layer when reaching this score.
  -e EPSILON, --epsilon EPSILON
                        Epsilon value for exploration.
  -mg MAX_GAMES, --max_games MAX_GAMES
                        Exit the simulation after max_games games.
  -ms MAX_SCORE, --max_score MAX_SCORE
                        Exit the simulation if a score of max_score is achieved.
  -msn MAX_SCORE_NUM, --max_score_num MAX_SCORE_NUM
                        Exit the simulation if a score of max_score is achieved max_num
                        times.
  -nls NEW_LAYER_SCORE, --new_layer_score NEW_LAYER_SCORE
                        Drop in a new layer at this score
  -nbg NU_BAD_GAMES, --nu_bad_games NU_BAD_GAMES
                        The number of games with no new high score.
  -nmm NU_MAX_MOVES, --nu_max_moves NU_MAX_MOVES
                        Maximum number of random moves injected by NuAlgo.
  -nps NU_PRINT_STATS, --nu_print_stats NU_PRINT_STATS
                        Print NuAlgo status information in the console.
  -ns NU_SCORE, --nu_score NU_SCORE
                        The nu algorithm is triggered when the score exceeds nu_score.
  -nv NU_VALUE, --nu_value NU_VALUE
                        The initial amount of randomness the nu algorithm injects.
  -nvm NU_VALUE_MAX, --nu_value_max NU_VALUE_MAX
                        Number of random moves to add to the nu pool if
                        nu_num_games_same_score_count_max is exceeded
  -s SPEED, --speed SPEED
                        Set the game speed.
  -sd SIM_DATA_DIR, --sim_data_dir SIM_DATA_DIR
                        Set a custom directory to store simulation results.
  -v AI_VERSION, --ai_version AI_VERSION
                        Load a previous simulation with version ai_version.
```

# Matplotlib Game Score Plot
The `asg.py` front end launches a matplatlib window that graphs out game score and average game score as the simulation runs.

![Screenshot of the Matplotlib Game Score Graph](/assets/images/snake/ai_metrics.png)

# Limitations and Lessons Learned
So what are the limits that this implementations is hitting? Well, the AI reaches what I call *level 1* where the length of the snake is over twice the width or height of the board. This is due to the strategy that the AI develops:

The strategy that the AI develops is to circle around the endge of the board. Once it is lined up with the food it moves through the center, eats the food and continues until it reaches the other side of the board. Then it circles again. This algorithm fails when the snake gets too long and it hits itself.

I found a paper, [Playing Atari with Deep Reinforcement Learning](#https://arxiv.org/pdf/1312.5602) which seems to identify and address this problem. Here's the first paragraph from the introduction:

*"Learning to control agents directly from high-dimensional sensory inputs like vision and speech is one of the long-standing challenges of reinforcement learning (RL). Most successful RL applications that operate on these domains have relied on hand-crafted features combined with linear value functions or policy representations. Clearly, the performance of such systems heavily relies on the quality of the feature representation."*

This describes the situation exactly. Here is what they developed:

*"We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards."*

I was quite surprised to discover that the AI Agent learns quickly and effectively with a very small and simple neural network. When I started I was using 512, 1024 and 2048 for the number of nodes and I was using multiple layers. What I found was that a single layer with only 80 nodes outperformed the simulations with large numbers of nodes and layers. It's truly impressive how effective even simple neural networks are at solving problems.

Food for thought!!! :)

# Links
* [This page](https://ai.osoyalce.com/pages/ai-snake-game.html)
* [My code on GitHub](https://github.com/NadimGhaznavi/ai)
* [AI Snake Game Tutorial on YouTube](https://ai.osoyalce.com/pages/ai-snake-game.html#limitations)
* [Patrick Loeber's code on GitHub](https://github.com/patrickloeber/snake-ai-pytorch)

# Credits and Acknowledgements

This code is based on a YouTube tutorial [Python + PyTorch + Pygame Reinforcement Learning – Train an AI to Play Snake](https://www.youtube.com/watch?v=L8ypSXwyBds&t=1042s&ab_channel=freeCodeCamp.org) by Patrick Loeber. You can access his original code [here](https://github.com/patrickloeber/snake-ai-pytorch) on GitHub.

Thank you Patrick!!! You are amazing!!!! :)



