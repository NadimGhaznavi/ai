---
title: AI Snake Game
---

# Table of Contents
* [Introduction and Scope](#introduction-and-scope)
* [My Motivation](#my-motivation)
* [Goals](#goals)
* [Technical Components](#technical-components)
* [Configuration File Management](#configuration-file-management)
* [Environment Setup](#environment-setup)
* [Running the Snake Game](#running-the-snake-game)
* [Running the AI Snake Game](#running-the-ai-snake-game)
* [Keyboard Shortcuts](#keyboard-shortcuts)
* [The Reward System](#the-reward-system)
* [Codebase Architectures](#codebase-architectures)
* [Neural Network Architectures](#neural-network-architectures)
  * [Failed Architectures - L2](#failed-architectures-l2)
  * [Failed Architectures - L10](#failed-architectures-l10)
  * [Failed Architectures - LX](#failed-architectures-lx)
  * [Failed Architectures - Evolutionary Networks](#failed-architectures-evolutionary-networks)
  * [Failed Architectures - Adding Layers on the Fly](#failed-architectures-adding-layers-on-the-fly)
  * [Failed Architectures - Adding Dropout Layers on the Fly](#failed-architectures-adding-dropout-layers-on-the-fly)
* [Codebase Components](#codebase-components)
* [Game Challenges](#game-challenges)
* [Command Line Options](#command-line-options)
* [Using Batch Scripts to Fine Tune Hyperparameters](#using-batch-scripts-to-fine-tune-hyperparameters)
* [Matplotlib Game Score Plot](#matplotlib-game-score-plot)
* [Matplotlib CNN Feature Maps](#matplotlib-cnn-feature-maps)
* [Highscore Files](#highscore-files)
* [Limitations and Lessons Learned](#limitations-and-lessons-learned)
* [Links](#Links)
* [Credits and Acknowledgements](#credits-and-acknowledgements)

# Introduction and Scope
This project is based on the classic *Snake Game* where the player uses the arrow keys to control a snake.

![AI Snake Game Screenshot](/assets/images/snake/ai_snake_game.png)

As the snake moves, the goal of the game is to maneuver the snake so that it *eats* the food. Every time the snake eats a block of food it grows one segment. The game ends when the snake hits the edge of the screen or the food. The score corresponds to the number of food chunks the snake has eaten.

The AI Snake Game has an AI controlling the snake. At the beginning of the game the AI is pretty terrible, but after about twenty-five games it has a length of eight and after a couple hundred or more games it can achieve scores of forty or more!

# My Motivation
It is clear to me that the *next new thing* is Artificial Intelligence. It's amazing how quickly it is pervading society! AI assistants, AI generated content writing, images and film. Self driving cars, movie recomendations and market analysis.

As a techie I wanted to learn how to create an AI and this project is my first foray into that arena. I did this project as an educational exercise to teach myself some of the technology that is under the hood.

# Goals
I'm always trying to improve my coding skills. I know that parts of the code are pretty messy and could use a serious cleanup and re-write, but my goals do **not** include writing perfect code.

* Learn more Python
* Learn about PyTorch and ML
* Learn more about GitHub pages 
* See how different neural networks affect the AI agent's learning process
* See how different hyper-parameters affect the AI agent's learning process
* See how adding layers *on-the-fly* affects the performance of the AI
* See how adding dropout layers affected the performance of the AI
* See how activating dropout layers on the fly affected the performance of the AI
* **Try to get an amazing highscore** :)

# Technical Components
This project is written in Python. It uses the following components:

* [Python](https://python.org) - The Python programming language
* [Git](https://git-scm.com/) - Distributed version control system, not used in the game, but used to get the game
* [PyGame](https://www.pygame.org/docs/) - Python library for creating graphical games and user interfaces
* [PyTorch](https://pytorch.org/get-started/locally/) - Backend library for AI development
* [Matplotlib](https://matplotlib.org/) - Visualization library
* [IPython](https://ipython.org/) - Used to connect PyGame and Matplotlib

# Configuration File Management

**IMPORTANT NOTE:** If you decide to download, run and tinker with this project I want to share a hard-learned lesson: **Neural networks are EXTREMELY sensitive to hyperparameters and the neural network architecture.** For this reason it's critical that you manage the YAML file with extreme care. A small change in configuration can make your AI go from a super star to a dismal failure. Do not modify the AISim.yml file, make a copy and then use the `--in` switch to point the `AISim.py` front-end at your custom configuration.

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
(ai_dev) nadim@mypc:~$ pip install PyYAML
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
environment, navigate to the snake directory and launch the *AISim.py* front end:
```
nadim@mypc:~$ . ai_dev/bin/activate
(ai_dev) nadim@mypc:~$ cd ai/snake
(ai_dev) nadim@mypc:~/ai/snake$ python AISim.py
```

# Keyboard Shortcuts
I've coded in some additional keyboard shortcuts into the AI Snake game:

Key       | Description
----------|-------------
 m        | Print the neural network to console
 a        | Speed the game up
 z        | Slow the game down
 h        | Stop displaying the game, while letting it continue to run
 i        | Resume displaying the game
 m        | Print the runtime neural network architecture
 p        | Pause the game
 d        | Resume the game, but don't update the game display
 u        | Resume the game and resume updating the game display
 spacebar | Resume the game
 q        | Quit the game

# The Reward System
These are the *rewards* in the AI Snake Game:

Reward Value | Reward Description
-------------|---------------------
 -10         | The snake hit the edge of the screen
 -10         | Exceeded the maximum number of moves
 -10+        | -10 to 0.2 * length of the snake
 +10         | Snake got a piece of food
 +1          | Moving towards the food
 -1          | Moving away from the food

These values are set in the configuration YAML file.

The *maximum number of moves* is also configured in the YAML file. The value in the YAML file is multiplied by 0.2 times the length of the snake. This is to discourage self collisions at higher scores.

At the end of each move, the AI Snake Game simulation calls the AI Agent's `train_short_memory()` function. The **only** thing this function does is call the *QTrainer's* `train_step()` function, where the weights and bias' are rebalanced.

# Codebase Architecture
The `AISim.py` is the main front end to the AI Snake Game. It's the code that you need to execute in order to run a simulation. 

I initially implemented a neural network architecture with three blocks; B1, B2, B3. By using the `--b1_nodes`, `--b1_layers`, `--b2_nodes` and so on, you can configure the number of nodes and layers in each block when you start the simulation. I have refactored the code a number of times since. Now most settings are configured in a YAML file.

After a lot of re-design and re-factoring of the code, I have settled on the following architecture for the AI Agent: The initial neural network architecture is selected with the ```-mo MODEL_TYPE```. Valid options are ```linear, rnn, cnn, cnnr``` and a few more that I'm working on. This is a moving target as I learn more about ML and implement additional architectures. 

Switch | Description
-------|--------------------------
 NONE  | The default is a simple linear network
 rnn   | A *recurrent neural network* which uses a custom feature map as an input
 cnn   | A *convolutional neural network* that uses a pixmap as the state input
 cnnr  | A *confolutional neural network / Long-Short-Term-Memory* hybrid model which also uses the pixmap as the state input

## Failed Architectures L2

In my quest to improve the AI Agent's ability to play the Snake Game, I have implemented what I called a *Level 2* neural network. This is basically a second, independent neural network that is only fed data when the game score in the level two range ie. scores from 41 on.

This approach was somewhat successful, but achieving scores above 50 proved difficult and finding the magic hyperparameters for the NuAlgo and Epsilon required a ton of simulation runs. In the end I abandoned this approach.

## Failed Architectures L10
* The *L10* level also includes a *ReplayMemory*, *Trainer* and *Epsilon Greedy* instances.
* This *L10* component handles Snake Game scores up to 10. When the AI achieves a score between 11 and 20 a new component, *L20* is created with it's own *ReplayMemory*, *Trainer* and *Epsilon Greedy* instances. The neural network architecture is the same as the *L10* component with respect to node numbers, layers, dropout layers etc.
* When the AI achieves a score of 20 a *L30* component is created... and so on, indefinitely.
* When a new *L* component is created, the weights and biases from the lower layer are copied in so it has the experience and the training from the lower neural network. After that, it operates independently.
* I disable the *Epsilon Greedy* instance by default and use my own *NuAlgo* instead
* Like the *Epsilon Greedy* object, the *NuAlgo* object inserts random moves into the game as part of the AI's *exploration/exploitation* training process. But unlike the Epsilon Greedy, the NuAlgo is dynamic: It injects random moves based on the AI's (poor) performance.

## Failed Architectures LX
The *LX* codebase was me taking the *L10* architecture further. Instead of having a neural network dedicated to a range of scores, *LX* spun up a neural network for each score above a theshold (e.g. 3). My rationale for this design was that a network dedicated to playing the game at a particular score would get better and better at playing the game at that specific score. One cool thing was how control was passed smoothly from one level to the next, but -overall- the performance was worse than having a single network for the entire game regardless of the score.

## Failed Architeures Evolutionary Networks
This was a variation on *LX* where, if (for example) the L7  neural network, which is dedicated to playing the game at score 7 fails a number of times. Then the L6 network is cloned and replaces L7. My thinking was that this would leverage the additional training that is received by lower level networks. Let me explain that further. If you have a codebase that has a neural network dedicated to playing at each score, then the L0 network will get the most training since every game starts with a score of 0, so the L0 is trained for every game. Whereas, the L10 network only gets trained when the score is 10. This didn't perform well either.

After some though and consideration, I believe that what I was basically doing was using the L0 network to play the entire game, because it would get promoted to L1. Then eventually it's promoted to L2, L3 etc.

## Failed Architectures Adding Layers On The Fly

I was curious about the effect of adding layers on the fly. You may want to experiment with this option using the `--b1_score`, `--b2_score` and `--b3_score` which are features that drop in a B1, B2 or B3 layer when the AI reaches a particular score. What I learned was that adding layers on-the-fly disrupts the performance of the AI (no big surprise). When adding a B1 layer i.e. one that matches the shape of the existing B1 layer is much less disruptive: The AI recovers relatively quickly and carries on. Adding a new B2 layer i.e. when you didn't have any B2 layers and the shape is different than the B1 layer is **extremely** disruptive to the perfomance of the neural network.

These features weren't helpful in achieving higher scores, so I removed the functionality.

## Failed Architecturs Adding Dropout Layers On The Fly

I implemented a `--dropout-static` switch that instructs the `AISim.py` to create PyTorch `nn.Dropout` layers with a *P Value* that is passed in with an argument to the `--dropout-static` switch. The code took care of inserting these *dropout layers* are in between the hidden B1, B2 and B3 hidden layers.

I implemented this feature to see if adding additional noise to the simulation would stop the AI from getting stuck in sub-optimal game strategy. It's stuck now: When the snake reaches a length that is more than twice the width of the board (I'm using a 20x20 board), then there is an added challenge. With my current setup, the AI can achieve scores of up to around 50, but not really any higher. At that point in the game, the AI has settled into a strategy of moving the snake around the edge of the screen and then cutting through the middle to get the food. It continues to the other edge and then circles again. While this strategy is good for scores up to 40, it fails to reach scores in the 60s because it ends up hitting itself.

# Codebase Components
Here's a breakdown of the files and directories and what they are.

## AIAgent.py
This file houses the *AI Agent* or the AI player.

## AiSim.py
This is the front end to the AI Snake game. It's the file you need to run to launch the AI Snake Game.

## AISnakeGameConfig.py
This file handles reading the AI Snake Game configuration settings from the *AISnakeGame.ini* file.

## AISim.yml
This file controls the settings for the AI simulation.

## AISnakeGame.py
This a modified version of the SnakeGame.py. It's been changed so that the AIAgent acts as the player instead of a human being.

## arial.ttf
The actual snake game and AI snake game uses this file to render the scores, moves and times shown at the top of the game screen.

## batch_scripts
This directory has a couple of batch scripts I wrote to run the AI Snake Game in batch mode while changing one or more parameters. You can use this to run a bunch of simulations overnight and then look at the highscore files to see how different settings affect the performance of the AI.

For example a [batch script](#using-batch-scripts-to-fine-tune-hyperparameters) that is running simulations with varying learning rates to see if tweaking that value will help for a given model.

I included these as examples. I encourage you to author your own and experiement!

## EpsilonAlgo.py
The epsilon algorithm is a standard algorithm used to inject a decreasing amount of randomness into the initial stages of the game. It implements the *exploration* part of *exploration/exploitation* in machine learning.

## next_ai_version.txt
This file holds a number that the code uses for the version of the simulation. It is incremented every time you run the `AISim.py` front end.

## NuAlgo.py
This is a class I wrote to try and optimize the learning behaviour of the AI Agent. By tweaking this code and the settings in the AISnakeGame.ini I have managed to train the AI to reach a high score of 80!! 

However, tuning the code and NuAlgo settings is very, very domain specific, so I have dropped this module and am no longer using the code.

## QTrainer.py
This is part of the reinforcement learning that is used in this code to train the AI. It houses the *optimizer* that tweaks the neural network settings as the game runs.

## README.md
A standard GitHub README file. It points at this page.

## ReplayMemory
This class implements a replay memory function, essentially a Python deque to store a fixed number of states which are used for training.

## sim_data
This directory is automatically created when you run the AISim.py script. Here are three example files that were created during a simulation run:

* 409_sim_checkpoint.ptc - A Simulation checkpoint file
* 409_sim_desc.txt - A file that contains some of the simulation settings
* 409_sim_highscore.csv - A CSV file with highscores from the simulation run

## sim_plots
This directory is also automatically created. The matplotlib figure for a simulation run is saved to this directory when the simulation ends. Having all of the plots in the same directory makes comparisons quick and easy.

## SnakeGameElement.py
This contains some helper classes used by the AISnakeGame.

## SimPlots.py
This has the `plot()` function that launches the *matplotlib* pop-up window that graphs out the game scores, mean score, score distribution, loss, loss reason and a nice pixmap of the state. This figure is updated in realtime as you run the simulation.

## SnakeGame.py
The original Snake Game that you can play.

# Game Challenges

The Snake Game becomes significantly more challenging when the Snake's length is more than twice the width of the board. At that point the strategy needed to continue to improve the scores doesn't just rely on finding the food, it also includes the challenge of moving in a manner such that collisions with the Snake itself are avoided. 

A very basic and simple neural network architecture consisting of one layer with 100 nodes is easily able to achieve scores in the low 40s on a 20x20 board. Reaching a highscore of 60 is almost impossible with this architecture.

# Command Line Options
I've implemented quite a few options to the `AISim.py` front end:

```
usage: AISim.py [-h] [-bs BLOCK_SIZE] [-ep EPSILON] [-in INI_FILE] [-ma MAX_EPOCHS]
                [-mo MODEL] [-ne NU_ENABLED] [-nu NU_MAX_EPOCHS] [-re RESTART] [-sp SPEED]
```
Running the `AISim.py` frontend with the `-h` switch provides a more detailed description of these options. **IMPORTANT NOTE:** I am still tinkering with this project, so like they say at Microsoft, *the implementation is the specification*...

# Using Batch Scripts to Fine Tune Hyperparameters

Fine tuning the hyperparameters and neural network architecture are key elements in finding successful neural network configurations. Scripting these *explorations* is a systematic and efficient way to execute this process. Here is an example of batch script I am using that does 10 simulation runs that vary the *learning rate* of the configuration from 0.0005 to 0.0015:
```
#!/bin/bash
#
CONFIG=configs/B1_100_nodes-B2_200_nodes-B3_400_nodes-dropout_0.2.ini

MAX_GAMES=800

# Learnign rate, testing from 0.0005 to 0.0015 and skipping 0.001,
# because I already have that simulation result.
COUNT=5

while [ $COUNT != 10 ]; do
	LR=0.000${COUNT}
	python AISim.py \
		-i $CONFIG \
		--max_games $MAX_GAMES \
		-l $LR 
	COUNT=$((COUNT+1))
done
COUNT=1
while [ $COUNT != 6 ]; do
	LR=0.001${COUNT}
	python AISim.py \
		-i $CONFIG \
		--max_games $MAX_GAMES \
		-l $LR 
	COUNT=$((COUNT+1))
done
```
# Highscore Files
The simulation runs produce a CSV high scores file which allows me to easily analyse the results of the batch runs:
```
$ echo; for x in $(ls *high*); do echo $x; cat $x; echo; done

800_sim_highscore.csv
Game Number,High Score
0,0
9,1
120,2
265,5

801_sim_highscore.csv
Game Number,High Score
0,0
15,1
171,2
247,3
268,4
292,5

802_sim_highscore.csv
Game Number,High Score
0,0
11,1
90,2
  .
  .
  .
```

# Matplotlib Game Score Plot
The `AISim.py` front end launches a matplatlib window that graphs out game score and average game score as the simulation runs. It also shows a bitmap representation of the board that the game maintains internally (and provides to the CNN and CNNR models as a state map). The reason the game was lost is either  because the snake hit the wall or the edge of the board, because the snake collided with itself or because the maximum number of moves was exceeded. These metrics are plotted as well in the bottom right bar chart. The distribution of scores is also shown in the bottom right bar chart.

![Screenshot of the Matplotlib Game Score Graph](/assets/images/snake/ai_metrics.png)

# Matplotlib CNN Feature Maps
If you use the ```cnnr``` model type, the code will also create and update a grid of images that display the CNN feature maps. I found this useful when tuning the CNN hyperparameters. 

Obviously the way **I** interpret these images is totally different than how the AI uses that data. So these images are mostly useful in identifying CNN hyperparameters which are bad. For example blank feature maps or feature maps with visual elements that don't correspond to anything in the image.

![Screenshot of the Matplotlib CNN Feature Maps](/assets/images/snake/cnn_feature_maps.png)

# Limitations and Lessons Learned
So what are the limits that this implementations is hitting? Well, the AI reaches what I call *level 1* where the length of the snake is over twice the width or height of the board. This is due to the strategy that the AI develops:

The strategy that the AI develops is to circle around the endge of the board. Once it is lined up with the food it moves through the center, eats the food and continues until it reaches the other side of the board. Then it circles again. This algorithm fails when the snake gets too long and it hits itself.

Modifying the reward system to include a small reward for moving towards the food and moving away from the food broke this pattern.

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



