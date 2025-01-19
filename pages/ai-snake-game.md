---
title: AI Snake Game
---

## Table of Contents

* [Introduction and Scope](#introduction-and-scope)
* [My Motivation](#my-motivation)

## Introduction and Scope

This project is based on the classic *Snake Game* where the player uses the arrow keys to control a snake.

![AI Snake Game Screenshot](/assets/images/snake/ai_snake_game.png)

As the snake moves, the goal of the game is to maneuver the snake so that it *eats* the food. Every time
the snake eats a block of food it grows one segment. The game ends when the snake hits the edge of the 
screen or the food. The score corresponds to the number of food chunks the snake has eaten.

The AI Snake Game has an AI controlling the snake. At the beginning of the game the AI is pretty terrible, 
but after about twenty-five games it has a length of eight and after a couple hundred or more games it can achieve
scores of forty or more!

## My Motivation

It is clear to me that the *next new thing* is Artificial Intelligence. It's amazing how quickly it is
pervading society! AI assistants, AI generated content writing, images and film. Self driving cars,
movie recomendations and market analysis.

As a techie I wanted to learn how to create an AI and this project is my first foray into that arena.
I did this project as an educational exercise to teach myself some of the technology that is under the
hood.

## Technical Components

This project is written in Python. It uses the following components:

* [Python](https://python.org) - The Python programming language
* [Git](https://git-scm.com/) - Distributed version control system
* [PyGame](https://www.pygame.org/docs/) - Python library for creating graphical games and user interfaces
* [

## Environment Setup

I strongly recommend setting up a *virtual environment* to run and modify the *AI Snake Game*. You will need
Python installed in order to do this. By using a virtual environment you won't be altering the overall state
of your Python installation. This will avoid screwing up programs and components on your system that use
Python.

While you can easily run this on Windows, the instructions I have provided are for Linux as that's the system
I run.

The command below creates a virtual environment called *ai_dev* in your current working directory.

```
python3 -m venv ai_dev
```

Next, you'll need to copy the code from this GitHub repository to your computer:
```
git clone https://github.com/nadimghaznavi/ai.git
```

This will download my *ai* git repository onto your system in a directory called *ai*.

You will need to install the Python libraries that the code uses. Again, you'll want
to do so in your virtual development environment. The first step is to activate the
environment:
```
. ai_dev/bin/activate
```
Your prompt will change, indicating that you're now in the virtual environment:

```
nadim@mypc:~$ . ai_dev/bin/activate
(ai_dev) nadim@mypc:~$
```
Once you have activated the virtual environment you will need to install some Python
libraries. You can use *pip* to do so.
```
(ai_dev) nadim@mypc:~$ pip install matplotlib
(ai_dev) nadim@mypc:~$ pip install torch --index-url https://download.pytorch.org/whl/cpu
(ai_dev) nadim@mypc:~$ pip install pygame
(ai_dev) nadim@mypc:~$ pip install IPython
```






