---
title: Arcade Learning Environment
---

# Table of Contents
* [Introduction and Scope](#introduction-and-scope)
* [Environment Setup](#environment-setup)
* [Links](#links)

# Introduction and Scope

The Arcade Learning Environment (ALE), commonly referred to as Atari, is a framework that allows researchers and hobbyists to develop AI agents for Atari 2600 ROMS. Its built on top of the Atari 2600 emulator Stella and separates the details of emulation from agent design.

# Environment Setup
# Links

# Environment Setup

The commands below show the creation of a Python virtual environment and the installation of various libraries, e.g. ale, matplotlib, tensorflow and torch libraries.
```
nadim@mypc:~$ python3 -m venv ai_ale
nadim@mypc:~$ . ai_ale/bin/activate
(ai_ale) nadim@mypc:~$ pip install torch --index-url https://download.pytorch.org/whl/cpu
(ai_ale) nadim@mypc:~$ pip install pyyaml
(ai_ale) nadim@mypc:~$ pip install ale-py
```

# Links

* [Gymnasium Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/)
* [ALE Installation](https://ale.farama.org/getting-started/)

## ALE Citations
In using the ALE, I'd like to cite the following two reference articles which were used to development the ALE framework that I'm using.
* M. G. Bellemare, Y. Naddaf, J. Veness and M. Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents, Journal of Artificial Intelligence Research, Volume 47, pages 253-279, 2013.
* M. C. Machado, M. G. Bellemare, E. Talvitie, J. Veness, M. J. Hausknecht, M. Bowling. Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents, Journal of Artificial Intelligence Research, Volume 61, pages 523-562, 2018.
