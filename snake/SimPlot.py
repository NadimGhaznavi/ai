

import matplotlib.pyplot as plt
from IPython import display
from IPython.utils import io
import os, sys
from scipy.interpolate import make_interp_spline
import numpy as np

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)

class SimPlot():
  def __init__(self, ini, log, stats):
    self.ini = ini
    self.log = log
    self.stats = stats
    plt.ion()
    self.fig, self.axs = plt.subplots(2, 1, figsize=(12,8), layout="tight")
    self.fig.suptitle('AI Sim (v' + str(self.ini.get('sim_num')) + ')')
    self.log.log('SimPlot initialization:     [OK]')

  def __del__(self):
    self.save()
    plt.close()

  def plot(self):
    self.update()
    
    with io.capture_output() as captured:
      display.display(plt.gcf())
    # Plot horizontal lines to differentiate the ranges of scores into groups of 10
    # Only plot the horizontal lines if there is a score that is equal to or greater than 10
    for y in range(10, 100, 10):
      if max(self.scores) >= y:
        plt.axhline(y=y, color='r', linestyle=':')
    # Plot the scores and the mean scores
    #self.axs[0].set_ylim(ymin=0)
    self.axs[0].set_ylabel('Score')
    self.axs[0].set_xlabel('Number of games')
    self.axs[0].plot(self.games, self.scores, color='blue')
    self.axs[0].plot(self.games, self.mean_scores, color='green')
    # Create a bar chart of the scores
    self.axs[1].set_ylabel('Number of Times')
    self.axs[1].set_xlabel('Score')
    self.axs[1].bar(self.bar_scores, self.bar_count, color='green')

    plt.show()
    plt.pause(0.1)
    display.clear_output(wait=True)
    
    

  def update(self):
    games = []
    scores = []
    mean_scores = []
    count = 0
    for x in self.stats.get('scores', 'all'):
      games.append(count)
      scores.append(x)
      if count == 0:
        mean_scores.append(x)
      else:
        mean_scores.append(round((mean_scores[count - 1] * count + x) / (count + 1), 2))
      count += 1
    self.games = games
    self.scores = scores
    self.mean_scores = mean_scores

    bar_scores = []
    bar_count = []
    for x in range(0,max(self.scores) + 1):
      bar_scores.append(str(x))
      if self.stats.exists('scores', x):
        bar_count.append(self.stats.get('scores', x))
      else:
        bar_count.append(0)
    self.bar_scores = bar_scores
    self.bar_count = bar_count

  def plot2(self):
    display.clear_output(wait=True)
    with io.capture_output() as captured:
      display.display(plt.gcf())
    plt.clf()
    plt.title('AI Sim (v' + str(self.ini.get('sim_num')) + ')')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    games = []
    scores = []
    mean_scores = []
    count = 0
    for x in self.stats.get('scores', 'all'):
      games.append(count)
      scores.append(x)
      if count == 0:
        mean_scores.append(x)
      else:
        mean_scores.append(round((mean_scores[count - 1] * count + x) / (count + 1), 2))
      count += 1
    games = np.array(games)
    scores = np.array(scores)
    mean_scores = np.array(mean_scores)
    
    # Plot horizontal lines to differentiate the ranges of scores into groups of 10
    # Only plot the horizontal lines if there is a score that is equal to or greater than 10
    for y in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
      if max(scores) >= y:
        plt.axhline(y=y, color='r', linestyle=':')
    
    # Plot vertical lines to break the results into comparable sections
    if max(games) < 200:
      for y in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
        if max(games) >= y:
          plt.axvline(x=y, color='r', linestyle=':')
    elif max(games) > 200 and max(games) <= 400:
      for y in [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]:
        if max(games) >= y:
          plt.axvline(x=y, color='r', linestyle=':')
    elif max(games) > 400 and max(games) <= 800:
      for y in [80, 160, 240, 320, 400, 480, 560, 640, 720, 800]:
        if max(games) >= y:
          plt.axvline(x=y, color='r', linestyle=':')
    else:
      for y in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
                2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000]:
        if max(games) >= y:
          plt.axvline(x=y, color='r', linestyle=':')

    # Smooth the scores into a nice curve if we have more than 10
    if len(scores) > 10:
      X_Y_Spline = make_interp_spline(games, scores)
      X = np.linspace(games.min(), games.max(), 50*len(games))
      Y = X_Y_Spline(X)
      plt.plot(X, Y)
    else:
      plt.plot(scores)
      #plt.plot(xnew(np_times), ynew(np_times), 'ro', markersize=2)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show()
    plt.pause(0.1)
    display.clear_output(wait=True)  

  def save(self):
    ini = self.ini
    plot_basename = ini.get('plot_basename')
    plot_dir = ini.get('plot_dir')
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = os.path.join(plot_dir, str(ini.get('sim_num')) + plot_basename)
    plt.savefig(plot_file)

