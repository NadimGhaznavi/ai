

import matplotlib.pyplot as plt
from IPython import display
from IPython.utils import io
import os, sys

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)

class SimPlot():
  def __init__(self, ini, log, stats):
    self.ini = ini
    self.log = log
    self.stats = stats
    plt.ion()
    self.fig, self.axs = plt.subplots(2, 1, figsize=(12,6), layout="tight", facecolor="#000000")
    self.fig.suptitle('AI Sim (v' + str(self.ini.get('sim_num')) + ')', color="#00FF00")
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
        self.axs[0].axhline(y=y, color='r', linestyle=(0, (1, 10)), linewidth=1)

    self.axs[0].set_facecolor('#002000')
    self.axs[1].set_facecolor('#002000')
    self.axs[0].tick_params(labelcolor='#00ff00')
    self.axs[1].tick_params(labelcolor='#00ff00')

    # Plot the scores and the mean scores
    #self.axs[0].set_ylim(ymin=0)
    self.axs[0].set_ylabel('Score', color='#00ff00')
    self.axs[0].set_xlabel('Number of Games', color='#00ff00')
    self.axs[0].plot(self.games, self.scores, color='blue', linewidth=1)
    self.axs[0].plot(self.games, self.mean_scores, color='#cccc00', linewidth=1)
    # Create a bar chart of the scores
    self.axs[1].set_ylabel('Score Count', color='#00ff00')
    self.axs[1].set_xlabel('Score', color='#00ff00')
    self.axs[1].bar(self.bar_scores, self.bar_count, color='#ff8f00')
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


  def save(self):
    ini = self.ini
    plot_basename = ini.get('plot_basename')
    plot_dir = ini.get('plot_dir')
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = os.path.join(plot_dir, str(ini.get('sim_num')) + plot_basename)
    plt.savefig(plot_file)

