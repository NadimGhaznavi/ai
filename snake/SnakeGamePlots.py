

import matplotlib.pyplot as plt
from IPython import display
from IPython.utils import io
import os, sys
from scipy.interpolate import make_interp_spline
import numpy as np

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig

class MyPlot():
  def __init__(self, ini):
    self.ini = ini
    self.plt = plt
    self.display = display
    self.plt.ion()
    self.plt.figure(figsize=(12,4), layout="tight")
    self.plt.title('Snake AI Training (v' + str(self.ini.get('ai_version')) + ')')
        
  def plot(self, scores, mean_scores, times, mean_times):
    display.clear_output(wait=True)
    with io.capture_output() as captured:
      display.display(plt.gcf())
    plt.clf()
    plt.title('Snake AI Training (v' + str(self.ini.get('ai_version')) + ')')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    # Plot horizontal lines to differentiate the ranges of scores into groups of 10
    # Only plot the horizontal lines if there is a score that is equal to or greater than 10
    for y in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
      if max(scores) > y:
        plt.axhline(y=y, color='r', linestyle=':')
    
    # Plot vertical lines to break the results into comparable sections
    if max(scores) > 100 and max(scores) <= 200:
      for y in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
        if max(scores) >= y:
          plt.axvline(x=y, color='r', linestyle=':')
    elif max(scores) > 200 and max(scores) <= 400:
      for y in [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]:
        if max(scores) >= y:
          plt.axvline(x=y, color='r', linestyle=':')
    elif max(scores) > 400 and max(scores) <= 800:
      for y in [80, 160, 240, 320, 400, 480, 560, 640, 720, 800]:
        if max(scores) >= y:
          plt.axvline(x=y, color='r', linestyle=':')
    else:
      for y in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
                2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000]:
        if max(scores) >= y:
          plt.axvline(x=y, color='r', linestyle=':')

    # Smooth the scores into a nice curve if we have more than 10
    if len(scores) > 10:
      np_scores = np.array(scores)
      
      # Create a games array to represent the game number
      games = []
      game_num = 1
      for x in scores:
        games.append(game_num)
        game_num += 1
      np_games =np.array(games)

      X_Y_Spline = make_interp_spline(np_games, np_scores)
      X = np.linspace(np_games.min(), np_games.max(), 50*len(games))
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
    plot_basename = ini.get('sim_plot_basename')
    data_dir = ini.get('sim_data_dir')
    plot_file = os.path.join(data_dir, str(ini.get('ai_version')) + plot_basename)
    plt.savefig(plot_file)

