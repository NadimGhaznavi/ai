

import matplotlib.pyplot as plt
from IPython import display
from IPython.utils import io
import os, sys

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
    plt.plot(scores)
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

