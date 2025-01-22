

import matplotlib.pyplot as plt
from IPython import display
from IPython.utils import io
import os, sys

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig

class MyPlot():
  def __init__(self, ai_version):
    ini = AISnakeGameConfig()
    self.sim_plot_basename = ini.get('sim_plot_basename')
    self.sim_data_dir = ini.get('sim_data_dir')
    self.sim_plot_file = str(ai_version) + self.sim_plot_basename
    self.sim_plot_file = os.path.join(self.sim_data_dir, self.sim_plot_file)
    plt.ion() # turn on interactive mode

  def annotate_axes(ax, text, fontsize=18):
      ax.text(0.5, 0.5, text, transform=ax.transAxes,
        ha="center", va="center", fontsize=fontsize, color="darkgrey")

  def plot(self, scores, mean_scores, times, mean_times, ai_version):
    display.clear_output(wait=True)
    with io.capture_output() as captured:
      display.display(plt.gcf())
    plt.clf()
    plt.title('Snake AI Training (v' + str(ai_version) + ')')
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

  def plot2(fig, spec, scores, mean_scores, times, mean_times, ai_version):
    plt.clf()
    plt.xlabel('Number of Games')
    fig.suptitle('Snake AI Training (v' + str(ai_version) + ') Metrics')
    ax0 = fig.add_subplot(spec[0])
    annotate_axes(ax0, 'Game Times')
    ax1 = fig.add_subplot(spec[1])
    annotate_axes(ax1, 'Game Scores')
    ax0.plot(scores, 'b-')
    ax1.plot(times, 'b-')
    plt.show()
    plt.pause(0.1)

  def save(self):
    os.makedirs(self.sim_data_dir, exist_ok=True)
    plt.savefig(self.sim_plot_file)


