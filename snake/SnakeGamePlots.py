

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
    # Generate a new matplotlib figure and spec
    self.plt = plt
    self.display = display
    self.plt.ion()
    
    self.plt.figure(figsize=(12,4), layout="tight")
    #self.fig = plt.figure(1)
    #self.fig, self.axis = plt.subplots(2, 1)
    self.plt.title('Snake AI Training (v' + str(ai_version) + ')')
    #self.fig, axs = self.plt.subplots(2, 1)
    #self.fig.tight_layout()
    #self.fig.set_size_inches(12, 8)
    #self.ax1 = axs[0]
    #self.ax2 = axs[1]
    
  def annotate_axes(ax, text, fontsize=18):
      ax.text(0.5, 0.5, text, transform=ax.transAxes,
        ha="center", va="center", fontsize=fontsize, color="darkgrey")

  def plot2(self, scores, mean_scores, times, mean_times, ai_version):
    self.display.clear_output(wait=True)
    with io.capture_output() as captured:
      self.display.display(self.plt.gcf())
    
    self.fig.clear()
    

    ax1 = self.ax1
    ax1.relim()
    ax1.autoscale_view()
    ax1.set_xlabel('Number of games')
    ax1.set_ylabel('Score')
    ax1.plot(scores)
    ax1.plot(mean_scores)  
    ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax1.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    ax2 = self.ax2
    ax2.relim()
    ax2.autoscale_view()
    ax2.set_xlabel('Number of games')
    ax2.set_ylabel('Time (s)')
    ax2.plot(times)
    ax2.plot(mean_times)
    ax2.text(len(times)-1, times[-1], str(times[-1]))
    ax2.text(len(mean_times)-1, mean_times[-1], str(mean_times[-1]))

    self.plt.show()
    self.plt.pause(0.1)
    self.display.clear_output(wait=True)  

    #self.plt.subplots_adjust(wspace=1.0, hspace=1.0)
    
    
  def save(self):
    os.makedirs(self.sim_data_dir, exist_ok=True)
    plt.savefig(self.sim_plot_file)


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
