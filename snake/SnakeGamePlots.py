

import matplotlib.pyplot as plt
from IPython import display
from IPython.utils import io

# Enable interactive mode
plt.ion()

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

def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
      ha="center", va="center", fontsize=fontsize, color="darkgrey")

def plot(scores, mean_scores, times, mean_times, ai_version):
  display.clear_output(wait=True)
  with io.capture_output() as captured:
    # Suppress the output to STDOUT in this context
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

