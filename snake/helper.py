import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, game_times, model_version):
  display.clear_output(wait=True)
  display.display(plt.gcf())
  plt.clf()
  plt.title('Training v' + str(model_version))
  plt.xlabel('Number of games')
  plt.ylabel('Score')
  plt.plot(scores)
  plt.plot(mean_scores)
  plt.ylim(ymin=0)
  plt.text(len(scores)-1, scores[-1], str(scores[-1]))
  plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

  #fig, ax1 = plt.subplots()
  #ax2 = ax1.twinx()
  #color = 'tab:green'
  #ax2.plot(game_times, color=color)
  #fig.tight_layout()

  plt.show()
  plt.pause(0.1)
