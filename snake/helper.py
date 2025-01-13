

import matplotlib.pyplot as plt
from IPython import display

# Enable interactive mode
plt.ion()

def plot2(scores, mean_scores, times, mean_times, ai_version):
  #display.clear_output(wait=True)
  #display.display(plt.gcf())
  
  #fig, axes = plt.subplots(2,1)
  #axes[0].plot(scores)
  #axes[0].plot(mean_scores)
  #axes[1].plot(times)
  #axes[1].plot(mean_times)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  line1, = ax.plot(scores, 'b-')
  line1.set_ydata(scores)
  fig.canvas.draw()
  fig.canvas.flush_events()
  
  plt.clf()
  plt.show()
  plt.pause(0.1)

#def plot(scores, mean_scores, ai_version):
def plot(scores, mean_scores, times, mean_times, ai_version):
  display.clear_output(wait=True)
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

