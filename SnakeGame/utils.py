import matplotlib.pyplot as plt
from IPython import display
import imageio
import os

plt.ion()

def plot_score(scores, mean_scores, gif_name='training_animation.gif'):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)

    # Save current plot as an image
    plt.savefig('current_plot.png')

    # Create GIF from the saved images
    if os.path.exists(gif_name):
        # If the GIF exists, append the new frame
        with imageio.get_writer(gif_name, mode='I', duration=0.5) as writer:
            writer.append_data(imageio.imread('current_plot.png'))
    else:
        # If the GIF doesn't exist, create it
        with imageio.get_writer(gif_name, mode='I', duration=0.5) as writer:
            writer.append_data(imageio.imread('current_plot.png'))

    # Remove the temporary image
    os.remove('current_plot.png')