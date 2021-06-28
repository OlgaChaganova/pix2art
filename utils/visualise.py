import numpy as np
from matplotlib import colors, pyplot as plt

def imshow_tensor(inp, title=None, plt_ax=plt, default=False):
    """Imshow for tensors"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


def show_images(edges, true, generated, n_images):
  fig, axs = plt.subplots(n_images, 3)
  plt.subplots_adjust(left=0, right=1.5, wspace=0.1, hspace=0.1, bottom=0, top=2)
  for i in range(n_images):
    imshow_tensor(edges.detach()[i], plt_ax=axs[i, 0])
    axs[i, 0].set_title('Edges')
    axs[i, 0].axis('off')

    imshow_tensor(true.detach()[i], plt_ax=axs[i, 1])
    axs[i, 1].set_title('Ground truth')
    axs[i, 1].axis('off')

    imshow_tensor(generated.detach()[i], plt_ax=axs[i, 2])
    axs[i, 2].set_title('Generated image')
    axs[i, 2].axis('off')
