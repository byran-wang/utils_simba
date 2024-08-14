import torch
import matplotlib.pyplot as plt

# image shape should be [height, width, channels] and in range [0, 1]
def show_img(img, normalize=False, title=None, off_axis=True):
    rgb_array = img
    if type(img) == torch.Tensor:
        rgb_tensor = img.detach().cpu()
        rgb_array = rgb_tensor.numpy()
        
    plt.imshow(rgb_array)
    if title:
        plt.title(title)
    if off_axis:
        plt.axis('off')  # Turn off axis labels
        plt.gca().set_axis_off()  # Turn off the axis
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # Remove padding
        plt.margins(0, 0)  # Set margins to zero
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Remove x-axis ticks
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Remove y-axis ticks
    plt.show()