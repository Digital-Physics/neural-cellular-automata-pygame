# modified from original source below:
# https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/automata/model.py
import torch
import torch.nn as nn

class CAModel(nn.Module):
    """Cell automata model.
    Parameters
    ----------
    n_channels : int
        Number of channels of the grid.
    hidden_channels : int
        Hidden channels that are related to the pixel-wise 1x1 convolution.
    fire_rate : float
        Number between 0 and 1. The lower it is the more likely it is for
        cells to be set to zero during the `stochastic_update` process.
    device : torch.device
        Determines on what device we perform all the computations.
    Attributes
    ----------
    update_module : nn.Sequential
        The only part of the network containing trainable parameters. Composed
        of 1x1 convolution, ReLu and 1x1 convolution.
    filters : torch.Tensor
        Constant tensor of shape `(3 * n_channels, 1, 3, 3)`.
    """

    def __init__(self, n_channels=16, hidden_channels=128, fire_rate=0.5, device=None):
        super().__init__()

        self.fire_rate = 0.5
        self.n_channels = n_channels
        self.device = device or torch.device("cpu")

        # Perceive step
        sobel_filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0

        sobel_filter_x = sobel_filter_ / scalar
        sobel_filter_y = sobel_filter_.t() / scalar
        identity_filter = torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=torch.float32,
        )
        filters = torch.stack(
            [identity_filter, sobel_filter_x, sobel_filter_y]
        )  # (3, 3, 3)
        filters = filters.repeat((n_channels, 1, 1))  # (3 * n_channels, 3, 3)
        self.filters = filters[:, None, ...].to(
            self.device
        )  # (3 * n_channels, 1, 3, 3)

        # Update step
        self.update_module = nn.Sequential(
            nn.Conv2d(
                3 * n_channels,
                hidden_channels,
                kernel_size=1,  # (1, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                n_channels,
                kernel_size=1,
                bias=False,
            ),
        )

        with torch.no_grad():
            self.update_module[2].weight.zero_()

        self.to(self.device)

    def perceive(self, x):
        """Approximate channelwise gradient and combine with the input.
        This is the only place where we include information on the
        neighboring cells. However, we are not using any learnable
        parameters here.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, 3 * n_channels, grid_size, grid_size)`.
        """
        return nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels)

    def update(self, x):
        """Perform update.
        Note that this is the only part of the forward pass that uses
        trainable parameters
        Paramters
        ---------
        x : torch.Tensor
            Shape `(n_samples, 3 * n_channels, grid_size, grid_size)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        """
        # this is just a forward pass on a trained NCA model and we won't need gradients for backpropagation/learning anymore
        # added this .no_grad() line so we don't use up memory which was adding to the garbage cleanup time when exiting room, i think
        with torch.no_grad():
            return self.update_module(x)
        # return self.update_module(x)

    @staticmethod
    def stochastic_update(x, fire_rate):
        """Run pixel-wise dropout.
        Unlike dropout there is no scaling taking place.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        fire_rate : float
            Number between 0 and 1. The higher the more likely a given cell
            updates.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        """
        device = x.device

        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
        return x * mask  # broadcasted over all channels

    @staticmethod
    def get_living_mask(x):
        """Identify living cells.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, 1, grid_size, grid_size)` and the
            dtype is bool.
        """
        return (
                nn.functional.max_pool2d(
                    x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1
                )
                > 0.1
        )

    def forward(self, x):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_sample, n_channels, grid_size, grid_size)`.
        """
        pre_life_mask = self.get_living_mask(x)

        y = self.perceive(x)
        dx = self.update(y)
        dx = self.stochastic_update(dx, fire_rate=self.fire_rate)

        x = x + dx

        post_life_mask = self.get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)

        return x * life_mask


###################
# other non-nca model helper functions

def to_rgb(img_rgba):
    """Convert RGBA image to RGB image.
    Parameters
    ----------
    img_rgba : torch.Tensor
        4D tensor of shape `(1, 4, size, size)` where the RGB channels
        were already multiplied by the alpha.
    Returns
    -------
    img_rgb : torch.Tensor
        4D tensor of shape `(1, 3, size, size)`.
    """
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    # print("to_rgb|rgb, a:", rgb.shape, a.shape)
    # print("a at middle", a[0][0][32][32])
    # print("output", torch.clamp(1.0 - a + rgb, 0, 1).shape)
    return torch.clamp(1.0 - a + rgb, 0, 1)


def make_seed(size, n_channels, pos=None, tensor=None):
    """Create a starting tensor for initialization or mid-play spawning of new NCA pattern.
    The only active pixel is going to be the middle pixel.
    Parameters
    ----------
    size : int
        The height and the width of the tensor.
    n_channels : int
        Overall number of channels. Note that it needs to be higher than 4
        since the first 4 channels represent RGBA.
    pos : tuple[int]
        optional; Position i, j to update in the current tensor that should accompany this argument
    tensor:
        optional; Current tensor that is being updated in the NCA that we want to modify
    Returns
    -------
    torch.Tensor
        4D float tensor of shape `(1, n_channels, size, size)`.
    """
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    if not pos:
        x[:, 3:, size // 2, size // 2] = 1
        return x
    else:
        tensor[:, 3:, pos[1], pos[0]] = 1


def update_rgb_tensor(tensor, clear_key_on, i, j):
    """make channels of certain cells in the NCA tensor input off"""
    if clear_key_on:
        for range_i in range(3):
            for range_j in range(3):
                for k in range(16):
                    # experimented with rgb, and rgba = 1, settled on all channels = 0
                    tensor[0][k][range_j + j][range_i + i] = 0
    return tensor


def check_erased(tensor):
    if torch.count_nonzero(tensor[0, :3]) == 0:
        return True
    else:
        return False