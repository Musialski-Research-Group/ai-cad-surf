from torch import nn
from .activations import AbsLayer

class Decoder(nn.Module):
    def __init__(self, udf=False):
        super(Decoder, self).__init__()
        # TODO: check if this matches in configs with the implementation for DiGS
        self.nl = nn.Identity() if not udf else AbsLayer()

    def forward(self, *args, **kwargs):
        res = self.fc_block(*args, **kwargs)
        res = self.nl(res)
        return res
