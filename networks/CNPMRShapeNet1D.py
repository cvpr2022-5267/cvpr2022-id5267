from torch import nn

from networks.CNPMR import CNPMR


class CNPMRShapeNet1D(CNPMR):
    def __init__(self, config):
        super(CNPMRShapeNet1D, self).__init__(config)
        self.decoder0 = nn.Sequential(
            nn.Linear(self.dim_w + self.dim_z, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.y_dim),
            nn.Tanh(),
        )
