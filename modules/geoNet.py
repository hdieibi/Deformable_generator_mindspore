import mindspore.nn as nn
from mindspore.ops import permute, reshape
from mindspore import ops


class GeoNet(nn.Cell):
    def __init__(self, zg_dim=64, nchan=3, imgsize=64, batchsize=9, kersize=[3,3,5,5], chb=16, fmd=[4,8,16,32,64], stride=[2, 2, 2, 2]):
        super(GeoNet, self).__init__()
        self.zg_dim = zg_dim
        self.nchan = nchan
        self.imgsize = imgsize
        self.bs = batchsize
        self.kersize = kersize
        self.chb = chb
        self.channelg = [chb * 8, chb * 4, chb * 2, chb * 1]
        self.fmd = fmd
        self.name = 'gengeo'
        self.base_geo = self.init_geo()
        self.dense = nn.SequentialCell(
            nn.Dense(self.zg_dim, self.fmd[0] * self.fmd[0] * self.channelg[0]),
            nn.ReLU()
        )
        self.deconv2d = nn.SequentialCell(
            nn.Conv2dTranspose(self.channelg[0], self.channelg[1], kersize[0], stride[0]),
            nn.ReLU(),
            nn.Conv2dTranspose(self.channelg[1], self.channelg[2], kersize[1], stride[1]),
            nn.ReLU(),
            nn.Conv2dTranspose(self.channelg[2], self.channelg[3], kersize[2], stride[2]),
            nn.ReLU(),
            nn.Conv2dTranspose(self.channelg[3], 2, kersize[3], stride[3]),
            nn.Tanh(),
            # nn.Upsample(size=(self.imgsize, self.imgsize), mode='bilinear'),
        )

    def init_geo(self):
        x = ops.linspace(-1, 1, self.imgsize)
        y = ops.linspace(-1, 1, self.imgsize)
        xx, yy = ops.meshgrid(x, y, indexing='ij')
        grid = ops.stack([yy, xx], axis=0)
        grid = ops.unsqueeze(grid, dim=0)
        return grid

    def construct(self, z):
        hc = self.dense(z)
        hc = reshape(hc, [self.bs, self.channelg[0], self.fmd[0], self.fmd[0]])
        gdfq = self.deconv2d(hc)
        gdf = gdfq + self.base_geo
        gdf = permute(gdf, (0, 2, 3, 1))
        return gdf
