import mindspore.nn as nn


class AppNet(nn.Cell):
    def __init__(self, za_dim=64, nchan=3, imgsize=64, batchsize=9, kersize=[3,3,5,5,3,3], chb=16, alpha=5/8, fmd=[4,8,16,32,64], stride=[1, 2, 2, 1, 1, 1]):
        super(AppNet, self).__init__()
        self.za_dim = za_dim
        self.nchan = nchan
        self.imgsize = imgsize
        self.bs = batchsize
        self.kersize = kersize
        self.chb = chb
        channelg = [chb * 8, chb * 4, chb * 4, chb * 4, chb * 2, chb * 1]
        self.alpha = alpha
        self.channela = [int(i * alpha) for i in channelg]
        self.fmd = fmd
        self.name = 'genapp'
        self.dense = nn.SequentialCell(
            nn.Dense(self.za_dim, self.fmd[0] * self.fmd[0] * self.channela[0]),
            nn.ReLU()
        )
        self.deconv2d = nn.SequentialCell(
            nn.Conv2dTranspose(self.channela[0], self.channela[1], kersize[0], stride[0]),
            nn.ReLU(),
            nn.Conv2dTranspose(self.channela[1], self.channela[2], kersize[1], stride[1]),
            nn.ReLU(),
            nn.Conv2dTranspose(self.channela[2], self.channela[3], kersize[2], stride[2]),
            nn.ReLU(),
            nn.Conv2dTranspose(self.channela[3], self.channela[4], kersize[3], stride[3]),
            nn.ReLU(),
            nn.Conv2d(self.channela[4], self.channela[5], kersize[4], stride[4], pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channela[5], self.nchan, kersize[5], stride[5], pad_mode='pad', padding=1),
            nn.Sigmoid(),
            nn.Upsample(size=(self.imgsize, self.imgsize), mode='bilinear'),
        )

    def construct(self, z):
        hc = self.dense(z)
        hc = hc.reshape([self.bs, self.channela[0], self.fmd[0], self.fmd[0]])
        gx = self.deconv2d(hc)
        return gx
