import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import grid_sample
from mindspore import grad, Parameter, Tensor


class DeformGenerator(nn.Cell):
    def __init__(self, App_net, Geo_net, batchsz, za_dim, zg_dim, sigmap, N, delta):
        super(DeformGenerator, self).__init__()
        self.deform_model = DeformModel(App_net, Geo_net, sigmap)
        self.mse = nn.MSE()
        Nbatch = int(N / batchsz)
        # self.zapp_all = Parameter(ops.randn([Nbatch, batchsz, za_dim]), requires_grad=False)
        # self.zgeo_all = Parameter(ops.randn([Nbatch, batchsz, zg_dim]), requires_grad=False)
        self.zapp_all = [ops.randn([batchsz, za_dim]) for _ in range(Nbatch)]
        self.zgeo_all = [ops.randn([batchsz, zg_dim]) for _ in range(Nbatch)]
        self.delta = delta
        self.batchsz = batchsz
        self.za_dim = za_dim
        self.zg_dim = zg_dim

    def construct(self, x, batch_idx):
        x_hat, app, geo = self.deform_model.decode(self.zapp_all[batch_idx], self.zgeo_all[batch_idx])
        self.langevin_infer(x, batch_idx)
        return x_hat, app, geo

    def update_delta(self, delta):
        if delta != self.delta:
            self.delta = delta

    def sample(self, zapp, zgeo):
        sample_x, sample_app, sample_geo = self.deform_model.decode(zapp, zgeo)
        return sample_x, sample_app, sample_geo

    def langevin_infer(self, x, batch_idx, iter=1):
        zapp = self.zapp_all[batch_idx]
        zgeo = self.zgeo_all[batch_idx]
        for _ in range(iter):
            grad_zapp, grad_zgeo = grad(self.deform_model, grad_position=(1,2))(x, zapp, zgeo)
            zapp_infer = zapp - 0.5 * self.delta * self.delta * grad_zapp + self.delta * ops.randn([self.batchsz, self.za_dim])
            zgeo_infer = zgeo - 0.5 * self.delta * self.delta * grad_zgeo + self.delta * ops.randn([self.batchsz, self.zg_dim])
            zapp = zapp_infer
            zgeo = zgeo_infer
        self.zapp_all[batch_idx] = zapp
        self.zgeo_all[batch_idx] = zgeo


class DeformModel(nn.Cell):
    def __init__(self, App_net, Geo_net, sigma):
        super(DeformModel, self).__init__()
        self.App_net = App_net
        self.Geo_net = Geo_net
        self.sigma = sigma
        self.mse = nn.MSE()

    def decode(self, zapp, zgeo):
        app = self.App_net(zapp)
        geo = self.Geo_net(zgeo)
        x = grid_sample(app, geo)
        return x, app, geo

    def construct(self, x, zapp, zgeo):
        x_hat, _, _ = self.decode(zapp, zgeo)
        loss = 1.0 / (2.0 * self.sigma * self.sigma) * self.mse(x_hat, x)
        return loss
