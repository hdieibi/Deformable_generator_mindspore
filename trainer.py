import mindspore as ms
import mindspore.nn as nn
from mindspore import ops


class Trainer:
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.lr)
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)

    def update_lr(self, lr):
        if lr != self.lr:
            self.lr = lr
            self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.lr)

    def train(self, x, idx):
        (loss, x_hat, app, geo), grads = self.grad_fn(x, idx)
        self.optimizer(grads)
        rec = {
            'gt': x,
            'rec': x_hat
        }
        return loss, rec

    def evaluate(self, x, idx):
        pass

    def forward_fn(self, x, idx):
        x_hat, app, geo = self.model(x, idx)
        loss = self.loss_fn(x_hat, x)
        return loss, x_hat, app, geo

    def rand_sample(self, num_sample):
        sample_zapp = ops.randn([num_sample, self.model.za_dim])
        sample_zgeo = ops.randn([num_sample, self.model.zg_dim])
        sample_x, sample_app, _ = self.model.sample(sample_zapp, sample_zgeo)
        sample = {
            'sample_img': sample_x,
            'sample_app': sample_app
        }
        return sample

    def sample_one_dim(self):
        npica = int(self.model.za_dim / 10)
        npicg = int(self.model.zg_dim / 10)

        '''
        (1) Plot the appearance basis functions
        '''
        app_samples = []
        for pic in range(npica):
            zgeop = ops.zeros((self.model.batchsz, self.model.zg_dim))
            zappp = ops.zeros((self.model.batchsz, self.model.za_dim))
            for d in range(10):
                zappp[d * 10:d * 10 + 10, pic * 10 + d] = ops.linspace(-10, 10, 10)
            app_sample_x, _, _ = self.model.sample(zappp, zgeop)
            app_samples.append(app_sample_x)
        '''
        (2) Plot the geometric basis functions
        '''
        geo_samples = []
        zappp = ops.zeros((self.model.batchsz, self.model.za_dim))
        for pic in range(npicg):
            zgeop = ops.zeros((self.model.batchsz, self.model.zg_dim))
            for d in range(10):
                zgeop[d * 10:d * 10 + 10, pic * 10 + d] = ops.linspace(-8, 8, 10)
            geo_sample_x, _, _ = self.model.sample(zappp, zgeop)
            geo_samples.append(geo_sample_x)
        return app_samples, geo_samples
