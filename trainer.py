import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
import random


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
        sample_x, sample_app, _ = self.model.sample(num_sample)
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
        zgeop = ops.zeros((self.model.batchsz, self.model.zg_dim))
        for pic in range(npica):
            for d in range(10):
                zappp = ops.zeros((self.model.batchsz, self.model.za_dim))
                zappp[:, pic * 10 + d] = ops.linspace(-10, 10, 9)
                app_sample_x, _, _ = self.model.deform_model.decode(zappp, zgeop)
                app_samples.append(app_sample_x)
        '''
        (2) Plot the geometric basis functions
        '''
        geo_samples = []
        zappp = ops.zeros((self.model.batchsz, self.model.za_dim))
        zappp = zappp - 1.
        for pic in range(npicg):
            for d in range(10):
                zgeop = ops.zeros((self.model.batchsz, self.model.zg_dim))
                zgeop[:, pic * 10 + d] = ops.linspace(-8, 8, 9)
                geo_sample_x, _, _ = self.model.deform_model.decode(zappp, zgeop)
                geo_samples.append(geo_sample_x)
        return app_samples, geo_samples

    def swap_app_geo(self):
        source_index = random.randint(0, 899)
        source_z_app = self.model.zapp_all[int(source_index / self.model.batchsz)][source_index%self.model.batchsz]
        source_z_app = ops.unsqueeze(source_z_app, 0)
        source_z_geo = self.model.zgeo_all[int(source_index / self.model.batchsz)][source_index % self.model.batchsz]
        source_z_geo = ops.unsqueeze(source_z_geo, 0)

        target_index = random.randint(0, 899)
        target_z_app = self.model.zapp_all[int(target_index / self.model.batchsz)][target_index % self.model.batchsz]
        target_z_app = ops.unsqueeze(target_z_app, 0)
        target_z_geo = self.model.zgeo_all[int(target_index / self.model.batchsz)][target_index % self.model.batchsz]
        target_z_geo = ops.unsqueeze(target_z_geo, 0)

        source_img, _, _ = self.model.deform_model.decode(source_z_app, source_z_geo)
        target_img, _, _ = self.model.deform_model.decode(target_z_app, target_z_geo)
        swapped_source_img, _, _ = self.model.deform_model.decode(source_z_app, target_z_geo)
        swapped_target_img, _, _ = self.model.deform_model.decode(target_z_app, source_z_geo)

        all_img1 = ops.cat([source_img[0], target_img[0]], axis=2)
        all_img2 = ops.cat([swapped_source_img[0], swapped_target_img[0]], axis=2)
        all_img = ops.cat([all_img1, all_img2], axis=1)
        return all_img
