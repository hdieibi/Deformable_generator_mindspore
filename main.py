from tqdm import trange
import os
import yaml
from trainer import Trainer
from data.face_data import FaceData
from datetime import datetime
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.dataset import GeneratorDataset
import mindspore.nn as nn
from modules.model import DeformGenerator
from modules.appNet import AppNet
from modules.geoNet import GeoNet
from utils.args import parse_args
from utils.visualize import visualize_train, visualize_evaluate, visualize_swap
import math


def train(model, dataset, args):
    size = dataset.get_dataset_size()
    model.set_train()
    lr = nn.piecewise_constant_lr(args['milestones'], args['lr'])
    # lr = 2.e-4
    trainer = Trainer(model, lr[0])
    log_dir = args['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    dtime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_path = os.path.join(log_dir, dtime)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    visualize_path = os.path.join(log_path, 'visualize')
    latent_z_path = os.path.join(log_path, 'latent_z.pkl')
    for epoch in trange(args['max_epoch']):
        trainer.update_lr(lr[epoch])
        trainer.model.update_delta(args['dfg_params']['delta'] * math.exp(-epoch))
        for batch_idx, data in enumerate(dataset.create_tuple_iterator(num_epochs=1)):
            loss, rec = trainer.train(data[0], batch_idx)

            if (batch_idx+1) % 10 == 0:
                loss, current = loss.asnumpy(), batch_idx
                print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")

            if (batch_idx+1) % 25 == 0:
                sample = trainer.rand_sample(args['batch_size'])
                visualize_train(rec, sample, visualize_path, epoch, batch_idx)

        if (epoch+1) % 100 == 0:
            save_checkpoint_path = os.path.join(log_path, 'dfg_model_epoch_{}.ckpt'.format(epoch))
            save_checkpoint(model, save_checkpoint_path)
    trainer.model.save_latent_z(latent_z_path)


def evaluate(model, dataset, args):
    ckpt_path = args['checkpoint']
    latent_z_path = args['latent']
    params_dict = load_checkpoint(ckpt_path)
    params_not_load, _ = load_param_into_net(model, params_dict)
    model.set_train(False)
    trainer = Trainer(model, 0)
    trainer.model.load_latent_z(latent_z_path)
    visualize_path = 'evaluate'
    sample_app, sample_geo = trainer.sample_one_dim()
    visualize_evaluate(sample_app, sample_geo, visualize_path)
    swap_img = trainer.swap_app_geo()
    visualize_swap(swap_img, visualize_path)


def main():
    opt = parse_args()
    with open(opt.config) as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    configs.update(vars(opt))
    mode = configs['mode']
    data = FaceData(configs['data_path'], mode)
    dataset = GeneratorDataset(data, column_names=['data'], shuffle=False)
    dataset = dataset.batch(batch_size=configs['batch_size'])
    app_net = AppNet()  # AppNet(**configs['app_params'])
    geo_net = GeoNet()  # GeoNet(**configs['geo_params'])
    model = DeformGenerator(app_net, geo_net, **configs['dfg_params'])
    if mode == 'train':
        train(model, dataset, configs)
    elif mode == 'eval':
        evaluate(model, dataset, configs)
    else:
        print("set wrong mode")


if __name__ == "__main__":
    main()
