import os
from mindspore import ops
import numpy as np
import matplotlib.pyplot as plt


def visualize_train(rec, sample, save_path, epoch, batch_idx):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    row = col = 3
    rec_path = os.path.join(save_path, 'rec_{}_{}.png'.format(epoch, batch_idx))
    row_elements = []
    for i in range(row):
        col_elements = []
        for j in range(col):
            index = row * i + j
            element = ops.cat([rec['gt'][index], rec['rec'][index]], axis=2)
            col_elements.append(element)
        row_elements.append(ops.cat(col_elements, axis=2))
    rec_img = ops.cat(row_elements, axis=1)
    rec_array = np.transpose(rec_img.asnumpy(), [1, 2, 0])
    plt.imsave(rec_path, rec_array)
    sample_path = os.path.join(save_path, 'sample_{}_{}.png'.format(epoch, batch_idx))
    row_elements = []
    for i in range(row):
        col_elements = []
        for j in range(col):
            index = row * i + j
            element = ops.cat([sample['sample_app'][index], sample['sample_img'][index]], axis=2)
            col_elements.append(element)
        row_elements.append(ops.cat(col_elements, axis=2))
    sample_img = ops.cat(row_elements, axis=1)
    sample_array = np.transpose(sample_img.asnumpy(), [1, 2, 0])
    plt.imsave(sample_path, sample_array)


def visualize_evaluate(sample_app, sample_geo, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    row = col = 3
    fea_dim = len(sample_app)
    for k in range(fea_dim):
        sample_app_path = os.path.join(save_path, 'sample_app_feat_{}.png'.format(k))
        row_elements = []
        for i in range(row):
            col_elements = []
            for j in range(col):
                index = row * i + j
                element = sample_app[k][index]
                col_elements.append(element)
            row_elements.append(ops.cat(col_elements, axis=2))
        sample_app_tensor = ops.cat(row_elements, axis=1)
        sample_app_array = np.transpose(sample_app.asnumpy(), [1, 2, 0])
        plt.imsave(sample_app_path, sample_app_array)
    sample_geo_path = os.path.join(save_path, 'sample_geo.png')
    row_elements = []
    for i in range(row):
        col_elements = []
        for j in range(col):
            index = row * i + j
            element = sample_geo[index]
            col_elements.append(element)
        row_elements.append(ops.cat(col_elements, axis=2))
    sample_geo = ops.cat(row_elements, axis=1)
    sample_geo_array = np.transpose(sample_geo.asnumpy(), [1, 2, 0])
    plt.imsave(sample_geo_path, sample_geo_array)
