import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import numpy as np
import torch
import os
import shutil


def scatter(x, colors, file_name, class_num):
    f = plt.figure(figsize=(226 / 15, 212 / 15))
    ax = plt.subplot(aspect='equal')
    color_pen = ['black', 'r']
    # relabel
    my_legend = ['Delta_C', 'Delta_M']
    label_set = []
    label_set.append(colors[0])
    for i in range(1, len(colors)):
        flag = 1
        for j in range(len(label_set)):
            if label_set[j] == colors[i]:
                flag = 0
                break
        if flag:
            label_set.append(colors[i])
    # draw
    for i in range(class_num):
        ax.scatter(x[colors == label_set[i], 0], x[colors == label_set[i], 1], lw=0, s=70, c=color_pen[i],
                   label=str(my_legend[i]))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('tight')
    ax.legend(loc='upper right')
    f.savefig(file_name + ".png", bbox_inches='tight')
    print(file_name + ' save finished')


def plot_TSNE(data, label, path, title, class_num):
    colors = label
    all_features_np = data
    tsne_features = TSNE(random_state=20190129).fit_transform(all_features_np)
    scatter(tsne_features, colors, os.path.join(path, title), class_num)


def visualization(v_tensors, outpath, elements=10):
    '''
    visualization of memory cells decoupling
    :param length: sequence length
    :param layers: stacked predictive layers
    :param c: variables
    :param m: variables
    :param path: save path
    :param elements: select top k element to visualization
    :return:
    '''
    if not os.path.exists('/data3/tsne/'):
        os.makedirs('/data3/tsne/')
    data = []
    label = []
    for i in range(v_tensors.shape[1]):
        # choose the most dominated variables to the similarity
        value1, index1 = torch.topk(v_tensors[i], elements)
        # c [c_topk, elements in m_topk pos]
        c_key = F.normalize(torch.cat([value1, c[layers * t + i, j, k, index2]], dim=0),
                            dim=0).detach().cpu().numpy().tolist()
        data.append(c_key)
        label.append(i)
    plot_TSNE(np.array(data), np.array(label), outpath, 'case_' + '_tsne_' + str(i), v_tensors.shape[1])
