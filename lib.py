import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, re
import torch, json
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from config import *

# lst file, [TICK, TIME, NOTE, IS_CIRCLE, IS_SLIDER, IS_SPINNER, IS_SLIDER_END, IS_SPINNER_END,
#               0,    1,    2,         3,         4,          5,             6,              7,
#            SLIDING, SPINNING, MOMENTUM, ANGULAR_MOMENTUM, EX1, EX2, EX3], length MAPTICKS
#                  8,        9,       10,               11,  12,  13,  14,
# wav file, [len(snapsize), MAPTICKS, 2, fft_size//4]
def read_npz(fn):
    with np.load(fn) as data:
        wav_data = data["wav"];
        wav_data = np.swapaxes(wav_data, 2, 3);
        train_data = wav_data;
        div_source = data["lst"][:, 0];
        div_source2 = data["lst"][:, 12:15];
        div_data = np.concatenate([divisor_array(div_source), div_source2], axis=1);
        lst_data = data["lst"][:, 2:10];
        # Change the 0/1 data to -1/1 to use tanh instead of softmax in the NN.
        # Somehow tanh works much better than softmax, even if it is a linear combination. Maybe because it is alchemy!
        lst_data = 2 * lst_data - 1;
        train_labels = lst_data;
    return train_data, div_data, train_labels;


def divisor_array(t):
    d_range = list(range(0, divisor));
    return np.array([[int(k % divisor == d) for d in d_range] for k in t]);


def read_npz_list():
    npz_list = [];
    for file in os.listdir(data_path):
        if file.endswith(".npz"):
            if 'train' in file or 'test' in file:
                continue
            npz_list.append(os.path.join(data_path, file));
    return npz_list;


def prefilter_data(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered):
    # Filter out slider ends from the training set, since we cannot reliably decide if a slider end is on a note.
    # Another way is to set 0.5 for is_note value, but that will break the validation algorithm.
    # Also remove the IS_SLIDER_END, IS_SPINNER_END columns which are left to be zeros.

    # Before: NOTE, IS_CIRCLE, IS_SLIDER, IS_SPINNER, IS_SLIDER_END, IS_SPINNER_END, SLIDING, SPINNING
    #            0,         1,         2,          3,             4,              5,       6,        7
    # After:  NOTE, IS_CIRCLE, IS_SLIDER, IS_SPINNER, SLIDING, SPINNING
    #            0,         1,         2,          3,       4,        5

    non_object_end_indices = [i for i, k in enumerate(train_labels_unfiltered) if k[4] == -1 and k[5] == -1];
    train_data = train_data_unfiltered[non_object_end_indices];
    div_data = div_data_unfiltered[non_object_end_indices];
    train_labels = train_labels_unfiltered[non_object_end_indices][:, [0, 1, 2, 3, 6, 7]];

    # should be (X, 7, 32, 2) and (X, 6) in default sampling settings
    # (X, fft_window_type, freq_point, magnitude/phase)
    return train_data, div_data, train_labels;


def preprocess_npzs(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered):
    train_data, div_data, train_labels = prefilter_data(train_data_unfiltered, div_data_unfiltered,
                                                        train_labels_unfiltered);
    # In this version, the train data is already normalized, no need to do it again here
    #     mean = train_data.mean(axisprefilter_=0)
    #     std = train_data.std(axis=0)
    #     train_data = (train_data - np.tile(mean, (train_data.shape[0], 1,1,1))) / np.tile(std, (train_data.shape[0], 1,1,1))

    # Make time intervals from training data
    if train_data.shape[0] % time_interval > 0:
        train_data = train_data[:-(train_data.shape[0] % time_interval)];
        div_data = div_data[:-(div_data.shape[0] % time_interval)];
        train_labels = train_labels[:-(train_labels.shape[0] % time_interval)];
    train_data2 = np.reshape(train_data,
                             (-1, time_interval, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    div_data2 = np.reshape(div_data, (-1, time_interval, div_data.shape[1]))
    train_labels2 = np.reshape(train_labels, (-1, time_interval, train_labels.shape[1]))
    return train_data2, div_data2, train_labels2;


def get_data_shape():
    for file in os.listdir(data_path):
        if file.endswith(".npz"):
            train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered = read_npz(os.path.join(data_path, file));
            train_data, div_data, train_labels = prefilter_data(train_data_unfiltered, div_data_unfiltered,
                                                                train_labels_unfiltered);
            # should be (X, 7, 32, 2) and (X, 6) in default sampling settings
            # (X, fft_window_type, freq_point, magnitude/phase)
            # X = 76255
            # print(train_data.shape, train_labels.shape);
            if train_data.shape[0] == 0:
                continue;
            return train_data.shape, div_data.shape, train_labels.shape;
    print("cannot find npz!! using default shape");
    return (-1, 7, 32, 2), (-1, 3 + divisor), (-1, 6);


def read_some_npzs_and_preprocess(npz_list):
    td_list = [];
    dd_list = [];
    tl_list = [];
    for fp in npz_list:
        if fp.endswith(".npz"):
            _td, _dd, _tl = read_npz(fp);
            #if _td.shape[1:] != train_shape[1:]:
            #    print("Warning: something wrong found in {}! shape = {}".format(fp, _td.shape));
            #    continue;
            td_list.append(_td);
            dd_list.append(_dd);
            tl_list.append(_tl);
    train_data_unfiltered = np.concatenate(td_list);
    div_data_unfiltered = np.concatenate(dd_list);
    train_labels_unfiltered = np.concatenate(tl_list);

    train_data2, div_data2, train_labels2 = preprocess_npzs(train_data_unfiltered, div_data_unfiltered,
                                                            train_labels_unfiltered);
    return train_data2, div_data2, train_labels2;


def train_test_split(train_data2, div_data2, train_labels2, test_split_count=233):
    new_train_data = train_data2[:-test_split_count];
    new_div_data = div_data2[:-test_split_count];
    new_train_labels = train_labels2[:-test_split_count];
    test_data = train_data2[-test_split_count:];
    test_div_data = div_data2[-test_split_count:];
    test_labels = train_labels2[-test_split_count:];
    return (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels);

def load_data():
    f_train = os.path.join(data_path, 'train.npz')
    with np.load(f_train) as data:
        train_spec = data["spec"][:,:,:,:,0]
        train_div = data["div"]
        train_label = data["label"]
    f_test = os.path.join(data_path, 'test.npz')
    with np.load(f_test) as data:
        test_spec = data["spec"][:,:,:,:,0]
        test_div = data["div"]
        test_label = data["label"]
    return (train_spec, train_div, train_label), (test_spec, test_div, test_label)


class Data2Torch(Dataset):
    def __init__(self, data, lim=-1):
        self.spec_data = data[0][:lim]
        self.div_data = data[1][:lim]
        self.label = data[2][:lim]

    def __getitem__(self, index):
        spec = self.spec_data[index]
        div = self.div_data[index]
        label = self.label[index]

        return torch.from_numpy(spec).float(), torch.from_numpy(np.array(div)).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.label)


# loss function, calculate the distane between two latent as the rating
def loss_func(pred, target):
    loss = F.cross_entropy(pred, target)
    return loss


#train_shape, div_shape, label_shape = get_data_shape();

#print(train_shape, div_shape, label_shape)