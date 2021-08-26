#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed



def DataLoader(dataset, answerset, data_version='old'):
    infile = open("{}".format(str(dataset)), "rb")
    data = pickle.load(infile)
    infile.close()
    if data_version=='old':
        answer = np.genfromtxt('{}'.format(answerset), usecols=[0,1], dtype=str)
    else:
        answer = np.genfromtxt('{}'.format(answerset), usecols=[0,1], dtype=str)
    return data, answer



def get_data_integrated(data_path, subset, data_size, begin_set, end_set, latter_cut_percentage=1, latter_cut_count=0, opt='default',
                        data_set_version='old'):
    """
    This function will return a tuple which contains the data for machine learning algorithm training.
    as default
    :param data_size: (double/float) percentage of one simulation--counted from the beginning. In other words, how far you want the simulation to go
    :param begin_set: (int) names of data sets are in order so we decide the first data set
    :param end_set: (int) names of data sets are in order so we decide the last data set
    :param latter_cut_percentage: (double/float) contrary to data_size(@param), this indicates which starting point you want the simulation to begin
    :param latter_cut_count: (int) similar to latter_cut_percentage except you gave exact number of frames counting from end point
    :param opt: (strings) how you want your data grouped.
                'default': you will get a tuple consisted with one feature list--full of frames and one label list--full of label corresponding to each frames in feature list
                'equilibrium': return same data structure as 'default' but number of each type of simulation are the same--50% OUT and 50% IN
                'data_grouped': return a tuple with 4 lists-two for features and two for labels. same type of simulations are stored in the same list.
    :param data_set_version:(string) which version data set you want to load
            'new'
            'old'
    :return:
        'default' or 'equilibrium' : (x,y)
        'data_grouped':(x_in,x_out,y_in,y_out)
    """
    if data_path[-1] != "/":
        return print("Please, make sure you used a / at the end of your data_path")

    x_in = []
    y_in = []
    x_out = []
    y_out = []
    
    # 250000    5000 in training and testing. 200000
    for k in range(begin_set, end_set + 1):
        x_ex = []
        y_ex = []
        if (k < 10):
            if data_set_version == 'old':
                x_ex, y_ex = DataLoader("{}data_set_00{}.bin".format(data_path, k),
                                        "{}cheatsheet_set_00{}.dat".format(data_path, k))
            elif data_set_version == 'new':
                x_ex, y_ex = DataLoader("{}subset_set{}_000.bin".format(data_path, k),
                                        "{}cheatsheet_set{}_magd_strided.dat".format(data_path, k),data_version='new')
            elif data_set_version == 'cluster':
                x_ex, y_ex = DataLoader("{}subset_set{}_{:03d}.bin".format(data_path, k, subset),
                                        "{}cheatsheet_set{}_magd_strided.dat".format(data_path, k), data_version='new')
        else:
            x_ex, y_ex = DataLoader("{}data_set_0{}.bin".format(data_path, k),
                                    "{}cheatsheet_set_0{}.dat".format(data_path, k))


        simulation_count = 0
        cut_length = latter_cut_count
        for label_info in y_ex:
            l=label_info[0] if data_set_version=='old' else int(label_info[0])
            y_label=label_info[1]
            frame_length = int((len(x_ex[l]) - 1) * data_size)  # exclude first frame
            if latter_cut_count == 0:
                cut_length = int(frame_length * latter_cut_percentage)
            simulation_count = simulation_count + 1
            if (y_label == 'IN'):
                x_in.append(
                    x_ex[l][1 + frame_length - cut_length:1 + frame_length])
                template = np.empty(cut_length).astype(str)
                template[:] = y_label
                y_in.append(template)
            else:
                x_out.append(
                    x_ex[l][1 + frame_length - cut_length:1 + frame_length])
                template = np.empty(cut_length).astype(str)
                template[:] = y_label
                y_out.append(template)
        del x_ex
        del y_ex
        del template
        

    if opt == 'label_grouped':
        return np.vstack(x_in), np.vstack(x_out), np.hstack(y_in), np.hstack(y_out)
    else:
        if opt == 'default':
            x_total = np.vstack((np.array(x_in), np.array(x_out)))
            y_total = np.vstack((y_in, y_out))
        elif opt == 'equilibrium':
            min_length = min(len(x_in), len(x_out))  # find the number of
            print("equilibrium ", min_length, " of each label")
            x_total = np.vstack((np.array(x_in[0:min_length]), np.array(
                x_out[0:min_length])))  # assemble features of IN simulations and OUT simulations
            y_total = np.vstack(
                (y_in[0:min_length], y_out[0:min_length]))  # assemble labels of IN simulations and OUT simulations
        else:
            raise Exception("not supported opt")
        x = np.vstack(x_total)
        y = np.hstack(np.array(y_total))
        return x, y



class DataSet(object):
    """Container class for a dataset (deprecated).

    THIS CLASS IS DEPRECATED. See
    [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
    for general migration instructions.
    """

    # @deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
    #                   ' from tensorflow/models.')
    def __init__(self,
                 features,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(
                'Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert features.shape[0] == labels.shape[0], (
                    'images.shape: %s labels.shape: %s' % (features.shape, labels.shape))
            self._num_examples = features.shape[0]

            # # Convert shape from [num examples, rows, columns, depth]
            # # to [num examples, rows*columns] (assuming depth == 1)
            # if reshape:
            #   assert features.shape[3] == 1
            #   features = features.reshape(features.shape[0],
            #                               features.shape[1] * features.shape[2])
            # if dtype == dtypes.float32:
            #   # Convert from [0, 255] -> [0.0, 1.0].
            #   features = features.astype(np.float32)
            #   features = np.multiply(features, 1.0 / 255.0)
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            ##print("first time and shuffle")
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            features_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                # print("one epoch is over shuffle")
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._features = self.features[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate(
                (features_rest_part, features_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._labels[start:end]

