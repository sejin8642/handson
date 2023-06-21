import os
# for error "not creating xla devices tf_xla_enable_xla_devices not set"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# TF_XLA_FLAGS is an environment variable used by TensorFlow's XLA (Accelerated Linear Algebra) 
# compiler to control its behavior. In this case, setting TF_XLA_FLAGS to 
# --tf_xla_enable_xla_devices enables the XLA compiler to use all available XLA devices, 
# such as GPUs or TPUs, for faster execution of TensorFlow computations.

# for error "Successfully opened dynamic library libcudart.so.10.1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import inspect
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

import tensorflow as tf
from tensorflow import keras
    
class TOC:
    def __init__(self):
        pass
    
    def __repr__(self):
        pages = (
            "Part I. The Fundamentals of Machine Learning\n"
            "  1 (pg 003). The Machine Learning Landscape\n"
            "  2 (pg 039). End-to-End Machine Learning Project\n"
            "  3 (pg 103). Classification\n"
            "  4 (pg 131). Training Models\n"
            "  5 (pg 175). Support Vector Machines\n"
            "  6 (pg 195). Decision Trees\n"
            "  7 (pg 211). Ensemble Learning and Random Forests\n"
            "  8 (pg 237). Dimensionality Reduction\n"
            "  9 (pg 259). Unsupervised Learning Techniques\n"
            "\n"
            "Part II. Neural Networks and Deep Learning\n"
            " 10 (pg 299). Introduction to Artificial Neural Networks with Keras\n"
            " 11 (pg 357). Training Deep Neural Networks\n"
            " 12 (pg 403). Custom Models and Training with TensorFlow\n"
            " 13 (pg 441). Loading and Preprocessing Data with TensorFlow\n"
            " 14 (pg 479). Deep Computer Vision Using Convolutional Neural Networks\n"
            " 15 (pg 537). Processing Sequences Using RNNs and CNNs\n"
            " 16 (pg 577). Natural Language Processing with RNNs and Attention\n"
            " 17 (pg 635). Autoencoders, GANs, and Diffusion Models\n"
            " 18 (pg 683). Reinforcement Learning\n"
            " 19 (pg 721). Training and Deploying TensorFlow Models at Scale\n"
            "")
        return pages

def list_attr(method = None, spacing=25):
    """
    prints and returns a list of attributes of the keras method
    
    
    Parameters
    ----------
    method: str
        name of the method from which to get attribute list (default None)
    spacing: int
        spacing amongst printed items of the list
    
    return
    ------
    List: str
        list of attribute strings
    """
    Method = method.capitalize()
    attr_list = []

    method = getattr(keras, method)
    Method = getattr(method, Method[:-1])
    
    for attr_name in dir(method):
        attr_class = getattr(method, attr_name)
        if inspect.isclass(attr_class) and issubclass(attr_class, Method):
            if attr_name[0].isupper():
                attr_list.append(attr_name)
                print(attr_name.ljust(spacing), end='')
                
    return attr_list

def mnist_data():
    """
    Loads MNIST data and returns train, valid, test datasets
    
    return
    ------
    numpy array: float64
        numpy arrays of X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def housing_data():
    """
    Loads california housing data and returns train, valid, test datasets. Train and valid datasets
    are scaled to have a mean of 0 and STD of 1 through StandardScaler().transform (fit is needed first)
    
    return
    ------
    numpy array: float64
        numpy arrays of X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    housing = fetch_california_housing()

    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

# chapter 11
def fashion_mnist_data():
    """
    Loads MNIST fashion data and returns train, valid, test datasets.
    
    return
    ------
    numpy array: float64
        numpy arrays of X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

# chapter 13
def Filepaths():
    """
    returns a list of data path strings for housing data (total of 20)
    for chapter 13
    """
    # get a list of 20 csv files in housing directory
    p = Path.home()/'handson'/'housing'
    filepaths = list(p.glob('**/data*'))
    
    # change the PosixPath format to string format for tf to accept
    filepaths = [str(x) for x in filepaths]
    filepaths.sort()
    return filepaths

def preprocess(line):
    """
    normalizes input housing data features (usually byte string) and returns 
    tensor objects of both features and target value separately
    for chapter 13
    
    Parameter:
    ---------
    line: byte str
        Byte string of housing data features and target value
        
    return
    ------
    normalized housing data sample: tensor objects
        tensor objects of separate normalized features and target value
    """
    
    # load means and stds of the training dataset
    with open('housing_means_stds.npy', 'rb') as f:
        X_mean = np.load(f)
        X_std = np.load(f)
        
    n_inputs = 8
    
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y

def preprocess_nothing(line):
    """
    returns tensor objects of both features and target value separately without any transform
    for chapter 13
    
    Parameter:
    ---------
    line: byte str
        Byte string of housing data features and target value
        
    return
    ------
    raw housing data sample: tensor objects
        tensor objects of separate normalized features and target value
    """
    n_inputs = 8
    
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return x, y

# returns dataset that is randomized, processed, and batch-grouped
def csv_reader_dataset(
    filepaths,
    repeat=1,
    n_readers=4,
    n_read_threads=None, 
    shuffle_buffer_size=10000,
    n_parse_threads=4,
    batch_size=32,
    normalization=True):
    """
    returns dataset that is randomized, processed, and batch-grouped
    for chapter 13
    
    Parameters:
    -----------
    filepaths: list[byte str]
        List of housing data file paths strings. It is obtained by Filepath function
    repeat: int
        Number of the original data plus extra duplicate data (default 1). Check out
        DatasetV2.repeat for more details
    n_readers: int
        The number of input elements of the dataset that will be processed concurrently 
        by dataset.interleave (default 4). If n_readers = 4, the first 4 elements of 
        dataset will be interleaved. After that, the next 4 will be interleaved and such 
        process will repeat until all data are interleaved
    n_read_threads: int
        if n_read_threads is at least 1, dataset.interleave creates a threadpool, which is 
        used to fetch inputs from cycle elements asynchronously and in parallel (default None)
        Check out DatasetV2.interleave docstrings for more details
    shuffle_buffer_size: int
        buffer size input for DatasetV2.shuffle (default 10000). If shuffle_buffer_size = 1,000,
        the first 1,000 data elements from the dataset becomes the selection pool from which
        random element is picked for the first element of the shuffled dataset. The next element
        (for this case, 1,001-th element of the entire dataset) fills out the empty spot of the 
        selection pool. Such process is repeated until all elements are selected for the 
        shuffled dataset. Check out DatasetV2.shuffle docstrings for more details
    n_parse_threads: int
        the number elements to process asynchronously in parallel by DatasetV2.map (default 4)
    batch_size: int
        batch size of combined consecutive data elements of the fully processed dataset 
        (default 32). Check out DatasetV2.batch for more details
    normalization: bool
        preprocessing normalization is applied by default. If it is False, then the output 
        will not normalized
        
    return
    ------
    dataset: tf dataset
        tf dataset that is randomized, processed, and batch-grouped
    """
    
    # filepaths: list of file path strings
    # create tensor dataset of file path bytestrings shuffled
    # argument shuffle=False will not shuffle (default is shuffle=None)
    dataset = tf.data.Dataset.list_files(filepaths)
    
    # create dataset that reads from n_readers files at a time and interleave their lines 
    # the first head row line in each file is skipped
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, 
        num_parallel_calls=n_read_threads)
    
    # normalization and conversion of data format
    if normalization:
        dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    else:
        dataset = dataset.map(preprocess_nothing, num_parallel_calls=n_parse_threads)
        
    dataset = dataset.shuffle(shuffle_buffer_size).repeat(repeat)
    return dataset.batch(batch_size).prefetch(1)

def housing_data_norm():
    """
    Loads housing data from housing directory and returns train, valid, and test datasets
    
    return
    ------
    numpy array: float32
        numpy arrays of datasets
    """
    filepaths = Filepaths()

    # split into train, valid, test sets
    train_filepaths = filepaths[:16]
    valid_filepaths = filepaths[16:19]
    test_filepaths = filepaths[19:]

    # raw data (not normalized)
    train_set = csv_reader_dataset(train_filepaths, normalization=False)
    valid_set = csv_reader_dataset(valid_filepaths, normalization=False)
    test_set = csv_reader_dataset(test_filepaths, normalization=False)
    
    # convert the tensorflow dataset format into numpy format
    X_train = np.concatenate([x[0] for x in train_set])
    y_train = np.concatenate([x[1] for x in train_set])

    X_valid = np.concatenate([x[0] for x in valid_set])
    y_valid = np.concatenate([x[1] for x in valid_set])

    X_test = np.concatenate([x[0] for x in test_set])
    y_test = np.concatenate([x[1] for x in test_set])
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def ocean_proximity_data():
    """
    generates ocean proximity data that include features, labels, and categories (str)
    for chapter 13
    
    return
    ------
        9 numpy arrays of features (float), categories (str), labels (float) of 
        train, valid, and test datasets
    """
    ocean_data_path = "./ocean_proximity.csv"

    dataset = tf.data.Dataset.list_files(ocean_data_path)
    dataset = dataset.interleave(lambda path: tf.data.TextLineDataset(path).skip(1))
    dataset = dataset.shuffle(20640) # buffer size 20640 is exactly the number of data lines
    
    # function to convert tf tensor of string data type into float data and byte str categories
    @tf.autograph.experimental.do_not_convert 
    def convert(line):
        defs = [0.]*8 + [tf.constant([], dtype=tf.float32)] + [tf.constant([], dtype=tf.string)]
        fields = tf.io.decode_csv(line, record_defaults=defs)
        data = tf.stack(fields[:-1])
        categories = tf.stack(fields[-1])
        return data, categories
    
    ocean_dataset = dataset.map(convert)
    
    # separate features, labels, and categories
    X_float = np.array([data[:-1] for data, categories in ocean_dataset.as_numpy_iterator()])
    y = np.array([data[-1:] for data, categories in ocean_dataset.as_numpy_iterator()])
    X_byteStrings = np.array([cat for data, cat in ocean_dataset.as_numpy_iterator()])
    
    # simple function to decode bytestring for vectorization 
    def decode(byteString):
        return byteString.decode()

    # obtains category data in string with 1 extra dimension form because
    # tf input data has two dimensions (number of data instance, size of features)
    byte_to_str = np.vectorize(decode)
    X_categories = byte_to_str(X_byteStrings)
    # X_categories = np.expand_dims(X_categories, axis=1)
    
    # indices to separate train, valid, test datasets according to separate powers of two
    ind1 = pow(2, 14)
    ind2 = ind1 + pow(2, 11)
    ind3 = ind2 + pow(2, 10)
    
    # return datasets
    xtr = X_float[:ind1] # X_train
    xctr = X_categories[:ind1] # X_categories_train
    ytr = y[:ind1] # y_train

    xv = X_float[ind1:ind2] # X_valid
    xcv = X_categories[ind1:ind2] # X_categories_valid
    yv = y[ind1:ind2] # y_valid

    xt = X_float[ind2:ind3] # X_test
    xct = X_categories[ind2:ind3] # X_categories_test
    yt = y[ind2:ind3] # y_test

    return xtr, xctr, ytr, xv, xcv, yv, xt, xct, yt

# chapter 14
def mnist_784():
    """
    MNIST-784 consists of 70,000 28x28 grayscale images of handwritten digits (0 to 9) and their 
    corresponding labels. mnist_784 function loads such images and returns numpy arrays of them
    
    return
    ------
    numpy array: float64
        numpy arrays of 28 x 28 images with one channel (greyscale image) and its targets
        train, valid, and test datasets are created
    """
    mnist = fetch_openml('mnist_784', parser='auto', version=1)
    X, y = mnist["data"], mnist["target"]
    
    # convert the loaded dataset into numpy format
    X = pd.DataFrame.to_numpy(X)
    y = y.to_numpy()
    
    # transform data format and adjust its dimensions
    X = X/255.0
    X = np.expand_dims(X, axis=-1) # adding extra dimension for one channel
    X = np.reshape(X, (70000, 28, 28, 1))
    y = keras.utils.to_categorical(y.astype('int32'), num_classes=10)

    # separate data into train, valid, test datasets
    X_train = X[:55000]
    y_train = y[:55000]
    X_valid = X[55000:65000]
    y_valid = y[55000:65000]
    X_test = X[65000:]
    y_test = y[65000:]
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def bus_rail_df():
    """
    load rail data from CTA ridership from 2016-01 to 2019-06 and transforms them into panda
    dataframes to be returned
    
    return
    ------
        panda dataframe of bus and rail data with service date as our index axis
    """
    # load csv data and trim it
    path = Path("CTA.Ridership.Daily.Boarding.Totals.csv")
    df = pd.read_csv(path, parse_dates=["service_date"])
    df.columns = ["date", "day_type", "bus", "rail", "total"] # shorter names
    df = df.sort_values("date").set_index("date")
    df = df.drop("total", axis=1) # no need for total, it's just bus + rail
    df = df.drop_duplicates() # remove duplicated months (2011-10 and 2014-07)
    return df

def rail_train_data():
    """
    load rail data from CTA ridership from 2016-01 to 2019-06 and returns TF dataset
    for chapter 15
    
    return
    ------
        train and valid tf dataset in 32 batch
    """
    # panda dataframe of bus and rail data
    df = bus_rail_df()
    
    # split the data into train, valid, text in unit of millions 
    rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
    rail_valid = df["rail"]["2019-01":"2019-05"] / 1e6
    rail_test = df["rail"]["2019-06":] / 1e6
    
    # predict the outcome based on the previous 56 days data (8 weeks)
    seq_length = 56

    # shuffled train dataset in batch of 32
    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        rail_train.to_numpy(),
        targets=rail_train[seq_length:],
        sequence_length=seq_length,
        batch_size=32,
        shuffle=True,
        seed=42
    )
    
    # valid dataset in batch of 32
    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        rail_valid.to_numpy(),
        targets=rail_valid[seq_length:],
        sequence_length=seq_length,
        batch_size=32
    )
    
    # test dataset in batch of 32
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        rail_test.to_numpy(),
        targets=rail_valid[seq_length:],
        sequence_length=seq_length,
        batch_size=32
    )
    
    return train_ds, valid_ds, test_ds

def bus_rail_train_data():
    """
    load bus and rail data from CTA ridership from 2016-01 to 2019-06 and returns TF dataset
    for chapter 15
    
    return
    ------
        train, valid tf dataset in 32 batch
    """
    # panda dataframe of bus and rail data
    df = bus_rail_df()
    
    # dataset rescale and modification
    df_mulvar = df[["bus", "rail"]] / 1e6 # use both bus & rail series as input
    df_mulvar["next_day_type"] = df["day_type"].shift(-1) # we know tomorrow's type
    df_mulvar = pd.get_dummies(df_mulvar) # one-hot encode the day type

    # split the data into train, valid, text in 
    mulvar_train = df_mulvar["2016-01":"2018-12"] 
    mulvar_valid = df_mulvar["2019-01":"2019-05"] 
    mulvar_test = df_mulvar["2019-06":] 
    
    # predict the outcome based on the previous 56 days data (8 weeks)
    seq_length = 56

    # shuffled train dataset in batch of 32
    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        mulvar_train.to_numpy(),
        targets=mulvar_train["rail"][seq_length:],
        sequence_length=seq_length,
        batch_size=32,
        shuffle=True,
        seed=42
    )
    
    # valid dataset in batch of 32
    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        mulvar_valid.to_numpy(),
        targets=mulvar_valid["rail"][seq_length:],
        sequence_length=seq_length,
        batch_size=32
    )
    
    # test dataset in batch of 32
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        mulvar_test.to_numpy(),
        targets=mulvar_valid["rail"][seq_length:],
        sequence_length=seq_length,
        batch_size=32
    )
    
    return train_ds, valid_ds, test_ds

def rail_mulvar_pd():
    """
    load rail data from CTA ridership from 2016-01 to 2019-06 and returns them in panda dataframes
    for chapter 15
    
    return
    ------
        train, valid, test panda dataframes for rail and both
    """
    # panda dataframe of bus and rail data
    df = bus_rail_df()
        
    df_mulvar = df[["bus", "rail"]] / 1e6 # use both bus & rail series as input
    df_mulvar["next_day_type"] = df["day_type"].shift(-1) # we know tomorrow's type
    df_mulvar = pd.get_dummies(df_mulvar) # one-hot encode the day type
    
    # split the data into train, valid, text in unit of millions 
    rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
    rail_valid = df["rail"]["2019-01":"2019-05"] / 1e6
    rail_test = df["rail"]["2019-06":] / 1e6

    mulvar_train = df_mulvar["2016-01":"2018-12"]
    mulvar_valid = df_mulvar["2019-01":"2019-05"]
    mulvar_test = df_mulvar["2019-06":]
    
    return rail_train, rail_valid, rail_test, mulvar_train, mulvar_valid, mulvar_test

def to_windows(dataset, length):
    """
    creates sub-sequences of dataset that are shifted by 1 from the preceding sequence.
    target values are the next elements to the last element of each sequence
    
    return
    ------
        TF dataset of batches (in length) of sequences and their target values
    """
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))

@tf.autograph.experimental.do_not_convert # WARNING:tensorflow:AutoGraph could not transform
def seq2seq_dataset(
    seq_length=56, 
    ahead=14, 
    target_col=1,
    batch_size=32):
    """
    load rail data from CTA ridership from 2016-01 to 2019-06 and returns sequences of input
    and output data
    for chapter 15
    
    return
    ------
        train, valid, test panda dataframes for rail and both
    """
    # panda dataframe of bus and rail data
    df = bus_rail_df()
        
    df_mulvar = df[["bus", "rail"]] / 1e6 # use both bus & rail series as input
    df_mulvar["next_day_type"] = df["day_type"].shift(-1) # we know tomorrow's type
    df_mulvar = pd.get_dummies(df_mulvar) # one-hot encode the day type
    
    # split the data into train, valid, text in unit of millions 
    mulvar = [
        df_mulvar["2016-01":"2018-12"], 
        df_mulvar["2019-01":"2019-05"], 
        df_mulvar["2019-06":]]
    
    # create sequences input and output of data
    for ind in range(3):
        ds = to_windows(tf.data.Dataset.from_tensor_slices(mulvar[ind]), ahead + 1)
        ds = to_windows(ds, seq_length).map(lambda S: (S[:, 0], S[:, 1:, 1]))
        if ind == 0:
            ds = ds.shuffle(8*batch_size, seed=42)
        mulvar[ind] = ds.batch(batch_size)

    return mulvar[0], mulvar[1], mulvar[2]

# custom
def cifar_10():
    """
    CIFAR-10 consists of 60,000 32x32 RBG images of 10 different objects (0 to 9) and their 
    corresponding labels. cifar_10 function loads such images and returns numpy arrays of them.
    categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
    
    return
    ------
    numpy array: float32
        numpy arrays of 32x32 RBG images (3 channels} and its targets
        train, valid, and test datasets are created
    """
    # fetch CIFAR-10 dataset
    cifar = fetch_openml('cifar_10', parser='auto')
    X_cifar, y_cifar = cifar["data"], cifar["target"]

    # convert the loaded dataset into numpy format
    X = pd.DataFrame.to_numpy(X_cifar)
    y = y_cifar.to_numpy()

    # transform data format and adjust its dimensions
    X_reshaped = np.reshape(X, (60000, 3, 32, 32)).transpose(0,2,3,1).astype("uint8")
    X_scaled = X_reshaped/255.0
    y = keras.utils.to_categorical(y.astype('int32'), num_classes=10)

    # separate data into train, valid, test datasets
    X_train = X_scaled[:50000]
    y_train = y[:50000]
    X_valid = X_scaled[50000:55000]
    y_valid = y[50000:55000]
    X_test = X_scaled[55000:]
    y_test = y[55000:]
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test
