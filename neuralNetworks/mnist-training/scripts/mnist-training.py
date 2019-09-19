import gzip
import pickle
import _pickle
import network
import numpy as np

def load_data():
    file = gzip.open("../data/mnist.pkl.gz", "rb")
    # tr_d, val_d, ts_d = pickle.load(file)
    file.seek(0)
    tr_d, val_d, ts_d = _pickle.load(file, encoding='latin1')
    return (tr_d, val_d, ts_d)

def prepare_data(data, vector_form):
    """data is a tuple which contains two lists : features and labels"""
    data_input = [ np.reshape(x,(784,1)) for x in data[0]]
    if vector_form:
        label = [ vectorized_result(y) for y in data[1]]
    else:
        label = data[1]
    new_data = list(zip(data_input, label))
    return new_data
def load_data_wrapper():
    tr_d, val_d, ts_d = load_data()
    training_data = prepare_data(tr_d, True)
    validation_data = prepare_data(val_d, False)
    testing_data = prepare_data(ts_d, False)
    return (training_data, validation_data, testing_data)

def vectorized_result(x):
    unit_vector = np.zeros((10,1))
    unit_vector[x] = 1.0
    return unit_vector

net = network.Network([784,30,10])

training_data, validation_data, testing_data = load_data_wrapper()

net.SGD(training_data, 30, 10, 3, testing_data)