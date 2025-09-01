import numpy as np


def split_data(
    x_train,
    y_train,
    mp_size,
    dp_size,
    rank,
):
    """The function for splitting the dataset uniformly across data parallel groups

    Parameters
    ----------
        x_train : np.ndarray float32
            the input feature of MNIST dataset in numpy array of shape (data_num, feature_dim)

        y_train : np.ndarray int32
            the label of MNIST dataset in numpy array of shape (data_num,)

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        rank : int
            the corresponding rank of the process

    Returns
    -------
        split_x_train : np.ndarray float32
            the split input feature of MNIST dataset in numpy array of shape (data_num/dp_size, feature_dim)

        split_y_train : np.ndarray int32
            the split label of MNIST dataset in numpy array of shape (data_num/dp_size, )

    Note
    ----
        please split the data uniformly across data parallel groups and
        do not shuffle the index as we will shuffle them later
    """

    data_num = x_train.shape[0]
    per_data_num = data_num // dp_size
    start_index = rank // mp_size * per_data_num
    end_index = start_index + per_data_num
    split_x_train = x_train[start_index:end_index]
    split_y_train = y_train[start_index:end_index]
    return split_x_train, split_y_train
