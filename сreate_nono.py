# let's look at a 10 by 10 board for example:
# so we want to get from each row/col an array of it's constraints
# then we want to complete it to be of length 5 (pad with zeros)
# since the max number of constraints is 5 for each row and col [1,1,1,1,1]
# then we want to normalize the data (divide by 10) and flatten all inputs.
# total of 20 (# of rows+cols) X 5 (len of each constraint) = 100
# so the input layer has 100 neurons

import numpy as np
import math

COL = 0
ROW = 1


def get_board_and_constraints(n=10):
    """ create a random board and get its constraints """

    # get board
    y = np.array([np.random.randint(2, size=n) for _ in range(n)])

    # get constraints for rows and for colls
    y_pad = np.zeros((y.shape[0] + 2, y.shape[1] + 2))
    y_pad[1:-1, 1:-1] = y

    cols = [np.diff(np.nonzero(c == 1)[0].reshape(-1, 2), axis=1)[:, 0]
            for c in np.abs(np.diff(y_pad, axis=0)).T[1:-1]]

    rows = [np.diff(np.nonzero(r == 1)[0].reshape(-1, 2), axis=1)[:, 0]
            for r in np.abs(np.diff(y_pad, axis=1))[1:-1]]

    return y, rows, cols



def padder(arr, n=10):
    """ pad the constraints to the length of math.ceil(n/2)"""
    max_blocks = math.ceil(n / 2)
    for i in range(len(arr)):
        arr[i] = np.pad(arr[i], (0, max_blocks - len(arr[i])), 'constant')


def get_set(n=10, size=500000):
    """ create a whole training set """
    X_set, Y_set = [], []
    for i in range(size):  # 500000
        # print(i)
        y, rows, cols = get_board_and_constraints(n)
        rows.extend(cols)
        const = rows
        # const = np.concatenate([rows, cols])
        padder(const, n)
        Y_set.append(y)
        X_set.append(const)

    preprocess(X_set, Y_set, n)
    return np.array(X_set), np.array(Y_set)


def preprocess(X_set, Y_set, board_size=10):
    """ normalize data and flatten the board """
    normal = board_size
    for i in range(len(X_set)):
        X_set[i] = np.true_divide(np.array(X_set[i]).flatten(), normal)
    for i in range(len(Y_set)):
        Y_set[i] = np.array(Y_set[i]).flatten()



def main():
    n = 3
    X_set, Y_set = get_set(n=n, size=100000)
    print(X_set)
    preprocess(X_set, Y_set, n)
    print(X_set)


if __name__ == '__main__':
    main()
