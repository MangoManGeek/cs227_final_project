import numpy as np

TIME = 30
# TIME = 3

def augmentation(x, y):
    size = x.shape
    x_list = []
    x_list.append(x)
    y_list = []
    y_list.append(y)
    #
    # shift = (np.max(x, axis=1) - np.min(x, axis=1)) * 0.01
    # print(shift.shape)
    # shift = np.repeat(shift, size[1], axis=1)
    # print(shift.shape)
    # shift = np.reshape(shift, [size[0], size[1], 1])
    # print(shift.shape)
    # print(shift)

    for i in range(TIME):
        uniform = np.random.uniform(-0.01, 0.01, size=size)
        # x_list.append(x + uniform + shift)
        # x_list.append(x + uniform - shift)
        x_list.append(x + uniform)
        # y_list.append(y)
        y_list.append(y)

    x = np.concatenate(x_list,axis=0)
    x = x.astype('float32')
    print(type(x[0, 0, 0]))
    y = np.concatenate(y_list,axis=0)
    return x, y
