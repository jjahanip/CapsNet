import os
import time
import h5py
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
import skimage.io as io
from skimage.transform import resize

import matplotlib.pyplot as plt

input_dir = r'E:\50_plex\tif\pipeline2\final'
bbxs_file = r'E:\50_plex\tif\pipeline2\classification_results/classification_table.csv'
parallel = False

margin = 5
image_size = (50, 50)
topN = 5000

# for not-existing channel put ''
biomarkers = {'DAPI': 'S1_R2C1.tif',
              'Histones': 'S1_R2C2.tif',
              'NeuN': 'S1_R2C2.tif',
              'S100': 'S1_R3C5.tif',
              'Olig2': 'S1_R1C9.tif',
              'Iba1': 'S1_R1C5.tif',
              'RECA1': 'S1_R1C6.tif'}


def zero_pad(image, dim):
    """
    pad zeros to the image in the first and second dimension
    :param image: image array [width*height*channel]
    :param dim: new dimension
    :return: padded image
    """
    pad_width = ((np.ceil((dim - image.shape[0]) / 2), np.floor((dim - image.shape[0]) / 2)),
                 (np.ceil((dim - image.shape[1]) / 2), np.floor((dim - image.shape[1]) / 2)),
                 (0, 0))
    return np.pad(image, np.array(pad_width, dtype=int), 'constant')


def to_square(image):
    """
    pad zeros to the image to make it square
    :param image: image array [width*height*channel]
    :param dim: new dimension
    :return: padded image
    """
    dim = max(image.shape[:2])
    if image.shape[0] >= image.shape[1]:
        pad_width = ((0, 0),
                     (np.ceil((dim - image.shape[1]) / 2), np.floor((dim - image.shape[1]) / 2)),
                     (0, 0))
    else:
        pad_width = ((np.ceil((dim - image.shape[0]) / 2), np.floor((dim - image.shape[0]) / 2)),
                     (0, 0),
                     (0, 0))
    return np.pad(image, np.array(pad_width, dtype=int), 'constant')


def main():
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default

    # read bbxs file
    assert os.path.isfile(bbxs_file), '{} not found!'.format(bbxs_file)
    # if file exist -> load
    bbxs_table = pd.read_csv(bbxs_file, sep=',')
    bbxs = bbxs_table[['xmin', 'ymin', 'xmax', 'ymax']].values
    labels = bbxs_table[['NeuN', 'S100', 'Olig2', 'Iba1', 'RECA1']].values.astype(int)

    # get bounding boxes in the center of brain
    # inside_cells = (bbxs[:, 0] >= 6000) & (bbxs[:, 2] <= 38000) & (bbxs[:, 1] >= 4500) & (bbxs[:, 3] <= 24000)
    inside_cells = (bbxs[:, 0] >= 20000) & (bbxs[:, 2] <= 24000) & (bbxs[:, 1] >= 13000) & (bbxs[:, 3] <= 17000)

    bbxs = bbxs[inside_cells, :]
    labels = labels[inside_cells, :]

    # remove cells with no class
    bbxs = bbxs[~np.all(labels == 0, axis=1)]
    classes = labels[~np.all(labels == 0, axis=1)]

    # shuffle the bounding boxes
    permutation = np.random.permutation(bbxs.shape[0])
    bbxs = bbxs[permutation, :]
    classes = classes[permutation, :]

    # get images
    image_size = io.imread_collection(os.path.join(input_dir, biomarkers['DAPI']), plugin='tifffile')[0].shape
    images = np.zeros((image_size[0], image_size[1], 7), dtype=np.uint16)

    # for each biomarker read the image and replace the black image if the channel is defined
    for i, bioM in enumerate(biomarkers.keys()):
        if biomarkers[bioM] != "":
            images[:, :, i] = io.imread(os.path.join(input_dir, biomarkers[bioM]))

    # from utils import bbxs_image
    # bbxs_image('all.tif', bbxs, images[:, :, 0].shape[::-1])

    # crop image (with extra margin) to get each sample (cell)
    def get_crop(image, bbx, margin=0):
        return image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :]

    ################### GENERATE IMAGES ###############################
    # get the crops
    cells = [get_crop(images, bbx, margin=margin) for bbx in bbxs]
    del images

    # zero pad to the maximum dim
    max_dim = max((max([cell.shape[:2] for cell in cells])))  # find maximum in each dimension
    if parallel:
        zero_pad_x = partial(zero_pad, dim=max_dim)
        with multiprocessing.Pool(processes=cpus) as pool:
            new_cells = pool.map(zero_pad_x, cells)
    else:
        new_cells = [zero_pad(cell, max_dim) for cell in cells]


    # resize image specific size
    if parallel:
        resize_x = partial(resize, output_shape=image_size, mode='constant', preserve_range=True, anti_aliasing=True)
        with multiprocessing.Pool(processes=cpus) as pool:
            new_new_cells = pool.map(resize_x, new_cells)
    else:
        # new_new_cells = [resize(cell, image_size, mode='constant', preserve_range=True, anti_aliasing=True) for cell in new_cells]
        new_new_cells = []
        for id, cell in enumerate(new_cells):
            temp = resize(cell, image_size, mode='constant', preserve_range=True)
            new_new_cells.append(temp)

    # visualize
    id = 140
    fig = plt.figure(figsize=(10,2))
    for i in range(7):
        plt.subplot(1, 7, i + 1)
        plt.imshow(new_cells[id][:, :, i], cmap='gray', vmin=0, vmax=np.max(new_cells[id]))
        plt.title(list(biomarkers)[i+2])
    plt.tight_layout()
    fig.suptitle('LABEL = {}'.format(biomarkers[np.argmax(classes[id])+2]))
    plt.show()

    cells = np.array(new_new_cells)

    # visualize
    # from utils import bbxs_image
    # bbxs_image(biomarkers[2] + '.tif', bbxs[labels[:, 0] == 1, :], images[:, :, 3].shape[::-1], color='red')

    # split dataset to train, test and validation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(cells, labels, test_size=0.2)

    with h5py.File('data.h5', 'w') as f:
        f.create_dataset('x_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('x_test', data=X_test)
        f.create_dataset('y_test', data=y_test)


if __name__ == '__main__':
    start = time.time()
    main()
    print('*' * 50)
    print('*' * 50)
    print('Pipeline finished successfully in {} seconds.'.format(time.time() - start))