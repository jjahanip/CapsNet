import os
import time
import h5py
import numpy as np
import pandas as pd
import multiprocessing
import skimage.io as io
from functools import partial
from skimage.transform import resize


input_dir = r'/brazos/roysam/50_plex/Set#1_S1/final'
bbxs_file = r'/brazos/roysam/50_plex/Set#1_S1/detection_results/bbxs_detection.txt'
channelInfo_file = r'/brazos/roysam/50_plex/Set#1_S1/scripts/channel_info.csv'
parallel = True

margin = 5
crop_size = (50, 50)
topN = 5000

biomarkers = ['DAPI', 'Histones', 'NeuN', 'S100', 'Olig2', 'Iba1', 'RECA1']


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


def get_crop(image, bbx, margin=0):
    """
    crop large image with extra margin
    :param image: large image
    :param bbx: [xmin, ymin, xmax, ymax]
    :param margin: margin from each side
    :return:
    """
    return image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :]


def main():

    if parallel:
        try:
            cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            cpus = 2  # arbitrary default
        pool = multiprocessing.Pool(processes=cpus)

    # read channel info table
    assert os.path.isfile(channelInfo_file), '{} not found!'.format(channelInfo_file)
    chInfo = pd.read_csv(channelInfo_file, sep=',')

    # get channels full address from channel info table
    channel_names = [chInfo.loc[chInfo['Biomarker'] == bioM]['Channel'].values[0] for bioM in biomarkers]
    channel_names = [os.path.join(input_dir, ch) for ch in channel_names]

    # get image collection
    im_coll = io.imread_collection(channel_names, plugin='tifffile')
    image_size = im_coll[0].shape[::-1]
    images = io.concatenate_images(im_coll)
    images = np.moveaxis(images, 0, -1)     # put channel as last dimension

    # read bbxs file
    assert os.path.isfile(bbxs_file), '{} not found!'.format(bbxs_file)
    # if file exist -> load
    bbxs_table = pd.read_csv(bbxs_file, sep='\t')
    bbxs = bbxs_table[['xmin', 'ymin', 'xmax', 'ymax']].values

    # Generate dataset
    X = [get_crop(images, bbx, margin=margin) for bbx in bbxs]
    del images

    # calculate mean intensity of each image -> we need it later for generating labels
    meanInt = np.array([np.mean(x, axis=(0, 1)) for x in X])
    meanInt = meanInt[:, 2:]         # we don't need DAPI and Histones for classification

    ## preprocess
    # zero pad to the maximum dim
    max_dim = max((max([cell.shape[:2] for cell in X])))  # find maximum in each dimension
    if parallel:
        zero_pad_x = partial(zero_pad, dim=max_dim)
        X = pool.map(zero_pad_x, X)
    else:
        X = [zero_pad(cell, max_dim) for cell in X]

    # resize image specific size
    if parallel:
        resize_x = partial(resize, output_shape=crop_size, mode='constant', preserve_range=True)
        X = pool.map(resize_x, X)
    else:
        X = [resize(cell, crop_size, mode='constant', preserve_range=True) for cell in X]

    if parallel:
        pool.close()

    # Generate test set
    X_test = np.array(X)
    Y_test = np.zeros_like(meanInt, dtype=int)
    Y_test[np.arange(len(meanInt)), meanInt.argmax(1)] = 1

    with h5py.File('data.h5', 'w') as f:
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('Y_test', data=Y_test)
        f.create_dataset('bbxs', data=bbxs_table)
        f.create_dataset('image_size', data=image_size)
        f.create_dataset('biomarkers', data=[x.encode('UTF8') for x in biomarkers])


if __name__ == '__main__':
    start = time.time()
    main()
    print('*' * 50)
    print('*' * 50)
    print('Pipeline finished successfully in {} seconds.'.format(time.time() - start))
