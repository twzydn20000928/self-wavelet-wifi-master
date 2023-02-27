import numpy as np
from tqdm import tqdm
from util import log_f_ch, load_mat
import os
def jitter(x, sigma=0.15):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0]))
    return np.multiply(x, factor[:, np.newaxis])

def rotation(x):
    flip = np.random.choice([-1, 1])
    return np.multiply(x, flip[np.newaxis,np.newaxis])

def magnitude_warp(x, sigma=0.4, knot=10):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2))
    warp_steps = (np.ones((x.shape[0],1))*(np.linspace(0, x.shape[1]-1., num=knot+2)))
    ret = np.zeros_like(x)
    warper = np.array([CubicSpline(warp_steps[dim, :], random_warps[dim,:])(orig_steps) for dim in range(x.shape[0])])
    ret = x * warper

    return ret

def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for dim in range(x.shape[0]):
        ret[dim,:] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), x[dim, starts:ends]).T
    return ret

def window_warp(x, window_ratio=0.1, scales=None):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    if scales is None:
        scales = [0.5, 2.]
    warp_scales = np.random.choice(scales)
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1)
    window_ends = (window_starts + warp_size).astype(int)
    ret = np.zeros_like(x)
    for dim in range(x.shape[0]):
        start_seg = x[dim, :window_starts]
        window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales)), window_steps,  x[dim,window_starts:window_ends])
        end_seg = x[dim, window_ends:]
        warped = np.concatenate((start_seg, window_seg, end_seg))
        ret[dim,:] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped)
    return ret

def mean_mix(input_data, data_path, file):
    index_list = ['01','02','03','04','06','07','08','09']
    # np.random.shuffle(index_list)
    name, _ = file.split('_')
    data_list = [np.expand_dims(input_data,axis=0)]
    # for file_i in index_list[:2]:
    #     file_path = os.path.join(data_path, f'{name}_{file_i}.h5')
    #     data_list.append(np.expand_dims(load_mat(file_path)['amp'],axis=0))
    for i in range(1):
        file_i = np.random.choice(index_list)
        file_path = os.path.join(data_path, f'{name}_{file_i}.h5')
        data_list.append(np.expand_dims(load_mat(file_path)['amp'], axis=0))
    return np.mean(np.concatenate(data_list, axis=0), axis=0)