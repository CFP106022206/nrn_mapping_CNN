import copy
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def standard_mask(map1, map2):
    score_x = np.zeros((map1[0].shape[0], map1[0].shape[0]))
    score_y = np.zeros((map1[0].shape[0], map1[0].shape[0]))
    score_z = np.zeros((map1[0].shape[0], map1[0].shape[0]))

    for i in range(score_x.shape[0]):
        for j in range(score_x.shape[0]):
            score_x[i][j] = 1-np.abs(map1[0][i][j]-map2[0][i][j])

    for i in range(score_x.shape[0]):
        for j in range(score_x.shape[0]):
            score_y[i][j] = 1-np.abs(map1[1][i][j]-map2[1][i][j])

    for i in range(score_x.shape[0]):
        for j in range(score_x.shape[0]):
            score_z[i][j] = 1-np.abs(map1[2][i][j]-map2[2][i][j])

    return (np.average(score_x) + np.average(score_y) + np.average(score_z))/3


def map_shift(map1, sx, sy):
    _map = copy.deepcopy(map1)
    _map = np.roll(_map, sx, 1)
    _map = np.roll(_map, sy, 0)
    if sy >= 0:
        _map[0:sy, :] = 0
    else:
        _map[map1.shape[0]+sy:map1.shape[0], :] = 0
    if sx >= 0:
        _map[:, 0:sx] = 0
    else:
        _map[:, map1.shape[1]+sx:map1.shape[0]] = 0
    return _map


def mask_test_gpu(map1, map2, max_score, shift=3):
    _map1x = tf.pad(tf.constant(map1[0].reshape(1, map1[0].shape[0], map1[0].shape[1], 1)),
                    paddings=[[0, 0], [shift, shift], [shift, shift], [0, 0]])
    _map1y = tf.pad(tf.constant(map1[1].reshape(1, map1[1].shape[0], map1[1].shape[1], 1)),
                    paddings=[[0, 0], [shift, shift], [shift, shift], [0, 0]])
    _map1z = tf.pad(tf.constant(map1[2].reshape(1, map1[2].shape[0], map1[2].shape[1], 1)),
                    paddings=[[0, 0], [shift, shift], [shift, shift], [0, 0]])
    temp = _map1x.numpy()
    _map2x = tf.constant(map2[0].reshape(map2[0].shape[0], map2[0].shape[1], 1, 1))
    _map2y = tf.constant(map2[1].reshape(map2[1].shape[0], map2[1].shape[1], 1, 1))
    _map2z = tf.constant(map2[2].reshape(map2[2].shape[0], map2[2].shape[1], 1, 1))

    inner1 = tf.nn.convolution(_map1x, _map2x).numpy().reshape(2*shift+1, 2*shift+1)
    inner2 = tf.nn.convolution(_map1y, _map2y).numpy().reshape(2*shift+1, 2*shift+1)
    inner3 = tf.nn.convolution(_map1z, _map2z).numpy().reshape(2*shift+1, 2*shift+1)

    score_map = np.zeros((2*shift+1, 2*shift+1, 2*shift+1))
    for i in range(2*shift+1):
        for j in range(2*shift+1):
            for k in range(2*shift+1):
                score_map[i][j][k] += inner1[j][k] + inner2[k][i] + inner3[i][j]

    best_shift = np.unravel_index(np.argmax(score_map), (2*shift+1, 2*shift+1, 2*shift+1))
    best_score = score_map[best_shift[0]][best_shift[1]][best_shift[2]]/max_score
    score1 = inner1[best_shift[1]][best_shift[2]]
    score2 = inner2[best_shift[2]][best_shift[0]]
    score3 = inner3[best_shift[0]][best_shift[1]]

    return best_score, best_shift, score1, score2, score3


def mask_substract(map1, map2, shift=10):
    _map1x = tf.pad(tf.constant(map1[0].reshape(map1[0].shape[0], map1[0].shape[1])),
                    paddings=[[shift, shift], [shift, shift]])
    _map1y = tf.pad(tf.constant(map1[1].reshape(map1[1].shape[0], map1[1].shape[1])),
                    paddings=[[shift, shift], [shift, shift]])
    _map1z = tf.pad(tf.constant(map1[2].reshape(map1[2].shape[0], map1[2].shape[1])),
                    paddings=[[shift, shift], [shift, shift]])

    _map2x = tf.constant(map2[0])
    _map2y = tf.constant(map2[1])
    _map2z = tf.constant(map2[2])

    inner1 = np.zeros((2*shift+1, 2*shift+1))
    inner2 = np.zeros((2*shift+1, 2*shift+1))
    inner3 = np.zeros((2*shift+1, 2*shift+1))

    for i in range(2*shift+1):
        for j in range(2*shift+1):
            inner1[i][j] = tf.reduce_sum(tf.abs(tf.subtract(_map1x[0+i:map1[0].shape[0]+i, 0+j:map1[0].shape[0]+j],
                                                            _map2x)))
            inner2[i][j] = tf.reduce_sum(tf.abs(tf.subtract(_map1y[0+i:map1[0].shape[0]+i, 0+j:map1[0].shape[0]+j],
                                                            _map2y)))
            inner3[i][j] = tf.reduce_sum(tf.abs(tf.subtract(_map1z[0+i:map1[0].shape[0]+i, 0+j:map1[0].shape[0]+j],
                                                            _map2z)))

    score_map = np.zeros((2*shift+1, 2*shift+1, 2*shift+1))
    for i in range(2*shift+1):
        for j in range(2*shift+1):
            for k in range(2*shift+1):
                score_map[i][j][k] += inner1[j][k] + inner2[k][i] + inner3[i][j]

    best_shift = np.unravel_index(np.argmin(score_map), (2*shift+1, 2*shift+1, 2*shift+1))
    best_score = score_map[best_shift[0]][best_shift[1]][best_shift[2]]

    return best_score, best_shift


def mask_test(map1, map2, shift=10):
    max_score = np.max((np.count_nonzero(map1), np.count_nonzero(map2)))
    shift_list = [[i, j, k] for i in range(-shift, shift+1) for j in range(-shift, shift+1) for k in range(-shift, shift+1)]
    best_score = 0.0
    best_shift = []

    for oper in shift_list:
        _map = np.array([map_shift(map2[0], oper[1], oper[2]),
                         map_shift(map2[1], oper[2], oper[0]),
                         map_shift(map2[2], oper[0], oper[1])])
        temp_score = np.sum(np.multiply(map1, _map))

        if temp_score > best_score:
            best_score = temp_score
            best_shift = oper

    return best_score/max_score, best_shift


def sig_ave_mask(map1, map2, optional=9):
    length = map1[0].shape[0] - optional + 1

    _score = 0
    _N = 0
    for i in range(length):
        for j in range(length):
            if map1[0][i + optional // 2][j + optional // 2] != -1 or map2[0][i + optional // 2][j + optional // 2] != -1:
                _N += 1
                _score += 1 - np.min(np.abs(map1[0][i + optional // 2][j + optional // 2] -
                                            map2[0][i:i + optional, j:j + optional]))
    score_x = _score/_N

    _score = 0
    _N = 0
    for i in range(length):
        for j in range(length):
            if map1[1][i + optional // 2][j + optional // 2] != -1 or map2[1][i + optional // 2][j + optional // 2] != -1:
                _N += 1
                _score += 1 - np.min(np.abs(map1[1][i + optional // 2][j + optional // 2] -
                                            map2[1][i:i + optional, j:j + optional]))
    score_y = _score/_N

    _score = 0
    _N = 0
    for i in range(length):
        for j in range(length):
            if map1[2][i + optional // 2][j + optional // 2] != -1 or map2[2][i + optional // 2][j + optional // 2] != -1:
                _N += 1
                _score += 1 - np.min(np.abs(map1[2][i + optional // 2][j + optional // 2] -
                                            map2[2][i:i + optional, j:j + optional]))
    score_z = _score/_N
    score = [score_x, score_y, score_z]
    score.sort()
    score = score[0] + score[1] + score[2]

    return score/3
