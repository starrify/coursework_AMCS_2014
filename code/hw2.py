#! /usr/bin/env python

# This is part of coursework for Applied Mathematics for Computer Science
# by Pengyu CHEN (pengyu[at]libstarrify.so)
# COPYLEFT, ALL WRONGS RESERVED.

import numpy
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec

data_file_path = './data/optdigit3.raw'


# For testing purpose only
def _test():
    try:
        with open(data_file_path, 'r') as f:
            lines = f.readlines()
            tmp_mat = []
            while lines:
                tmp_lines = lines[:32]
                # there's a blank line between each digits
                lines = lines[33:]
                tmp_str = ''.join(tmp_lines).replace('\n', '')
                tmp_vec = [1. if x == '1' else 0. for x in tmp_str]
                tmp_mat += [tmp_vec]
                pass
            Xt = numpy.matrix(tmp_mat)
        pass
    except:
        raise('error reading %s' % data_file_path)

    N, M = Xt.shape
    # according to the data
    assert(N == 572)
    assert(M == 1024)
    mean = Xt.mean(0)
    Xt_reg = Xt - mean.repeat(N, axis=0)
    V, D_diag, Ut = numpy.linalg.svd(Xt_reg, full_matrices=False)
    feature_dim = 2
    X_proj = Xt_reg * Ut.getT()[:, :feature_dim]
    X_proj_back = Xt_reg * Ut.getT()[:, :feature_dim] * Ut[:feature_dim, :]

    sample_x = [-6, -4, -2, 0, 2, 4]
    sample_y = [-4, -2, 0, 2, 4]
    sample_points = numpy.matrix([[x, y] for x in sample_x for y in sample_y])
    sample_idx = [
        numpy.argmin([
            numpy.inner(dis, dis) for dis in X_proj - p.repeat(N, axis=0)])
        for p in sample_points]
    X_sample = X_proj[sample_idx]
    assert(feature_dim == 2)

    fig = plt.figure()
    subplot = fig.add_subplot(1, 1, 1)
    subplot.plot(
        X_proj.getT()[0].getA1(), X_proj.getT()[1].getA1(),
        marker='.', linestyle='None')
    subplot.plot(
        X_sample.getT()[0].getA1(), X_sample.getT()[1].getA1(),
        marker='o', color='r', linestyle='None')
    subplot.set_xlabel('First Principal Component')
    subplot.set_ylabel('Second Principal Component')
    fig.show()

    fig = plt.figure()
    grids = gridspec.GridSpec(
        len(sample_y), len(sample_x),
        wspace=0.1, hspace=0.1)
    for yid in range(len(sample_y)):
        for xid in range(len(sample_x)):
            flatid = yid * len(sample_x) + xid
            figid = sample_idx[flatid]
            raw = Xt[figid]
            raw = raw.reshape(32, 32)
            grid = grids[flatid]
            subplot = plt.Subplot(fig, grid)
            subplot.set_xticks([])
            subplot.set_yticks([])
            # list of color maps:
            # http://matplotlib.org/examples/color/colormaps_reference.html
            subplot.imshow(
                raw,
                extent=[0, 1, 0, 1],
                cmap='binary',
                interpolation='gaussian')
            fig.add_subplot(subplot)
    fig.show()

    fig = plt.figure()
    grids = gridspec.GridSpec(
        len(sample_y), len(sample_x),
        wspace=0.1, hspace=0.1)
    for yid in range(len(sample_y)):
        for xid in range(len(sample_x)):
            flatid = yid * len(sample_x) + xid
            figid = sample_idx[flatid]
            raw = X_proj_back[figid]
            raw = raw.reshape(32, 32)
            grid = grids[flatid]
            subplot = plt.Subplot(fig, grid)
            subplot.set_xticks([])
            subplot.set_yticks([])
            # list of color maps:
            # http://matplotlib.org/examples/color/colormaps_reference.html
            subplot.imshow(
                raw,
                extent=[0, 1, 0, 1],
                cmap='binary')
            fig.add_subplot(subplot)
    fig.show()

 
    plt.show()
    pass


def main():
    _test()
    pass


if __name__ == '__main__':
    main()
