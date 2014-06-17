#! /usr/bin/env python

# This is part of coursework for Applied Mathematics for Computer Science
# by Pengyu CHEN (pengyu[at]libstarrify.so)
# COPYLEFT, ALL WRONGS RESERVED.

import functools
import operator
import numpy
import scipy.linalg
import scipy.stats
import matplotlib.pyplot as plt


def gaussian_dist_2D(
        mu=numpy.array([0, 0]), sigma=numpy.array([(1, 0), (0, 1)]),
        size=1, size_one_list=False):
    assert(numpy.linalg.det(sigma) != 0)
    sigma_sqrt = scipy.linalg.sqrtm(sigma)
    ret = [
        numpy.dot(numpy.random.normal(size=(1, 2)), sigma_sqrt) + mu
        for i in range(size)]
    if size == 1 and not size_one_list:
        ret = ret[0]
    else:
        ret = numpy.vstack(ret)
    return ret


def EM_2D(data, comp_size, max_iter=256, epsilon=1e-6, report_step=None):
    data_size, data_dim = data.shape
    assert(data_dim == 2)
    p_assign = numpy.zeros((data_size, comp_size))
    p = numpy.ones(comp_size) / comp_size
    mu = [numpy.random.normal(size=data_dim) for i in range(comp_size)]
    sigma = [numpy.eye(data_dim) for i in range(comp_size)]

    for _i in range(max_iter):
        # E step
        for i in range(data_size):
            p_assign[i] = numpy.array([
                p[j] * scipy.stats.multivariate_normal.pdf(
                    data[i], mu[j], sigma[j])
                for j in range(comp_size)])
            p_assign[i] /= numpy.sum(p_assign[i])

        old_params = [numpy.copy(x) for x in [p, mu, sigma]]
        # M step
        p = numpy.sum(p_assign, axis=0) / data_size
        for i in range(comp_size):
            mu[i] = (
                numpy.dot(data.transpose(), p_assign[:, i]) / data_size /
                p[i] if p[i] else -numpy.ones(comp_size))
            data_norm = data - numpy.repeat([mu[i]], data_size, axis=0)
            sigma[i] = (
                functools.reduce(
                    numpy.dot, [
                        data_norm.transpose(),
                        numpy.diag(p_assign[:, i]),
                        data_norm])
                / data_size / p[i]
                if p[i] else -numpy.ones(comp_size, comp_size))

        params = [p, mu, sigma]
        diff = sum([
            numpy.linalg.norm(x - y)
            for x, y in zip(old_params, params)])

        def report_status():
            print('Iteration %d: diff=%e' % (_i, diff))
            pass

        if diff < epsilon:
            report_status()
            break
        if report_step and _i % report_step == 0:
            report_status()
            pass

    return p, mu, sigma


# For testing purpose only

def _test():
    numpy.random.seed(0xdeadbeef)

    sample_size = 1000
    param_p = [0.2, 0.5, 0.3]
    param_size = [int(sample_size * p) for p in param_p]
    param_mu = [numpy.array(mu) for mu in [(0, 0), (4, 1), (-2, 3)]]
    param_sigma = [numpy.array(sigma) for sigma in [
        ((1, 0.7), (0.7, 1)),
        ((1.2, -0.4), (-0.4, 1.2)),
        ((1, 0.2), (0.2, 1)),
        ]]

    # Schwartzian transform
    param_p, param_size, param_mu, param_sigma = zip(*sorted(
        zip(param_p, param_size, param_mu, param_sigma),
        key=operator.itemgetter(0),
        reverse=True))

    comp_size = len(param_size)
    grouped_data = [
        gaussian_dist_2D(mu, sigma, size)
        for size, mu, sigma in zip(param_size, param_mu, param_sigma)]
    est_p, est_mu, est_sigma = EM_2D(
        numpy.vstack(grouped_data), comp_size, report_step=5)

    print('sample size: %d' % sample_size)

    def tmp_print(msg_0, msg_1, data):
        print(msg_0)
        for msg, d in zip(msg_1, data):
            print(msg)
            for x in d:
                print(x)
        pass

    tmp_print(
        '\nGround truth:',
        ['p:', 'mu:', 'sigma:'],
        [param_p, param_mu, param_sigma])
    tmp_print(
        '\nEstimation:',
        ['p:', 'mu:', 'sigma:'],
        [est_p, est_mu, est_sigma])

    fig = plt.figure()
    plt.axis('equal')
    for data in grouped_data:
        plt.plot(
            data.transpose()[0], data.transpose()[1],
            marker='o', linestyle='None')

    fig.show()
    plt.show()
    pass


def main():
    _test()
    pass


if __name__ == '__main__':
    main()
