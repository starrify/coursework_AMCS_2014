#! /usr/bin/env python

# This is part of coursework for Applied Mathematics for Computer Science
# by Pengyu CHEN (pengyu[at]libstarrify.so)
# COPYLEFT, ALL WRONGS RESERVED.

import functools
import numpy
import scipy.linalg
import scipy.stats
import matplotlib.pyplot as plt


def LM(x0, f, df, ddf, max_iter=256, epsilon=1e-6, report_step=None):
    x = x0
    dim = x.shape[0]
    mu = 1
    f_s_k, q_k = 0, 0

    for _i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        ddfx = ddf(x)
        if numpy.linalg.norm(dfx) < epsilon:
            break
        while True:
            try:
                M = ddfx + mu * numpy.identity(dim)
                L = numpy.linalg.cholesky(M)
                break
            except numpy.linalg.LinAlgError:
                mu *= 4
        s = numpy.linalg.solve(M, -dfx)
        new_f_s_k = f(x + s)
        new_q_k = fx + dfx.transpose().dot(s) + \
            s.transpose().dot(ddfx).dot(s) / 2
        r_k = (new_f_s_k - f_s_k) / (new_q_k - q_k)
        f_s_k, q_k = new_f_s_k, new_q_k
        if r_k < 0.25:
            mu *= 4
        elif r_k > 0.75:
            mu /= 2
        if r_k > 0:
            x += s

        if report_step and _i % report_step == 0:
            print(
                'Iteration %d: mu=%e, ||g_k||=%e' %
                (_i, mu, numpy.linalg.norm(dfx)))
            pass

    return x


# For testing purpose only

def _test_plot(domain, func_base, func_fit, data_set, legend_title=''):
    fig = plt.figure()
    base_range = func_base(domain)
    fit_range = func_fit(domain)
    plt.plot(domain, base_range, label='Base function')
    plt.plot(domain, fit_range, label='Fitting result')
    data_x, data_y = data_set
    plt.plot(
        data_x, data_y,
        marker='o', linestyle='None', label='Sample points')
    plt.legend(title=legend_title)

    margin = 0.5
    plt.xlim(numpy.amin(domain) - margin, numpy.amax(domain) + margin)
    plt.ylim(numpy.amin(base_range) - margin, numpy.amax(fit_range) + margin)
    fig.show()
    pass


def _test():
    numpy.random.seed(0xdeadbeef)
    dim = 4

    assert(dim == 4)
    nsample = 256
    N = 10
    gauss_scale = 0.16
    func_base = numpy.sin
    domain_l, domain_r = 0 * numpy.pi, 2 * numpy.pi
    domain = numpy.linspace(domain_l, domain_r, num=nsample)
    data_x_gen = lambda N: numpy.linspace(domain_l, domain_r, num=N)
    data_y_gen = lambda data_x: [func_base(x) for x in data_x]
    data_y_noisified_gen = lambda data_x, scale: [
        func_base(x) + numpy.random.normal(scale=scale) for x in data_x]
    data_x = data_x_gen(N)
    data_y_noisified = data_y_noisified_gen(data_x, gauss_scale)
    data_set = data_x, data_y_noisified

    X = numpy.array([[i ** j for j in range(dim)] for i in data_x])
    # y0 = numpy.array([numpy.random.uniform() for i in range(dim)])
    y0 = data_y_noisified
    k = 0.3

    f = lambda x: numpy.linalg.norm(X.dot(x) - y0) ** 2 + \
        k / 2 * numpy.linalg.norm(x) ** 2
    df = lambda x: 2 * X.transpose().dot(X.dot(x) - y0) + k * x
    ddf = lambda x: 2 * X.transpose().dot(X) + \
        k * numpy.identity(X.shape[1])

    A = LM(numpy.zeros(dim), f, df, ddf, report_step=1)

    func_fit = numpy.polynomial.Polynomial(A)
    msg = 'Polynomial fitting using LM algorithm'
    _test_plot(domain, func_base, func_fit, data_set, msg)
    plt.show()
    pass


def main():
    _test()
    pass


if __name__ == '__main__':
    main()
