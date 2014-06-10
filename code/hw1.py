#! /usr/bin/env python

# This is part of coursework for Applied Mathematics for Computer Science
# by Pengyu CHEN (pengyu[at]libstarrify.so)
# COPYLEFT, ALL WRONGS RESERVED.

import numpy
import matplotlib.pyplot as plt


def polynomial_fitting_least_square(degree, data_set, log_lambda=None):
    """
    Gives polynomial least squares fitting with regularization of given data
    set.
    """
    X0, Y0 = data_set
    N, = X0.shape
    M = degree
    X = numpy.matrix([[X0[i] ** j for j in range(M + 1)] for i in range(N)])
    Y = numpy.matrix(Y0).getT()

    XTX = X.getT() * X
    if log_lambda is not None:
        _lambda = numpy.exp(log_lambda)
        XTX += numpy.identity(M + 1) * _lambda
    A = XTX.getI() * X.getT() * Y
    return A.A1


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

    margin = 1
    plt.xlim(numpy.amin(domain) - margin, numpy.amax(domain) + margin)
    plt.ylim(numpy.amin(base_range) - margin, numpy.amax(fit_range) + margin)
    fig.show()
    pass


def _test_fit(N, M, log_lambda, gauss_scale, nsample):
    func_base = numpy.sin
    domain_l, domain_r = 0, 2 * numpy.pi
    domain = numpy.linspace(domain_l, domain_r, num=nsample)
    data_x_gen = lambda N: numpy.linspace(domain_l, domain_r, num=N)
    data_y_gen = lambda data_x: [func_base(x) for x in data_x]
    data_y_noisified_gen = lambda data_x, scale: [
        func_base(x) + numpy.random.normal(scale=scale) for x in data_x]

    data_x = data_x_gen(N)
    data_y_noisified = data_y_noisified_gen(data_x, gauss_scale)
    data_set = data_x, data_y_noisified
    func_fit_A = polynomial_fitting_least_square(M, data_set, log_lambda)
    func_fit = numpy.polynomial.Polynomial(func_fit_A)
    legend_title = 'M=%d, N=%d' % (M, N)
    if log_lambda is not None:
        legend_title += ', log(lambda)=%f' % log_lambda
    _test_plot(domain, func_base, func_fit, data_set, legend_title)
    pass


def _test():
    numpy.random.seed(0xdeadbeef)
    gauss_scale = 0.09
    nsample = 256
    _test_fit(10, 3, None, gauss_scale, nsample)
    _test_fit(10, 9, None, gauss_scale, nsample)
    _test_fit(15, 9, None, gauss_scale, nsample)
    _test_fit(100, 9, None, gauss_scale, nsample)
    _test_fit(10, 9, -4, gauss_scale, nsample)

    plt.show()
    pass


def main():
    _test()
    pass


if __name__ == '__main__':
    main()
