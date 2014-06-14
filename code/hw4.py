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

def _test():
    numpy.random.seed(0xdeadbeef)
    dim = 4

    A = numpy.array([[i ** j for j in range(dim)] for i in range(dim)])
    y0 = numpy.array([numpy.random.uniform() for i in range(dim)])
    k = 0.3

    f = lambda x: numpy.linalg.norm(A * x - y0) ** 2 + \
        k / 2 * numpy.linalg.norm(x) ** 2
    df = lambda x: 2 * A.transpose().dot(A.dot(x) - y0) + k * x
    ddf = lambda x: 2 * A.transpose().dot(A) + \
        k * numpy.identity(A.shape[1])

    x = LM(numpy.zeros(dim), f, df, ddf, report_step=1)
    pass


def main():
    _test()
    pass


if __name__ == '__main__':
    main()
