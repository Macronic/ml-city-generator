#!/usr/bin/env python3

"""Calculates the Kernel Inception Distance (KID) to evalulate GANs

"""

import os

import pathlib

import sys

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



import numpy as np

import torch

from sklearn.metrics.pairwise import polynomial_kernel

from scipy import linalg

from PIL import Image

from torch.nn.functional import adaptive_avg_pool2d



from fid_metric import get_activations, preprocess_images





def calculate_kid(fake, true, batch_size, model_type='inception'):

    """Calculates the KID of two paths"""

    pths = [true, fake]



    if model_type == 'inception':

        act_true = get_activations(preprocess_images(true, False), batch_size)

        act_false = get_activations(preprocess_images(fake, False), batch_size)

    elif model_type == 'lenet':

        pass



    kid_values = polynomial_mmd_averages(act_true, act_false, n_subsets=100)

    return kid_values[0].mean(), kid_values[0].std()



def _sqn(arr):

    flat = np.ravel(arr)

    return flat.dot(flat)





def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,#n_subsets=50, subset_size=1000,

                            ret_var=True, output=sys.stdout, **kernel_args):

    m = min(codes_g.shape[0], codes_r.shape[0])

    if subset_size > len(codes_g):

        subset_size = len(codes_g)

        n_subsets = 1

        

    mmds = np.zeros(n_subsets)

    if ret_var:

        vars = np.zeros(n_subsets)

    choice = np.random.choice

    



    for i in range(n_subsets):

        g = codes_g[choice(len(codes_g), subset_size, replace=False)]

        r = codes_r[choice(len(codes_r), subset_size, replace=False)]

        o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)

        if ret_var:

            mmds[i], vars[i] = o

        else:

            mmds[i] = o

            

    return (mmds, vars) if ret_var else mmds





def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,

                   var_at_m=None, ret_var=True):

    # use  k(x, y) = (gamma <x, y> + coef0)^degree

    # default gamma is 1 / dim

    X = codes_g

    Y = codes_r



    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)

    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)

    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)



    return _mmd2_and_variance(K_XX, K_XY, K_YY,

                              var_at_m=var_at_m, ret_var=ret_var)



def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,

                       mmd_est='unbiased', block_size=1024,

                       var_at_m=None, ret_var=True):

    # based on

    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py

    # but changed to not compute the full kernel matrix at once

    m = K_XX.shape[0]

    assert K_XX.shape == (m, m)

    assert K_XY.shape == (m, m)

    assert K_YY.shape == (m, m)

    if var_at_m is None:

        var_at_m = m



    # Get the various sums of kernels that we'll use

    # Kts drop the diagonal, but we don't need to compute them explicitly

    if unit_diagonal:

        diag_X = diag_Y = 1

        sum_diag_X = sum_diag_Y = m

        sum_diag2_X = sum_diag2_Y = m

    else:

        diag_X = np.diagonal(K_XX)

        diag_Y = np.diagonal(K_YY)



        sum_diag_X = diag_X.sum()

        sum_diag_Y = diag_Y.sum()



        sum_diag2_X = _sqn(diag_X)

        sum_diag2_Y = _sqn(diag_Y)



    Kt_XX_sums = K_XX.sum(axis=1) - diag_X

    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y

    K_XY_sums_0 = K_XY.sum(axis=0)

    K_XY_sums_1 = K_XY.sum(axis=1)



    Kt_XX_sum = Kt_XX_sums.sum()

    Kt_YY_sum = Kt_YY_sums.sum()

    K_XY_sum = K_XY_sums_0.sum()



    if mmd_est == 'biased':

        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)

                + (Kt_YY_sum + sum_diag_Y) / (m * m)

                - 2 * K_XY_sum / (m * m))

    else:

        assert mmd_est in {'unbiased', 'u-statistic'}

        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))

        if mmd_est == 'unbiased':

            mmd2 -= 2 * K_XY_sum / (m * m)

        else:

            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))



    if not ret_var:

        return mmd2



    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X

    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y

    K_XY_2_sum = _sqn(K_XY)



    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)

    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)



    m1 = m - 1

    m2 = m - 2

    zeta1_est = (

        1 / (m * m1 * m2) * (

            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)

        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)

        + 1 / (m * m * m1) * (

            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)

        - 2 / m**4 * K_XY_sum**2

        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)

        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum

    )

    zeta2_est = (

        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)

        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)

        + 2 / (m * m) * K_XY_2_sum

        - 2 / m**4 * K_XY_sum**2

        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)

        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum

    )

    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est

               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)



    return mmd2, var_est





if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--true', type=str, required=True,

                        help=('Path to the true images'))

    parser.add_argument('--fake', type=str, nargs='+', required=True,

                        help=('Path to the generated images'))

    parser.add_argument('--batch-size', type=int, default=50,

                        help='Batch size to use')

    parser.add_argument('--dims', type=int, default=2048,

                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),

                        help=('Dimensionality of Inception features to use. '

                              'By default, uses pool3 features'))

    parser.add_argument('-c', '--gpu', default='', type=str,

                        help='GPU to use (leave blank for CPU only)')

    parser.add_argument('--model', default='inception', type=str,

                        help='inception or lenet')

    args = parser.parse_args()

    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    paths = [args.true] + args.fake



    results = calculate_kid_given_paths(paths, args.batch_size, args.gpu != '', args.dims, model_type=args.model)

    for p, m, s in results:

        print('KID (%s): %.3f (%.3f)' % (p, m, s))
