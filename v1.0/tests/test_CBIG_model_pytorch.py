#!/usr/bin/env python3
#
# test_CBIG_model_pytorch.py
# 2022-06-21
"""
Written by _ and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import os
import sys
import unittest

import numpy as np
import torch

try:
    import CBIG_model_pytorch
except ModuleNotFoundError as e:
    for root, dirs, files in os.walk(os.path.realpath(f"{__file__}/../..")):
        if "CBIG_model_pytorch.py" in files:
            sys.path.insert(0, root)
            break
    import CBIG_model_pytorch



def random_ndarray(shape=None, n_dims=None, max_dims=5, seed=None):
    '''Create a random numpy.ndarray

    Args:
        shape (tuple): shape of numpy.ndarray (default: None)
        n_dims (int): number of dimensions, size of each dimension would be 
            randomly determined (default: None)
        max_dims (int): maximum number of dimensions, if determined randomly 
            (default: 5)
        seed (int): seed for generating random numbers

    Returns:
        numpy.ndarray
    '''
    if seed is not None and not isinstance(seed, int):
        raise TypeError(f"{seed = }, {type(seed) = }, should be an integer.")
    rng = np.random.default_rng(seed)
    if shape is None:
        if n_dims is None:
            n_dims = rng.integers(1, 1 + max_dims)
        shape = tuple(rng.integers(1, 1 + max_dims) for _ in range(n_dims))
    A = rng.random(shape)
    return A

def random_tensor(shape=None, n_dims=None, max_dims=5, seed=None):
    '''Create a random torch.Tensor

    Args:
        shape (tuple): shape of torch.Tensor (default: None)
        n_dims (int): number of dimensions, size of each dimension would be 
            randomly determined (default: None)
        max_dims (int): maximum number of dimensions, if determined randomly 
            (default: 5)
        seed (int): seed for generating random numbers (default: None)

    Returns:
        torch.Tensor
    '''
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError(f"{seed = }, {type(seed) = }, should be an integer.")
        torch.manual_seed(seed)
    if shape is None:
        if n_dims is None:
            n_dims = torch.randint(1, 1 + max_dims, (1, 1))
        shape = tuple(torch.randint(1, 1 + max_dims, (1, 1)) for _ in range(n_dims))
    A = torch.rand(shape)
    return A

def insert_blanks(A, proportion=0.1, seed=None, verbose=False):
    '''Randomly insert missing values into a numpy.ndarray or torch.Tensor

    Args:
        A (numpy.ndarray or torch.Tensor): input array
        proportion (float): proportion of elements to convert to missing 
            values (default: 0.1)
        seed (int): seed for generating random numbers (default: None)
        verbose (bool): print number of elements and number of missing values 
            (default: False)

    Returns:
        numpy.ndarray or torch.Tensor with missing values
    '''
    if seed is not None and not isinstance(seed, int):
        raise TypeError(f"{seed = }, {type(seed) = }, should be an integer.")

    if isinstance(A, torch.Tensor):
        lib = torch
        n = lib.numel(A)
        if seed is not None:
            lib.manual_seed(seed)
    elif isinstance(A, np.ndarray):
        lib = np
        n = lib.size(A)
        if seed is not None:
            lib.random.seed(seed)
    else:
        raise TypeError(f"{A = }, {type(A) = }, "
                        "should be a torch.tensor or numpy.ndarray.")

    n_nans = int(n * proportion)
    mask = lib.zeros(n, dtype=bool)
    mask[:n_nans] = True

    if isinstance(A, torch.Tensor):
        mask = mask[lib.randperm(n)]
    elif isinstance(A, np.ndarray):
        lib.random.shuffle(mask)

    mask = lib.reshape(mask, A.shape)
    A[mask] = lib.nan

    if verbose:
        print(f"     {n = }")
        print(f"{n_nans = }")

    return A



class Futures():
    """
    Functions that can be refactored.
    Some functions can be made obsolete.
    Try to preserve the existing APIs.
    """

    @classmethod
    def covariance_rowwise(self, A, B):
        '''rowwise covariance computation

        Args:
            A (ndarray): first array for covariance computaion
            B (ndarray): second array for covariance computaion

        Returns:
            ndarray: rowwise covariance between two array
        '''
        n = A.shape[1]
        cov = np.cov(A, B, rowvar=False)
        top_right_quadrant = cov[:n, n:]
        return np.squeeze(top_right_quadrant)

    @classmethod
    def torch_nanmean(self, x):
        '''Calculate mean and omit NAN 

        Args:
            x (torch.tensor): input data

        Returns:
            torch.Tensor: mean value (omit NAN)
        '''
        return torch.nanmean(x)

    @classmethod
    def msenanloss(self, input, target):
        '''Calculate MSE (mean absolute error) and omit NAN 

        Args:
            input (torch.tensor): predicted value
            target (torch.tensor): original value

        Returns:
            torch.Tensor: MSE loss (omit NAN)
        '''
        return torch.nanmean((input - target) ** 2)



class TestCBIGModelPytorch(unittest.TestCase):

    def test_random_ndarray(self):
        A = random_ndarray(shape=(5, 4), seed=1729)
        want = np.array([[0.03074203, 0.16845748, 0.29395045, 0.57625191],
                         [0.81237188, 0.97635631, 0.46500147, 0.14406921],
                         [0.91589172, 0.36004323, 0.05323127, 0.16150129],
                         [0.26354111, 0.68625911, 0.51000833, 0.94696381],
                         [0.1357902,  0.36624053, 0.32442991, 0.2249961 ]])
        np.testing.assert_allclose(A, want)
        B = random_ndarray(shape=(5, 3), seed=1729)
        want = np.array([[0.03074203, 0.16845748, 0.29395045],
                         [0.57625191, 0.81237188, 0.97635631],
                         [0.46500147, 0.14406921, 0.91589172],
                         [0.36004323, 0.05323127, 0.16150129],
                         [0.26354111, 0.68625911, 0.51000833]])
        np.testing.assert_allclose(B, want)

    def test_random_tensor(self):
        A = random_tensor(shape=(4, 3), seed=42)
        want = torch.Tensor([[0.882269263, 0.915003955, 0.382863760],
                             [0.959305644, 0.390448213, 0.600895345],
                             [0.256572485, 0.793641329, 0.940771461],
                             [0.133185923, 0.934598088, 0.593579650]])
        torch.testing.assert_close(A, want)
        B = random_tensor(shape=(4, 3), seed=1729)
        want = torch.Tensor([[0.312599957, 0.379095435, 0.308669269],
                             [0.073580086, 0.421601593, 0.069053531],
                             [0.233219326, 0.404656231, 0.216237605],
                             [0.992694557, 0.412752330, 0.593822539]])
        torch.testing.assert_close(B, want)

    def test_insert_blanks(self):
        A = random_ndarray(shape=(5, 4), seed=1729)
        A = insert_blanks(A, seed=1729)
        want = np.array([[0.03074203, 0.16845748,     np.nan, 0.57625191],
                         [0.81237188, 0.97635631, 0.46500147, 0.14406921],
                         [0.91589172, 0.36004323, 0.05323127,     np.nan],
                         [0.26354111, 0.68625911, 0.51000833, 0.94696381],
                         [0.1357902,  0.36624053, 0.32442991, 0.2249961 ]])
        B = random_tensor(shape=(4, 3), seed=42)
        B = insert_blanks(B, seed=1729)
        want = torch.Tensor([[0.882269263, 0.915003955, 0.382863760],
                             [0.959305644, 0.390448213, 0.600895345],
                             [0.256572485,   torch.nan, 0.940771461],
                             [0.133185923, 0.934598088, 0.593579650]])
        np.testing.assert_allclose(B, want)

    def test_sum_of_mul_fixed(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        got = CBIG_model_pytorch.sum_of_mul(A, B)
        want = np.array([17, 53])
        np.testing.assert_allclose(got, want)

    def test_sum_of_mul_fixed_missing_values(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [np.nan, 8]])
        got = CBIG_model_pytorch.sum_of_mul(A, B)
        want = np.array([17, np.nan])
        np.testing.assert_allclose(got, want)

    def test_sum_of_mul_diff_shapes(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8], [9, 10]])
        C = np.array([[11, 12, 13], [14, 15, 16]])
        with self.assertRaises(ValueError):
            _ = CBIG_model_pytorch.sum_of_mul(A, B)
        with self.assertRaises(ValueError):
            _ = CBIG_model_pytorch.sum_of_mul(A, C)

    def test_covariance_rowwise_fixed(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        got1 = CBIG_model_pytorch.covariance_rowwise(A, B)
        got2 = Futures.covariance_rowwise(A, B)
        want = np.array([[2, 2], [2, 2]])
        np.testing.assert_allclose(got1, want)
        np.testing.assert_allclose(got2, want)

    def test_covariance_rowwise_fixed_missing_values(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, np.nan]])
        got1 = CBIG_model_pytorch.covariance_rowwise(A, B)
        got2 = Futures.covariance_rowwise(A, B)
        want = np.array([[2, np.nan], [2, np.nan]])
        np.testing.assert_allclose(got1, want)
        np.testing.assert_allclose(got2, want)

    def test_covariance_rowwise_rand(self):
        A = random_ndarray(shape=(5, 4), seed=1729)
        B = random_ndarray(shape=(5, 3), seed=1729)
        got1 = CBIG_model_pytorch.covariance_rowwise(A, B)
        got2 = Futures.covariance_rowwise(A, B)
        want = np.array([[ 0.073428,  0.024875,  0.129824],
                         [ 0.052897,  0.051925,  0.042133],
                         [ 0.003119,  0.017686, -0.026029],
                         [-0.030098, -0.080904, -0.112042]])
        np.testing.assert_allclose(got1, want, atol=5e-07)
        np.testing.assert_allclose(got2, want, atol=5e-07)

    def test_covariance_rowwise_rand_missing_values(self):
        A = random_ndarray(shape=(5, 4), seed=1729)
        B = random_ndarray(shape=(5, 3), seed=1729)
        A = insert_blanks(A, seed=1729)
        got1 = CBIG_model_pytorch.covariance_rowwise(A, B)
        got2 = Futures.covariance_rowwise(A, B)
        want = np.array([[0.073428, 0.024875, 0.129824],
                         [0.052897, 0.051925, 0.042133],
                         [  np.nan,   np.nan,   np.nan],
                         [  np.nan,   np.nan,   np.nan]])
        np.testing.assert_allclose(got1, want, atol=5e-07)
        np.testing.assert_allclose(got2, want, atol=5e-07)

    def test_covariance_rowwise_diff_nrows(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8], [9, 10]])
        C = np.array([[11, 12, 13], [14, 15, 16]])
        with self.assertRaises(ValueError):
            _ = CBIG_model_pytorch.covariance_rowwise(A, B)
        with self.assertRaises(ValueError):
            _ = Futures.covariance_rowwise(A, B)
        with self.assertRaises(ValueError):
            _ = CBIG_model_pytorch.covariance_rowwise(B, C)
        with self.assertRaises(ValueError):
            _ = Futures.covariance_rowwise(B, C)

    def test_demean_norm_fixed(self):
        A = np.array([[1, 2], [3, 4]])
        got = CBIG_model_pytorch.demean_norm(A)
        want = np.array([[-0.707107,  0.707107],
                         [-0.707107,  0.707107]])
        np.testing.assert_allclose(got, want, atol=5e-07)

    def test_demean_norm_fixed_missing_values(self):
        A = np.array([[1, 2], [3, np.nan]])
        with self.assertRaises(ValueError):
            _ = CBIG_model_pytorch.demean_norm(A)

    @unittest.skip("TODO after understanding the function")
    def test_stacking(self):
        raise NotImplementedError

    def test_torch_nanmean_fixed(self):
        A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        mask = torch.isnan(A)
        got1 = CBIG_model_pytorch.torch_nanmean(A, mask=mask)
        got2 = torch.nanmean(A)
        want = torch.tensor(2.5)
        torch.testing.assert_close(got1, want)
        torch.testing.assert_close(got2, want)

    def test_torch_nanmean_fixed_missing_values(self):
        A = torch.tensor([[1, 2], [3, torch.nan]], dtype=torch.float32)
        mask = torch.isnan(A)
        got1 = CBIG_model_pytorch.torch_nanmean(A, mask=mask)
        got2 = torch.nanmean(A)
        want = torch.tensor(2.0)
        torch.testing.assert_close(got1, want)
        torch.testing.assert_close(got2, want)

    def test_msenanloss_fixed(self):
        A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        B = torch.tensor([[5, 5], [5, 5]], dtype=torch.float32)
        got1 = CBIG_model_pytorch.msenanloss(A, B)
        got2 = Futures.msenanloss(A, B)
        want = torch.tensor(7.5)
        torch.testing.assert_close(got1, want)
        torch.testing.assert_close(got2, want)

    def test_msenanloss_fixed_missing_values(self):
        A = torch.tensor([[1, 2], [3, torch.nan]], dtype=torch.float32)
        B = torch.tensor([[5, torch.nan], [5, 5]], dtype=torch.float32)
        got1 = CBIG_model_pytorch.msenanloss(A, B)
        got2 = Futures.msenanloss(A, B)
        want = torch.tensor(10.0)
        torch.testing.assert_close(got1, want)
        torch.testing.assert_close(got2, want)

    def test_msenanloss_rand(self):
        A = random_tensor(shape=(4, 3), seed=42)
        B = random_tensor(shape=(4, 3), seed=1729)
        got1 = CBIG_model_pytorch.msenanloss(A, B)
        got2 = Futures.msenanloss(A, B)
        want = torch.tensor(0.281120330)
        torch.testing.assert_close(got1, want)
        torch.testing.assert_close(got2, want)

    def test_msenanloss_rand_missing_values(self):
        A = random_tensor(shape=(4, 3), seed=42)
        B = random_tensor(shape=(4, 3), seed=1729)
        A = insert_blanks(A, seed=1729)
        got1 = CBIG_model_pytorch.msenanloss(A, B)
        got2 = Futures.msenanloss(A, B)
        want = torch.tensor(0.292921335)
        torch.testing.assert_close(got1, want)
        torch.testing.assert_close(got2, want)

    def test_msenanloss_diff_shapes(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8], [9, 10]])
        C = np.array([[11, 12, 13], [14, 15, 16]])
        with self.assertRaises(ValueError):
            _ = CBIG_model_pytorch.msenanloss(A, B)
        with self.assertRaises(ValueError):
            _ = Futures.msenanloss(A, B)
        with self.assertRaises(ValueError):
            _ = CBIG_model_pytorch.msenanloss(B, C)
        with self.assertRaises(ValueError):
            _ = Futures.msenanloss(B, C)



if __name__ == "__main__":
    unittest.main()



