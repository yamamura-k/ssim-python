import gc
import numpy as np
import torch
from libcpp import ssim_batch_recursive, ssim_batch_recursive_put, ssim_batch_full#, ssim_batch_hybrid

from ssim_python import (_ssim_batch_full, _ssim_batch_hybrid,
                         _ssim_batch_recursive, create_window)
# from ssim_python.lib.lib_cython_2 import (_ssim_batch_full, _ssim_batch_hybrid,
#                                           _ssim_batch_recursive, create_window)


def stopwatch(func):
    import time

    def wrapper(*args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        start = time.time()
        for _ in range(10):
            result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start} seconds")
        return result

    return wrapper


def calc_diff(s1, s2):
    return np.sqrt(np.power(s1 - s2, 2).sum())


def set_diag(matrix, value):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    for i in range(matrix.shape[0]):
        matrix[i, i] = value
    return matrix

def test_all_gpu():
    torch.set_num_threads(4)

    device = "cuda:0"
    window_size = 11
    W = 64
    H = 64
    R1 = 100
    R2 = 75
    with torch.no_grad():
        images = torch.rand(300, 3, W, H, device=device)
        n_images = images.size(0)
        n_channels = images.size(1)
        window = create_window(window_size, n_channels).to(device)
        s_full = stopwatch(_ssim_batch_full)
        s_full_t = stopwatch(ssim_batch_full)
        
        s_hybrid = stopwatch(_ssim_batch_hybrid)
        # s_hybrid_t = stopwatch(ssim_batch_hybrid)
        
        s_recursive = stopwatch(_ssim_batch_recursive)
        s_recursive_t = stopwatch(ssim_batch_recursive)
        s_recursive_p = stopwatch(ssim_batch_recursive_put)

        similarity_f = set_diag(s_full(images, window, window_size), 0)
        similarity_h = set_diag(s_hybrid(images, window, window_size, n_channels, 0), 0)
        similarity_f_t = set_diag(s_full_t(images, window, window_size), 0)
        # similarity_h_t = set_diag(s_hybrid_t(images, window, window_size, n_channels, 0), 0)
        similarity_r = set_diag(
            s_recursive(images, 0, n_images, 0, n_images, R1, window, window_size), 0
        )
        similarity_r_2 = set_diag(
            s_recursive(images, 0, n_images, 0, n_images, R2, window, window_size), 0
        )
        similarity_rt_1 = set_diag(
            s_recursive_t(images, 0, n_images, 0, n_images, R1, window, window_size), 0
        )
        similarity_rt_2 = set_diag(
            s_recursive_t(images, 0, n_images, 0, n_images, R2, window, window_size), 0
        )
        similarity_rp_1 = set_diag(
            s_recursive_p(
                torch.zeros((n_images, n_images), device=device),
                images,
                0,
                n_images,
                0,
                n_images,
                25,
                window,
                window_size,
            ),
            0,
        )
        similarity_rp_2 = set_diag(
            s_recursive_p(
                torch.zeros((n_images, n_images), device=device),
                images,
                0,
                n_images,
                0,
                n_images,
                25,
                window,
                window_size,
            ),
            0,
        )
        assert(calc_diff(similarity_f, similarity_h) < 1e-5)
        assert(calc_diff(similarity_f, similarity_f_t) < 1e-5)
        assert(calc_diff(similarity_f, similarity_r) < 1e-5)
        assert(calc_diff(similarity_f, similarity_r_2) < 1e-5)
        assert(calc_diff(similarity_f, similarity_rt_1) < 1e-5)
        assert(calc_diff(similarity_f, similarity_rt_2) < 1e-5)
        assert(calc_diff(similarity_f, similarity_rp_1) < 1e-5)
        assert(calc_diff(similarity_f, similarity_rp_2) < 1e-5)
