import gc
import numpy as np
import torch
from libcpp import ssim_batch_recursive, ssim_batch_full

from ssim_python import _ssim_batch_full, _ssim_batch_recursive, create_window


def stopwatch(func, n_trial=10):
    import time

    def wrapper(*args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        start = time.time()
        for _ in range(n_trial):
            result = func(*args, **kwargs)
        print(f"{func.__name__:<25}: {(time.time() - start):.5f} sec ({n_trial} runs)")
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


torch.set_num_threads(4)

device = "cuda:0"
window_size = 11
n_trial = 100
W = 64
H = 64
SIZE = 100
R1 = 50
R2 = 25
with torch.no_grad():
    for size in range(10, SIZE, 10):
        print(f"\n[ size: {size} ]")
        images = torch.rand(size, 3, W, H, device=device)
        n_images = images.size(0)
        n_channels = images.size(1)
        window = create_window(window_size, n_channels).to(device)
        s_full = stopwatch(_ssim_batch_full, n_trial)
        s_recursive = stopwatch(_ssim_batch_recursive, n_trial)

        s_full_t = stopwatch(ssim_batch_full, n_trial)
        s_recursive_t = stopwatch(ssim_batch_recursive, n_trial)

        print("-" * 10 + " Python Implementation " + "-" * 10)
        similarity_f = set_diag(s_full(images, window, window_size), 0)
        similarity_r = set_diag(
            s_recursive(images, 0, n_images, 0, n_images, R1, window, window_size), 0
        )
        similarity_r_2 = set_diag(
            s_recursive(images, 0, n_images, 0, n_images, R2, window, window_size), 0
        )

        print()
        print("-" * 10 + " C++ Implementation " + "-" * 10)
        similarity_f_t = set_diag(s_full_t(images, window, window_size), 0)
        assert calc_diff(similarity_f, similarity_f_t) < 1e-5
        assert calc_diff(similarity_f, similarity_r) < 1e-5
        assert calc_diff(similarity_f, similarity_r_2) < 1e-5
        print()
        for r in range(10, n_images + 1, 10):
            print(f"Recursive/Size: {r:>3}/{size:>3}")
            for mode in range(5):
                similarity_rt_1 = set_diag(
                    s_recursive_t(images, n_images, r, window, window_size, mode), 0
                )
                assert calc_diff(similarity_f, similarity_rt_1) < 1e-5
        print()
