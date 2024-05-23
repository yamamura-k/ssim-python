from math import exp

import numpy as np
import torch
import torch.nn.functional as F


def gaussian(window_size: int, sigma: float):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(
    img1, img2, window, window_size: int, channel: int, size_average: bool = True
):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean().cpu().numpy()
    else:
        return ssim_map.mean(1).mean(1).mean(1).cpu().numpy()


def _ssim_batch_full(images, window, window_size: int = 11):
    (bs, channel, width, height) = images.size()
    mu = F.conv2d(images, window, padding=window_size // 2, groups=channel)
    mu_sq = mu.pow(2)
    # #bbox x #bbox x channel x height x width になるように軸を増やして要素積を取る
    mu1_mu2 = (mu.unsqueeze(0) * mu.unsqueeze(1)).view(-1, channel, width, height)
    sigma_sq = (
        F.conv2d(images.pow(2), window, padding=window_size // 2, groups=channel)
        - mu_sq
    )
    images_12 = (images.unsqueeze(0) * images.unsqueeze(1)).view(
        -1, channel, width, height
    )
    sigma12 = (
        F.conv2d(images_12, window, padding=window_size // 2, groups=channel) - mu1_mu2
    )
    mu1_mu2 = mu1_mu2.view(bs, bs, channel, width, height)
    sigma12 = sigma12.view(bs, bs, channel, width, height)
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu_sq.unsqueeze(0) + mu_sq.unsqueeze(1) + C1)
        * (sigma_sq.unsqueeze(0) + sigma_sq.unsqueeze(1) + C2)
    )

    ssim_matrix = ssim_map.mean(-1).mean(-1).mean(-1)
    # ssim_matrix[range(bs), range(bs)] = 0.0
    return ssim_matrix.cpu().numpy()


def _ssim_batch_hybrid(images, window, window_size, channel, q_matrix_threshold=100):
    n_candidates = len(images)
    if n_candidates > q_matrix_threshold:  # should determine based on GPU memory
        similarity = np.zeros((n_candidates, n_candidates), dtype=np.float32)
        for i in range(1, n_candidates):
            _base_bbox = images[i].expand(i, -1, -1, -1)
            similarity[i, :i] = _ssim(
                _base_bbox,
                images[:i],
                window,
                window_size=window_size,
                channel=channel,
                size_average=False,
            )
        similarity += similarity.T
    else:  # 行列を分解して計算するのもあり
        similarity = _ssim_batch_full(images, window, window_size=window_size)
    return similarity


def _ssim_batch_recursive(
    images, l1, r1, l2, r2, q_matrix_threshold: int, window, window_size: int = 11
):
    length1 = r1 - l1
    length2 = r2 - l2
    if max(length1, length2) > q_matrix_threshold:
        m1 = (l1 + r1) // 2
        m2 = (l2 + r2) // 2
        _similarity1 = _ssim_batch_recursive(
            images, l1, m1, l2, m2, q_matrix_threshold, window, window_size
        )
        _similarity2 = _ssim_batch_recursive(
            images, m1, r1, m2, r2, q_matrix_threshold, window, window_size
        )
        _similarity12 = _ssim_batch_recursive(
            images, m1, r1, l2, m2, q_matrix_threshold, window, window_size
        )
        _similarity21 = _ssim_batch_recursive(
            images, l1, m1, m2, r2, q_matrix_threshold, window, window_size
        )
        similarity = np.concatenate(
            [
                np.concatenate([_similarity1, _similarity21], axis=1),
                np.concatenate([_similarity12, _similarity2], axis=1),
            ],
            axis=0,
        )
    else:
        images1 = images[l1:r1]
        images2 = images[l2:r2]
        (bs1, channel, width, height) = images1.size()
        (bs2, _, _, _) = images2.size()
        mu1 = F.conv2d(images1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(images2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        # #bbox x #bbox x channel x height x width になるように軸を増やして要素積を取る
        mu1_mu2 = (mu1.unsqueeze(1) * mu2.unsqueeze(0)).view(-1, channel, width, height)
        sigma1_sq = (
            F.conv2d(images1.pow(2), window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(images2.pow(2), window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        images_12 = (images1.unsqueeze(1) * images2.unsqueeze(0)).view(
            -1, channel, width, height
        )
        sigma12 = (
            F.conv2d(images_12, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        mu1_mu2 = mu1_mu2.view(bs1, bs2, channel, width, height)
        sigma12 = sigma12.view(bs1, bs2, channel, width, height)
        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq.unsqueeze(1) + mu2_sq.unsqueeze(0) + C1)
            * (sigma1_sq.unsqueeze(1) + sigma2_sq.unsqueeze(0) + C2)
        )

        ssim_matrix = ssim_map.mean(-1).mean(-1).mean(-1)
        similarity = ssim_matrix.cpu().numpy()
    return similarity
