#include <torch/extension.h>
namespace Idx = torch::indexing;
namespace F = torch::nn::functional;

torch::Tensor _ssim_batch_recursive(
    torch::Tensor images, int l1, int r1, int l2, int r2, int q_matrix_threshold, torch::Tensor window, int window_size
) {
    torch::Tensor similarity;
    int length1 = r1 - l1;
    int length2 = r2 - l2;
    if (std::max(length1, length2) > q_matrix_threshold) {
        int m1 = (l1 + r1) / 2;
        int m2 = (l2 + r2) / 2;
        torch::Tensor _similarity1 = _ssim_batch_recursive(
            images, l1, m1, l2, m2, q_matrix_threshold, window, window_size
        );
        torch::Tensor _similarity2 = _ssim_batch_recursive(
            images, m1, r1, m2, r2, q_matrix_threshold, window, window_size
        );
        torch::Tensor _similarity12 = _ssim_batch_recursive(
            images, m1, r1, l2, m2, q_matrix_threshold, window, window_size
        );
        torch::Tensor _similarity21 = _ssim_batch_recursive(
            images, l1, m1, m2, r2, q_matrix_threshold, window, window_size
        );
        similarity = torch::cat(
            {
                torch::cat({_similarity1, _similarity21}, 1),
                torch::cat({_similarity12, _similarity2}, 1),
            },
            0
        );
    } else {
        torch::Tensor images1 = images.index({Idx::Slice(l1, r1, Idx::None)});
        torch::Tensor images2 = images.index({Idx::Slice(l2, r2, Idx::None)});
        int bs1 = images1.size(0);
        int channel = images1.size(1);
        int width = images1.size(2);
        int height = images1.size(3);
        int bs2 = images2.size(0);
        torch::Tensor mu1 = F::conv2d(images1, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
        torch::Tensor mu2 = F::conv2d(images2, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
        torch::Tensor mu1_sq = mu1.pow(2);
        torch::Tensor mu2_sq = mu2.pow(2);

        // #bbox x #bbox x channel x height x width になるように軸を増やして要素積を取る
        torch::Tensor mu1_mu2 = (mu1.unsqueeze(1) * mu2.unsqueeze(0)).view({-1, channel, width, height});
        torch::Tensor sigma1_sq = (
            F::conv2d(images1.pow(2), window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
            - mu1_sq
        );
        torch::Tensor sigma2_sq = (
            F::conv2d(images2.pow(2), window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
            - mu2_sq
        );
        torch::Tensor images_12 = (images1.unsqueeze(1) * images2.unsqueeze(0)).view(
            {-1, channel, width, height}
        );
        torch::Tensor sigma12 = (
            F::conv2d(images_12, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
            - mu1_mu2
        );

        mu1_mu2 = mu1_mu2.view({bs1, bs2, channel, width, height});
        sigma12 = sigma12.view({bs1, bs2, channel, width, height});
        float C1 = 0.01 * 0.01;
        float C2 = 0.03 * 0.03;

        torch::Tensor ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq.unsqueeze(1) + mu2_sq.unsqueeze(0) + C1)
            * (sigma1_sq.unsqueeze(1) + sigma2_sq.unsqueeze(0) + C2)
        );

        similarity = ssim_map.mean(-1).mean(-1).mean(-1);
    }
    return similarity;
}
torch::Tensor _ssim_batch_recursive_put(
    torch::Tensor similarity, torch::Tensor images, int l1, int r1, int l2, int r2, int q_matrix_threshold, torch::Tensor window, int window_size
) {
    int length1 = r1 - l1;
    int length2 = r2 - l2;
    if (std::max(length1, length2) > q_matrix_threshold) {
        int m1 = (l1 + r1) / 2;
        int m2 = (l2 + r2) / 2;
        _ssim_batch_recursive_put(
            similarity, images, l1, m1, l2, m2, q_matrix_threshold, window, window_size
        );
        _ssim_batch_recursive_put(
            similarity, images, m1, r1, m2, r2, q_matrix_threshold, window, window_size
        );
        _ssim_batch_recursive_put(
            similarity, images, m1, r1, l2, m2, q_matrix_threshold, window, window_size
        );
        _ssim_batch_recursive_put(
            similarity, images, l1, m1, m2, r2, q_matrix_threshold, window, window_size
        );
    } else {
        torch::Tensor images1 = images.index({Idx::Slice(l1, r1, Idx::None)});
        torch::Tensor images2 = images.index({Idx::Slice(l2, r2, Idx::None)});
        int bs1 = images1.size(0);
        int channel = images1.size(1);
        int width = images1.size(2);
        int height = images1.size(3);
        int bs2 = images2.size(0);
        torch::Tensor mu1 = F::conv2d(images1, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
        torch::Tensor mu2 = F::conv2d(images2, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
        torch::Tensor mu1_sq = mu1.pow(2);
        torch::Tensor mu2_sq = mu2.pow(2);

        // #bbox x #bbox x channel x height x width になるように軸を増やして要素積を取る
        torch::Tensor mu1_mu2 = (mu1.unsqueeze(1) * mu2.unsqueeze(0)).view({-1, channel, width, height});
        torch::Tensor sigma1_sq = (
            F::conv2d(images1.pow(2), window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
            - mu1_sq
        );
        torch::Tensor sigma2_sq = (
            F::conv2d(images2.pow(2), window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
            - mu2_sq
        );
        torch::Tensor images_12 = (images1.unsqueeze(1) * images2.unsqueeze(0)).view(
            {-1, channel, width, height}
        );
        torch::Tensor sigma12 = (
            F::conv2d(images_12, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
            - mu1_mu2
        );

        mu1_mu2 = mu1_mu2.view({bs1, bs2, channel, width, height});
        sigma12 = sigma12.view({bs1, bs2, channel, width, height});
        float C1 = 0.01 * 0.01;
        float C2 = 0.03 * 0.03;

        torch::Tensor ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq.unsqueeze(1) + mu2_sq.unsqueeze(0) + C1)
            * (sigma1_sq.unsqueeze(1) + sigma2_sq.unsqueeze(0) + C2)
        );

        similarity.index_put_({Idx::Slice(l1, r1, Idx::None), Idx::Slice(l2, r2, Idx::None)}, ssim_map.mean(-1).mean(-1).mean(-1));
    }
    return similarity;
}


torch::Tensor _ssim(
    torch::Tensor img1, torch::Tensor img2, torch::Tensor window, int window_size, int channel
) {
    torch::Tensor mu1 = F::conv2d(img1, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
    torch::Tensor mu2 = F::conv2d(img2, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel));

    torch::Tensor mu1_sq = mu1.pow(2);
    torch::Tensor mu2_sq = mu2.pow(2);
    torch::Tensor mu1_mu2 = mu1 * mu2;

    torch::Tensor sigma1_sq = (
        F::conv2d(img1 * img1, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_sq
    );
    torch::Tensor sigma2_sq = (
        F::conv2d(img2 * img2, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu2_sq
    );
    torch::Tensor sigma12 = (
        F::conv2d(img1 * img2, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
        - mu1_mu2
    );

    float C1 = 0.01 * 0.01;
    float C2 = 0.03 * 0.03;

    torch::Tensor ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    );

    return ssim_map.mean(1).mean(1).mean(1);
}


torch::Tensor  _ssim_batch_full(torch::Tensor images, torch::Tensor window, int window_size) {
    int bs = images.size(0);
    int channel = images.size(1);
    int width = images.size(2);
    int height = images.size(3);
    torch::Tensor mu = F::conv2d(images, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
    torch::Tensor mu_sq = mu.pow(2);
    // # #bbox x #bbox x channel x height x width になるように軸を増やして要素積を取る
    torch::Tensor mu1_mu2 = (mu.unsqueeze(0) * mu.unsqueeze(1)).view({-1, channel, width, height});
    torch::Tensor sigma_sq = (
        F::conv2d(images.pow(2), window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel))
        - mu_sq
    );
    torch::Tensor images_12 = (images.unsqueeze(0) * images.unsqueeze(1)).view(
        {-1, channel, width, height}
    );
    torch::Tensor sigma12 = (
        F::conv2d(images_12, window, F::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_mu2
    );
    mu1_mu2 = mu1_mu2.view({bs, bs, channel, width, height});
    sigma12 = sigma12.view({bs, bs, channel, width, height});
    float C1 = 0.01 * 0.01;
    float C2 = 0.03 * 0.03;
    torch::Tensor ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu_sq.unsqueeze(0) + mu_sq.unsqueeze(1) + C1)
        * (sigma_sq.unsqueeze(0) + sigma_sq.unsqueeze(1) + C2)
    );

    return ssim_map.mean(-1).mean(-1).mean(-1);
}


torch::Tensor _ssim_batch_hybrid(torch::Tensor images, torch::Tensor window, int window_size, int channel, int q_matrix_threshold) {
    int n_candidates = images.size(0);
    torch::Tensor similarity;
    if (n_candidates > q_matrix_threshold) {
        similarity = torch::zeros({n_candidates, n_candidates});
        for(ssize_t i = 1; i < n_candidates; ++i) {
            torch::Tensor _base_bbox = images.index({i}).unsqueeze(0);
            similarity.index_put_(
                {i, Idx::Slice(0, i, Idx::None)}, 
                _ssim(
                    _base_bbox,
                    images.index({Idx::Slice(0, i, Idx::None)}),
                    window,
                    window_size,
                    channel
                )
            );
        }
        similarity = similarity + similarity.permute({1, 0});
    } else {
        similarity = _ssim_batch_full(images, window, window_size);
    }
    return similarity;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ssim_batch_recursive", &_ssim_batch_recursive, "_ssim_batch_recursive");
  m.def("ssim_batch_recursive_put", &_ssim_batch_recursive_put, "_ssim_batch_recursive_put");
  m.def("ssim_batch_full", &_ssim_batch_full, "_ssim_batch_full");
  m.def("ssim_batch_hybrid", &_ssim_batch_hybrid, "_ssim_batch_hybrid");
}