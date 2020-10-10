import numpy as np


LAMBDA = .0

# parent_sum = np.array([1.0, 1.5])


def compute_score(G, H, L):
    H = max(H, 1e-6)
    return G**2 / (H + L)


def compute_split_score(G, H, G_l, H_l, G_r, H_r, L=LAMBDA):
    p = compute_score(G, H, L)
    l = compute_score(G_l, H_l, L)
    r = compute_score(G_r, H_r, L)
    # print('gain:', l + r)
    return l + r - p


def evaluate(g_hist, h_hist, feature_values, parent_sum):
    rows = g_hist.shape[0]
    if len(g_hist.shape) == 1:
        g_hist = g_hist.reshape(rows, 1)
        h_hist = h_hist.reshape(rows, 1)

    cols = g_hist.shape[1]
    assert cols == h_hist.shape[1]
    assert cols == feature_values.shape[1]

    g_p, h_p = (parent_sum[0], parent_sum[1])

    score = .0
    split_ind = -1
    split_val = np.NaN

    print("Histo", g_hist, "\n\n", h_hist)
    print('\nForward\n')

    for i in range(cols):
        g_l = .0
        h_l = .0
        print("feature:", i)
        for j in range(rows):
            g_l = g_l + g_hist[j, i]
            h_l = h_l + h_hist[j, i]

            g_r = g_p - g_l
            h_r = h_p - h_l

            new = compute_split_score(g_p, h_p, g_l, h_l, g_r, h_r)
            print('loss_chg:', new)
            if new > score:
                split_ind = i
                score = new
                split_val = feature_values[j, i]

    print('score:', score, 'ind:', split_ind, 'val:', split_val)
    print('\nBackward\n')

    for i in range(cols):
        g_l = .0
        h_l = .0
        print("feature:", i)
        for j in range(rows - 1, -1, -1):
            g_l = g_l + g_hist[j, i]
            h_l = h_l + h_hist[j, i]

            g_r = g_p - g_l
            h_r = h_p - h_l

            new = compute_split_score(g_p, h_p, g_l, h_l, g_r, h_r)
            print('loss_chg:', new)
            if new > score:
                split_ind = i
                score = new
                split_val = feature_values[j, i]

    print('score:', score, 'ind:', split_ind, 'val:', split_val)


def test_root_split():
    feature_values = np.array(
        [[0.30, 0.67, 1.64],
         [0.32, 0.77, 1.95],
         [0.29, 0.70, 1.80],
         [0.32, 0.75, 1.85],
         [0.18, 0.59, 1.69],
         [0.25, 0.74, 2.00],
         [0.26, 0.74, 1.98],
         [0.26, 0.71, 1.83]]
    ).T
    g_hist = np.array(
        [
            [0.8314, 1.7989, 3.3846],
            [2.9277, 1.8429, 1.2443],
            [1.6380, 1.5657, 2.8111],
            [2.1322, 3.2927, 0.5899],
            [1.5185, 2.0686, 2.4278],
            [1.5105, 2.6922, 1.8122],
            [0.0000, 4.3245, 1.6903],
            [2.4012, 3.6136, 0.0000]
        ]
    ).T
    h_hist = np.array(
        [
            [0.7147, 3.7312, 3.4598],
            [3.5886, 2.4152, 1.9019],
            [2.9174, 2.5107, 2.4776],
            [3.0651, 3.8540, 0.9866],
            [1.6263, 3.1844, 3.0950],
            [2.1403, 4.2217, 1.5437],
            [0.0000, 5.7955, 2.1103],
            [4.4754, 3.4303, 0.0000]
        ]
    ).T
    root_sum = [6.4, 12.8]
    evaluate(g_hist, h_hist, feature_values, root_sum)


def test_basic():
    g_hist = np.array([-0.5, 0.5]).T
    h_hist = np.array([0.5, 0.5]).T
    feature_values = np.array([1.0, 2.0]).T
    evaluate(g_hist, h_hist, feature_values)


if __name__ == '__main__':
    test_root_split()
    pass
