from sklearn.datasets import fetch_openml
import numpy as np
from numpy.linalg import norm
import time

print("データをダウンロード中")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

def get_histogram(image_data):
    flat_data = image_data.flatten()
    seed_value = int(time.time())
    np.random.seed(seed_value)
    c = 0
    all_diffs = []

    for i in range(10000000):
        ra = np.random.randint(0, 784)
        ix = flat_data[ra]
        all_diffs.append(abs(c-ix))
        c = ix

    hist, _ = np.histogram(all_diffs, bins=256, range=(0, 256))
    return hist / np.sum(hist)

def cos_sim_filtered(hist_a, hist_b):
    a = hist_a[1:]
    b = hist_b[1:]
    return np.dot(a, b) / (norm(a) * norm(b))

mask_1 = (mnist.target == '1')
hist_1 = get_histogram(mnist.data[mask_1][0])
time.sleep(1.0)
mask_8 = (mnist.target == '8')
hist_8 = get_histogram(mnist.data[mask_8][0])

target_image = mnist.data[mask_1][1]
hist_target_1 = get_histogram(target_image)

target_image = mnist.data[mask_8][1]
hist_target_8 = get_histogram(target_image)

print("1との類似度:", cos_sim_filtered(hist_target_1, hist_1))
print("8との類似度:", cos_sim_filtered(hist_target_8, hist_8))
print("1と8の区別:", cos_sim_filtered(hist_1, hist_8))