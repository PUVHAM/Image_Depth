import cv2
import numpy as np
from src.metrics import WindowBasedMetrics

def get_similarity_function(similiarity_type):
    if similiarity_type == 'l1':
        similarity_func = WindowBasedMetrics.l1_distance
    elif similiarity_type == 'l2':
        similarity_func = WindowBasedMetrics.l2_distance
    elif similiarity_type == 'cosine':
        similarity_func = WindowBasedMetrics.cosine_similarity
    elif similiarity_type == 'correlation':
        similarity_func = WindowBasedMetrics.correlation_coefficient
    return similarity_func

def window_based_matching(left_img_path, right_img_path, 
                          similiarity_type,
                          disparity_range, 
                          kernel_size=5, 
                          scale=16):
    left  = cv2.imread(left_img_path, 0)
    right = cv2.imread(right_img_path, 0)

    left  = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)

    similarity_func = get_similarity_function(similiarity_type=similiarity_type)

    for y in range(kernel_half, height-kernel_half):
        for x in range(kernel_half, width-kernel_half):
            # Find j where cost has minimum value
            disparity = 0
            cost_optimal = -10000

            for j in range(disparity_range):
                d = x - j
                cost = -10000
                if (d - kernel_half) > 0:
                    wp = left[(y-kernel_half):(y+kernel_half)+1, (x-kernel_half):(x+kernel_half)+1]
                    wqd = right[(y-kernel_half):(y+kernel_half)+1, (d-kernel_half):(d+kernel_half)+1]

                    wp_flattened = wp.flatten()
                    wqd_flattened = wqd.flatten()

                    cost = similarity_func(wp_flattened, wqd_flattened)

                if cost > cost_optimal:
                    cost_optimal = cost
                    disparity = j

            depth[y, x] = disparity * scale

    depth = depth.astype(np.uint8)

    return depth, cv2.applyColorMap(depth, cv2.COLORMAP_JET)