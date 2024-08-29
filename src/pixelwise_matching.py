import cv2
import numpy as np
from src.metrics import PixelWiseMetrics

def pixel_wise_matching(left_img_path, right_img_path, similiarity_type, disparity_range, scale=16):
    # Read left , right images then convert to grayscale
    left = cv2.imread(left_img_path, 0).astype(dtype=np.float32)
    right = cv2.imread(right_img_path, 0).astype(dtype=np.float32)
    
    height, width = left.shape[:2]
    
    # Metric option
    if similiarity_type == 'l1':
        similiarity_metric = PixelWiseMetrics.l1_distance
    if similiarity_type == 'l2':
        similiarity_metric = PixelWiseMetrics.l2_distance
        
    # Precompute the cost for all disparities
    max_value = 255
    costs = np.full((height, width, disparity_range), max_value, dtype=np.float32)
    for j in range(disparity_range):
        left_d = left[:,j:width]
        right_d = right[:,0:width-j]
        costs[:, j:width, j] = similiarity_metric(left_d, right_d)

    # Find the disparity with the minimum cost
    min_cost_indices = np.argmin(costs, axis=2)

    # Set the disparity map
    depth = min_cost_indices * scale
    depth = depth.astype(np.uint8)

    return depth, cv2.applyColorMap(depth, cv2.COLORMAP_JET)