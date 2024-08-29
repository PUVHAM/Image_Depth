import numpy as np
from abc import ABC, abstractmethod

class Metrics(ABC):
    @abstractmethod
    def l1_distance(x, y):
        pass
    
    @abstractmethod
    def l2_distance(x, y):
        pass

class PixelWiseMetrics(Metrics):
    @staticmethod
    def l1_distance(x, y):
        return abs(x - y)
    
    @staticmethod
    def l2_distance(x, y):
        return (x - y) ** 2

class WindowBasedMetrics(Metrics):
    @staticmethod
    def l1_distance(x, y):
        return -1 * np.sum(np.abs(x - y))
    
    @staticmethod
    def l2_distance(x, y):
        return -1 * np.sqrt(np.sum((x - y) ** 2))
    
    @staticmethod
    def cosine_similarity(x, y):
        numerator = x.dot(y)
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        return numerator / denominator if denominator != 0 else 0
    
    @staticmethod
    def correlation_coefficient(x, y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
        return numerator / denominator if denominator != 0 else 0
