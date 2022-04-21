import numpy as np
from numba import jit
from utils import utils

class msckf:
    def __init__(self):
        u = utils() # makes all the utils available as u.something
        pass

    def _initialize_IMU_vars(self):
        self.imu_next = 0
        self.gravity = np.array([0,0,-9.81], dtype = np.float32)
