import numpy as np
import cv2


class OptimizationConfigEuRoC(object):
    """
    Configuration parameters for 3d feature position optimization.
    """
    def __init__(self):
        self._vio_translation_threshold__ = -1.0  # 0.2
        self._vio_huber_epsilon__ = 0.01
        self._vio_estimation_precision__ = 5e-7
        self._vio_initial_damping__ = 1e-3
        self._vio_outer_loop_max_iteration__ = 5 # 10
        self._vio_inner_loop_max_iteration__ = 5 # 10

class ConfigEuRoC(object):
    def __init__(self):
        # feature position optimization
        self._vio_optimization_config__ = OptimizationConfigEuRoC()

        ## image processor
        self._vio_grid_row__ = 4
        self._vio_grid_col__ = 5
        self._vio_grid_num__ = self._vio_grid_row__ * self._vio_grid_col__
        self._vio_grid_min_feature_num__ = 3
        self._vio_grid_max_feature_num__ = 5
        self._vio_fast_threshold__ = 15
        self._vio_ransac_threshold__ = 3
        self._vio_stereo_threshold__ = 5
        self._vio_max_iteration__ = 30
        self._vio_track_precision__ = 0.01
        self._vio_pyramid_levels__ = 3
        self._vio_patch_size__ = 15
        self._vio_win_size__ = (self._vio_patch_size__, self._vio_patch_size__)

        self._vio_lk_params__ = dict(
            winSize=self._vio_win_size__,
            maxLevel=self._vio_pyramid_levels__,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                self._vio_max_iteration__, 
                self._vio_track_precision__),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        

        ## msckf vio
        # _vio_gravity__
        self._vio_gravity_acc__ = 9.81
        self._vio_gravity__ = np.array([0.0, 0.0, -self._vio_gravity_acc__])

        # Framte rate of the stereo images. This variable is only used to 
        # determine the timing threshold of each iteration of the filter.
        self._vio_frame_rate__ = 20

        # Maximum number of camera states to be stored
        self._vio_max_cam_state_size__ = 20

        # The position uncertainty threshold is used to determine
        # when to reset the system online. Otherwise, the ever-increaseing
        # uncertainty will make the estimation unstable.
        # Note this online reset will be some dead-reckoning.
        # Set this threshold to nonpositive to disable online reset.
        self._vio_position_std_threshold__ = 8.0

        # Threshold for determine keyframes
        self._vio_rotation_threshold__ = 0.2618
        self._vio_translation_threshold__ = 0.4
        self._vio_tracking_rate_threshold__ = 0.5

        # Noise related parameters (Use variance instead of standard deviation)
        self._vio_gyro_noise__ = 0.005 ** 2
        self._vio_acc_noise__ = 0.05 ** 2
        self._vio_gyro_bias_noise__ = 0.001 ** 2
        self._vio_acc_bias_noise__ = 0.01 ** 2
        self._vio_observation_noise__ = 0.035 ** 2

        # initial state
        self._vio_velocity__ = np.zeros(3)

        # The initial covariance of orientation and position can be
        # set to 0. But for _vio_velocity__, bias and extrinsic parameters, 
        # there should be nontrivial uncertainty.
        self._vio_velocity_cov__ = 0.25
        self._vio_gyro_bias_cov__ = 0.01
        self._vio_acc_bias_cov__ = 0.01
        self._vio_extrinsic_rotation_cov__ = 3.0462e-4
        self._vio_extrinsic_translation_cov__ = 2.5e-5

        ## calibration parameters
        # T_imu_cam: takes a vector from the IMU frame to the cam frame.
        # _vio_T_cn_cnm1__: takes a vector from the cam0 frame to the cam1 frame.
        # see https://github.com/ethz-asl/kalibr/wiki/yaml-formats
        self._vio_T_imu_cam0__ = np.array([
            [ 0.014865542981794,   0.999557249008346,  -0.025774436697440,   0.065222909535531],
            [-0.999880929698575,   0.014967213324719,   0.003756188357967,  -0.020706385492719],
            [ 0.004140296794224,   0.025715529947966,   0.999660727177902,  -0.008054602460030],
            [                 0,                   0,                   0,   1.000000000000000]])
        self._vio_cam0_camera_model__ = 'pinhole'
        self._vio_cam0_distortion_model__ = 'radtan'
        self._vio_cam0_distortion_coeffs__ = np.array(
            [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
        self._vio_cam0_intrinsics__ = np.array([458.654, 457.296, 367.215, 248.375])
        self._vio_cam0_resolution__ = np.array([752, 480])

        self._vio_T_imu_cam1__ = np.array([
            [ 0.012555267089103,   0.999598781151433,  -0.025389800891747,  -0.044901980682509],
            [-0.999755099723116,   0.013011905181504,   0.017900583825251,  -0.020569771258915],
            [ 0.018223771455443,   0.025158836311552,   0.999517347077547,  -0.008638135126028],
            [                 0,                   0,                   0,   1.000000000000000]])
        self._vio_T_cn_cnm1__ = np.array([
            [ 0.999997256477881,   0.002312067192424,   0.000376008102415,  -0.110073808127187],
            [-0.002317135723281,   0.999898048506644,   0.014089835846648,   0.000399121547014],
            [-0.000343393120525,  -0.014090668452714,   0.999900662637729,  -0.000853702503357],
            [                 0,                   0,                   0,   1.000000000000000]])
        self._vio_cam1_camera_model__ = 'pinhole'
        self._vio_cam1_distortion_model__ = 'radtan'
        self._vio_cam1_distortion_coeffs__ = np.array(
            [-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05])
        self._vio_cam1_intrinsics__ = np.array([457.587, 456.134, 379.999, 255.238])
        self._vio_cam1_resolution__ = np.array([752, 480])
        # self.baseline = 

        self._vio_T_imu_body__ = np.eye(4)