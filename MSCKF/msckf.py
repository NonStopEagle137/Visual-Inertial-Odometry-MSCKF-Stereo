import numpy as np
from scipy.stats import chi2

from utils import *
from feature import Feature

import time
from collections import namedtuple
from jit_utils import (_process_model, _predict_new_state, _propaget_state_Covariance, _state_augmentation,
                        _fastInv, _fastNorm, _fastQR, _fastSolve, _fastSVD)





class IMUState(object):
    # id for next IMU state
    _vio_next_id__ = 0

    # Gravity vector in the world frame
    _vio_gravity__ = np.array([0., 0., -9.81])

    # Transformation offset from the IMU frame to the body frame. 
    # The transformation takes a vector from the IMU frame to the 
    # body frame. The _vio_z__ axis of the body frame should point upwards.
    # Normally, this transform should be identity.
    _vio_T_imu_body__ = Isometry3d(np.identity(3), np.zeros(3))

    def __init__(self, new_id=None):
        # An unique identifier for the IMU state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the IMU (body) frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the IMU (body) frame in the world frame.
        self._vio_position__ = np.zeros(3)
        # Velocity of the IMU (body) frame in the world frame.
        self.velocity = np.zeros(3)

        # Bias for measured angular velocity and acceleration.
        self._vio_gyro_bias__ = np.zeros(3)
        self.acc_bias = np.zeros(3)

        # These three variables should have the same physical
        # interpretation with `orientation`, `_vio_position__`, and
        # `velocity`. There three variables are used to modify
        # the transition matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)
        self.velocity_null = np.zeros(3)

        # Transformation between the IMU and the left camera (cam0)
        self.R_imu_cam0 = np.identity(3)
        self.t_cam0_imu = np.zeros(3)


class CAMState(object):
    # Takes a vector from the cam0 frame to the cam1 frame.
    _vio_R_cam0_cam1__ = None
    _vio_t_cam0_cam1__ = None

    def __init__(self, new_id=None):
        # An unique identifier for the CAM state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the camera frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the camera frame in the world frame.
        self._vio_position__ = np.zeros(3)

        # These two variables should have the same physical
        # interpretation with `orientation` and `_vio_position__`.
        # There two variables are used to modify the _vio_measurement__
        # Jacobian matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)

        
class StateServer(object):
    """
    Store one IMU states and several camera states for constructing 
    _vio_measurement__ model.
    """
    def __init__(self):
        self._vio_imu_state__ = IMUState()
        self._vio_cam_states__ = dict()   # <CAMStateID, CAMState>, ordered dict

        # State covariance matrix
        self._vio_state_cov__ = np.zeros((21, 21))
        self._vio_continuous_noise_cov__ = np.zeros((12, 12))



class MSCKF(object):
    def __init__(self, config):
        self.config = config
        self.optimization_config = config._vio_optimization_config__

        # IMU data buffer
        # This is buffer is used to handle the unsynchronization or
        # transfer delay between IMU and Image messages.
        self.imu_msg_buffer = []

        # State vector
        self.state_server = StateServer()
        # Features used
        self.map_server = dict()   # <FeatureID, Feature>

        # Chi squared test table.
        # Initialize the chi squared test table with confidence level 0.95.
        self.chi_squared_test_table = dict()
        for _vio_i__ in range(1, 100):
            self.chi_squared_test_table[_vio_i__] = chi2.ppf(0.05, _vio_i__)

        # Set the initial IMU state.
        # The intial orientation and _vio_position__ will be set to the origin implicitly.
        # But the initial velocity and bias can be set by parameters.
        # TODO: is it reasonable to set the initial bias to 0?
        self.state_server._vio_imu_state__.velocity = config._vio_velocity__
        self.reset_state_cov()

        _vio_continuous_noise_cov__ = np.identity(12)
        _vio_continuous_noise_cov__[:3, :3] *= self.config._vio_gyro_noise__
        _vio_continuous_noise_cov__[3:6, 3:6] *= self.config._vio_gyro_bias_noise__
        _vio_continuous_noise_cov__[6:9, 6:9] *= self.config._vio_acc_noise__
        _vio_continuous_noise_cov__[9:, 9:] *= self.config._vio_acc_bias_noise__
        self.state_server._vio_continuous_noise_cov__ = _vio_continuous_noise_cov__

        # Gravity vector in the world frame
        IMUState._vio_gravity__ = config._vio_gravity__

        # Transformation between the IMU and the left camera (cam0)
        _vio_T_cam0_imu__ = np.linalg.inv(config._vio_T_imu_cam0__)
        self.state_server._vio_imu_state__.R_imu_cam0 = _vio_T_cam0_imu__[:3, :3].T
        self.state_server._vio_imu_state__.t_cam0_imu = _vio_T_cam0_imu__[:3, 3]

        # Extrinsic parameters of camera and IMU.
        _vio_T_cam0_cam1__ = config._vio_T_cn_cnm1__
        CAMState._vio_R_cam0_cam1__ = _vio_T_cam0_cam1__[:3, :3]
        CAMState._vio_t_cam0_cam1__ = _vio_T_cam0_cam1__[:3, 3]
        Feature._vio_R_cam0_cam1__ = CAMState._vio_R_cam0_cam1__
        Feature._vio_t_cam0_cam1__ = CAMState._vio_t_cam0_cam1__
        IMUState._vio_T_imu_body__ = Isometry3d(
            config._vio_T_imu_body__[:3, :3],
            config._vio_T_imu_body__[:3, 3])

        # Tracking rate.
        self.tracking_rate = None

        # Indicate if the _vio_gravity__ vector is set.
        self.is_gravity_set = False
        # Indicate if the received image is the first one. The system will 
        # _vio_start__ after receiving the first image.
        self.is_first_img = True

    def imu_callback(self, imu_msg):
        """
        Callback function for the imu message.
        """
        # IMU msgs are pushed backed into a buffer instead of being processed 
        # immediately. The IMU msgs are processed when the next image is  
        # available, in which way, we can easily handle the transfer delay.
        self.imu_msg_buffer.append(imu_msg)

        if not self.is_gravity_set:
            if len(self.imu_msg_buffer) >= 200:
                self.initialize_gravity_and_bias()
                self.is_gravity_set = True

    def feature_callback(self, feature_msg):
        """
        Callback function for _vio_feature__ measurements.
        """
        if not self.is_gravity_set:
            return
        _vio_start__ = time.time()

        # Start the system if the first image is received.
        # The frame where the first image is received will be the origin.
        if self.is_first_img:
            self.is_first_img = False
            self.state_server._vio_imu_state__.timestamp = feature_msg.timestamp

        _vio_t__ = time.time()

        # Propogate the IMU state.
        # that are received before the image _vio_msg__.
        self.batch_imu_processing(feature_msg.timestamp)

        #print('---batch_imu_processing    ', time.time() - _vio_t__)
        _vio_t__ = time.time()

        # Augment the state vector.
        self.state_augmentation(feature_msg.timestamp)

        #print('---state_augmentation      ', time.time() - _vio_t__)
        _vio_t__ = time.time()

        # Add new observations for existing features or new features 
        # in the map server.
        self.add_feature_observations(feature_msg)

        #print('---add_feature_observations', time.time() - _vio_t__)
        _vio_t__ = time.time()

        # Perform _vio_measurement__ update if necessary.
        # And prune features and camera states.
        self.remove_lost_features()

        #print('---remove_lost_features    ', time.time() - _vio_t__)
        _vio_t__ = time.time()

        self.prune_cam_state_buffer()

        #print('---prune_cam_state_buffer  ', time.time() - _vio_t__)
        #print('---msckf elapsed:          ', time.time() - _vio_start__, f'({feature_msg.timestamp})')

        try:
            # Publish the odometry.
            return self.publish(feature_msg.timestamp)
        finally:
            # Reset the system if necessary.
            self.online_reset()

    def initialize_gravity_and_bias(self):
        """
        Initialize the IMU bias and initial orientation based on the 
        first few IMU readings.
        """
        _vio_sum_angular_vel__ = np.zeros(3)
        _vio_sum_linear_acc__ = np.zeros(3)
        for _vio_msg__ in self.imu_msg_buffer:
            _vio_sum_angular_vel__ += _vio_msg__.angular_velocity
            _vio_sum_linear_acc__ += _vio_msg__.linear_acceleration

        _vio_gyro_bias__ = _vio_sum_angular_vel__ / len(self.imu_msg_buffer)
        self.state_server._vio_imu_state__._vio_gyro_bias__ = _vio_gyro_bias__

        # This is the _vio_gravity__ in the IMU frame.
        _vio_gravity_imu__ = _vio_sum_linear_acc__ / len(self.imu_msg_buffer)

        # Initialize the initial orientation, so that the estimation
        # is consistent with the inertial frame.
        _vio_gravity_norm__ = np.linalg.norm(_vio_gravity_imu__)
        IMUState._vio_gravity__ = np.array([0., 0., -_vio_gravity_norm__])

        self.state_server._vio_imu_state__.orientation = from_two_vectors(
            -IMUState._vio_gravity__, _vio_gravity_imu__)

    # Filter related functions
    # (batch_imu_processing, process_model, predict_new_state)
    def batch_imu_processing(self, time_bound):
        """
        Propogate the state
        """
        _vio_used_imu_msg_count__ = 0
        for _vio_msg__ in self.imu_msg_buffer:
            _vio_imu_time__ = _vio_msg__.vio_timestamp__
            if _vio_imu_time__ < self.state_server._vio_imu_state__.timestamp:
                _vio_used_imu_msg_count__ += 1
                continue
            if _vio_imu_time__ > time_bound:
                break

            # Execute process model.
            self.process_model(
                _vio_imu_time__, _vio_msg__.angular_velocity, _vio_msg__.linear_acceleration)
            _vio_used_imu_msg_count__ += 1

            # Update the state info
            self.state_server._vio_imu_state__.timestamp = _vio_imu_time__

        self.state_server._vio_imu_state__.id = IMUState._vio_next_id__
        IMUState._vio_next_id__ += 1

        # Remove all used IMU msgs.
        self.imu_msg_buffer = self.imu_msg_buffer[_vio_used_imu_msg_count__:]

    

    def process_model(self, time, m_gyro, m_acc):
        _vio_imu_state__ = self.state_server._vio_imu_state__
        _dt_ = time - _vio_imu_state__.timestamp

        _vio_gyro__ = m_gyro - _vio_imu_state__._vio_gyro_bias__
        _vio_acc__ = m_acc - _vio_imu_state__.acc_bias

        # Compute discrete transition and noise covariance matrix
        _vio_R_w_i__ = to_rotation(_vio_imu_state__.orientation)
        _vio_F__,_vio_G__, _vio_Fdt__, _vio_Fdt_square__, _vio_Fdt_cube__, _vio_Phi__  =  _process_model(_vio_gyro__,_vio_R_w_i__,_vio_acc__, _dt_)
        
        # _vio_F__ = np.zeros((21, 21))
        # _vio_G__ = np.zeros((21, 12))

        # 

        # _vio_F__[:3, :3] = -skew(_vio_gyro__)
        # _vio_F__[:3, 3:6] = -np.identity(3)
        # _vio_F__[6:9, :3] = -_vio_R_w_i__.T @ skew(_vio_acc__)
        # _vio_F__[6:9, 9:12] = -_vio_R_w_i__.T
        # _vio_F__[12:15, 6:9] = np.identity(3)

        # _vio_G__[:3, :3] = -np.identity(3)
        # _vio_G__[3:6, 3:6] = np.identity(3)
        # _vio_G__[6:9, 6:9] = -_vio_R_w_i__.T
        # _vio_G__[9:12, 9:12] = np.identity(3)

        # # Approximate matrix exponential to the 3rd order, which can be 
        # # considered to be accurate enough assuming _dt_ is within 0.01s.
        # _vio_Fdt__ = _vio_F__ * _dt_
        # _vio_Fdt_square__ = _vio_Fdt__ @ _vio_Fdt__
        # _vio_Fdt_cube__ = _vio_Fdt_square__ @ _vio_Fdt__
        # _vio_Phi__ = np.identity(21) + _vio_Fdt__ + _vio_Fdt_square__/2. + _vio_Fdt_cube__/6.

        # Propogate the state using 4th order Runge-Kutta
        self.predict_new_state(_dt_, _vio_gyro__, _vio_acc__)

        # Modify the transition matrix
        _vio_R_kk_1__ = to_rotation(_vio_imu_state__.orientation_null)
        _vio_Phi__[:3, :3] = to_rotation(_vio_imu_state__.orientation) @ _vio_R_kk_1__.T

        _vio_u__ = _vio_R_kk_1__ @ IMUState._vio_gravity__
        # _vio_s__ = (_vio_u__.T @ _vio_u__).inverse() @ _vio_u__.T
        # _vio_s__ = np.linalg.inv(_vio_u__[:, None] * _vio_u__) @ _vio_u__
        _vio_s__ = _vio_u__ / (_vio_u__ @ _vio_u__)

        _A1_ = _vio_Phi__[6:9, :3]
        _w1_ = skew(_vio_imu_state__.velocity_null - _vio_imu_state__.velocity) @ IMUState._vio_gravity__
        _vio_Phi__[6:9, :3] = _A1_ - (_A1_ @ _vio_u__ - _w1_)[:, None] * _vio_s__

        _A2_ = _vio_Phi__[12:15, :3]
        _w2_ = skew(_dt_*_vio_imu_state__.velocity_null+_vio_imu_state__.position_null - 
            _vio_imu_state__._vio_position__) @ IMUState._vio_gravity__
        _vio_Phi__[12:15, :3] = _A2_ - (_A2_ @ _vio_u__ - _w2_)[:, None] * _vio_s__

        # Propogate the state covariance matrix.

        self.state_server._vio_state_cov__[:21, :21] = _propaget_state_Covariance(self.state_server._vio_continuous_noise_cov__,
                                                        self.state_server._vio_state_cov__,
                                                         _vio_G__, _vio_Phi__, _dt_)[:21, :21]
        # _vio_Q__ = _vio_Phi__ @ _vio_G__ @ self.state_server._vio_continuous_noise_cov__ @ _vio_G__.T @ _vio_Phi__.T * _dt_
        # self.state_server._vio_state_cov__[:21, :21] = (
        #     _vio_Phi__ @ self.state_server._vio_state_cov__[:21, :21] @ _vio_Phi__.T + _vio_Q__)

        if len(self.state_server._vio_cam_states__) > 0:
            self.state_server._vio_state_cov__[:21, 21:] = (
                _vio_Phi__ @ self.state_server._vio_state_cov__[:21, 21:])
            self.state_server._vio_state_cov__[21:, :21] = (
                self.state_server._vio_state_cov__[21:, :21] @ _vio_Phi__.T)

        # Fix the covariance to be symmetric
        self.state_server._vio_state_cov__ = (
            self.state_server._vio_state_cov__ + self.state_server._vio_state_cov__.T) / 2.

        # Update the state correspondes to null space.
        self.state_server._vio_imu_state__.orientation_null = _vio_imu_state__.orientation
        self.state_server._vio_imu_state__.position_null = _vio_imu_state__._vio_position__
        self.state_server._vio_imu_state__.velocity_null = _vio_imu_state__.velocity

    def predict_new_state(self, _dt_, _vio_gyro__, _vio_acc__):

        _vio_q__ = self.state_server._vio_imu_state__.orientation
        _vio_v__ = self.state_server._vio_imu_state__.velocity
        _vio_p__ = self.state_server._vio_imu_state__._vio_position__
        _vio_gravity__ = IMUState._vio_gravity__
        _vio_q__, _vio_v__, _vio_p__ = _predict_new_state(_dt_, _vio_gyro__, _vio_acc__,
                                         _vio_q__, _vio_v__, _vio_p__, _vio_gravity__)
        self.state_server._vio_imu_state__.orientation = _vio_q__
        self.state_server._vio_imu_state__.velocity = _vio_v__
        self.state_server._vio_imu_state__._vio_position__ = _vio_p__
        
    
    # Measurement update
    # (state_augmentation, add_feature_observations)
    def state_augmentation(self, time):
        _vio_imu_state__ = self.state_server._vio_imu_state__
        _vio_R_i_c__ = _vio_imu_state__.R_imu_cam0
        _vio_t_c_i__ = _vio_imu_state__.t_cam0_imu

        # Add a new camera state to the state server.
        _vio_R_w_i__ = to_rotation(_vio_imu_state__.orientation)
        _vio_R_w_c__ = _vio_R_i_c__ @ _vio_R_w_i__
        _vio_t_c_w__ = _vio_imu_state__._vio_position__ + _vio_R_w_i__.T @ _vio_t_c_i__

        _vio_cam_state__ = CAMState(_vio_imu_state__.id)
        _vio_cam_state__.timestamp = time
        _vio_cam_state__.orientation = to_quaternion(_vio_R_w_c__)
        _vio_cam_state__._vio_position__ = _vio_t_c_w__
        _vio_cam_state__.orientation_null = _vio_cam_state__.orientation
        _vio_cam_state__.position_null = _vio_cam_state__._vio_position__
        self.state_server._vio_cam_states__[_vio_imu_state__.id] = _vio_cam_state__

        _vio_stateCovShape__ = self.state_server._vio_state_cov__.shape[0]
        _vio_state_covNew__ = np.zeros((_vio_stateCovShape__+6, _vio_stateCovShape__+6))
        _vio_state_covNew__ = _state_augmentation(_vio_R_i_c__, _vio_R_w_i__, _vio_t_c_i__, self.state_server._vio_state_cov__, _vio_stateCovShape__)
        # Fix the covariance to be symmetric
        self.state_server._vio_state_cov__ = (_vio_state_covNew__ + _vio_state_covNew__.T) / 2.

    def add_feature_observations(self, feature_msg):
        _vio_state_id__ = self.state_server._vio_imu_state__.id
        _vio_curr_feature_num__ = len(self.map_server)
        _vio_tracked_feature_num__ = 0
        #raise ValueError
        for _vio_feature__ in feature_msg.vio_features:
            if _vio_feature__.id not in self.map_server:
                # This is a new _vio_feature__.
                _vio_map_feature__ = Feature(_vio_feature__.id, self.optimization_config)
                _vio_map_feature__.observations[_vio_state_id__] = np.array([
                    _vio_feature__.u0, _vio_feature__.v0, _vio_feature__.u1, _vio_feature__.v1])
                self.map_server[_vio_feature__.id] = _vio_map_feature__
            else:
                # This is an old _vio_feature__.
                self.map_server[_vio_feature__.id].observations[_vio_state_id__] = np.array([
                    _vio_feature__.u0, _vio_feature__.v0, _vio_feature__.u1, _vio_feature__.v1])
                _vio_tracked_feature_num__ += 1

        self.tracking_rate = _vio_tracked_feature_num__ / (_vio_curr_feature_num__+1e-5)

    def measurement_jacobian(self, cam_state_id, _vio_feature_id__):
        """
        This function is used to compute the _vio_measurement__ Jacobian
        for a single _vio_feature__ observed at a single camera frame.
        """
        # Prepare all the required data.
        _vio_cam_state__ = self.state_server._vio_cam_states__[cam_state_id]
        _vio_feature__ = self.map_server[_vio_feature_id__]

        # Cam0 pose.
        _vio_R_w_c0__ = to_rotation(_vio_cam_state__.orientation)
        _vio_t_c0_w__ = _vio_cam_state__._vio_position__

        # Cam1 pose.
        _vio_R_w_c1__ = CAMState._vio_R_cam0_cam1__ @ _vio_R_w_c0__
        _vio_t_c1_w__ = _vio_t_c0_w__ - _vio_R_w_c1__.T @ CAMState._vio_t_cam0_cam1__

        # 3d _vio_feature__ _vio_position__ in the world frame.
        # And its observation with the stereo cameras.
        _vio_p_w__ = _vio_feature__._vio_position__
        _vio_z__ = _vio_feature__.observations[cam_state_id]

        # Convert the _vio_feature__ _vio_position__ from the world frame to
        # the cam0 and cam1 frame.
        _vio_p_c0__ = _vio_R_w_c0__ @ (_vio_p_w__ - _vio_t_c0_w__)
        _vio_p_c1__ = _vio_R_w_c1__ @ (_vio_p_w__ - _vio_t_c1_w__)

        # Compute the Jacobians.
        _vio_dz_dpc0__ = np.zeros((4, 3))
        _vio_dz_dpc0__[0, 0] = 1 / _vio_p_c0__[2]
        _vio_dz_dpc0__[1, 1] = 1 / _vio_p_c0__[2]
        _vio_dz_dpc0__[0, 2] = -_vio_p_c0__[0] / (_vio_p_c0__[2] * _vio_p_c0__[2])
        _vio_dz_dpc0__[1, 2] = -_vio_p_c0__[1] / (_vio_p_c0__[2] * _vio_p_c0__[2])

        _vio_dz_dpc1__ = np.zeros((4, 3))
        _vio_dz_dpc1__[2, 0] = 1 / _vio_p_c1__[2]
        _vio_dz_dpc1__[3, 1] = 1 / _vio_p_c1__[2]
        _vio_dz_dpc1__[2, 2] = -_vio_p_c1__[0] / (_vio_p_c1__[2] * _vio_p_c1__[2])
        _vio_dz_dpc1__[3, 2] = -_vio_p_c1__[1] / (_vio_p_c1__[2] * _vio_p_c1__[2])

        _vio_dpc0_dxc__ = np.zeros((3, 6))
        _vio_dpc0_dxc__[:, :3] = skew(_vio_p_c0__)
        _vio_dpc0_dxc__[:, 3:] = -_vio_R_w_c0__

        _vio_dpc1_dxc__ = np.zeros((3, 6))
        _vio_dpc1_dxc__[:, :3] = CAMState._vio_R_cam0_cam1__ @ skew(_vio_p_c0__)
        _vio_dpc1_dxc__[:, 3:] = -_vio_R_w_c1__

        _vio_dpc0_dpg__ = _vio_R_w_c0__
        _vio_dpc1_dpg__ = _vio_R_w_c1__

        _vio_H_x__ = _vio_dz_dpc0__ @ _vio_dpc0_dxc__ + _vio_dz_dpc1__ @ _vio_dpc1_dxc__   # shape: (4, 6)
        _vio_H_f__ = _vio_dz_dpc0__ @ _vio_dpc0_dpg__ + _vio_dz_dpc1__ @ _vio_dpc1_dpg__   # shape: (4, 3)

        # Modifty the _vio_measurement__ Jacobian to ensure observability constrain.
        _vio_A__ = _vio_H_x__   # shape: (4, 6)
        _vio_u__ = np.zeros(6)
        _vio_u__[:3] = to_rotation(_vio_cam_state__.orientation_null) @ IMUState._vio_gravity__
        _vio_u__[3:] = skew(_vio_p_w__ - _vio_cam_state__.position_null) @ IMUState._vio_gravity__

        _vio_H_x__ = _vio_A__ - (_vio_A__ @ _vio_u__)[:, None] * _vio_u__ / (_vio_u__ @ _vio_u__)
        _vio_H_f__ = -_vio_H_x__[:4, 3:6]

        # Compute the residual.
        _vio_r__ = _vio_z__ - np.array([*_vio_p_c0__[:2]/_vio_p_c0__[2], *_vio_p_c1__[:2]/_vio_p_c1__[2]])

        # _vio_H_x__: shape (4, 6)
        # _vio_H_f__: shape (4, 3)
        # _vio_r__  : shape (4,)
        return _vio_H_x__, _vio_H_f__, _vio_r__

    def feature_jacobian(self, _vio_feature_id__, _vio_cam_state_ids__):
        """
        This function computes the Jacobian of all measurements viewed 
        in the given camera states of this _vio_feature__.
        """
        _vio_feature__ = self.map_server[_vio_feature_id__]

        # Check how many camera states in the provided camera id 
        # camera has actually seen this _vio_feature__.
        _vio_valid_cam_state_ids__ = []
        for _vio_cam_id__ in _vio_cam_state_ids__:
            if _vio_cam_id__ in _vio_feature__.observations:
                _vio_valid_cam_state_ids__.append(_vio_cam_id__)

        _vio_jacobian_row_size__ = 4 * len(_vio_valid_cam_state_ids__)

        _vio_cam_states__ = self.state_server._vio_cam_states__
        _vio_H_xj__ = np.zeros((_vio_jacobian_row_size__, 
            21+len(self.state_server._vio_cam_states__)*6))
        _vio_H_fj__ = np.zeros((_vio_jacobian_row_size__, 3))
        _vio_r_j__ = np.zeros(_vio_jacobian_row_size__)

        _vio_stack_count__ = 0
        for _vio_cam_id__ in _vio_valid_cam_state_ids__:
            _vio_H_xi__, _vio_H_fi__, _vio_r_i__ = self.measurement_jacobian(_vio_cam_id__, _vio_feature__.id)

            # Stack the Jacobians.
            _vio_idx__ = list(self.state_server._vio_cam_states__.keys()).index(_vio_cam_id__)
            _vio_H_xj__[_vio_stack_count__:_vio_stack_count__+4, 21+6*_vio_idx__:21+6*(_vio_idx__+1)] = _vio_H_xi__
            _vio_H_fj__[_vio_stack_count__:_vio_stack_count__+4, :3] = _vio_H_fi__
            _vio_r_j__[_vio_stack_count__:_vio_stack_count__+4] = _vio_r_i__
            _vio_stack_count__ += 4

        # Project the residual and Jacobians onto the nullspace of _vio_H_fj__.
        # svd of _vio_H_fj__
        _vio_U__, _vio____, _vio____ = _fastSVD(_vio_H_fj__)
        _vio_A__ = _vio_U__[:, 3:]

        _vio_H_x__ = _vio_A__.T @ _vio_H_xj__
        _vio_r__ = _vio_A__.T @ _vio_r_j__

        return _vio_H_x__, _vio_r__

    def measurement_update(self, H, _vio_r__):
        if len(H) == 0 or len(_vio_r__) == 0:
            return

        # Decompose the final Jacobian matrix to reduce computational
        # complexity as in Equation (28), (29).
        if H.shape[0] > H.shape[1]:
            # QR decomposition
            _vio_Q__, _vio_R__ = _fastQR(H)  # if M > N, return (M, N), (N, N)
            _vio_H_thin__ = _vio_R__         # shape (N, N)
            _vio_r_thin__ = _vio_Q__.T @ _vio_r__   # shape (N,)
        else:
            _vio_H_thin__ = H   # shape (M, N)
            _vio_r_thin__ = _vio_r__   # shape (M)

        # Compute the Kalman gain.
        _vio_P__ = self.state_server._vio_state_cov__
        _vio_S__ = _vio_H_thin__ @ _vio_P__ @ _vio_H_thin__.T + (self.config._vio_observation_noise__ * 
            np.identity(len(_vio_H_thin__)))
        _vio_K_transpose__ = _fastSolve(_vio_S__, _vio_H_thin__ @ _vio_P__)
        _vio_K__ = _vio_K_transpose__.T   # shape (N, _vio_K__)

        # Compute the error of the state.
        _vio_delta_x__ = _vio_K__ @ _vio_r_thin__

        # Update the IMU state.
        _vio_delta_x_imu__ = _vio_delta_x__[:21]

        # if (np.linalg.norm(_vio_delta_x_imu__[6:9]) > 0.5 or 
        #     np.linalg.norm(_vio_delta_x_imu__[12:15]) > 1.0):
        #     print('[Warning] Update change is too large')

        _vio_dq_imu__ = small_angle_quaternion(_vio_delta_x_imu__[:3])
        _vio_imu_state__ = self.state_server._vio_imu_state__
        _vio_imu_state__.orientation = quaternion_multiplication(
            _vio_dq_imu__, _vio_imu_state__.orientation)
        _vio_imu_state__._vio_gyro_bias__ += _vio_delta_x_imu__[3:6]
        _vio_imu_state__.velocity += _vio_delta_x_imu__[6:9]
        _vio_imu_state__.acc_bias += _vio_delta_x_imu__[9:12]
        _vio_imu_state__._vio_position__ += _vio_delta_x_imu__[12:15]

        _vio_dq_extrinsic__ = small_angle_quaternion(_vio_delta_x_imu__[15:18])
        _vio_imu_state__.R_imu_cam0 = to_rotation(_vio_dq_extrinsic__) @ _vio_imu_state__.R_imu_cam0
        _vio_imu_state__.t_cam0_imu += _vio_delta_x_imu__[18:21]

        # Update the camera states.
        for _vio_i__, (_vio_cam_id__, _vio_cam_state__) in enumerate(
                self.state_server._vio_cam_states__.items()):
            _vio_delta_x_cam__ = _vio_delta_x__[21+_vio_i__*6:27+_vio_i__*6]
            _vio_dq_cam__ = small_angle_quaternion(_vio_delta_x_cam__[:3])
            _vio_cam_state__.orientation = quaternion_multiplication(
                _vio_dq_cam__, _vio_cam_state__.orientation)
            _vio_cam_state__._vio_position__ += _vio_delta_x_cam__[3:]

        # Update state covariance.
        _vio_I_KH__ = np.identity(len(_vio_K__)) - _vio_K__ @ _vio_H_thin__
        # _vio_state_cov__ = _vio_I_KH__ @ self.state_server._vio_state_cov__ @ _vio_I_KH__.T + (
        #     _vio_K__ @ _vio_K__.T * self.config.observation_noise)
        _vio_state_cov__ = _vio_I_KH__ @ self.state_server._vio_state_cov__   # ?

        # Fix the covariance to be symmetric
        self.state_server._vio_state_cov__ = (_vio_state_cov__ + _vio_state_cov__.T) / 2.

    def gating_test(self, H, _vio_r__, dof):
        _P1_ = H @ self.state_server._vio_state_cov__ @ H.T
        _P2_ = self.config._vio_observation_noise__ * np.identity(len(H))
        _vio_gamma__ = _vio_r__ @ _fastSolve(_P1_+_P2_, _vio_r__)

        if(_vio_gamma__ < self.chi_squared_test_table[dof]):
            return True
        else:
            return False

    def remove_lost_features(self):
        # Remove the features that lost track.
        # BTW, find the _vio_size__ the final Jacobian matrix and residual vector.
        _vio_jacobian_row_size__ = 0
        _vio_invalid_feature_ids__ = []
        _vio_processed_feature_ids__ = []

        for _vio_feature__ in self.map_server.values():
            # Pass the features that are still being tracked.
            if self.state_server._vio_imu_state__.id in _vio_feature__.observations:
                continue
            if len(_vio_feature__.observations) < 3:
                _vio_invalid_feature_ids__.append(_vio_feature__.id)
                continue

            # Check if the _vio_feature__ can be initialized if it has not been.
            if not _vio_feature__.is_initialized:
                # Ensure there is enough translation to triangulate the _vio_feature__
                if not _vio_feature__.check_motion(self.state_server._vio_cam_states__):
                    _vio_invalid_feature_ids__.append(_vio_feature__.id)
                    continue

                # Intialize the _vio_feature__ _vio_position__ based on all current available 
                # measurements.
                _vio_ret__ = _vio_feature__.initialize_position(self.state_server._vio_cam_states__)
                if _vio_ret__ is False:
                    _vio_invalid_feature_ids__.append(_vio_feature__.id)
                    continue

            _vio_jacobian_row_size__ += (4 * len(_vio_feature__.observations) - 3)
            _vio_processed_feature_ids__.append(_vio_feature__.id)

        # Remove the features that do not have enough measurements.
        for _vio_feature_id__ in _vio_invalid_feature_ids__:
            del self.map_server[_vio_feature_id__]

        # Return if there is no lost _vio_feature__ to be processed.
        if len(_vio_processed_feature_ids__) == 0:
            return

        _vio_H_x__ = np.zeros((_vio_jacobian_row_size__, 
            21+6*len(self.state_server._vio_cam_states__)))
        _vio_r__ = np.zeros(_vio_jacobian_row_size__)
        _vio_stack_count__ = 0

        # Process the features which lose track.
        for _vio_feature_id__ in _vio_processed_feature_ids__:
            _vio_feature__ = self.map_server[_vio_feature_id__]

            _vio_cam_state_ids__ = []
            for _vio_cam_id__, _vio_measurement__ in _vio_feature__.observations.items():
                _vio_cam_state_ids__.append(_vio_cam_id__)

            _vio_H_xj__, _vio_r_j__ = self.feature_jacobian(_vio_feature__.id, _vio_cam_state_ids__)

            if self.gating_test(_vio_H_xj__, _vio_r_j__, len(_vio_cam_state_ids__)-1):
                _vio_H_x__[_vio_stack_count__:_vio_stack_count__+_vio_H_xj__.shape[0], :_vio_H_xj__.shape[1]] = _vio_H_xj__
                _vio_r__[_vio_stack_count__:_vio_stack_count__+len(_vio_r_j__)] = _vio_r_j__
                _vio_stack_count__ += _vio_H_xj__.shape[0]

            # Put an upper bound on the row _vio_size__ of _vio_measurement__ Jacobian,
            # which helps guarantee the executation time.
            if _vio_stack_count__ > 1500:
                break

        _vio_H_x__ = _vio_H_x__[:_vio_stack_count__]
        _vio_r__ = _vio_r__[:_vio_stack_count__]

        # Perform the _vio_measurement__ update step.
        self.measurement_update(_vio_H_x__, _vio_r__)

        # Remove all processed features from the map.
        for _vio_feature_id__ in _vio_processed_feature_ids__:
            del self.map_server[_vio_feature_id__]

    def find_redundant_cam_states(self):
        # Move the iterator to the key _vio_position__.
        _vio_cam_state_pairs__ = list(self.state_server._vio_cam_states__.items())

        _vio_key_cam_state_idx__ = len(_vio_cam_state_pairs__) - 4
        _vio_cam_state_idx__ = _vio_key_cam_state_idx__ + 1
        _vio_first_cam_state_idx__ = 0

        # Pose of the key camera state.
        _vio_key_position__ = _vio_cam_state_pairs__[_vio_key_cam_state_idx__][1]._vio_position__
        _vio_key_rotation__ = to_rotation(
            _vio_cam_state_pairs__[_vio_key_cam_state_idx__][1].orientation)

        _vio_rm_cam_state_ids__ = []

        # Mark the camera states to be removed based on the
        # motion between states.
        for _vio_i__ in range(2):
            _vio_position__ = _vio_cam_state_pairs__[_vio_cam_state_idx__][1]._vio_position__
            _vio_rotation__ = to_rotation(
                _vio_cam_state_pairs__[_vio_cam_state_idx__][1].orientation)
            
            _vio_distance__ = np.linalg.norm(_vio_position__ - _vio_key_position__)
            _vio_angle__ = 2 * np.arccos(to_quaternion(
                _vio_rotation__ @ _vio_key_rotation__.T)[-1])

            if _vio_angle__ < 0.2618 and _vio_distance__ < 0.4 and self.tracking_rate > 0.5:
                _vio_rm_cam_state_ids__.append(_vio_cam_state_pairs__[_vio_cam_state_idx__][0])
                _vio_cam_state_idx__ += 1
            else:
                _vio_rm_cam_state_ids__.append(_vio_cam_state_pairs__[_vio_first_cam_state_idx__][0])
                _vio_first_cam_state_idx__ += 1
                _vio_cam_state_idx__ += 1

        # Sort the elements in the output list.
        _vio_rm_cam_state_ids__ = sorted(_vio_rm_cam_state_ids__)
        return _vio_rm_cam_state_ids__


    def prune_cam_state_buffer(self):
        if len(self.state_server._vio_cam_states__) < self.config._vio_max_cam_state_size__:
            return

        # Find two camera states to be removed.
        _vio_rm_cam_state_ids__ = self.find_redundant_cam_states()

        # Find the _vio_size__ of the Jacobian matrix.
        _vio_jacobian_row_size__ = 0
        for _vio_feature__ in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this _vio_feature__.
            _vio_involved_cam_state_ids__ = []
            for _vio_cam_id__ in _vio_rm_cam_state_ids__:
                if _vio_cam_id__ in _vio_feature__.observations:
                    _vio_involved_cam_state_ids__.append(_vio_cam_id__)

            if len(_vio_involved_cam_state_ids__) == 0:
                continue
            if len(_vio_involved_cam_state_ids__) == 1:
                del _vio_feature__.observations[_vio_involved_cam_state_ids__[0]]
                continue

            if not _vio_feature__.is_initialized:
                # Check if the _vio_feature__ can be initialize.
                if not _vio_feature__.check_motion(self.state_server._vio_cam_states__):
                    # If the _vio_feature__ cannot be initialized, just remove
                    # the observations associated with the camera states
                    # to be removed.
                    for _vio_cam_id__ in _vio_involved_cam_state_ids__:
                        del _vio_feature__.observations[_vio_cam_id__]
                    continue

                _vio_ret__ = _vio_feature__.initialize_position(self.state_server._vio_cam_states__)
                if _vio_ret__ is False:
                    for _vio_cam_id__ in _vio_involved_cam_state_ids__:
                        del _vio_feature__.observations[_vio_cam_id__]
                    continue

            _vio_jacobian_row_size__ += 4*len(_vio_involved_cam_state_ids__) - 3

        # Compute the Jacobian and residual.
        _vio_H_x__ = np.zeros((_vio_jacobian_row_size__, 21+6*len(self.state_server._vio_cam_states__)))
        _vio_r__ = np.zeros(_vio_jacobian_row_size__)

        _vio_stack_count__ = 0
        for _vio_feature__ in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this _vio_feature__.
            _vio_involved_cam_state_ids__ = []
            for _vio_cam_id__ in _vio_rm_cam_state_ids__:
                if _vio_cam_id__ in _vio_feature__.observations:
                    _vio_involved_cam_state_ids__.append(_vio_cam_id__)

            if len(_vio_involved_cam_state_ids__) == 0:
                continue

            _vio_H_xj__, _vio_r_j__ = self.feature_jacobian(_vio_feature__.id, _vio_involved_cam_state_ids__)

            if self.gating_test(_vio_H_xj__, _vio_r_j__, len(_vio_involved_cam_state_ids__)):
                _vio_H_x__[_vio_stack_count__:_vio_stack_count__+_vio_H_xj__.shape[0], :_vio_H_xj__.shape[1]] = _vio_H_xj__
                _vio_r__[_vio_stack_count__:_vio_stack_count__+len(_vio_r_j__)] = _vio_r_j__
                _vio_stack_count__ += _vio_H_xj__.shape[0]

            for _vio_cam_id__ in _vio_involved_cam_state_ids__:
                del _vio_feature__.observations[_vio_cam_id__]

        _vio_H_x__ = _vio_H_x__[:_vio_stack_count__]
        _vio_r__ = _vio_r__[:_vio_stack_count__]

        # Perform _vio_measurement__ update.
        self.measurement_update(_vio_H_x__, _vio_r__)

        for _vio_cam_id__ in _vio_rm_cam_state_ids__:
            _vio_idx__ = list(self.state_server._vio_cam_states__.keys()).index(_vio_cam_id__)
            _vio_cam_state_start__ = 21 + 6*_vio_idx__
            _vio_cam_state_end__ = _vio_cam_state_start__ + 6

            # Remove the corresponding rows and columns in the state
            # covariance matrix.
            _vio_state_cov__ = self.state_server._vio_state_cov__.copy()
            if _vio_cam_state_end__ < _vio_state_cov__.shape[0]:
                _vio_size__ = _vio_state_cov__.shape[0]
                _vio_state_cov__[_vio_cam_state_start__:-6, :] = _vio_state_cov__[_vio_cam_state_end__:, :]
                _vio_state_cov__[:, _vio_cam_state_start__:-6] = _vio_state_cov__[:, _vio_cam_state_end__:]
            self.state_server._vio_state_cov__ = _vio_state_cov__[:-6, :-6]

            # Remove this camera state in the state vector.
            del self.state_server._vio_cam_states__[_vio_cam_id__]

    def reset_state_cov(self):
        """
        Reset the state covariance.
        """
        _vio_state_cov__ = np.zeros((21, 21))
        _vio_state_cov__[ 3: 6,  3: 6] = self.config._vio_gyro_bias_cov__ * np.identity(3)
        _vio_state_cov__[ 6: 9,  6: 9] = self.config._vio_velocity_cov__ * np.identity(3)
        _vio_state_cov__[ 9:12,  9:12] = self.config._vio_acc_bias_cov__ * np.identity(3)
        _vio_state_cov__[15:18, 15:18] = self.config._vio_extrinsic_rotation_cov__ * np.identity(3)
        _vio_state_cov__[18:21, 18:21] = self.config._vio_extrinsic_translation_cov__ * np.identity(3)
        self.state_server._vio_state_cov__ = _vio_state_cov__

    def reset(self):
        """
        Reset the VIO to initial status.
        """
        # Reset the IMU state.
        _vio_imu_state__ = IMUState()
        _vio_imu_state__.id = self.state_server._vio_imu_state__.id
        _vio_imu_state__.R_imu_cam0 = self.state_server._vio_imu_state__.R_imu_cam0
        _vio_imu_state__.t_cam0_imu = self.state_server._vio_imu_state__.t_cam0_imu
        self.state_server._vio_imu_state__ = _vio_imu_state__

        # Remove all existing camera states.
        self.state_server._vio_cam_states__.clear()

        # Reset the state covariance.
        self.reset_state_cov()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Clear the IMU _vio_msg__ buffer.
        self.imu_msg_buffer.clear()

        # Reset the starting flags.
        self.is_gravity_set = False
        self.is_first_img = True

    def online_reset(self):
        """
        Reset the system online if the uncertainty is too large.
        """
        # Never perform online reset if _vio_position__ std threshold is non-positive.
        if self.config._vio_position_std_threshold__ <= 0:
            return

        # Check the uncertainty of positions to determine if 
        # the system can be reset.
        _vio_position_x_std__ = np.sqrt(self.state_server._vio_state_cov__[12, 12])
        _vio_position_y_std__ = np.sqrt(self.state_server._vio_state_cov__[13, 13])
        _vio_position_z_std__ = np.sqrt(self.state_server._vio_state_cov__[14, 14])

        if max(_vio_position_x_std__, _vio_position_y_std__, _vio_position_z_std__ 
            ) < self.config._vio_position_std_threshold__:
            return

        # print('Start online reset...')

        # Remove all existing camera states.
        self.state_server._vio_cam_states__.clear()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Reset the state covariance.
        self.reset_state_cov()

    def publish(self, time):
        _vio_imu_state__ = self.state_server._vio_imu_state__
        # print('+++publish:')
        # print('   timestamp:', _vio_imu_state__.timestamp)
        # print('   orientation:', _vio_imu_state__.orientation)
        # print('   _vio_position__:', _vio_imu_state__._vio_position__)
        # print('   velocity:', _vio_imu_state__.velocity)
        # print()
        
        _vio_T_i_w__ = Isometry3d(
            to_rotation(_vio_imu_state__.orientation).T,
            _vio_imu_state__._vio_position__)
        _vio_T_b_w__ = IMUState._vio_T_imu_body__ * _vio_T_i_w__ * IMUState._vio_T_imu_body__.inverse()
        _vio_body_velocity__ = IMUState._vio_T_imu_body__._vio_R__ @ _vio_imu_state__.velocity

        _vio_R_w_c__ = _vio_imu_state__.R_imu_cam0 @ _vio_T_i_w__._vio_R__.T
        _vio_t_c_w__ = _vio_imu_state__._vio_position__ + _vio_T_i_w__._vio_R__ @ _vio_imu_state__.t_cam0_imu
        _vio_T_c_w__ = Isometry3d(_vio_R_w_c__.T, _vio_t_c_w__)

        return namedtuple('vio_result', ['timestamp', 'pose', 'velocity', 'cam0_pose'])(
            time, _vio_T_b_w__, _vio_body_velocity__, _vio_T_c_w__)