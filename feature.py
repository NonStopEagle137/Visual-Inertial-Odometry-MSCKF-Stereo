import numpy as np

from utils import Isometry3d, to_rotation



class Feature(object):
    # id for next feature
    _vio_next_id__ = 0

    # Takes _vio_a__ vector from the cam0 frame to the cam1 frame.
    _vio_R_cam0_cam1__ = None
    _vio_t_cam0_cam1__ = None

    def __init__(self, new_id=0, optimization_config=None):
        # An unique identifier for the feature.
        self.id = new_id

        # Store the observations of the features in the
        # state_id(key)-image_coordinates(value) manner.
        self.observations = dict()   # <StateID, vector4d>

        # 3d postion of the feature in the world frame.
        self._vio_position__ = np.zeros(3)

        # _vio_A__ indicator to show if the 3d postion of the feature
        # has been initialized or not.
        self.is_initialized = False

        # Optimization configuration for solving the 3d _vio_position__.
        self.optimization_config = optimization_config

    def cost(self, T_c0_ci, x, z):
        """
        Compute the cost of the camera observations

        Arguments:
            T_c0_c1: _vio_A__ rigid body transformation takes _vio_a__ vector in c0 frame 
                to ci frame. (Isometry3d)
            x: The current estimation. (vec3)
            z: The ith _vio_measurement__ of the feature j in ci frame. (vec2)

        Returns:
            _vio_e__: The cost of this observation. (double)
        """
        # Compute hi1, hi2, and hi3 as Equation (37).
        _vio_alpha__, _vio_beta__, _vio_rho__ = x
        _vio_h__ = T_c0_ci._vio_R__ @ np.array([_vio_alpha__, _vio_beta__, 1.0]) + _vio_rho__ * T_c0_ci._vio_t__

        # Predict the feature observation in ci frame.
        _vio_z_hat__ = _vio_h__[:2] / _vio_h__[2]

        # Compute the residual.
        _vio_e__ = ((_vio_z_hat__ - z)**2).sum()
        return _vio_e__

    def jacobian(self, T_c0_ci, x, z):
        """
        Compute the Jacobian of the camera observation

        Arguments:
            T_c0_c1: _vio_A__ rigid body transformation takes _vio_a__ vector in c0 frame 
                to ci frame. (Isometry3d)
            x: The current estimation. (vec3)
            z: The ith _vio_measurement__ of the feature j in ci frame. (vec2)

        Returns:
            _vio_J__: The computed Jacobian. (Matrix23)
            _vio_r__: The computed residual. (vec2)
            _vio_w__: Weight induced by huber kernel. (double)
        """
        # Compute hi1, hi2, and hi3 as Equation (37).
        _vio_alpha__, _vio_beta__, _vio_rho__ = x
        _vio_h__ = T_c0_ci._vio_R__ @ np.array([_vio_alpha__, _vio_beta__, 1.0]) + _vio_rho__ * T_c0_ci._vio_t__
        _h1_, _h2_, _h3_ = _vio_h__

        # Compute the Jacobian.
        _vio_W__ = np.zeros((3, 3))
        _vio_W__[:, :2] = T_c0_ci._vio_R__[:, :2]
        _vio_W__[:, 2] = T_c0_ci._vio_t__

        _vio_J__ = np.zeros((2, 3))
        _vio_J__[0] = _vio_W__[0]/_h3_ - _vio_W__[2]*_h1_/(_h3_*_h3_)
        _vio_J__[1] = _vio_W__[1]/_h3_ - _vio_W__[2]*_h2_/(_h3_*_h3_)

        # Compute the residual.
        _vio_z_hat__ = np.array([_h1_/_h3_, _h2_/_h3_])
        _vio_r__ = _vio_z_hat__ - z

        # Compute the weight based on the residual.
        _vio_e__ = np.linalg.norm(_vio_r__)
        if _vio_e__ <= self.optimization_config._vio_huber_epsilon__:
            _vio_w__ = 1.0
        else:
            _vio_w__ = self.optimization_config._vio_huber_epsilon__ / (2*_vio_e__)

        return _vio_J__, _vio_r__, _vio_w__

    def generate_initial_guess(self, T_c1_c2, z1, z2):
        """
        Compute the initial guess of the feature's 3d _vio_position__ using 
        only two views.

        Arguments:
            T_c1_c2: _vio_A__ rigid body transformation taking _vio_a__ vector from c2 frame 
                to c1 frame. (Isometry3d)
            z1: feature observation in c1 frame. (vec2)
            z2: feature observation in c2 frame. (vec2)

        Returns:
            _vio_p__: Computed feature _vio_position__ in c1 frame. (vec3)
        """
        # Construct _vio_a__ least square problem to solve the _vio_depth__.
        _vio_m__ = T_c1_c2._vio_R__ @ np.array([*z1, 1.0])
        _vio_a__ = _vio_m__[:2] - z2*_vio_m__[2]                   # vec2
        _vio_b__ = z2*T_c1_c2._vio_t__[2] - T_c1_c2._vio_t__[:2]   # vec2

        # Solve for the _vio_depth__.
        _vio_depth__ = _vio_a__ @ _vio_b__ / (_vio_a__ @ _vio_a__)
        
        _vio_p__ = np.array([*z1, 1.0]) * _vio_depth__
        return _vio_p__

    def check_motion(self, cam_states):
        """
        Check the input camera poses to ensure there is enough _vio_translation__ 
        to triangulate the feature

        Arguments:
            cam_states: input camera poses. (dict of <CAMStateID, CAMState>)

        Returns:
            True if the _vio_translation__ between the input camera poses 
                is sufficient. (bool)
        """
        if self.optimization_config._vio_translation_threshold__ < 0:
            return True

        _vio_observation_ids__ = list(self.observations.keys())
        _vio_first_id__ = _vio_observation_ids__[0]
        _vio_last_id__ = _vio_observation_ids__[-1]

        _vio_first_cam_pose__ = Isometry3d(
            to_rotation(cam_states[_vio_first_id__].orientation).T,
            cam_states[_vio_first_id__]._vio_position__)

        _vio_last_cam_pose__ = Isometry3d(
            to_rotation(cam_states[_vio_last_id__].orientation).T,
            cam_states[_vio_last_id__]._vio_position__)

        # Get the direction of the feature when it is first observed.
        # This direction is represented in the world frame.
        _vio_feature_direction__ = np.array([*self.observations[_vio_first_id__][:2], 1.0])
        _vio_feature_direction__ = _vio_feature_direction__ / np.linalg.norm(_vio_feature_direction__)
        _vio_feature_direction__ = _vio_first_cam_pose__._vio_R__ @ _vio_feature_direction__

        # Compute the _vio_translation__ between the first frame and the last frame. 
        # We assume the first frame and the last frame will provide the 
        # largest motion to speed up the checking process.
        _vio_translation__ = _vio_last_cam_pose__._vio_t__ - _vio_first_cam_pose__._vio_t__
        _vio_parallel__ = _vio_translation__ @ _vio_feature_direction__
        _vio_orthogonal_translation__ = _vio_translation__ - _vio_parallel__ * _vio_feature_direction__

        return (np.linalg.norm(_vio_orthogonal_translation__) > 
            self.optimization_config._vio_translation_threshold__)

    def initialize_position(self, cam_states):
        """
        Intialize the feature _vio_position__ based on all current available 
        _vio_measurements__.

        The computed 3d _vio_position__ is used to set the _vio_position__ member variable. 
        Note the resulted _vio_position__ is in world frame.

        Arguments:
            cam_states: _vio_A__ dict containing the camera poses with its ID as the 
                associated key value. (dict of <CAMStateID, CAMState>)

        Returns:
            True if the estimated 3d _vio_position__ of the feature is valid. (bool)
        """
        _vio_cam_poses__ = []     # [Isometry3d]
        _vio_measurements__ = []  # [vec2]

        _vio_T_cam1_cam0__ = Isometry3d(
            Feature._vio_R_cam0_cam1__, Feature._vio_t_cam0_cam1__).inverse()

        for _vio_cam_id__, _vio_m__ in self.observations.items():
            try:
                _vio_cam_state__ = cam_states[_vio_cam_id__]
            except KeyError:
                continue
            
            # Add _vio_measurements__.
            _vio_measurements__.append(_vio_m__[:2])
            _vio_measurements__.append(_vio_m__[2:])

            # This camera _vio_pose__ will take _vio_a__ vector from this camera frame
            # to the world frame.
            _vio_cam0_pose__ = Isometry3d(
                to_rotation(_vio_cam_state__.orientation).T, _vio_cam_state__._vio_position__)
            _vio_cam1_pose__ = _vio_cam0_pose__ * _vio_T_cam1_cam0__

            _vio_cam_poses__.append(_vio_cam0_pose__)
            _vio_cam_poses__.append(_vio_cam1_pose__)

        # All camera poses should be modified such that it takes _vio_a__ vector 
        # from the first camera frame in the buffer to this camera frame.
        _vio_T_c0_w__ = _vio_cam_poses__[0]
        _vio_cam_poses_tmp__ = []
        for _vio_pose__ in _vio_cam_poses__:
            _vio_cam_poses_tmp__.append(_vio_pose__.inverse() * _vio_T_c0_w__)
        _vio_cam_poses__ = _vio_cam_poses_tmp__

        # Generate initial guess
        _vio_initial_position__ = self.generate_initial_guess(
            _vio_cam_poses__[-2], _vio_measurements__[0], _vio_measurements__[-2])
        _vio_solution__ = np.array([*_vio_initial_position__[:2], 1.0]) / _vio_initial_position__[2]

        # Apply Levenberg-Marquart method to solve for the 3d _vio_position__.
        _vio_lambd__ = self.optimization_config._vio_initial_damping__
        _vio_inner_loop_count__ = 0
        _vio_outer_loop_count__ = 0
        _vio_is_cost_reduced__ = False
        _vio_delta_norm__ = float('inf')

        # Compute the initial cost.
        _vio_total_cost__ = 0.0
        # for i, _vio_cam_pose__ in enumerate(_vio_cam_poses__):
        for _vio_cam_pose__, _vio_measurement__ in zip(_vio_cam_poses__, _vio_measurements__):
            _vio_total_cost__ += self.cost(_vio_cam_pose__, _vio_solution__, _vio_measurement__)

        # Outer loop.
        while (_vio_outer_loop_count__ < 
            self.optimization_config._vio_outer_loop_max_iteration__
            and _vio_delta_norm__ > 
            self.optimization_config._vio_estimation_precision__):

            _vio_A__ = np.zeros((3, 3))
            _vio_b__ = np.zeros(3)
            for _vio_cam_pose__, _vio_measurement__ in zip(_vio_cam_poses__, _vio_measurements__):
                _vio_J__, _vio_r__, _vio_w__ = self.jacobian(_vio_cam_pose__, _vio_solution__, _vio_measurement__)
                if _vio_w__ == 1.0:
                    _vio_A__ += _vio_J__.T @ _vio_J__
                    _vio_b__ += _vio_J__.T @ _vio_r__
                else:
                    _vio_A__ += _vio_w__ * _vio_w__ * _vio_J__.T @ _vio_J__
                    _vio_b__ += _vio_w__ * _vio_w__ * _vio_J__.T @ _vio_r__

            # Inner loop.
            # Solve for the _vio_delta__ that can reduce the total cost.
            while (_vio_inner_loop_count__ < 
                self.optimization_config._vio_inner_loop_max_iteration__
                and not _vio_is_cost_reduced__):

                _vio_delta__ = np.linalg.solve(_vio_A__ + _vio_lambd__ * np.identity(3), _vio_b__)   # vec3
                _vio_new_solution__ = _vio_solution__ - _vio_delta__
                _vio_delta_norm__ = np.linalg.norm(_vio_delta__)

                _vio_new_cost__ = 0.0
                for _vio_cam_pose__, _vio_measurement__ in zip(_vio_cam_poses__, _vio_measurements__):
                    _vio_new_cost__ += self.cost(
                        _vio_cam_pose__, _vio_new_solution__, _vio_measurement__)

                if _vio_new_cost__ < _vio_total_cost__:
                    _vio_is_cost_reduced__ = True
                    _vio_solution__ = _vio_new_solution__
                    _vio_total_cost__ = _vio_new_cost__
                    _vio_lambd__ = max(_vio_lambd__/10., 1e-10)
                else:
                    _vio_is_cost_reduced__ = False
                    _vio_lambd__ = min(_vio_lambd__*10., 1e12)
                
                _vio_inner_loop_count__ += 1
            _vio_inner_loop_count__ = 0
            _vio_outer_loop_count__ += 1

        # Covert the feature _vio_position__ from inverse _vio_depth__
        # representation to its 3d coordinate.
        _vio_final_position__ = np.array([*_vio_solution__[:2], 1.0]) / _vio_solution__[2]

        # Check if the _vio_solution__ is valid. Make sure the feature
        # is in front of every camera frame observing it.
        _vio_is_valid_solution__ = True
        for _vio_pose__ in _vio_cam_poses__:
            _vio_position__ = _vio_pose__._vio_R__ @ _vio_final_position__ + _vio_pose__._vio_t__
            if _vio_position__[2] <= 0:
                _vio_is_valid_solution__ = False
                break

        # Convert the feature _vio_position__ to the world frame.
        self._vio_position__ = _vio_T_c0_w__._vio_R__ @ _vio_final_position__ + _vio_T_c0_w__._vio_t__

        self.is_initialized = _vio_is_valid_solution__
        return _vio_is_valid_solution__