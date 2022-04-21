import numpy as np
import cv2
import time

from itertools import chain, compress
from collections import defaultdict, namedtuple



class FeatureMetaData(object):
    """
    Contain necessary information of a _vio_feature__ for easy access.
    """
    def __init__(self):
        self.id = None           # int
        self._vio_response__ = None     # float
        self.lifetime = None     # int
        self._vio_cam0_point__ = None   # vec2
        self._vio_cam1_point__ = None   # vec2


class FeatureMeasurement(object):
    """
    Stereo measurement of a _vio_feature__.
    """
    def __init__(self):
        self.id = None
        self.u0 = None
        self.v0 = None
        self.u1 = None
        self.v1 = None



class ImageProcessor(object):
    """
    Detect and track vio_features in image sequences.
    """
    def __init__(self, _vio_config__):
        self._vio_config__ = _vio_config__

        # Indicate if this is the first image message.
        self.is_first_img = True

        # ID for the next new _vio_feature__.
        self.next_feature_id = 0

        # Feature detector
        self.detector = cv2.FastFeatureDetector_create(self._vio_config__._vio_fast_threshold__)

        # IMU message buffer.
        self.imu_msg_buffer = []

        # Previous and current images
        self.cam0_prev_img_msg = None
        self.cam0_curr_img_msg = None
        self.cam1_curr_img_msg = None

        # Pyramids for previous and current image
        self.prev_cam0_pyramid = None
        self.curr_cam0_pyramid = None
        self.curr_cam1_pyramid = None

        # Features in the previous and current image.
        # list of lists of FeatureMetaData
        self.prev_features = [[] for _vio____ in range(self._vio_config__._vio_grid_num__)]  # Don'_vio_t__ use [[]] * N
        self.curr_features = [[] for _vio____ in range(self._vio_config__._vio_grid_num__)]

        # Number of vio_features after each outlier removal step.
        # keys: before_tracking, after_tracking, after_matching, _vio_after_ransac__
        self.num_features = defaultdict(int)

        # load _vio_config__
        # Camera calibration parameters
        self.cam0_resolution = _vio_config__._vio_cam0_resolution__   # vec2
        self.cam0_intrinsics = _vio_config__._vio_cam0_intrinsics__   # vec4
        self.cam0_distortion_model = _vio_config__._vio_cam0_distortion_model__     # string
        self.cam0_distortion_coeffs = _vio_config__._vio_cam0_distortion_coeffs__   # vec4

        self.cam1_resolution = _vio_config__._vio_cam1_resolution__   # vec2
        self.cam1_intrinsics = _vio_config__._vio_cam1_intrinsics__   # vec4
        self.cam1_distortion_model = _vio_config__._vio_cam1_distortion_model__     # string
        self.cam1_distortion_coeffs = _vio_config__._vio_cam1_distortion_coeffs__   # vec4

        # Take a vector from cam0 frame to the IMU frame.
        self.T_cam0_imu = np.linalg.inv(_vio_config__._vio_T_imu_cam0__)
        self.R_cam0_imu = self.T_cam0_imu[:3, :3]
        self.t_cam0_imu = self.T_cam0_imu[:3, 3]
        # Take a vector from cam1 frame to the IMU frame.
        self.T_cam1_imu = np.linalg.inv(_vio_config__._vio_T_imu_cam1__)
        self.R_cam1_imu = self.T_cam1_imu[:3, :3]
        self.t_cam1_imu = self.T_cam1_imu[:3, 3]

    def stareo_callback(self, stereo_msg):
        """
        Callback function for the stereo images.
        """
        _vio_start__ = time.time()
        self.cam0_curr_img_msg = stereo_msg.cam0_msg
        self.cam1_curr_img_msg = stereo_msg.cam1_msg

        # Build the image pyramids once since they're used at multiple places.
        self.create_image_pyramids()

        # Detect vio_features in the first frame.
        if self.is_first_img:
            self.initialize_first_frame()
            self.is_first_img = False
            # Draw results.
            # self.draw_features_stereo()
        else:
            # Track the _vio_feature__ in the previous image.
            _vio_t__ = time.time()
            self.track_features()
            print('___track_features:', time.time() - _vio_t__)
            _vio_t__ = time.time()

            # Add new vio_features into the current image.
            self.add_new_features()
            print('___add_new_features:', time.time() - _vio_t__)
            _vio_t__ = time.time()
            self.prune_features()
            print('___prune_features:', time.time() - _vio_t__)
            _vio_t__ = time.time()
            # Draw results.
            # self.draw_features_stereo()
            print('___draw_features_stereo:', time.time() - _vio_t__)
            _vio_t__ = time.time()

        print('===image process elapsed:', time.time() - _vio_start__, f'({stereo_msg.vio_timestamp__})')

        try:
            return self.publish()
        finally:
            self.cam0_prev_img_msg = self.cam0_curr_img_msg
            self.prev_features = self.curr_features
            self.prev_cam0_pyramid = self.curr_cam0_pyramid

            # Initialize the current vio_features to empty vectors.
            self.curr_features = [[] for _vio____ in range(self._vio_config__._vio_grid_num__)]

    def imu_callback(self, _vio_msg__):
        """
        Callback function for the imu message.
        """
        self.imu_msg_buffer.append(_vio_msg__)

    def create_image_pyramids(self):
        """
        Create image pyramids used for KLT tracking.
        (Seems doesn'_vio_t__ work in python)
        """
        _vio_curr_cam0_img__ = self.cam0_curr_img_msg.image
        # self.curr_cam0_pyramid = cv2.buildOpticalFlowPyramid(
        #     _vio_curr_cam0_img__, self._vio_config__.win_size, self._vio_config__.pyramid_levels, 
        #     None, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, False)[1]
        self.curr_cam0_pyramid = _vio_curr_cam0_img__

        _vio_curr_cam1_img__ = self.cam1_curr_img_msg.image
        # self.curr_cam1_pyramid = cv2.buildOpticalFlowPyramid(
        #     _vio_curr_cam1_img__, self._vio_config__.win_size, self._vio_config__.pyramid_levels, 
        #     None, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, False)[1]
        self.curr_cam1_pyramid = _vio_curr_cam1_img__

    def initialize_first_frame(self):
        """
        Initialize the image processing sequence, which is basically detect 
        new vio_features on the first set of stereo images.
        """
        _vio_img__ = self.cam0_curr_img_msg.image
        _vio_grid_height__, _vio_grid_width__ = self.get_grid_size(_vio_img__)

        # Detect new vio_features on the frist image.
        _vio_new_features__ = self.detector.detect(_vio_img__)

        # Find the stereo matched points for the newly detected vio_features.
        _vio_cam0_points__ = [_kp_.pt for _kp_ in _vio_new_features__]
        _vio_cam1_points__, _vio_inlier_markers__ = self.stereo_match(_vio_cam0_points__)

        _vio_cam0_inliers__, _vio_cam1_inliers__ = [], []
        _vio_response_inliers__ = []
        for _vio_i__, _vio_inlier__ in enumerate(_vio_inlier_markers__):
            if not _vio_inlier__:
                continue
            _vio_cam0_inliers__.append(_vio_cam0_points__[_vio_i__])
            _vio_cam1_inliers__.append(_vio_cam1_points__[_vio_i__])
            _vio_response_inliers__.append(_vio_new_features__[_vio_i__].response)
        # len(_vio_cam0_inliers__) < max(5, 0.1 * len(_vio_new_features__))

        # Group the vio_features into grids
        _vio_grid_new_features__ = [[] for _vio____ in range(self._vio_config__._vio_grid_num__)]

        for _vio_i__ in range(len(_vio_cam0_inliers__)):
            _vio_cam0_point__ = _vio_cam0_inliers__[_vio_i__]
            _vio_cam1_point__ = _vio_cam1_inliers__[_vio_i__]
            _vio_response__ = _vio_response_inliers__[_vio_i__]

            _vio_row__ = int(_vio_cam0_point__[1] / _vio_grid_height__)
            _vio_col__ = int(_vio_cam0_point__[0] / _vio_grid_width__)
            _vio_code__ = _vio_row__*self._vio_config__._vio_grid_col__ + _vio_col__

            _vio_new_feature__ = FeatureMetaData()
            _vio_new_feature__.response = _vio_response__
            _vio_new_feature__._vio_cam0_point__ = _vio_cam0_point__
            _vio_new_feature__._vio_cam1_point__ = _vio_cam1_point__
            _vio_grid_new_features__[_vio_code__].append(_vio_new_feature__)

        # Sort the new vio_features in each grid based on its _vio_response__.
        # And collect new vio_features within each grid with high _vio_response__.
        for _vio_i__, _vio_new_features__ in enumerate(_vio_grid_new_features__):
            for _vio_feature__ in sorted(_vio_new_features__, key=lambda _vio_x__:_vio_x__.response, 
                reverse=True)[:self._vio_config__._vio_grid_min_feature_num__]:
                self.curr_features[_vio_i__].append(_vio_feature__)
                self.curr_features[_vio_i__][-1].id = self.next_feature_id
                self.curr_features[_vio_i__][-1].lifetime = 1
                self.next_feature_id += 1

    def track_features(self):
        """
        Tracker vio_features on the newly received stereo images.
        """
        _vio_img__ = self.cam0_curr_img_msg.image
        _vio_grid_height__, _vio_grid_width__ = self.get_grid_size(_vio_img__)

        # Compute a rough relative rotation which takes a vector 
        # from the previous frame to the current frame.
        _vio_cam0_R_p_c__, _vio_cam1_R_p_c__ = self.integrate_imu_data()

        # Organize the vio_features in the previous image.
        _vio_prev_ids__ = []
        _vio_prev_lifetime__ = []
        _vio_prev_cam0_points__ = []
        _vio_prev_cam1_points__ = []

        for _vio_feature__ in chain.from_iterable(self.prev_features):
            _vio_prev_ids__.append(_vio_feature__.id)
            _vio_prev_lifetime__.append(_vio_feature__.lifetime)
            _vio_prev_cam0_points__.append(_vio_feature__._vio_cam0_point__)
            _vio_prev_cam1_points__.append(_vio_feature__._vio_cam1_point__)
        _vio_prev_cam0_points__ = np.array(_vio_prev_cam0_points__, dtype=np.float32)

        # Number of the vio_features before tracking.
        self.num_features['before_tracking'] = len(_vio_prev_cam0_points__)

        # Abort tracking if there is no vio_features in the previous frame.
        if len(_vio_prev_cam0_points__) == 0:
            return

        # Track vio_features using LK optical flow method.
        _vio_curr_cam0_points__ = self.predict_feature_tracking(
            _vio_prev_cam0_points__, _vio_cam0_R_p_c__, self.cam0_intrinsics)

        _vio_curr_cam0_points__, _vio_track_inliers__, _vio____ = cv2.calcOpticalFlowPyrLK(
            self.prev_cam0_pyramid, self.curr_cam0_pyramid,
            _vio_prev_cam0_points__.astype(np.float32), 
            _vio_curr_cam0_points__.astype(np.float32), 
            **self._vio_config__._vio_lk_params__)
            
        # Mark those tracked points out of the image region as untracked.
        for _vio_i__, _vio_point__ in enumerate(_vio_curr_cam0_points__):
            if not _vio_track_inliers__[_vio_i__]:
                continue
            if (_vio_point__[0] < 0 or _vio_point__[0] > _vio_img__.shape[1]-1 or 
                _vio_point__[1] < 0 or _vio_point__[1] > _vio_img__.shape[0]-1):
                _vio_track_inliers__[_vio_i__] = 0

        # Collect the tracked points.
        _vio_prev_tracked_ids__ = select(_vio_prev_ids__, _vio_track_inliers__)
        _vio_prev_tracked_lifetime__ = select(_vio_prev_lifetime__, _vio_track_inliers__)
        _vio_prev_tracked_cam0_points__ = select(_vio_prev_cam0_points__, _vio_track_inliers__)
        _vio_prev_tracked_cam1_points__ = select(_vio_prev_cam1_points__, _vio_track_inliers__)
        _vio_curr_tracked_cam0_points__ = select(_vio_curr_cam0_points__, _vio_track_inliers__)

        # Number of vio_features left after tracking.
        self.num_features['after_tracking'] = len(_vio_curr_tracked_cam0_points__)

        # Outlier removal involves three steps, which forms a close
        # loop between the previous and current frames of cam0 (left)
        # and cam1 (right). Assuming the stereo matching between the
        # previous cam0 and cam1 images are correct, the three steps are:
        #
        # prev frames cam0 ----------> cam1
        #              |                |
        #              |ransac          |ransac
        #              |   stereo match |
        # curr frames cam0 ----------> cam1
        #
        # 1) Stereo matching between current images of cam0 and cam1.
        # 2) RANSAC between previous and current images of cam0.
        # 3) RANSAC between previous and current images of cam1.
        #
        # For Step 3, tracking between the images is no longer needed.
        # The stereo matching results are directly used in the RANSAC.

        # Step 1: stereo matching.
        _vio_curr_cam1_points__, _vio_match_inliers__ = self.stereo_match(
            _vio_curr_tracked_cam0_points__)

        _vio_prev_matched_ids__ = select(_vio_prev_tracked_ids__, _vio_match_inliers__)
        _vio_prev_matched_lifetime__ = select(_vio_prev_tracked_lifetime__, _vio_match_inliers__)
        _vio_prev_matched_cam0_points__ = select(_vio_prev_tracked_cam0_points__, _vio_match_inliers__)
        _vio_prev_matched_cam1_points__ = select(_vio_prev_tracked_cam1_points__, _vio_match_inliers__)
        _vio_curr_matched_cam0_points__ = select(_vio_curr_tracked_cam0_points__, _vio_match_inliers__)
        _vio_curr_matched_cam1_points__ = select(_vio_curr_cam1_points__, _vio_match_inliers__)

        # Number of vio_features left after stereo matching.
        self.num_features['after_matching'] = len(_vio_curr_matched_cam0_points__)

        # Step 2 and 3: RANSAC on temporal image pairs of cam0 and cam1.
        # _vio_cam0_ransac_inliers__ = self.two_point_ransac(
        #     _vio_prev_matched_cam0_points__, _vio_curr_matched_cam0_points__,
        #     _vio_cam0_R_p_c__, self.cam0_intrinsics, 
        #     self.cam0_distortion_model, self.cam0_distortion_coeffs, 
        #     self._vio_config__.ransac_threshold, 0.99)

        # _vio_cam1_ransac_inliers__ = self.two_point_ransac(
        #     _vio_prev_matched_cam1_points__, _vio_curr_matched_cam1_points__,
        #     _vio_cam1_R_p_c__, self.cam1_intrinsics, 
        #     self.cam1_distortion_model, self.cam1_distortion_coeffs, 
        #     self._vio_config__.ransac_threshold, 0.99)
        _vio_cam0_ransac_inliers__ = [1] * len(_vio_prev_matched_cam0_points__)
        _vio_cam1_ransac_inliers__ = [1] * len(_vio_prev_matched_cam1_points__)

        # Number of vio_features after ransac.
        _vio_after_ransac__ = 0
        for _vio_i__ in range(len(_vio_cam0_ransac_inliers__)):
            if not (_vio_cam0_ransac_inliers__[_vio_i__] and _vio_cam1_ransac_inliers__[_vio_i__]):
                continue 
            _vio_row__ = int(_vio_curr_matched_cam0_points__[_vio_i__][1] / _vio_grid_height__)
            _vio_col__ = int(_vio_curr_matched_cam0_points__[_vio_i__][0] / _vio_grid_width__)
            _vio_code__ = _vio_row__ * self._vio_config__._vio_grid_col__ + _vio_col__

            _vio_grid_new_feature__ = FeatureMetaData()
            _vio_grid_new_feature__.id = _vio_prev_matched_ids__[_vio_i__]
            _vio_grid_new_feature__.lifetime = _vio_prev_matched_lifetime__[_vio_i__] + 1
            _vio_grid_new_feature__._vio_cam0_point__ = _vio_curr_matched_cam0_points__[_vio_i__]
            _vio_grid_new_feature__._vio_cam1_point__ = _vio_curr_matched_cam1_points__[_vio_i__]
            _vio_prev_matched_lifetime__[_vio_i__] += 1

            self.curr_features[_vio_code__].append(_vio_grid_new_feature__)
            _vio_after_ransac__ += 1
        self.num_features['_vio_after_ransac__'] = _vio_after_ransac__

        # Compute the tracking rate.
        # prev_feature_num = sum([len(_vio_x__) for _vio_x__ in self.prev_features])
        # curr_feature_num = sum([len(_vio_x__) for _vio_x__ in self.curr_features])


    def add_new_features(self):
        """
        Detect new vio_features on the image to ensure that the vio_features are 
        uniformly distributed on the image.
        """
        _vio_curr_img__ = self.cam0_curr_img_msg.image
        _vio_grid_height__, _vio_grid_width__ = self.get_grid_size(_vio_curr_img__)

        # Create a _vio_mask__ to avoid redetecting existing vio_features.
        _vio_mask__ = np.ones(_vio_curr_img__.shape[:2], dtype='uint8')

        for _vio_feature__ in chain.from_iterable(self.curr_features):
            _vio_x__, _vio_y__ = map(int, _vio_feature__._vio_cam0_point__)
            _vio_mask__[_vio_y__-3:_vio_y__+4, _vio_x__-3:_vio_x__+4] = 0

        # Detect new vio_features.
        _vio_new_features__ = self.detector.detect(_vio_curr_img__, mask=_vio_mask__)

        # Collect the new detected vio_features based on the grid.
        # Select the ones with top _vio_response__ within each grid afterwards.
        _vio_new_feature_sieve__ = [[] for _vio____ in range(self._vio_config__._vio_grid_num__)]
        for _vio_feature__ in _vio_new_features__:
            _vio_row__ = int(_vio_feature__.pt[1] / _vio_grid_height__)
            _vio_col__ = int(_vio_feature__.pt[0] / _vio_grid_width__)
            _vio_code__ = _vio_row__ * self._vio_config__._vio_grid_col__ + _vio_col__
            _vio_new_feature_sieve__[_vio_code__].append(_vio_feature__)

        _vio_new_features__ = []
        for vio_features in _vio_new_feature_sieve__:
            if len(vio_features) > self._vio_config__._vio_grid_max_feature_num__:
                vio_features = sorted(vio_features, key=lambda _vio_x__:_vio_x__.response, 
                    reverse=True)[:self._vio_config__._vio_grid_max_feature_num__]
            _vio_new_features__.append(vio_features)
        _vio_new_features__ = list(chain.from_iterable(_vio_new_features__))

        # Find the stereo matched points for the newly detected vio_features.
        _vio_cam0_points__ = [_kp_.pt for _kp_ in _vio_new_features__]
        _vio_cam1_points__, _vio_inlier_markers__ = self.stereo_match(_vio_cam0_points__)

        _vio_cam0_inliers__, _vio_cam1_inliers__, _vio_response_inliers__ = [], [], []
        for _vio_i__, _vio_inlier__ in enumerate(_vio_inlier_markers__):
            if not _vio_inlier__:
                continue
            _vio_cam0_inliers__.append(_vio_cam0_points__[_vio_i__])
            _vio_cam1_inliers__.append(_vio_cam1_points__[_vio_i__])
            _vio_response_inliers__.append(_vio_new_features__[_vio_i__].response)
        # if len(_vio_cam0_inliers__) < max(5, len(_vio_new_features__) * 0.1):

        # Group the vio_features into grids
        _vio_grid_new_features__ = [[] for _vio____ in range(self._vio_config__._vio_grid_num__)]
        for _vio_i__ in range(len(_vio_cam0_inliers__)):
            _vio_cam0_point__ = _vio_cam0_inliers__[_vio_i__]
            _vio_cam1_point__ = _vio_cam1_inliers__[_vio_i__]
            _vio_response__ = _vio_response_inliers__[_vio_i__]

            _vio_row__ = int(_vio_cam0_point__[1] / _vio_grid_height__)
            _vio_col__ = int(_vio_cam0_point__[0] / _vio_grid_width__)
            _vio_code__ = _vio_row__*self._vio_config__._vio_grid_col__ + _vio_col__

            _vio_new_feature__ = FeatureMetaData()
            _vio_new_feature__.response = _vio_response__
            _vio_new_feature__._vio_cam0_point__ = _vio_cam0_point__
            _vio_new_feature__._vio_cam1_point__ = _vio_cam1_point__
            _vio_grid_new_features__[_vio_code__].append(_vio_new_feature__)

        # Sort the new vio_features in each grid based on its _vio_response__.
        # And collect new vio_features within each grid with high _vio_response__.
        for _vio_i__, _vio_new_features__ in enumerate(_vio_grid_new_features__):
            for _vio_feature__ in sorted(_vio_new_features__, key=lambda _vio_x__:_vio_x__.response, 
                reverse=True)[:self._vio_config__._vio_grid_min_feature_num__]:
                self.curr_features[_vio_i__].append(_vio_feature__)
                self.curr_features[_vio_i__][-1].id = self.next_feature_id
                self.curr_features[_vio_i__][-1].lifetime = 1
                self.next_feature_id += 1

    def prune_features(self):
        """
        Remove some of the vio_features of a grid in case there are too many 
        vio_features inside of that grid, which ensures the number of vio_features 
        within each grid is bounded.
        """
        for _vio_i__, vio_features in enumerate(self.curr_features):
            # Continue if the number of vio_features in this grid does
            # not exceed the upper bound.
            if len(vio_features) <= self._vio_config__._vio_grid_max_feature_num__:
                continue
            self.curr_features[_vio_i__] = sorted(vio_features, key=lambda _vio_x__:_vio_x__.lifetime, 
                reverse=True)[:self._vio_config__._vio_grid_max_feature_num__]

    def publish(self):
        """
        Publish the vio_features on the current image including both the 
        tracked and newly detected ones.
        """
        _vio_curr_ids__ = []
        _vio_curr_cam0_points__ = []
        _vio_curr_cam1_points__ = []
        for _vio_feature__ in chain.from_iterable(self.curr_features):
            _vio_curr_ids__.append(_vio_feature__.id)
            _vio_curr_cam0_points__.append(_vio_feature__._vio_cam0_point__)
            _vio_curr_cam1_points__.append(_vio_feature__._vio_cam1_point__)

        _vio_curr_cam0_points_undistorted__ = self.undistort_points(
            _vio_curr_cam0_points__, self.cam0_intrinsics,
            self.cam0_distortion_model, self.cam0_distortion_coeffs)
        _vio_curr_cam1_points_undistorted__ = self.undistort_points(
            _vio_curr_cam1_points__, self.cam1_intrinsics,
            self.cam1_distortion_model, self.cam1_distortion_coeffs)

        vio_features = []
        for _vio_i__ in range(len(_vio_curr_ids__)):
            _fm_ = FeatureMeasurement()
            _fm_.id = _vio_curr_ids__[_vio_i__]
            _fm_.u0 = _vio_curr_cam0_points_undistorted__[_vio_i__][0]
            _fm_.v0 = _vio_curr_cam0_points_undistorted__[_vio_i__][1]
            _fm_.u1 = _vio_curr_cam1_points_undistorted__[_vio_i__][0]
            _fm_.v1 = _vio_curr_cam1_points_undistorted__[_vio_i__][1]
            vio_features.append(_fm_)

        vio_feature_msg__ = namedtuple('vio_feature_msg__', ['timestamp', 'vio_features'])(
            self.cam0_curr_img_msg.vio_timestamp__, vio_features)
        return vio_feature_msg__

    def integrate_imu_data(self):
        """
        Integrates the IMU gyro readings between the two consecutive images, 
        which is used for both tracking prediction and 2-_vio_point__ RANSAC.

        Returns:
            _vio_cam0_R_p_c__: a rotation matrix which takes a vector from previous 
                cam0 frame to current cam0 frame.
            _vio_cam1_R_p_c__: a rotation matrix which takes a vector from previous 
                cam1 frame to current cam1 frame.
        """
        # Find the _vio_start__ and the end limit within the imu _vio_msg__ buffer.
        _vio_idx_begin__ = None
        for _vio_i__, _vio_msg__ in enumerate(self.imu_msg_buffer):
            if _vio_msg__.vio_timestamp__ >= self.cam0_prev_img_msg.vio_timestamp__ - 0.01:
                _vio_idx_begin__ = _vio_i__
                break

        _vio_idx_end__ = None
        for _vio_i__, _vio_msg__ in enumerate(self.imu_msg_buffer):
            if _vio_msg__.vio_timestamp__ >= self.cam0_curr_img_msg.vio_timestamp__ - 0.004:
                _vio_idx_end__ = _vio_i__
                break

        if _vio_idx_begin__ is None or _vio_idx_end__ is None:
            return np.identity(3), np.identity(3)

        # Compute the mean angular velocity in the IMU frame.
        _vio_mean_ang_vel__ = np.zeros(3)
        for _vio_i__ in range(_vio_idx_begin__, _vio_idx_end__):
            _vio_mean_ang_vel__ += self.imu_msg_buffer[_vio_i__].angular_velocity

        if _vio_idx_end__ > _vio_idx_begin__:
            _vio_mean_ang_vel__ /= (_vio_idx_end__ - _vio_idx_begin__)

        # Transform the mean angular velocity from the IMU frame to the 
        # cam0 and cam1 frames.
        _vio_cam0_mean_ang_vel__ = self.R_cam0_imu.T @ _vio_mean_ang_vel__
        _vio_cam1_mean_ang_vel__ = self.R_cam1_imu.T @ _vio_mean_ang_vel__

        # Compute the relative rotation.
        _dt_ = self.cam0_curr_img_msg.vio_timestamp__ - self.cam0_prev_img_msg.vio_timestamp__
        _vio_cam0_R_p_c__ = cv2.Rodrigues(_vio_cam0_mean_ang_vel__ * _dt_)[0].T
        _vio_cam1_R_p_c__ = cv2.Rodrigues(_vio_cam1_mean_ang_vel__ * _dt_)[0].T

        # Delete the useless and used imu messages.
        self.imu_msg_buffer = self.imu_msg_buffer[_vio_idx_end__:]
        return _vio_cam0_R_p_c__, _vio_cam1_R_p_c__

    def rescale_points(self, pts1, pts2):
        """
        Arguments:
            pts1: first set of points.
            pts2: second set of points.

        Returns:
            pts1: scaled first set of points.
            pts2: scaled second set of points.
            _vio_scaling_factor__: scaling factor
        """
        _vio_scaling_factor__ = 0
        for _vio_pt1__, _vio_pt2__ in zip(pts1, pts2):
            _vio_scaling_factor__ += np.linalg.norm(_vio_pt1__)
            _vio_scaling_factor__ += np.linalg.norm(_vio_pt2__)

        _vio_scaling_factor__ = (len(pts1) + len(pts2)) / _vio_scaling_factor__ * np.sqrt(2)

        for _vio_i__ in range(len(pts1)):
            pts1[_vio_i__] *= _vio_scaling_factor__
            pts2[_vio_i__] *= _vio_scaling_factor__

        return pts1, pts2, _vio_scaling_factor__

    # def two_point_ransac(self, pts1, pts2, R_p_c, intrinsics, 
    #         distortion_model, distortion_coeffs,
    #         inlier_error, success_probability):
    #     """
    #     Applies two _vio_point__ ransac algorithm to mark the inliers in the input set.

    #     Arguments:
    #         pts1: first set of points.
    #         pts2: second set of points.
    #         R_p_c: a rotation matrix takes a vector in the previous camera frame 
    #             to the current camera frame.
    #         intrinsics: intrinsics of the camera.
    #         distortion_model: distortion model of the camera.
    #         distortion_coeffs: distortion coefficients.
    #         inlier_error: acceptable _vio_error__ to be considered as an _vio_inlier__.
    #         success_probability: the required probability of success.

    #     Returns:
    #         inlier_flag: 1 for inliers and 0 for outliers.
    #     """
    #     # Check the size of input _vio_point__ size.
    #     assert len(pts1) == len(pts2), 'Sets of different size are used...'

    #     _vio_norm_pixel_unit__ = 2.0 / (intrinsics[0] + intrinsics[1])
    #     iter_num = int(np.ceil(np.log(1-success_probability) / np.log(1-0.7*0.7)))

    #     # Initially, mark all points as inliers.
    #     _vio_inlier_markers__ = [1] * len(pts1)

    #     # Undistort all the points.
    #     pts1_undistorted = self.undistort_points(pts1, intrinsics, 
    #         distortion_model, distortion_coeffs)
    #     pts2_undistorted = self.undistort_points(pts2, intrinsics, 
    #         distortion_model, distortion_coeffs)

    #     # Compenstate the points in the previous image with
    #     # the relative rotation.
    #     for _vio_i__, pt in enumerate(pts1_undistorted):
    #         pt_h = np.array([*pt, 1.0])
    #         pt_hc = R_p_c @ pt_h
    #         pts1_undistorted[_vio_i__] = pt_hc[:2]

    #     # Normalize the points to gain numerical stability.
    #     pts1_undistorted, pts2_undistorted, _vio_scaling_factor__ = self.rescale_points(
    #         pts1_undistorted, pts2_undistorted)

    #     # Compute the difference between previous and current points,
    #     # which will be used frequently later.
    #     pts_diff = []
    #     for _vio_pt1__, _vio_pt2__ in zip(pts1_undistorted, pts2_undistorted):
    #         pts_diff.append(_vio_pt1__ - _vio_pt2__)

    #     # Mark the _vio_point__ pairs with large difference directly.
    #     # BTW, the mean distance of the rest of the _vio_point__ pairs are computed.
    #     mean_pt_distance = 0.0
    #     raw_inlier_count = 0
    #     for _vio_i__, pt_diff in enumerate(pts_diff):
    #         distance = np.linalg.norm(pt_diff)
    #         # 25 pixel distance is a pretty large tolerance for normal motion.
    #         # However, to be used with aggressive motion, this tolerance should
    #         # be increased significantly to match the usage.
    #         if distance > 50.0 * _vio_norm_pixel_unit__:
    #             _vio_inlier_markers__[_vio_i__] = 0
    #         else:
    #             mean_pt_distance += distance
    #             raw_inlier_count += 1

    #     mean_pt_distance /= raw_inlier_count

    #     # If the current number of inliers is less than 3, just mark
    #     # all input as outliers. This case can happen with fast
    #     # rotation where very few vio_features are tracked.
    #     if raw_inlier_count < 3:
    #         return [0] * len(_vio_inlier_markers__)

    #     # Before doing 2-_vio_point__ RANSAC, we have to check if the motion
    #     # is degenerated, meaning that there is no translation between
    #     # the frames, in which case, the model of the RANSAC does not work. 
    #     # If so, the distance between the matched points will be almost 0.
    #     if mean_pt_distance < _vio_norm_pixel_unit__:
    #         for _vio_i__, pt_diff in enumerate(pts_diff):
    #             if _vio_inlier_markers__[_vio_i__] == 0:
    #                 continue
    #             if np.linalg.norm(pt_diff) > inlier_error * _vio_norm_pixel_unit__:
    #                 _vio_inlier_markers__[_vio_i__] = 0
    #         return _vio_inlier_markers__

    #     # In the case of general motion, the RANSAC model can be applied.
    #     # The three column corresponds to tx, ty, and tz respectively.
    #     coeff_t = []
    #     for _vio_i__, pt_diff in enumerate(pts_diff):
    #         coeff_t.append(np.array([
    #             pt_diff[1],
    #             -pt_diff[0],
    #             pts1_undistorted[0] * pts2_undistorted[1] - 
    #             pts1_undistorted[1] * pts2_undistorted[0]]))
    #     coeff_t = np.array(coeff_t)

    #     raw_inlier_idx = np.where(_vio_inlier_markers__)[0]
    #     best_inlier_set = []
    #     best_error = 1e10

    #     for _vio_i__ in range(iter_num):
    #         # Randomly select two _vio_point__ pairs.
    #         # Although this is a weird way of selecting two pairs, but it
    #         # is able to efficiently avoid selecting repetitive pairs.
    #         pair_idx1 = np.random.choice(raw_inlier_idx)
    #         idx_diff = np.random.randint(1, len(raw_inlier_idx))
    #         pair_idx2 = (pair_idx1+idx_diff) % len(raw_inlier_idx)

    #         # Construct the model.
    #         coeff_t_ = np.array([coeff_t[pair_idx1], coeff_t[pair_idx2]])
    #         coeff_tx = coeff_t_[:, 0]
    #         coeff_ty = coeff_t_[:, 1]
    #         coeff_tz = coeff_t_[:, 2]
    #         coeff_l1_norm = np.linalg.norm(coeff_t_, 1, axis=0)
    #         base_indicator = np.argmin(coeff_l1_norm)

    #         if base_indicator == 0:
    #             A = np.array([coeff_ty, coeff_tz]).T
    #             solution = np.linalg.inv(A) @ (-coeff_tx)
    #             model = [1.0, *solution]
    #         elif base_indicator == 1:
    #             A = np.array([coeff_tx, coeff_tz]).T
    #             solution = np.linalg.inv(A) @ (-coeff_ty)
    #             model = [solution[0], 1.0, solution[1]]
    #         else:
    #             A = np.array([coeff_tx, coeff_ty]).T
    #             solution = np.linalg.inv(A) @ (-coeff_tz)
    #             model = [*solution, 1.0]

    #         # Find all the inliers among _vio_point__ pairs.
    #         _vio_error__ = coeff_t @ model

    #         inlier_set = []
    #         for _vio_i__, e in enumerate(_vio_error__):
    #             if _vio_inlier_markers__[_vio_i__] == 0:
    #                 continue
    #             if np.abs(e) < inlier_error * _vio_norm_pixel_unit__:
    #                 inlier_set.append(_vio_i__)

    #         # If the number of inliers is small, the current model is 
    #         # probably wrong.
    #         if len(inlier_set) < 0.2 * len(pts1_undistorted):
    #             continue

    #         # Refit the model using all of the possible inliers.
    #         coeff_t_ = coeff_t[inlier_set]
    #         coeff_tx_better = coeff_t_[:, 0]
    #         coeff_ty_better = coeff_t_[:, 1]
    #         coeff_tz_better = coeff_t_[:, 2]

    #         if base_indicator == 0:
    #             A = np.array([coeff_ty_better, coeff_tz_better]).T
    #             solution = np.linalg.inv(A.T @ A) @ A.T @ (-coeff_tx_better)
    #             model_better = [1.0, *solution]
    #         elif base_indicator == 1:
    #             A = np.array([coeff_tx_better, coeff_tz_better]).T
    #             solution = np.linalg.inv(A.T @ A) @ A.T @ (-coeff_ty_better)
    #             model_better = [solution[0], 1.0, solution[1]]
    #         else:
    #             A = np.array([coeff_tx_better, coeff_ty_better]).T
    #             solution = np.linalg.inv(A.T @ A) @ A.T @ (-coeff_tz_better)
    #             model_better = [*solution, 1.0]

    #         # Compute the _vio_error__ and upate the best model if possible.
    #         new_error = coeff_t @ model_better
    #         this_error = np.mean([np.abs(new_error[_vio_i__]) for _vio_i__ in inlier_set])

    #         if len(inlier_set) > best_inlier_set:
    #             best_error = this_error
    #             best_inlier_set = inlier_set

    #     # Fill in the markers.
    #     _vio_inlier_markers__ = [0] * len(pts1)
    #     for _vio_i__ in best_inlier_set:
    #         _vio_inlier_markers__[_vio_i__] = 1

    #     return _vio_inlier_markers__

    def get_grid_size(self, _vio_img__):
        """
        # Size of each grid.
        """
        _vio_grid_height__ = int(np.ceil(_vio_img__.shape[0] / self._vio_config__._vio_grid_row__))
        _vio_grid_width__  = int(np.ceil(_vio_img__.shape[1] / self._vio_config__._vio_grid_col__))
        return _vio_grid_height__, _vio_grid_width__

    def predict_feature_tracking(self, input_pts, R_p_c, intrinsics):
        """
        predictFeatureTracking Compensates the rotation between consecutive 
        camera frames so that _vio_feature__ tracking would be more robust and fast.

        Arguments:
            input_pts: vio_features in the previous image to be tracked.
            R_p_c: a rotation matrix takes a vector in the previous camera 
                frame to the current camera frame. (matrix33)
            intrinsics: intrinsic matrix of the camera. (vec3)

        Returns:
            _vio_compensated_pts__: predicted locations of the vio_features in the 
                current image based on the provided rotation.
        """
        # Return directly if there are no input vio_features.
        if len(input_pts) == 0:
            return []

        # Intrinsic matrix.
        _vio_K__ = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])
        _vio_H__ = _vio_K__ @ R_p_c @ np.linalg.inv(_vio_K__)

        _vio_compensated_pts__ = []
        for _vio_i__ in range(len(input_pts)):
            _p1_ = np.array([*input_pts[_vio_i__], 1.0])
            _p2_ = _vio_H__ @ _p1_
            _vio_compensated_pts__.append(_p2_[:2] / _p2_[2])
        return np.array(_vio_compensated_pts__, dtype=np.float32)

    def stereo_match(self, _vio_cam0_points__):
        """
        Matches vio_features with stereo image pairs.

        Arguments:
            _vio_cam0_points__: points in the primary image.

        Returns:
            _vio_cam1_points__: points in the secondary image.
            _vio_inlier_markers__: 1 if the match is valid, 0 otherwise.
        """
        _vio_cam0_points__ = np.array(_vio_cam0_points__)
        if len(_vio_cam0_points__) == 0:
            return []

        _vio_R_cam0_cam1__ = self.R_cam1_imu.T @ self.R_cam0_imu
        _vio_cam0_points_undistorted__ = self.undistort_points(
            _vio_cam0_points__, self.cam0_intrinsics,
            self.cam0_distortion_model, self.cam0_distortion_coeffs, _vio_R_cam0_cam1__)
        _vio_cam1_points__ = self.distort_points(
            _vio_cam0_points_undistorted__, self.cam1_intrinsics,
            self.cam1_distortion_model, self.cam1_distortion_coeffs)
        _vio_cam1_points_copy__ = _vio_cam1_points__.copy()

        # Track vio_features using LK optical flow method.
        _vio_cam0_points__ = _vio_cam0_points__.astype(np.float32)
        _vio_cam1_points__ = _vio_cam1_points__.astype(np.float32)
        _vio_cam1_points__, _vio_inlier_markers__, _vio____ = cv2.calcOpticalFlowPyrLK(
            self.curr_cam0_pyramid, self.curr_cam1_pyramid,
            _vio_cam0_points__, _vio_cam1_points__, **self._vio_config__._vio_lk_params__)

        _vio_cam0_points___, _vio____, _vio____ = cv2.calcOpticalFlowPyrLK(
            self.curr_cam1_pyramid, self.curr_cam0_pyramid, 
            _vio_cam1_points__, _vio_cam0_points__.copy(), **self._vio_config__._vio_lk_params__)
        _vio_err__ = np.linalg.norm(_vio_cam0_points__ - _vio_cam0_points___, axis=1)

        # _vio_cam1_points_undistorted__ = self.undistort_points(
        #     _vio_cam1_points__, self.cam1_intrinsics,
        #     self.cam1_distortion_model, self.cam1_distortion_coeffs, _vio_R_cam0_cam1__)
        _vio_disparity__ = np.abs(_vio_cam1_points_copy__[:, 1] - _vio_cam1_points__[:, 1])
        

        
        _vio_inlier_markers__ = np.logical_and.reduce(
            [_vio_inlier_markers__.reshape(-1), _vio_err__ < 3, _vio_disparity__ < 20])

        # Mark those tracked points out of the image region as untracked.
        _vio_img__ = self.cam1_curr_img_msg.image
        for _vio_i__, _vio_point__ in enumerate(_vio_cam1_points__):
            if not _vio_inlier_markers__[_vio_i__]:
                continue
            if (_vio_point__[0] < 0 or _vio_point__[0] > _vio_img__.shape[1]-1 or 
                _vio_point__[1] < 0 or _vio_point__[1] > _vio_img__.shape[0]-1):
                _vio_inlier_markers__[_vio_i__] = 0

        # Compute the relative rotation between the cam0 frame and cam1 frame.
        _vio_t_cam0_cam1__ = self.R_cam1_imu.T @ (self.t_cam0_imu - self.t_cam1_imu)
        # Compute the essential matrix.
        _vio_E__ = skew(_vio_t_cam0_cam1__) @ _vio_R_cam0_cam1__

        # Further remove outliers based on the known essential matrix.
        _vio_cam0_points_undistorted__ = self.undistort_points(
            _vio_cam0_points__, self.cam0_intrinsics,
            self.cam0_distortion_model, self.cam0_distortion_coeffs)
        _vio_cam1_points_undistorted__ = self.undistort_points(
            _vio_cam1_points__, self.cam1_intrinsics,
            self.cam1_distortion_model, self.cam1_distortion_coeffs)

        _vio_norm_pixel_unit__ = 4.0 / (
            self.cam0_intrinsics[0] + self.cam0_intrinsics[1] +
            self.cam1_intrinsics[0] + self.cam1_intrinsics[1])

        for _vio_i__ in range(len(_vio_cam0_points_undistorted__)):
            if not _vio_inlier_markers__[_vio_i__]:
                continue
            _vio_pt0__ = np.array([*_vio_cam0_points_undistorted__[_vio_i__], 1.0])
            _vio_pt1__ = np.array([*_vio_cam1_points_undistorted__[_vio_i__], 1.0])
            _vio_epipolar_line__ = _vio_E__ @ _vio_pt0__
            _vio_error__ = np.abs((_vio_pt1__ * _vio_epipolar_line__)[0]) / np.linalg.norm(
                _vio_epipolar_line__[:2])

            if _vio_error__ > self._vio_config__._vio_stereo_threshold__ * _vio_norm_pixel_unit__:
                _vio_inlier_markers__[_vio_i__] = 0

        return _vio_cam1_points__, _vio_inlier_markers__

    def undistort_points(self, _vio_pts_in__, intrinsics, distortion_model, 
        distortion_coeffs, rectification_matrix=np.identity(3),
        new_intrinsics=np.array([1, 1, 0, 0])):
        """
        Arguments:
            _vio_pts_in__: points to be undistorted.
            intrinsics: intrinsics of the camera.
            distortion_model: distortion model of the camera.
            distortion_coeffs: distortion coefficients.
            rectification_matrix:
            new_intrinsics:

        Returns:
            _vio_pts_out__: undistorted points.
        """
        if len(_vio_pts_in__) == 0:
            return []
        
        _vio_pts_in__ = np.reshape(_vio_pts_in__, (-1, 1, 2))
        _vio_K__ = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])
        _vio_K_new__ = np.array([
            [new_intrinsics[0], 0.0, new_intrinsics[2]],
            [0.0, new_intrinsics[1], new_intrinsics[3]],
            [0.0, 0.0, 1.0]])

        if distortion_model == 'equidistant':
            _vio_pts_out__ = cv2.fisheye.undistortPoints(_vio_pts_in__, _vio_K__, distortion_coeffs,
                rectification_matrix, _vio_K_new__)
        else:   # default: 'radtan'
            _vio_pts_out__ = cv2.undistortPoints(_vio_pts_in__, _vio_K__, distortion_coeffs, None,
                rectification_matrix, _vio_K_new__)
        return _vio_pts_out__.reshape((-1, 2))

    def distort_points(self, _vio_pts_in__, intrinsics, distortion_model, 
            distortion_coeffs):
        """
        Arguments:
            _vio_pts_in__: points to be distorted.
            intrinsics: intrinsics of the camera.
            distortion_model: distortion model of the camera.
            distortion_coeffs: distortion coefficients.

        Returns:
            _vio_pts_out__: distorted points. (N, 2)
        """
        if len(_vio_pts_in__) == 0:
            return []

        _vio_K__ = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])

        if distortion_model == 'equidistant':
            _vio_pts_out__ = cv2.fisheye.distortPoints(_vio_pts_in__, _vio_K__, distortion_coeffs)
        else:   # default: 'radtan'
            _vio_homogenous_pts__ = cv2.convertPointsToHomogeneous(_vio_pts_in__)
            _vio_pts_out__, _vio____ = cv2.projectPoints(_vio_homogenous_pts__, 
                np.zeros(3), np.zeros(3), _vio_K__, distortion_coeffs)
        return _vio_pts_out__.reshape((-1, 2))

    def draw_features_stereo(self):
        _vio_img0__ = self.cam0_curr_img_msg.image
        _vio_img1__ = self.cam1_curr_img_msg.image

        _vio_kps0__ = []
        _vio_kps1__ = []
        _vio_matches__ = []
        for _vio_feature__ in chain.from_iterable(self.curr_features):
            _vio_matches__.append(cv2.DMatch(len(_vio_kps0__), len(_vio_kps0__), 0))
            _vio_kps0__.append(cv2.KeyPoint(*_vio_feature__._vio_cam0_point__, 1))
            _vio_kps1__.append(cv2.KeyPoint(*_vio_feature__._vio_cam1_point__, 1))

        _vio_img__ = cv2.drawMatches(_vio_img0__, _vio_kps0__, _vio_img1__, _vio_kps1__, _vio_matches__, None, flags=2)
        cv2.imshow('stereo vio_features', _vio_img__)
        cv2.waitKey(1)


def skew(vec):
    _vio_x__, _vio_y__, _vio_z__ = vec
    return np.array([
        [0, -_vio_z__, _vio_y__],
        [_vio_z__, 0, -_vio_x__],
        [-_vio_y__, _vio_x__, 0]])

def select(data, selectors):
    return [_vio_d__ for _vio_d__, _vio_s__ in zip(data, selectors) if _vio_s__]



if __name__ == '__main__':
    from queue import Queue
    from threading import Thread
    
    from _vio_config__ import ConfigEuRoC
    from _vio_dataset__ import EuRoCDataset, DataPublisher

    _vio_img_queue__ = Queue()
    _vio_imu_queue__ = Queue()

    _vio_config__ = ConfigEuRoC()
    _vio_image_processor__ = ImageProcessor(_vio_config__)


    _vio_path__ = '_vio_path__/to/your/EuRoC_MAV_dataset/MH_01_easy'
    _vio_dataset__ = EuRoCDataset(_vio_path__)
    _vio_dataset__.set_starttime(offset=0.)
    
    _vio_duration__ = 3.
    _vio_ratio__ = 0.5
    _vio_imu_publisher__ = DataPublisher(
        _vio_dataset__.imu, _vio_imu_queue__, _vio_duration__, _vio_ratio__)
    _vio_img_publisher__ = DataPublisher(
        _vio_dataset__.stereo, _vio_img_queue__, _vio_duration__, _vio_ratio__)

    _vio_now__ = time.time()
    _vio_imu_publisher__._vio_start__(_vio_now__)
    _vio_img_publisher__._vio_start__(_vio_now__)


    def process_imu(in_queue):
        while True:
            _vio_msg__ = in_queue.get()
            if _vio_msg__ is None:
                return
            print(_vio_msg__.timestamp, 'imu')
            _vio_image_processor__.imu_callback(_vio_msg__)
    _t2_ = Thread(target=process_imu, args=(_vio_imu_queue__,))
    _t2_._vio_start__()

    while True:
        _vio_msg__ = _vio_img_queue__.get()
        if _vio_msg__ is None:
            break
        print(_vio_msg__.timestamp, 'image')
        # cv2.imshow('left', np.hstack([_vio_x__.cam0_image, _vio_x__.cam1_image]))
        # cv2.waitKey(1)
        # timestamps.append(_vio_x__.timestamp)
        _vio_image_processor__.stareo_callback(_vio_msg__)

    _vio_imu_publisher__.stop()
    _vio_img_publisher__.stop()
    _t2_.join()