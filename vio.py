
from queue import Queue
from threading import Thread

from config import ConfigEuRoC
from image import ImageProcessor
from msckf import MSCKF



class VIO(object):
    def __init__(self, _vio_config__, _vio_img_queue__, _vio_imu_queue__, _vio_viewer__=None):
        self._vio_config__ = _vio_config__
        self._vio_viewer__ = _vio_viewer__

        self._vio_img_queue__ = _vio_img_queue__
        self._vio_imu_queue__ = _vio_imu_queue__
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(_vio_config__)
        self.msckf = MSCKF(_vio_config__)

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

    def process_img(self):
        while True:
            _vio_img_msg__ = self._vio_img_queue__.get()
            if _vio_img_msg__ is None:
                self.feature_queue.put(None)
                return
            # print('_vio_img_msg__', _vio_img_msg__.timestamp)

            if self._vio_viewer__ is not None:
                self._vio_viewer__.update_image(_vio_img_msg__.cam0_image)
            print('here',_vio_img_msg__)
            _vio_feature_msg__ = self.image_processor.stareo_callback(_vio_img_msg__)

            if _vio_feature_msg__ is not None:
                self.feature_queue.put(_vio_feature_msg__)

    def process_imu(self):
        while True:
            _vio_imu_msg__ = self._vio_imu_queue__.get()
            if _vio_imu_msg__ is None:
                return
            # print('_vio_imu_msg__', _vio_imu_msg__.timestamp)

            self.image_processor.imu_callback(_vio_imu_msg__)
            self.msckf.imu_callback(_vio_imu_msg__)

    def process_feature(self):
        while True:
            _vio_feature_msg__ = self.feature_queue.get()
            if _vio_feature_msg__ is None:
                return
            print('_vio_feature_msg__', _vio_feature_msg__.timestamp)
            _vio_result__ = self.msckf.feature_callback(_vio_feature_msg__)

            if _vio_result__ is not None and self._vio_viewer__ is not None:
                self._vio_viewer__.update_pose(_vio_result__.cam0_pose)
        


if __name__ == '__main__':
    import time
    import argparse

    from dataset import EuRoCDataset, DataPublisher
    from viewer import Viewer

    _vio_parser__ = argparse.ArgumentParser()
    _vio_parser__.add_argument('--path', type=str, default='path/to/your/EuRoC_MAV_dataset/MH_01_easy', 
        help='Path of EuRoC MAV _vio_dataset__.')
    _vio_parser__.add_argument('--view', action='store_true', help='Show trajectory.')
    _vio_args__ = _vio_parser__.parse_args()

    if _vio_args__.view:
        _vio_viewer__ = Viewer()
    else:
        _vio_viewer__ = None

    _vio_dataset__ = EuRoCDataset(_vio_args__.path)
    _vio_dataset__.set_starttime(offset=40.)   # start from static state


    _vio_img_queue__ = Queue()
    _vio_imu_queue__ = Queue()
    # gt_queue = Queue()

    _vio_config__ = ConfigEuRoC()
    _vio_msckf_vio__ = VIO(_vio_config__, _vio_img_queue__, _vio_imu_queue__, _vio_viewer__=_vio_viewer__)


    _vio_duration__ = float('inf')
    _vio_ratio__ = 0.4  # make it smaller if image processing and MSCKF computation is slow
    _vio_imu_publisher__ = DataPublisher(
        _vio_dataset__.imu, _vio_imu_queue__, _vio_duration__, _vio_ratio__)
    _vio_img_publisher__ = DataPublisher(
        _vio_dataset__.stereo, _vio_img_queue__, _vio_duration__, _vio_ratio__)

    _vio_now__ = time.time()
    _vio_imu_publisher__.start(_vio_now__)
    _vio_img_publisher__.start(_vio_now__)