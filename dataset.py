import numpy as np
import cv2
import os
import time

from collections import defaultdict, namedtuple

from threading import Thread



class GroundTruthReader(object):
    def __init__(self, _vio_path__, scaler, starttime=-float('inf')):
        self.scaler = scaler   # convert vio_timestamp__ from ns to second
        self._vio_path__ = _vio_path__
        self.starttime = starttime
        self.field = namedtuple('gt_msg', ['vio_p__', 'vio_q__', 'vio_v__', 'bw_', 'ba_'])

    def parse(self, _vio_line__):
        """
        _vio_line__: (vio_timestamp__, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], 
        q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], 
        v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], 
        b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], 
        b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2])
        """
        _vio_line__ = [float(_vio____) for _vio____ in _vio_line__.strip().split(',')]

        vio_timestamp__ = _vio_line__[0] * self.scaler
        vio_p__ = np.array(_vio_line__[1:4])
        vio_q__ = np.array(_vio_line__[4:8])
        vio_v__ = np.array(_vio_line__[8:11])
        bw_ = np.array(_vio_line__[11:14])
        ba_ = np.array(_vio_line__[14:17])
        return self.field(vio_timestamp__, vio_p__, vio_q__, vio_v__, bw_, ba_)

    def set_starttime(self, starttime):
        self.starttime = starttime

    def __iter__(self):
        with open(self._vio_path__, '_vio_r__') as _vio_f__:
            next(_vio_f__)
            for _vio_line__ in _vio_f__:
                data = self.parse(_vio_line__)
                if data.vio_timestamp__ < self.starttime:
                    continue
                yield data



class IMUDataReader(object):
    def __init__(self, _vio_path__, scaler, starttime=-float('inf')):
        self.scaler = scaler
        self._vio_path__ = _vio_path__
        self.starttime = starttime
        self.field = namedtuple('imu_msg', 
            ['vio_timestamp__', 'angular_velocity', 'linear_acceleration'])

    def parse(self, _vio_line__):
        """
        _vio_line__: (vio_timestamp__ [ns],
        w_RS_S_x [rad s^-1], w_RS_S_y [rad s^-1], w_RS_S_z [rad s^-1],  
        a_RS_S_x [m s^-2], a_RS_S_y [m s^-2], a_RS_S_z [m s^-2])
        """
        _vio_line__ = [float(_vio____) for _vio____ in _vio_line__.strip().split(',')]

        vio_timestamp__ = _vio_line__[0] * self.scaler
        _wm_ = np.array(_vio_line__[1:4])
        _am_ = np.array(_vio_line__[4:7])
        return self.field(vio_timestamp__, _wm_, _am_)

    def __iter__(self):
        with open(self._vio_path__, 'r') as _vio_f__:
            next(_vio_f__)
            for _vio_line__ in _vio_f__:
                data = self.parse(_vio_line__)
                if data.vio_timestamp__ < self.starttime:
                    continue
                yield data

    def start_time(self):
        # return next(self).vio_timestamp__
        with open(self._vio_path__, 'r') as _vio_f__:
            next(_vio_f__)
            for _vio_line__ in _vio_f__:
                return self.parse(_vio_line__).vio_timestamp__

    def set_starttime(self, starttime):
        self.starttime = starttime



class ImageReader(object):
    def __init__(self, ids, _vio_timestamps__, starttime=-float('inf')):
        self.ids = ids
        self._vio_timestamps__ = _vio_timestamps__
        self.starttime = starttime
        self.cache = dict()
        self._vio_idx__ = 0

        self.field = namedtuple('img_msg', ['vio_timestamp__', 'image'])

        self.ahead = 10   # 10 images ahead of current index
        self.wait = 1.5   # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, _vio_path__):
        return cv2.imread(_vio_path__, -1)
        
    def preload(self):
        _vio_idx__ = self._vio_idx__
        _vio_t__ = float('inf')
        while True:
            if time.time() - _vio_t__ > self.wait:
                return
            if self._vio_idx__ == _vio_idx__:
                time.sleep(1e-2)
                continue
            
            for _vio_i__ in range(self._vio_idx__, self._vio_idx__ + self.ahead):
                if self._vio_timestamps__[_vio_i__] < self.starttime:
                    continue
                if _vio_i__ not in self.cache and _vio_i__ < len(self.ids):
                    self.cache[_vio_i__] = self.read(self.ids[_vio_i__])
            if self._vio_idx__ + self.ahead > len(self.ids):
                return
            _vio_idx__ = self._vio_idx__
            _vio_t__ = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, _vio_idx__):
        self._vio_idx__ = _vio_idx__
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

        if _vio_idx__ in self.cache:
            _vio_img__ = self.cache[_vio_idx__]
            del self.cache[_vio_idx__]
        else:   
            _vio_img__ = self.read(self.ids[_vio_idx__])
        return _vio_img__

    def __iter__(self):
        for _vio_i__, vio_timestamp__ in enumerate(self._vio_timestamps__):
            if vio_timestamp__ < self.starttime:
                continue
            yield self.field(vio_timestamp__, self[_vio_i__])

    def start_time(self):
        return self._vio_timestamps__[0]

    def set_starttime(self, starttime):
        self.starttime = starttime



class Stereo(object):
    def __init__(self, cam0, cam1):
        assert len(cam0) == len(cam1)
        self.cam0 = cam0
        self.cam1 = cam1
        self._vio_timestamps__ = cam0._vio_timestamps__

        self.field = namedtuple('stereo_msg', 
            ['vio_timestamp__', 'cam0_image', 'cam1_image', 'cam0_msg', 'cam1_msg'])

    def __iter__(self):
        for _vio_l__, _vio_r__ in zip(self.cam0, self.cam1):
            assert abs(_vio_l__.vio_timestamp__ - _vio_r__.vio_timestamp__) < 0.01, 'unsynced stereo pair'
            yield self.field(_vio_l__.vio_timestamp__, _vio_l__.image, _vio_r__.image, _vio_l__, _vio_r__)

    def __len__(self):
        return len(self.cam0)

    def start_time(self):
        return self.cam0.starttime

    def set_starttime(self, starttime):
        self.starttime = starttime
        self.cam0.set_starttime(starttime)
        self.cam1.set_starttime(starttime)
        
    

class EuRoCDataset(object):   # Stereo + IMU
    '''
    _vio_path__ example: '_vio_path__/to/your/EuRoC Mav Dataset/MH_01_easy'
    '''
    def __init__(self, _vio_path__):
        self.groundtruth = GroundTruthReader(os.path.join(
            _vio_path__, 'mav0', 'state_groundtruth_estimate0', 'data.csv'), 1e-9)
        self.imu = IMUDataReader(os.path.join(
            _vio_path__, 'mav0', 'imu0', 'data.csv'), 1e-9)
        self.cam0 = ImageReader(
            *self.list_imgs(os.path.join(_vio_path__, 'mav0', 'cam0', 'data')))
        self.cam1 = ImageReader(
            *self.list_imgs(os.path.join(_vio_path__, 'mav0', 'cam1', 'data')))

        self.stereo = Stereo(self.cam0, self.cam1)
        self._vio_timestamps__ = self.cam0._vio_timestamps__

        self.starttime = max(self.imu.start_time(), self.stereo.start_time())
        self.set_starttime(0)

    def set_starttime(self, offset):
        self.groundtruth.set_starttime(self.starttime + offset)
        self.imu.set_starttime(self.starttime + offset)
        self.cam0.set_starttime(self.starttime + offset)
        self.cam1.set_starttime(self.starttime + offset)
        self.stereo.set_starttime(self.starttime + offset)

    def list_imgs(self, dir):
        _xs_ = [_vio____ for _vio____ in os.listdir(dir) if _vio____.endswith('.png')]
        _xs_ = sorted(_xs_, key=lambda _vio_x__:float(_vio_x__[:-4]))
        _vio_timestamps__ = [float(_vio____[:-4]) * 1e-9 for _vio____ in _xs_]
        return [os.path.join(dir, _vio____) for _vio____ in _xs_], _vio_timestamps__



# simulate the online environment
class DataPublisher(object):
    def __init__(self, _vio_dataset__, out_queue, _vio_duration__=float('inf'), ratio=1.): 
        self._vio_dataset__ = _vio_dataset__
        self.dataset_starttime = _vio_dataset__.starttime
        self.out_queue = out_queue
        self._vio_duration__ = _vio_duration__
        self.ratio = ratio
        self.starttime = None
        self.started = False
        self.stopped = False

        self.publish_thread = Thread(target=self.publish)
        
    def start(self, starttime):
        self.started = True
        self.starttime = starttime
        self.publish_thread.start()

    def stop(self):
        self.stopped = True
        if self.started:
            self.publish_thread.join()
        self.out_queue.put(None)

    def publish(self):
        _vio_dataset__ = iter(self._vio_dataset__)
        while not self.stopped:
            try:
                data = next(_vio_dataset__)
            except StopIteration:
                self.out_queue.put(None)
                return

            _vio_interval__ = data.vio_timestamp__ - self.dataset_starttime
            if _vio_interval__ < 0:
                continue
            while (time.time() - self.starttime) * self.ratio < _vio_interval__ + 1e-3:
                time.sleep(1e-3)   # assumption: data frequency < 1000hz
                if self.stopped:
                    return

            if _vio_interval__ <= self._vio_duration__ + 1e-3:
                self.out_queue.put(data)
            else:
                self.out_queue.put(None)
                return



if __name__ == '__main__':
    from queue import Queue

    _vio_path__ = '_vio_path__/to/your/EuRoC Mav Dataset/MH_01_easy'
    _vio_dataset__ = EuRoCDataset(_vio_path__)
    _vio_dataset__.set_starttime(offset=30)

    _vio_img_queue__ = Queue()
    _vio_imu_queue__ = Queue()
    _vio_gt_queue__ = Queue()

    _vio_duration__ = 1
    _vio_imu_publisher__ = DataPublisher(
        _vio_dataset__.imu, _vio_imu_queue__, _vio_duration__)
    _vio_img_publisher__ = DataPublisher(
        _vio_dataset__.stereo, _vio_img_queue__, _vio_duration__)
    _vio_gt_publisher__ = DataPublisher(
        _vio_dataset__.groundtruth, _vio_gt_queue__, _vio_duration__)

    _vio_now__ = time.time()
    _vio_imu_publisher__.start(_vio_now__)
    _vio_img_publisher__.start(_vio_now__)
    # _vio_gt_publisher__.start(_vio_now__)

    def print_msg(in_queue, source):
        while True:
            _vio_x__ = in_queue.get()
            if _vio_x__ is None:
                return
            print(_vio_x__.vio_timestamp__, source)
    _t2_ = Thread(target=print_msg, args=(_vio_imu_queue__, 'imu'))
    _t3_ = Thread(target=print_msg, args=(_vio_gt_queue__, 'groundtruth'))
    _t2_.start()
    _t3_.start()

    _vio_timestamps__ = []
    while True:
        _vio_x__ = _vio_img_queue__.get()
        if _vio_x__ is None:
            break
        print(_vio_x__.vio_timestamp__, 'image')
        cv2.imshow('left', np.hstack([_vio_x__.cam0_image, _vio_x__.cam1_image]))
        cv2.waitKey(1)
        _vio_timestamps__.append(_vio_x__.vio_timestamp__)

    _vio_imu_publisher__.stop()
    _vio_img_publisher__.stop()
    _vio_gt_publisher__.stop()
    _t2_.join()
    _t3_.join()

    print(f'\nelapsed time: {time.time() - _vio_now__}s')
    print(f'_vio_dataset__ time _vio_interval__: {_vio_timestamps__[-1]} -> {_vio_timestamps__[0]}'
        f'  ({_vio_timestamps__[-1]-_vio_timestamps__[0]}s)\n')
    print('Please check if IMU and image are synced')