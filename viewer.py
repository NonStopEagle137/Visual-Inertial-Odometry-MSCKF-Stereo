import numpy as np
import OpenGL.GL as gl
import pangolin
import cv2

from multiprocessing import Queue, Process



class Viewer(object):
    def __init__(self):
        self.image_queue = Queue()
        self.pose_queue = Queue()

        self.view_thread = Process(target=self.view)
        self.view_thread.start()

    def update_pose(self, _vio_pose__):
        if _vio_pose__ is None:
            return
        self.pose_queue.put(_vio_pose__.matrix())

    def update_image(self, _vio_image__):
        if _vio_image__ is None:
            return
        elif _vio_image__.ndim == 2:
            _vio_image__ = np.repeat(_vio_image__[..., np.newaxis], 3, axis=2)
        self.image_queue.put(_vio_image__)
            

    def view(self):
        pangolin.CreateWindowAndBind('Viewer', 1024, 768)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc (gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        _vio_viewpoint_x__ = 0
        _vio_viewpoint_y__ = -7  
        _vio_viewpoint_z__ = -18 
        _vio_viewpoint_f__ = 1000

        _vio_proj__ = pangolin.ProjectionMatrix(
            1024, 768, _vio_viewpoint_f__, _vio_viewpoint_f__, 512, 389, 0.1, 300)
        _vio_look_view__ = pangolin.ModelViewLookAt(
            _vio_viewpoint_x__, _vio_viewpoint_y__, _vio_viewpoint_z__, 0, 0, 0, 0, -1, 0)

        # Camera Render Object (for view / scene browsing)
        _vio_scam__ = pangolin.OpenGlRenderState(_vio_proj__, _vio_look_view__)

        # Add named OpenGL viewport to window and provide 3D Handler
        _vio_dcam__ = pangolin.CreateDisplay()
        _vio_dcam__.SetBounds(0.0, 1.0, 175 / 1024., 1.0, -1024 / 768.)
        _vio_dcam__.SetHandler(pangolin.Handler3D(_vio_scam__))

        # _vio_image__
        _vio_width__, _vio_height__ = 376, 240
        _vio_dimg__ = pangolin.Display('_vio_image__')
        _vio_dimg__.SetBounds(0, _vio_height__ / 768., 0.0, _vio_width__ / 1024., 1024 / 768.)
        _vio_dimg__.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        _vio_texture__ = pangolin.GlTexture(_vio_width__, _vio_height__, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        _vio_image__ = np.ones((_vio_height__, _vio_width__, 3), 'uint8')

        # _vio_axis__
        _vio_axis__ = pangolin.Renderable()
        _vio_axis__.Add(pangolin.Axis())


        _vio_trajectory__ = DynamicArray()
        _vio_camera__ = None
        _vio_image__ = None
        #print('Pose Queue : ', list(self.pose_queue))

        while not pangolin.ShouldQuit():
            if not self.pose_queue.empty():
                while not self.pose_queue.empty():
                    _vio_pose__ = self.pose_queue.get()
                _vio_trajectory__.append(_vio_pose__[:3, 3])
                _vio_camera__ = _vio_pose__

            if not self.image_queue.empty():
                while not self.image_queue.empty():
                    _vio_img__ = self.image_queue.get()
                _vio_img__ = _vio_img__[::-1, :, ::-1]
                _vio_img__ = cv2.resize(_vio_img__, (_vio_width__, _vio_height__))
                _vio_image__ = _vio_img__.copy()

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            _vio_dcam__.Activate(_vio_scam__)


            # draw _vio_axis__
            _vio_axis__.Render()

            # draw current _vio_camera__
            if _vio_camera__ is not None:
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawCameras(np.array([_vio_camera__]), 0.5)

            # show _vio_trajectory__
            if len(_vio_trajectory__) > 0:
                gl.glPointSize(2)
                gl.glColor3f(0.0, 0.0, 0.0)
                pangolin.DrawPoints(_vio_trajectory__.array())

            # show _vio_image__
            if _vio_image__ is not None:
                _vio_texture__.Upload(_vio_image__, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                _vio_dimg__.Activate()
                gl.glColor3f(1.0, 1.0, 1.0)
                _vio_texture__.RenderToViewport()
                
            pangolin.FinishFrame()




class DynamicArray(object):
    def __init__(self, _vio_shape__=3):
        if isinstance(_vio_shape__, int):
            _vio_shape__ = (_vio_shape__,)
        assert isinstance(_vio_shape__, tuple)

        self.data = np.zeros((1000, *_vio_shape__))
        self._vio_shape__ = _vio_shape__
        self.ind = 0

    def clear(self):
        self.ind = 0

    def append(self, _vio_x__):
        self.extend([_vio_x__])
    
    def extend(self, xs):
        if len(xs) == 0:
            return
        assert np.array(xs[0]).shape == self._vio_shape__

        if self.ind + len(xs) >= len(self.data):
            self.data.resize(
                (2 * len(self.data), *self._vio_shape__) , refcheck=False)

        if isinstance(xs, np.ndarray):
            self.data[self.ind:self.ind+len(xs)] = xs
        else:
            for _vio_i__, _vio_x__ in enumerate(xs):
                self.data[self.ind+_vio_i__] = _vio_x__
            self.ind += len(xs)

    def array(self):
        return self.data[:self.ind]

    def __len__(self):
        return self.ind

    def __getitem__(self, _vio_i__):
        assert _vio_i__ < self.ind
        return self.data[_vio_i__]

    def __iter__(self):
        for _vio_x__ in self.data[:self.ind]:
            yield _vio_x__




if __name__ == '__main__':
    import g2o
    import time

    _vio_viewer__ = Viewer()
    _vio_viewer__.update_pose(g2o.Isometry3d())