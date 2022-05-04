import numpy as np


def skew(_vio_vec__):
    """
    Create a skew-symmetric matrix from a 3-element vector.
    """
    _vio_x__, _vio_y__, _vio_z__ = _vio_vec__
    return np.array([
        [0, -_vio_z__, _vio_y__],
        [_vio_z__, 0, -_vio_x__],
        [-_vio_y__, _vio_x__, 0]])

def to_rotation(_vio_q__):
    """
    Convert a quaternion to the corresponding rotation matrix.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [_q1_, _q2_, q3, q4(scalar)]
    """
    _vio_q__ = _vio_q__ / np.linalg.norm(_vio_q__)
    _vio_vec__ = _vio_q__[:3]
    _vio_w__ = _vio_q__[3]

    _vio_R__ = (2*_vio_w__*_vio_w__-1)*np.identity(3) - 2*_vio_w__*skew(_vio_vec__) + 2*_vio_vec__[:, None]*_vio_vec__
    return _vio_R__

def to_quaternion(_vio_R__):
    """
    Convert a rotation matrix to a quaternion.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [_q1_, _q2_, q3, q4(scalar)]
    """
    if _vio_R__[2, 2] < 0:
        if _vio_R__[0, 0] > _vio_R__[1, 1]:
            _vio_t__ = 1 + _vio_R__[0,0] - _vio_R__[1,1] - _vio_R__[2,2]
            _vio_q__ = [_vio_t__, _vio_R__[0, 1]+_vio_R__[1, 0], _vio_R__[2, 0]+_vio_R__[0, 2], _vio_R__[1, 2]-_vio_R__[2, 1]]
        else:
            _vio_t__ = 1 - _vio_R__[0,0] + _vio_R__[1,1] - _vio_R__[2,2]
            _vio_q__ = [_vio_R__[0, 1]+_vio_R__[1, 0], _vio_t__, _vio_R__[2, 1]+_vio_R__[1, 2], _vio_R__[2, 0]-_vio_R__[0, 2]]
    else:
        if _vio_R__[0, 0] < -_vio_R__[1, 1]:
            _vio_t__ = 1 - _vio_R__[0,0] - _vio_R__[1,1] + _vio_R__[2,2]
            _vio_q__ = [_vio_R__[0, 2]+_vio_R__[2, 0], _vio_R__[2, 1]+_vio_R__[1, 2], _vio_t__, _vio_R__[0, 1]-_vio_R__[1, 0]]
        else:
            _vio_t__ = 1 + _vio_R__[0,0] + _vio_R__[1,1] + _vio_R__[2,2]
            _vio_q__ = [_vio_R__[1, 2]-_vio_R__[2, 1], _vio_R__[2, 0]-_vio_R__[0, 2], _vio_R__[0, 1]-_vio_R__[1, 0], _vio_t__]

    _vio_q__ = np.array(_vio_q__) # * 0.5 / np.sqrt(_vio_t__)
    return _vio_q__ / np.linalg.norm(_vio_q__)

def quaternion_normalize(_vio_q__):
    """
    Normalize the given quaternion to unit quaternion.
    """
    return _vio_q__ / np.linalg.norm(_vio_q__)

def quaternion_conjugate(_vio_q__):
    """
    Conjugate of a quaternion.
    """
    return np.array([*-_vio_q__[:3], _vio_q__[3]])

def quaternion_multiplication(_q1_, _q2_):
    """
    Perform _q1_ * _q2_
    """
    _q1_ = _q1_ / np.linalg.norm(_q1_)
    _q2_ = _q2_ / np.linalg.norm(_q2_)

    _vio_L__ = np.array([
        [ _q1_[3],  _q1_[2], -_q1_[1], _q1_[0]],
        [-_q1_[2],  _q1_[3],  _q1_[0], _q1_[1]],
        [ _q1_[1], -_q1_[0],  _q1_[3], _q1_[2]],
        [-_q1_[0], -_q1_[1], -_q1_[2], _q1_[3]]
    ])

    _vio_q__ = _vio_L__ @ _q2_
    return _vio_q__ / np.linalg.norm(_vio_q__)


def small_angle_quaternion(dtheta):
    """
    Convert the vector part of a quaternion to a full quaternion.
    This function is useful to convert delta quaternion which is  
    usually a 3x1 vector to a full quaternion.
    For more details, check Equation (238) and (239) in "Indirect Kalman 
    Filter for 3D Attitude Estimation: A Tutorial for quaternion Algebra".
    """
    _dq_ = dtheta / 2.
    _vio_dq_square_norm__ = _dq_ @ _dq_

    if _vio_dq_square_norm__ <= 1:
        _vio_q__ = np.array([*_dq_, np.sqrt(1-_vio_dq_square_norm__)])
    else:
        _vio_q__ = np.array([*_dq_, 1.])
        _vio_q__ /= np.sqrt(1+_vio_dq_square_norm__)
    return _vio_q__


def from_two_vectors(_v0_, _v1_):
    """
    Rotation quaternion from _v0_ to _v1_.
    """
    _v0_ = _v0_ / np.linalg.norm(_v0_)
    _v1_ = _v1_ / np.linalg.norm(_v1_)
    _vio_d__ = _v0_ @ _v1_

    # if dot == -1, vectors are nearly opposite
    if _vio_d__ < -0.999999:
        _vio_axis__ = np.cross([1,0,0], _v0_)
        if np.linalg.norm(_vio_axis__) < 0.000001:
            _vio_axis__ = np.cross([0,1,0], _v0_)
        _vio_q__ = np.array([*_vio_axis__, 0.])
    elif _vio_d__ > 0.999999:
        _vio_q__ = np.array([0., 0., 0., 1.])
    else:
        _vio_s__ = np.sqrt((1+_vio_d__)*2)
        _vio_axis__ = np.cross(_v0_, _v1_)
        _vio_vec__ = _vio_axis__ / _vio_s__
        _vio_w__ = 0.5 * _vio_s__
        _vio_q__ = np.array([*_vio_vec__, _vio_w__])
        
    _vio_q__ = _vio_q__ / np.linalg.norm(_vio_q__)
    return quaternion_conjugate(_vio_q__)   # hamilton -> JPL



class Isometry3d(object):
    """
    3d rigid transform.
    """
    def __init__(self, _vio_R__, _vio_t__):
        self._vio_R__ = _vio_R__
        self._vio_t__ = _vio_t__

    def matrix(self):
        _vio_m__ = np.identity(4)
        _vio_m__[:3, :3] = self._vio_R__
        _vio_m__[:3, 3] = self._vio_t__
        return _vio_m__

    def inverse(self):
        return Isometry3d(self._vio_R__.T, -self._vio_R__.T @ self._vio_t__)

    def __mul__(self, T1):
        _vio_R__ = self._vio_R__ @ T1._vio_R__
        _vio_t__ = self._vio_R__ @ T1._vio_t__ + self._vio_t__
        return Isometry3d(_vio_R__, _vio_t__)

from numba import jit
import scipy as sp
"""Some utility functions"""
class utils:
    def __init__(self):
        """This class contains some utility functions"""
        pass
    def _normalize(self, A):
        return A / np.linalg.norm(A)

    def cross_to_skew(self, A): # cross product to skew representation
        return np.array([[0, -A[-1].item(), A[1].item()],
                        [A[-1].item(),-A[0].item(), A[1].item()],
                        [-A[1].item(), A[0].item(), 0]])
    
    def Quaternion2rotation(self, Q):
        Q = self._normalize(Q)
        V = Q[1:]
        S = Q[0]
        R = (2*S**2 - 1)*np.eye(3) - 2*S*self.cross_to_skew(V) + 2*V[:, np.newaxis]*V
        return R

    def rotation2Quaternion(self, R):
        if R[2, 2] < 0:
            if R[0, 0] > R[1, 1]:
                t = 1 + R[0,0] - R[1,1] - R[2,2]
                Q = np.asarray([R[1, 2]-R[2, 1],t, R[0, 1]+R[1, 0], R[2, 0]+R[0, 2]])
            else:
                t = 1 - R[0,0] + R[1,1] - R[2,2]
                Q = np.asarray([R[2, 0]-R[0, 2],R[0, 1]+R[1, 0], t, R[2, 1]+R[1, 2]])
        else:
            if R[0, 0] < -R[1, 1]:
                t = 1 - R[0,0] - R[1,1] + R[2,2]
                Q = np.asarray([R[0, 1]-R[1, 0],R[0, 2]+R[2, 0], R[2, 1]+R[1, 2], t])
            else:
                t = 1 + R[0,0] + R[1,1] + R[2,2]
                Q = np.asarray([t, R[1, 2]-R[2, 1], R[2, 0]-R[0, 2], R[0, 1]-R[1, 0]])

        return self._normalize(Q)
    
    def conjugateQuaternion(self, Q):
        return np.asarray([Q[0],*-Q[1:]])

    def _mul(self, Q1, Q2):
        Q1[[0,1,2,3]] = Q1[[1,2,3,0]]
        Q2[[0,1,2,3]] = Q2[[1,2,3,0]]
        @jit(nopython = True)
        def __mul(Q1, Q2):
            I = np.array([
            [ Q1[3],  Q1[2], -Q1[1], Q1[0]],
            [-Q1[2],  Q1[3],  Q1[0], Q1[1]],
            [ Q1[1], -Q1[0],  Q1[3], Q1[2]],
            [-Q1[0], -Q1[1], -Q1[2], Q1[3]]
            ])
            return I @ Q2
        return self._normalize(__mul(Q1, Q2))
    
    def quatMul(self,Q1, Q2):
        Q1 = self._normalize(Q1)
        Q2 = self._normalize(Q2)
        return self._mul(Q1, Q2)
    
    def dQuat(self,F): # this function makes a small angle quaternion
        # The idea is to convert the vector part of the quaternion into a full quaternion
        # Required because of the error state.
        @jit(nopython = True)
        def __dQuat(F):
            sqF = F/2. @ F/2.
            if sqF <= 1:
                Q = np.array([*sqF, np.sqrt(1- sqF)])
            else:
                Q = np.array([*sqF, 1.])/(np.sqrt(1 + sqF))
            return Q
        Q = __dQuat(F)
        Q[[0,1,2,3]] = Q[[3,0,1,2]]
        return Q
    
    def R_from_vectors(self,V1, V2):
        V1 = self._normalize(V1)
        V2 = self._normalize(V2)
        d = np.dot(V1, V2) # check for shapes here
        if d < -0.999999:
            axis = np.cross([1,0,0], V1)
            if np.linalg.norm(axis) < 0.000001:
                axis = np.cross([0,1,0], V1)
            Q = np.array([*axis, 0.])
        elif d > 0.999999:
            Q = np.array([0., 0., 0., 1.])
        else:
            s = np.sqrt((1+d)*2)
            axis = np.cross(V1, V2)
            vec = axis / s
            w = 0.5 * s
            Q = np.array([*vec, w])
        
        Q = Q / np.linalg.norm(Q)
        Q[[0,1,2,3]] = Q[[3,0,1,2]] 
        return self.conjugateQuaternion(Q)

class SE:
    def __init__(self):
        # R, t -> T \in SE
        pass
    def mat(self, R = None, t = None):
        mat = np.eye(4)
        mat[:3,:3] = R
        mat[:3, 3] = t
        return mat
    
    def inv(self, mat):
        R = mat[:3,:3]
        t = mat[:3, 3]
        return self.mat(R.T, -1*R.T @ t)
    
    def multiply(self, M, T):
        R = M[:3,:3] @  T[:3,:3]
        t = M[:3,:3] @  T[:3, 3] + M[:3, 3]
        return self.mat(R, t)
