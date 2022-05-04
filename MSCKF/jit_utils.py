import numpy as np
from numba import jit

"""This file contains all the JIT-TTED functions that speed up the computation of some of the expensive operations"""
"""Some JIT functions for speed"""
@jit(nopython = True)
def _process_model(_vio_gyro__,
    _vio_R_w_i__,_vio_acc__, _dt_):

    _vio_F__ = np.zeros((21, 21))
    _vio_G__ = np.zeros((21, 12))

    _vio_x__, _vio_y__, _vio_z__ = _vio_gyro__
    _vio_gyroSkew__ = np.array([
        [0, -_vio_z__, _vio_y__],
        [_vio_z__, 0, -_vio_x__],
        [-_vio_y__, _vio_x__, 0]])
    
    _vio_x__, _vio_y__, _vio_z__ = _vio_acc__
    _vio_accSkew__ = np.array([
        [0, -_vio_z__, _vio_y__],
        [_vio_z__, 0, -_vio_x__],
        [-_vio_y__, _vio_x__, 0]])

    _vio_F__[:3, :3] = -_vio_gyroSkew__
    _vio_F__[:3, 3:6] = -np.identity(3)
    _vio_F__[6:9, :3] = -_vio_R_w_i__.T @ _vio_accSkew__
    _vio_F__[6:9, 9:12] = -_vio_R_w_i__.T
    _vio_F__[12:15, 6:9] = np.identity(3)

    _vio_G__[:3, :3] = -np.identity(3)
    _vio_G__[3:6, 3:6] = np.identity(3)
    _vio_G__[6:9, 6:9] = -_vio_R_w_i__.T
    _vio_G__[9:12, 9:12] = np.identity(3)

    # Approximate matrix exponential to the 3rd order, which can be 
    # considered to be accurate enough assuming _dt_ is within 0.01s.
    _vio_Fdt__ = _vio_F__ * _dt_
    _vio_Fdt_square__ = _vio_Fdt__ @ _vio_Fdt__
    _vio_Fdt_cube__ = _vio_Fdt_square__ @ _vio_Fdt__
    _vio_Phi__ = np.identity(21) + _vio_Fdt__ + _vio_Fdt_square__/2. + _vio_Fdt_cube__/6.
    
    return _vio_F__,_vio_G__, _vio_Fdt__, _vio_Fdt_square__, _vio_Fdt_cube__, _vio_Phi__


@jit(nopython = True)
def _predict_new_state(_dt_, _vio_gyro__, _vio_acc__, _vio_q__, _vio_v__, _vio_p__, _vio_gravity__):
        _vio_gyro_norm__ = np.linalg.norm(_vio_gyro__)
        _vio_Omega__ = np.zeros((4, 4))

        _vio_x__, _vio_y__, _vio_z__ = _vio_gyro__
        _vio_gyroSkew__ = np.array([
            [0, -_vio_z__, _vio_y__],
            [_vio_z__, 0, -_vio_x__],
            [-_vio_y__, _vio_x__, 0]])

        _vio_Omega__[:3, :3] = -_vio_gyroSkew__
        _vio_Omega__[:3, 3] = _vio_gyro__
        _vio_Omega__[3, :3] = -_vio_gyro__

        
        

        if _vio_gyro_norm__ > 1e-5:
            _vio_dq_dt__ = (np.cos(_vio_gyro_norm__*_dt_*0.5) * np.identity(4) + 
                np.sin(_vio_gyro_norm__*_dt_*0.5)/_vio_gyro_norm__ * _vio_Omega__) @ _vio_q__
            _vio_dq_dt2__ = (np.cos(_vio_gyro_norm__*_dt_*0.25) * np.identity(4) + 
                np.sin(_vio_gyro_norm__*_dt_*0.25)/_vio_gyro_norm__ * _vio_Omega__) @ _vio_q__
        else:
            _vio_dq_dt__ = np.cos(_vio_gyro_norm__*_dt_*0.5) * (np.identity(4) + 
                _vio_Omega__*_dt_*0.5) @ _vio_q__
            _vio_dq_dt2__ = np.cos(_vio_gyro_norm__*_dt_*0.25) * (np.identity(4) + 
                _vio_Omega__*_dt_*0.25) @ _vio_q__

        #_vio_dR_dt_transpose__ = to_rotation(_vio_dq_dt__).T
        _vio_dq_dt__ = _vio_dq_dt__ / np.linalg.norm(_vio_dq_dt__)
        _vio_vec__ = _vio_dq_dt__[:3]
        _vio_w__ = _vio_dq_dt__[3]

        _vio_x__, _vio_y__, _vio_z__ = _vio_vec__
        _vio_vecSkew__ = np.array([
            [0, -_vio_z__, _vio_y__],
            [_vio_z__, 0, -_vio_x__],
            [-_vio_y__, _vio_x__, 0]])

        _vio_dR_dt_transpose__ = ((2*_vio_w__*_vio_w__-1)*np.identity(3) - 2*_vio_w__*_vio_vecSkew__ + 2*np.expand_dims(_vio_vec__, axis = 1)*_vio_vec__).T
        
        _vio_dq_dt2__ = _vio_dq_dt2__ / np.linalg.norm(_vio_dq_dt2__)
        _vio_vec__ = _vio_dq_dt2__[:3]
        _vio_w__ = _vio_dq_dt2__[3]

        

        _vio_dR_dt2_transpose__ = ((2*_vio_w__*_vio_w__-1)*np.identity(3) - 2*_vio_w__*_vio_vecSkew__ + 2*np.expand_dims(_vio_vec__, axis = 1)*_vio_vec__).T
        

        # k1 = f(tn, yn)
        _vio_k1_p_dot__ = _vio_v__

        _vio_q__ = _vio_q__ / np.linalg.norm(_vio_q__)
        _vio_vec__ = _vio_q__[:3]
        _vio_w__ = _vio_q__[3]

        _vio_R__ = (2*_vio_w__*_vio_w__-1)*np.identity(3) - 2*_vio_w__*_vio_vecSkew__ + 2*np.expand_dims(_vio_vec__, axis = 1)*_vio_vec__

        _vio_k1_v_dot__ = _vio_R__.T @ _vio_acc__ + _vio_gravity__

        # k2 = f(tn+_dt_/2, yn+k1*_dt_/2)
        _vio_k1_v__ = _vio_v__ + _vio_k1_v_dot__*_dt_/2.
        _vio_k2_p_dot__ = _vio_k1_v__
        _vio_k2_v_dot__ = _vio_dR_dt2_transpose__ @ _vio_acc__ + _vio_gravity__
        
        # k3 = f(tn+_dt_/2, yn+k2*_dt_/2)
        _vio_k2_v__ = _vio_v__ + _vio_k2_v_dot__*_dt_/2
        _vio_k3_p_dot__ = _vio_k2_v__
        _vio_k3_v_dot__ = _vio_dR_dt2_transpose__ @ _vio_acc__ + _vio_gravity__
        
        # k4 = f(tn+_dt_, yn+k3*_dt_)
        _vio_k3_v__ = _vio_v__ + _vio_k3_v_dot__*_dt_
        _vio_k4_p_dot__ = _vio_k3_v__
        _vio_k4_v_dot__ = _vio_dR_dt_transpose__ @ _vio_acc__ + _vio_gravity__

        # yn+1 = yn + _dt_/6*(k1+2*k2+2*k3+k4)
        _vio_q__ = _vio_dq_dt__ / np.linalg.norm(_vio_dq_dt__)
        _vio_v__ = _vio_v__ + (_vio_k1_v_dot__ + 2*_vio_k2_v_dot__ + 2*_vio_k3_v_dot__ + _vio_k4_v_dot__)*_dt_/6.
        _vio_p__ = _vio_p__ + (_vio_k1_p_dot__ + 2*_vio_k2_p_dot__ + 2*_vio_k3_p_dot__ + _vio_k4_p_dot__)*_dt_/6.

        return _vio_q__, _vio_v__, _vio_p__

@jit(nopython = True)
def _propaget_state_Covariance(_vio_continuous_noise_cov__,_vio_state_cov__, _vio_G__, _vio_Phi__, _dt_):
    _vio_Q__ = np.ascontiguousarray(_vio_Phi__) @ np.ascontiguousarray(_vio_G__) @ np.ascontiguousarray(_vio_continuous_noise_cov__) @ np.ascontiguousarray(_vio_G__.T) @ np.ascontiguousarray(_vio_Phi__.T) * _dt_
    _vio_state_cov__[:21, :21] = (
        np.ascontiguousarray(_vio_Phi__) @ np.ascontiguousarray(_vio_state_cov__[:21, :21]) @ np.ascontiguousarray(_vio_Phi__.T) + np.ascontiguousarray(_vio_Q__))
    return _vio_state_cov__

@jit(nopython = True)
def _state_augmentation(_vio_R_i_c__, _vio_R_w_i__, _vio_t_c_i__, _vio_state_cov__, _vio_stateCovShape__):
    # Update the covariance matrix of the state.
    # To simplify computation, the matrix _vio_J__ below is the nontrivial block
    # in Equation (16) of "MSCKF" paper.
    _vio_J__ = np.zeros((6, 21))
    _vio_J__[:3, :3] = _vio_R_i_c__
    _vio_J__[:3, 15:18] = np.identity(3)
    _vio_t_w_i__  = np.ascontiguousarray(_vio_R_w_i__.T) @ np.ascontiguousarray(_vio_t_c_i__)

    _vio_x__, _vio_y__, _vio_z__ = _vio_t_w_i__
    _vio_t_w_iSkew__ = np.array([
        [0, -_vio_z__, _vio_y__],
        [_vio_z__, 0, -_vio_x__],
        [-_vio_y__, _vio_x__, 0]])

    _vio_J__[3:6, :3] = _vio_t_w_iSkew__
    _vio_J__[3:6, 12:15] = np.identity(3)
    _vio_J__[3:6, 18:21] = np.identity(3)

    # Resize the state covariance matrix.
    # old_rows, old_cols = self.state_server._vio_state_cov__.shape
    _vio_old_size__ =  _vio_stateCovShape__  # symmetric
    _vio_state_covNew__ = np.zeros((_vio_old_size__+6, _vio_old_size__+6))
    _vio_state_covNew__[:_vio_old_size__, :_vio_old_size__] = _vio_state_cov__

    # Fill in the augmented state covariance.
    _vio_state_covNew__[_vio_old_size__:, :_vio_old_size__] = np.ascontiguousarray(_vio_J__) @ np.ascontiguousarray(_vio_state_covNew__[:21, :_vio_old_size__])
    _vio_state_covNew__[:_vio_old_size__, _vio_old_size__:] = _vio_state_covNew__[_vio_old_size__:, :_vio_old_size__].T
    _vio_state_covNew__[_vio_old_size__:, _vio_old_size__:] = np.ascontiguousarray(_vio_J__) @ np.ascontiguousarray(_vio_state_covNew__[:21, :21]) @ np.ascontiguousarray(_vio_J__.T)
    return _vio_state_covNew__

@jit(nopython = True)
def _fastSVD(A):
    return np.linalg.svd(A)

@jit(nopython = True)
def _fastQR(A):
    return np.linalg.qr(A)

@jit(nopython = True)
def _fastSolve(A, b):
    return np.linalg.solve(A, b)

@jit(nopython = True)
def _fastNorm(A):
    return np.linalg.norm(A)

@jit(nopython = True)
def _fastInv(A):
    return np.linalg.inv(A)
