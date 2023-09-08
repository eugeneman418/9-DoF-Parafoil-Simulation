import numpy as np
import numpy.linalg as la

def frame_change(roll, pitch, yaw):
    """
    Frame of reference transformation from ground frame to payload/parafoil frame.
    roll = phi
    pitch = theta
    yaw = psi

    Parameters
    ----------
    roll : TYPE
        DESCRIPTION.
    pitch : TYPE
        DESCRIPTION.
    yaw : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #roll = phi
    #pitch = theta
    #yaw = psi
    return np.array([
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)],
        [np.sin(roll) * np.sin(pitch)* np.cos(yaw) - np.cos(roll) * np.sin(yaw), np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw), np.cos(pitch) * np.sin(yaw)],
        [np.cos(roll) * np.sin(pitch)* np.cos(yaw) + np.sin(roll) * np.sin(yaw), np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw), np.cos(roll)* np.cos(pitch)]
        ])