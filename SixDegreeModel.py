import numpy as np

def inertial_velocity_func(T_IB, body_frame_velocity):
    """
    T_IB: the transformation matrix from inertial to body frame
    body_frame_velocity: [u,v,w]

    return: d(x,y,z)/dt = T_IB^T [u,v,w]
    """

    return T_IB.T @ body_frame_velocity

def euler_angle_rate_func(euler_angle, body_frame_angular_velocity):
    """
    euler_angle: [roll (phi), pitch (theta), yaw (psi)]
    body_frame_angular_velocity: [p,q,r]

    return: d(phi, theta, psi)/dt
    """

    phi, theta, psi = tuple(euler_angle)

    sin_phi = np.sin(phi)
    tan_theta = np.tan(theta)
    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)

    A = np.array([
        [1, sin_phi * tan_theta, cos_phi * tan_theta],
        [0, cos_phi, -sin_phi],
        [0, sin_phi/cos_theta, cos_phi/cos_theta]
    ])

    return A @ body_frame_angular_velocity

def T_IB_func(euler_angle):
    """
    euler_angle: [roll (phi), pitch (theta), yaw (psi)]

    return: transformation matrix from inertial to body frame
    """

    phi, theta, psi = tuple(euler_angle)
    cos_theta = np.cos(theta)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    sin_theta = np.sin(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    return np.array([
        [cos_theta*cos_psi, cos_theta*sin_psi, -sin_theta],
        [sin_phi*sin_theta*cos_psi-cos_phi*sin_psi, sin_phi*sin_theta*sin_psi+cos_phi*cos_psi, sin_phi*cos_theta],
        [cos_phi*sin_theta*cos_psi+sin_phi*sin_psi, cos_phi*sin_theta*sin_psi-sin_phi*cos_psi, cos_phi*cos_theta]
    ])

def body_frame_acceleration_func(m, F_W, F_A, F_S, F_AM, S_omega, body_frame_velocity):
    """
    m: total mass
    F_W: weight force
    F_S: payload drag force
    F_AM: apparent mass force
    S_omega: cross product matrix with body frame angular velocity (p,q,r)
    body_frame_velocity: [u,v,w]

    return: d(u,v,w)/dt
    """

    return 1/m * (F_W + F_A + F_S - F_AM) - S_omega @ body_frame_velocity

def body_frame_angular_acceleration_func(I_T, M_A, M_AM, S_CGP, F_A, S_CGC, F_S, S_CGM, F_AM, S_omega, body_frame_angular_velocity):
    """
    I_T: total inertia matrix
    M_A: aerodynamics moment
    M_AM: apparent mass moment
    S_CGP: cross product_matrix with vector from mass to aerodynamic center
    F_A: aerodynamic force
    S_CGC: cross product matrix with vector from mass center to canopy rotation point, though the paper writes S_CGS
    F_S: payload drag force
    S_CGM: cross product matrix with vector from mass center to apparent mass center
    F_AM: apparent mass force
    S_omega: cross product matrix with body frame angular velocity
    body_frame_angular_velocity: p,q r
    """

    return np.linalg.inv(I_T) @ (M_A + M_AM + S_CGP @ F_A + S_CGC @ F_S + S_CGM @ F_AM - S_omega @ I_T @ body_frame_angular_velocity)

def cross_product_matrix(vec):
    x,y,z = tuple(vec)
    return np.array([
        [0,-z,y],
        [z,0,-x],
        [-y,x,0]
    ])

def F_W_func(m,g,euler_angle):
    """
    m: mass
    g: gravitational constant
    euler_angle: [phi, theta, psi]
    """
    phi, theta, psi = tuple(euler_angle)
    sin_theta = np.sin(theta)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    return m*g*np.array([-sin_theta, sin_phi * cos_theta, cos_phi * cos_theta])

def canopy_frame_velocity_func(T_BC, body_frame_velocity, S_omega, delta_xyz_c, delta_xyz_p, T_IB, V_Wind):
    """
    T_BC: transformation from body to canopy frame
    body_frame_velocity: [u,v,w]
    S_omega: cross product matrix with body frame angular velocity
    delta_xyz_c: displacement vector from mass center to canopy rotation point
    delta_xyz_p: displacement vector from mass center to aerodynamic center
    T_IB: transformation matrix from inertial to body frame
    V_Wind: wind vector in inertial frame

    return: velocity in canopy frame
    """

    return T_BC @ (body_frame_velocity + S_omega @ (delta_xyz_c + T_BC.T @ delta_xyz_p) - T_IB @ V_Wind)

def alpha_func(canopy_frame_velocity):
    """
    canopy_frame_velocity: tilde [u,v,w]

    return: angle of attack
    """
    u, v, w = tuple(canopy_frame_velocity)
    return np.arctan2(w,u)

def beta_func(canopy_frame_velocity, V_A):
    """
    canopy_frame_velocity: tilde [u,v,w]
    V_A: aerodynamic speed

    return: angle of sideslip
    """

    u, v, w = tuple(canopy_frame_velocity)
    return np.arcsin(v, V_A)

def V_A_func(canopy_frame_velocity):
    """
    canopy_frame_velocity: tilde [u,v,w]

    return: aerodynamic speed of canopy
    """

    return np.linalg.norm(canopy_frame_velocity)

def T_BC_func(Gamma):
    """
    Gamma: rigging angle/canopy incidence angle

    return: transformation matrix from body to canopy frame
    """
    cos_Gamma = np.cos(Gamma)
    sin_Gamma = np.sin(Gamma)
    return np.array([
        [cos_Gamma, 0 , -sin_Gamma],
        [0, 1, 0],
        [sin_Gamma, 0, cos_Gamma]
    ])

def F_A_func(rho, V_S, S_P, T_AB, alpha, beta, C_D0, C_Dalpha, C_Dalpha2, C_Ybeta, C_L0, C_Lalpha, C_Lalpha3):
    """
    rho: air density
    V_S: aerodynamics velocity of payload
    S_P: canopy surface area
    T_AB: transformation matrix from aerodynamics to body frame
    alpha: angle of attack
    beta: angle os slideslip

    return: aerodynamics force
    """

    return -0.5 * rho * V_S**2 * S_P * T_AB @ np.array([
        C_D0 + C_Dalpha * alpha + C_Dalpha2 * alpha**2,
        C_Ybeta * beta,
        C_L0 + C_Lalpha * alpha + C_Lalpha3 * alpha**3
    ])

def T_AB_func(alpha, beta):
    """
    alpha: angle of attack
    beta: angle of sideslip
    euler_angle: phi, theta, psi

    return: transformation matrix from aerodynamics to body frame

    Note: in the paper there is an inconsistency where T_AB is refered to as
    transformation matrix from the aerodynamis to canopy frame. I think this is a mistake
    because to convert to canopy frame,
    the matrix would probably include the rigging angle/canopy incidence angle.
    It's also weird how the matrix isn't unitary.
    I suspect the paper is wrong, as the euler angle is defined in the ground frame
    Therefore I used the definition from Parafoil Control Authority for Landing on Titian paper
    """
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    sin_alpha = np.sin(alpha)
    sin_beta = np.sin(beta)


    return np.array([
        [cos_alpha*cos_beta, -cos_alpha*sin_beta, -sin_alpha],
        [sin_beta, cos_beta, 0],
        [sin_alpha*cos_beta+sin_phi*sin_psi, -sin_alpha*sin_beta, cos_alpha]
    ])

