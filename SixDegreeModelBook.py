import numpy as np

def F_g_func(m,g,euler_angle):
    """
    m: mass
    g: gravitational constant
    euler_angle: [phi, theta, psi]

    return: weight force vector
    """
    phi, theta, psi = tuple(euler_angle)
    sin_theta = np.sin(theta)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    return m*g*np.array([-sin_theta, sin_phi * cos_theta, cos_phi * cos_theta])

def F_a_func(Q,S,Rwb, alpha, beta, delta_s, C_D0, C_Dalpha2, C_Ddelta_s, C_Ybeta, C_L0, C_Lalpha, C_Ldelta_s):
    """
    Q: dynamic pressure, not sure about the sign though
    S: canopy area
    Rwb: transformation matrix from wind to body frame
    alpha: angle of attack
    beta: angle of sideslip
    delta_s: symmetric deflection
    """

    return Q * S * Rwb @ np.array([
        C_D0 + C_Dalpha2 * alpha**2 + C_Ddelta_s * delta_s,
        C_Ybeta * beta,
        C_L0 + C_Lalpha * alpha + C_Ldelta_s * delta_s
    ])

def Rwb_func(alpha, beta):
    """
    alpha: angle of attack
    beta: angle of sideslip

    return: transformation matrix from wind to body frame
    """
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    sin_alpha = np.sin(alpha)
    sin_beta = np.sin(beta)
    return np.array([
        [cos_alpha*cos_beta, cos_alpha*sin_beta, -sin_alpha],
        [-sin_beta, cos_beta, 0],
        [sin_alpha*cos_beta, sin_alpha*sin_beta, cos_alpha]
    ])

def alpha_func(Va):
    """
    Va: airspeed vector

    return: angle of attack
    """
    vx, vy, vz = tuple(Va)
    return  np.arctan2(vz,vx)

def beta_func(Va):
    """
    Va: airspeed vector

    return: angle of attack
    """
    vx, vy, vz = tuple(Va)
    return np.arctan2(vy, np.sqrt(vx**2 + vz**2))

def Va_func(body_frame_velocity, Rnb, wind):
    """
    body_frame_velocity: [u,v,w]
    Rnb: transformation matrix from inertial to body frame
    wind: wind velocity vector in inertial frame
    """
    return body_frame_velocity - Rnb @ wind

def Ma_func(rho, Va, S, b, c, alpha, beta, delta_a, body_frame_angular_velocity, C_lbeta, C_lp, C_lr, C_ldelta_a, C_m0, C_malpha, C_mq, C_nbeta, C_np, C_nr, C_ndelta_a):
    """
    rho: air density
    Va: airspeed vector
    S: canopy area
    b: span
    c: chord
    alpha: angle of attack
    beta: angle of sideslip
    delta_a: asymmetric deflection
    body_frame_angular_velocity
    
    return: aerodynamic moment
    """
    Va = np.linalg.norm(Va)
    p,q,r = tuple(body_frame_angular_velocity)
    return 0.5 * rho * Va**2 * S * np.array([
        b * (C_lbeta * beta + 0.5*b/Va * C_lp * p + 0.5 * b/Va * C_lr * r + C_ldelta_a * delta_a),
        c * (C_m0 + C_malpha * alpha + 0.5*c/Va * C_mq * q),
        b * (C_nbeta * beta + 0.5 * b / Va * C_np * p + 0.5 * b/Va * C_nr * r + C_ndelta_a * delta_a)
    ])

def Rnb_func(euler_angle):
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

def Rbp_func(mu):
    """
    mu: rigging angle/canopy incidence angle

    return: transformation matrix from body to canopy frame
    """
    cos_mu = np.cos(mu)
    sin_mu = np.sin(mu)
    return np.array([
        [cos_mu, 0 , -sin_mu],
        [0, 1, 0],
        [sin_mu, 0, cos_mu]
    ])

def canopy_frame_angular_velocity_func(Rbp, body_frame_angular_velocity):
    """
    Rbp: transformation matrix from body to canopy frame
    body_frame_angular_velocity: p,q,r

    return: tilde [p,q,r]
    """
    return Rbp @ body_frame_angular_velocity

def canopy_frame_angular_acceleration_func(Rbp, body_frame_angular_accleration):
    return Rbp @ body_frame_angular_accleration




