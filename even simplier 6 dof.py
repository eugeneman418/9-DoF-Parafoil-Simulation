import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
np.random.seed(0)

def geo_to_body(roll, pitch, yaw):
    """
    geographical to body frame transformation matrix generator
    """
    s_phi = np.sin(roll)
    c_phi = np.cos(roll)
    s_theta = np.sin(pitch)
    c_theta = np.cos(pitch)
    s_psi = np.sin(yaw)
    c_psi = np.cos(yaw)
    return np.array([
        [c_psi * c_theta, s_psi * c_theta, -s_theta],
        [c_psi * s_theta * s_phi - s_psi * c_phi, s_psi * s_theta * s_phi + c_psi * c_phi, c_theta * s_phi],
        [c_psi * s_theta * c_phi + s_psi * s_phi, s_psi * s_theta * c_phi - c_psi * s_phi, c_theta * c_phi],
    ])

def body_to_geo(roll, pitch, yaw):
    return geo_to_body(roll, pitch, yaw).T

def body_to_wind(aoa, aos):
    """
    body to wind frame transformation matrix generator
    aoa: angle of attack
    aos: angle of sideslip
    """
    s_a = np.sin(aoa)
    c_a = np.cos(aoa)
    s_b = np.sin(aos)
    c_b = np.cos(aos)
    return np.array([
        [c_a * c_b, -s_b, s_a * c_b],
        [c_a * s_b, c_b, s_a * s_b],
        [-s_a, 0, c_a],
    ])

def wind_to_body(aoa, aos):
    return body_to_wind(aoa, aos).T

def mass_matrix(mass, density, span, chord):
    """matrix capturing mass of payload, canopy, entrapped air"""
    inflation = 0.09 * density * span * chord**2 #mass entrapped air
    return (mass + inflation) * np.eye(3)

def weight_force(mass_mat, roll, pitch, yaw, g=9.81):
    return geo_to_body(roll, pitch, yaw) @ mass_mat @ np.array([0,0,-g])

def skew_symmetric(vector):
    """generate skew symmetric matrix from a 3D vector"""
    p,q,r = tuple(vector)
    return np.array([
        [0, -r, q],
        [r, 0, -p],
        [-q, p, 0],
    ])

def airspeed(linear_velocity, wind, roll, pitch, yaw, return_manitude=True):
    """
    Computes the airspeed vector Va
    linear_velocity: 3D vector of the linear veolicty states (u,v,w)
    angles: 3D vector of roll, pitch, yaw in body frame
    wind: wind velocity vector (3D) in geographical frame
    """
    if return_manitude:
        return np.linalg.norm(linear_velocity - geo_to_body(roll, pitch, yaw) @ wind, ord=2)
    else:
        return linear_velocity - geo_to_body(roll, pitch, yaw) @ wind

def dynamic_pressure(airspeed, density):
    """
    Computes the dynamic pressure Q
    airspeed: airspeed modulus Va
    density: rho
    """
    return density/2 * airspeed**2

def aerodynamic_force(dynamic_pressure, aoa, aos, surface_area, sym_def, cd0, cl0, cd_aoa_squared, cl_aoa, cy_aos, cd_sym, cl_sym):
    """
    generate the aerodynamaic force vector in body frame
    linear_velocity: 3D vector of the linear veolicty states (u,v,w)
    dynamic_pressure: dynamic pressure Q
    aoa: angle of attack
    aos: angle of sideslip
    surface_area: parafoil surfacec area
    sym_def: symmetric deflection, (left actuation + right actuation)/2
    cd0: steady level drag coefficient
    cl0: steady level lift coefficient
    cd_aoa_squared: the contribution of aoa^2 on cd
    cl_aoa: the contribution of aoa on cl
    cy_aos: some contribution of aos on aerodynamic force in Y direction
    cd_sym: contribution of symmetric deflection on cd
    cl_sym: contribution of symmetric delfection on cl
    """
    R = body_to_wind(aoa, aos).T
    f = np.array([
        cd0 + cd_aoa_squared * aoa**2 + cd_sym * sym_def,
        cy_aos * aos,
        cl0 + cl_aoa * aoa + cl_sym * sym_def,
    ])
    return -dynamic_pressure * surface_area * R @ f

def aerodynamic_moment(angular_v, dynamic_pressure, surface_area, aoa, aos, airspeed, asym_def, span, chord, cl_aos,  cl_p, cl_r, cl_asym, cm0, cm_aoa, cm_q, cn_aos, cn_p, cn_r, cn_asym):
    """
    angular_v: state vectors for angular velocities (p,q,r)
    dynamic_pressure: dynamic pressure Q
    surface_area: surface area of parafoil
    aoa: angle of attack alpha
    aos: angle of sideslide beta
    airspeed: magnitude of airspeed vector Va
    asym_def: asymmetric deflection (right actuation - left actuation)
    span: span/length of parafoil b
    chord: chord/width c
    cl_aos: contribution of aos to aerodynamic moment in l direction
    cl_p: contribution of p to aerodynamic moment in l direction
    cl_r: contribution of r to aerodynamic moment in l direction
    cl_asym: contribution of asym_def to aerodynamic moment in l direction
    cm: base aerodynamic moment in m direction
    cm_aoa: contribution of aoa to aerodynamic moment in m direction
    cm_q: contribution of q to aerodynamic moment in m direction
    cn_aos: contribution of aos to aerodynamic moment in n direction
    cn_p: contribution of p to aerodynamic moment in n direction
    cn_r: contribution of r to aerodynamic moment in n direction
    cn_asym: contribution of asym_def to aerodynamic moment in n direction
    """
    p, q, r = tuple(angular_v)
    return dynamic_pressure * surface_area * np.array([
            span * (cl_aos * aos + span/(2 * airspeed) * cl_p * p + span/(2 * airspeed) * cl_r * r + cl_asym * asym_def),
            chord * (cm0 + cm_aoa * aoa + chord/(2 * airspeed) * cm_q * q),
            span * (cn_aos * aos + span/(2 * airspeed) * cn_p * p + span/(2 * airspeed) * cn_r * r + cn_asym * asym_def),
        ])

