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

def body_to_parafoil(rigging):
    """
    Generates the change of frame matrix from body to parafoil frame
    rigging: rigging angle mu
    """
    return np.array([
        [np.cos(rigging), 0, -np.sin(rigging)],
        [0, 1, 0],
        [np.sin(rigging), 0, np.cos(rigging)],
    ])

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

def apparent_mass(density, arc, span, chord, thickness, rigging):
    aspect = span / chord
    arc_to_span = arc / span

    relative_thickness = thickness / chord

    A = 0.666 * density * (1 + 8 / 3 * arc_to_span ** 2) * thickness ** 2 * span
    B = 0.267 * density * (1 + 2 * (arc_to_span / relative_thickness) ** 2 * aspect ** 2 * (
                1 - relative_thickness ** 2)) * thickness ** 2 * chord
    C = 0.785 * density * np.sqrt(1 + 2 * arc_to_span ** 2 * (1 - relative_thickness ** 2)) * aspect / (
                1 + aspect) * chord ** 2 * span
    return body_to_parafoil(rigging).T @ np.diag([A, B, C]) @ body_to_parafoil(rigging)

def mass_matrix(mass, density, span, chord):
    """matrix capturing mass of payload, canopy, entrapped air"""
    inflation = 0.09 * density * span * chord**2 #mass entrapped air
    return (mass + inflation) * np.eye(3)

def apparent_inertia(density, arc, span, chord, thickness, rigging):
    """
    Generate apparent moment of inertia matrix in body frame
    """
    R = body_to_parafoil(rigging)
    aspect_ratio = span/chord
    arc_span_ratio = arc/span
    relative_thickness = thickness/chord
    Ia = 0.055 * density * aspect_ratio/(1 + aspect_ratio) * chord**2 * span**3
    Ib = 0.0308 * density * aspect_ratio/(1 + aspect_ratio) * (1 + np.pi/6 * (1 + aspect_ratio) * aspect_ratio * (arc_span_ratio * relative_thickness)**2 ) * chord**4 * span
    Ic = 0.0555 * density * (1 + 8 * arc_span_ratio**2) * thickness**2 * span**3
    return R.T @ np.diag([Ia, Ib, Ic]) @ R


def skew_symmetric(vector):
    """generate skew symmetric matrix from a 3D vector"""
    p,q,r = tuple(vector)
    return np.array([
        [0, -r, q],
        [r, 0, -p],
        [-q, p, 0],
    ])

def weight_force(angles, mass, g=9.81):
    row, pitch, yaw = tuple(angles)
    return geo_to_body(row,pitch,yaw) @ (-mass * g * np.array([0,0,1]))

def dynamic_pressure(airspeed, density):
    """
    Computes the dynamic pressure Q
    airspeed: airspeed Va
    density: rho
    """
    return density/2 * airspeed**2

def airspeed(linear_velocity, angles, wind, return_manitude=True):
    """
    Computes the airspeed vector Va
    linear_velocity: 3D vector of the linear veolicty states (u,v,w)
    angles: 3D vector of roll, pitch, yaw in body frame
    wind: wind velocity vector (3D) in geographical frame
    """
    roll, pitch, yaw = tuple(angles)
    if return_manitude:
        return np.linalg.norm(linear_velocity - geo_to_body(roll, pitch, yaw) @ wind, ord=2)
    else:
        return linear_velocity - geo_to_body(roll, pitch, yaw) @ wind

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

def euler_rate_to_anguler(phi, theta, ang_v):
    return np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ]) @ ang_v

m = 2.4
a = 0.1
b = 1.35
c = 0.75
t = 0.075
S = 1
rho = 1.293
mu = -12 * np.pi /180
wind = np.array([0,0,0])
I = np.array([[0.42,0,0.03], [0,0.4,0], [0.03,0,0.053]])
r_bm = np.array([0.046,0,-1.11])
C_D0 = 0.25
C_Dalpha2 = 0.12
C_Ybeta = -0.23
C_L0 = 0.091
C_Lalpha = 0.9
C_m0 = 0.35
C_malpha = -0.72
C_mq = -1.49
C_lbeta = -0.036
C_lp = -0.84
C_lr = -0.082
C_lasym = -0.0035
C_nbeta = -0.0015
C_np = -0.082
C_nr = -0.27
C_nasym = 0.0115
alpha = 5*np.pi/180
beta = 2 * np.pi/180


def dynamics(time, state):
    u,v,w, p,q,r, x,y,z, phi,theta,psi = tuple(state)

    lin_v = np.array([u,v,w])
    ang_v = np.array([p,q,r])
    position = np.array([x,y,z])
    angle = np.array([phi, theta, psi])

    Iam = apparent_mass(rho, a, b, c, t, mu)
    Iai = apparent_inertia(rho, a, b, c, t, mu)
    total_mass = mass_matrix(m, rho, b,c) + Iam
    Cross_rbm = skew_symmetric(r_bm)
    Cross_omega = skew_symmetric(ang_v)

    Va = airspeed(lin_v, angle,wind)
    Q = dynamic_pressure(Va, rho)
    wind_force = Cross_omega @ Iam @ geo_to_body(phi,theta,psi) @ wind
    B1 = aerodynamic_force(Q, alpha,beta,S,0,C_D0,C_L0,C_Dalpha2,C_Lalpha,C_Ybeta, 0,0) + weight_force(angle, m) - Cross_omega @ total_mass @ lin_v + Cross_omega @ Iam @ Cross_rbm @ ang_v + wind_force
    B2 = aerodynamic_moment(ang_v,Q,S,alpha,beta,Va,0,b,c,C_lbeta,C_lp,C_lr,C_lasym,C_m0,C_malpha, C_mq, C_nbeta, C_np,C_nr, C_nasym) - (Cross_omega @ (I + Iai) - Cross_rbm @ Cross_omega @ Iam @ Cross_rbm) @ ang_v - Cross_rbm @ Cross_omega @ Iam @ lin_v + Cross_rbm @ wind_force
    B = np.concatenate((B1,B2))
    A = np.block([
        [total_mass, -Iam @ Cross_rbm],
        [Cross_rbm @ Iam, I+Iai - Cross_rbm @ Iam @ Cross_rbm]])


    velocity = geo_to_body(phi, theta, psi).T @ lin_v
    return np.concatenate((np.linalg.inv(A) @ B, velocity, euler_rate_to_anguler(phi, theta, ang_v)))

def simplified_dynamics(time, state):
    u,v,w, p,q,r, x,y,z, phi,theta,psi = tuple(state)

    lin_v = np.array([u,v,w])
    ang_v = np.array([p,q,r])
    angle = np.array([phi, theta, psi])

    total_mass = mass_matrix(m, rho, b,c)
    Cross_omega = skew_symmetric(ang_v)

    Va = airspeed(lin_v, angle,wind)
    Q = dynamic_pressure(Va, rho)
    B1 = weight_force(angle, m) - Cross_omega @ total_mass @ lin_v
    B2 = aerodynamic_moment(ang_v,Q,S,alpha,beta,Va,0,b,c,C_lbeta,C_lp,C_lr,C_lasym,C_m0,C_malpha, C_mq, C_nbeta, C_np,C_nr, C_nasym) - Cross_omega @ I @ ang_v
    B = np.concatenate((B1,B2))
    A = np.block([
        [total_mass, np.zeros((3,3))],
        [np.zeros((3,3)), I]])


    velocity = geo_to_body(phi, theta, psi).T @ lin_v
    return np.concatenate((np.linalg.inv(A) @ B, velocity, euler_rate_to_anguler(phi, theta, ang_v)))



results = sp.integrate.solve_ivp(dynamics, [0,90], np.array([0,0,10,0,0,0,0,0,1000,0.0,-np.pi,0.0]))
path = results.y[-6:-3,:]
ax = plt.figure().add_subplot(projection='3d')

# Prepare arrays x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
x,y,z = path[0,:], path[1,:], path[2,:]


ax.plot(x, y, z, label='flight path')
ax.legend()

plt.show()

t = np.linspace(0,90,num = len(z))
plt.title("t vs z")
plt.plot(t,  z)
plt.show()