# -*- coding: utf-8 -*-
"""

Implemetation of nonlinear functionality from the "Nonlinear Flight Dynamics"
notebook

"""


import numpy as np
from scipy.optimize import fsolve

RHO_SL_SLUG_FT3 = 0.00238
G_FT_S2 = 32.174

def eom_6dof(time, state, aircraft, output_mode = False, rho = RHO_SL_SLUG_FT3, g = G_FT_S2):
    """
    Implement the 6DOF aircraft equations of motion assuming flat Earth and constant density.

    Parameters
    ==========

    time : float
        Current time, supplied for purposes of computing aircraft mass property and aerodynamic information
    state : array
        State vector with order [u, v, w, p, q, r, phi, theta, psi, xe, ye, ze] corresponding to four sets
        of equations: Body-frame translational, body-frame rotational, rotational kinematic, 
        and translational kinematic
    aircraft : object
        An aircraft class that returns its mass properties, thrust forces, and aerodynamic coefficients as
        functions of the time and state vector via the methods "get_mass_props," "get_thrust," and 
        "get_aero_coeffs" respectively
    output_mode : bool, optional
        Flag to return selected outputs rather than the state vector time derivativel, default is False
    rho : float, optional
        At-altitude density. The default is a sea level value of 0.00238 slug/ft^3; for SI units change 
        to 1.225 kg/m^3
    g : float, optional
        Gravitational acceleration. The default is a standard value of 32.17 ft/s^2; for SI units change
        to 9.81 m/s

    Returns
    =======
    state_dot : array
        State derivative time rate of change. Returned when output_mode = False
    outputs : array
        Miscallaneous outputs to support additional results or debugging; returned when output_mode = True
    """

    # Unpack states
    u, v, w, p, q, r, phi, theta, psi, xe, ye, ze = state

    # Get mass properties and aerodynamic forces
    mass, Ixx, Iyy, Izz, Ixz = aircraft.get_mass_props(time, state)
    Tx, Tz = aircraft.get_thrust(time, state)
    Cx, Cy, Cz, Cl, Cm, Cn = aircraft.get_aero_coeffs(time, state)

    # Dimensionalize
    qinf = 0.5*rho*(u*u + v*v + w*w)
    Fx = qinf*aircraft.sref*Cx
    Fy = qinf*aircraft.sref*Cy
    Fz = qinf*aircraft.sref*Cz
    L = qinf*aircraft.sref*aircraft.bref*Cl
    M = qinf*aircraft.sref*aircraft.cref*Cm
    N = qinf*aircraft.sref*aircraft.bref*Cn

    # Some trig shorthand
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)
    sin_psi = np.sin(psi)
    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    cos_psi = np.cos(psi)
    tan_theta = sin_theta/cos_theta

    # Translational equations
    udot = r*v - q*w - g*sin_theta + (Fx + Tx)/mass
    vdot = p*w - r*u + g*cos_theta*sin_phi + Fy/mass
    wdot = q*u - p*v + g*cos_theta*cos_phi + (Fz + Tz)/mass

    # Rotational equations; the solution for pdot and rdot coupled by Ixz is obtained
    # from a 2x2 matrix inversion
    qdot = (M - p*r*(Ixx - Izz) - (p*p - r*r)*Ixz)/Iyy
    pdot, rdot = np.linalg.solve([[Ixx, -Ixz], [-Ixz, Izz]], 
                                 [L - q*r*(Izz - Iyy) + q*p*Ixz,
                                  N - p*q*(Iyy - Ixx) - q*r*Ixz])

    # Rotational kinematics
    phidot = p + tan_theta*(q*sin_phi + r*cos_phi)
    thetadot = q*cos_phi - r*sin_phi
    psidot = (q*sin_phi + r*cos_phi)/cos_theta

    # Translational kinematics
    xedot = (u*cos_psi*cos_theta + v*(cos_psi*sin_theta*sin_phi - sin_psi*cos_phi)
             + w*(cos_psi*sin_theta*cos_phi + sin_psi*sin_phi))
    yedot = (u*sin_psi*cos_theta + v*(sin_psi*sin_theta*sin_phi + cos_psi*cos_phi)
             + w*(sin_psi*sin_theta*cos_phi - cos_psi*sin_phi))
    zedot = -u*sin_theta + v*cos_theta*sin_phi + w*cos_theta*cos_phi

    if output_mode:
        
        V = np.sqrt(u*u + v*v + w*w)
        beta = np.arcsin(v/V)
        alpha = np.arcsin(w/V)
        
        ax = Fx/(mass*g)
        ay = Fy/(mass*g)
        az = Fz/(mass*g)
        
        roll_control = aircraft.roll_control(time, state)
        pitch_control = aircraft.pitch_control(time, state)
        yaw_control = aircraft.yaw_control(time, state)
        return np.array([V, alpha, beta, 
                         p, q, r, 
                         phi, theta, psi,
                         xe, ye, ze, 
                         ax, ay, az,
                         roll_control, pitch_control, yaw_control])
    else:
        return np.array([udot, vdot, wdot, 
                         pdot, qdot, rdot,
                         phidot, thetadot, psidot,
                         xedot, yedot, zedot])
                                      
    

class AircraftLinear:

    def __init__(self):

        self.mass = None
        self.Ixx = None
        self.Iyy = None
        self.Izz = None
        self.Ixz = None

        self.sref = None
        self.bref = None
        self.cref = None

        self.Tx = 0.
        self.Tz = 0.

        self.Cx0 = 0.
        self.Cz0 = 0.
        self.Cm0 = 0.

        self.Cxa = 0.
        self.Cza = 0.
        self.Cma = 0.

        self.Cxdp = 0.
        self.Czdp = 0.
        self.Cmdp = 0.

        self.Cmq = 0.

        self.Cyb = 0.
        self.Clb = 0.
        self.Cnb = 0.

        self.Cyda = 0.
        self.Clda = 0.
        self.Cnda = 0.

        self.Cydr = 0.
        self.Cldr = 0.
        self.Cndr = 0.

        self.Clp = 0.
        self.Cnp = 0.

        self.Clr = 0.
        self.Cnr = 0.

        self.roll_control = lambda time, state: 0.
        self.pitch_control = lambda time, state: 0.
        self.yaw_control = lambda time, state: 0.

    def set_mass_props(self, mass, Ixx, Iyy, Izz, Ixz = 0.):

        self.mass = mass
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.Ixz = Ixz

    def set_ref_dims(self, sref, bref, cref):

        self.sref = sref
        self.bref = bref
        self.cref = cref

    def set_thrust(self, Tx, Tz = 0.):

        self.Tx = Tx
        self.Tz = Tz

    def set_aero_coeffs(self, **kwargs):

        self.Cx0 = kwargs.get('Cx0', 0.)
        self.Cz0 = kwargs.get('Cz0', 0.)
        self.Cm0 = kwargs.get('Cm0', 0.)

        self.Cxa = kwargs.get('Cxa', 0.)
        self.Cza = kwargs.get('Cza', 0.)
        self.Cma = kwargs.get('Cma', 0.)

        self.Czq = kwargs.get('Czq', 0.)
        self.Cmq = kwargs.get('Cmq', 0.)

        self.Cxdp = kwargs.get('Cxdp', 0.)
        self.Czdp = kwargs.get('Czdp', 0.)
        self.Cmdp = kwargs.get('Cmdp', 0.)

        self.Cyb = kwargs.get('Cyb', 0.)
        self.Clb = kwargs.get('Clb', 0.)
        self.Cnb = kwargs.get('Cnb', 0.)

        self.Cyda = kwargs.get('Cyda', 0.)
        self.Clda = kwargs.get('Clda', 0.)
        self.Cnda = kwargs.get('Cnda', 0.)

        self.Cydr = kwargs.get('Cydr', 0.)
        self.Cldr = kwargs.get('Cldr', 0.)
        self.Cndr = kwargs.get('Cndr', 0.)

        self.Clp = kwargs.get('Clp', 0.)
        self.Cnp = kwargs.get('Cnp', 0.)

        self.Clr = kwargs.get('Clr', 0.)
        self.Cnr = kwargs.get('Cnr', 0.)
                              

    def set_aero_controls(self, 
                          roll = lambda time, state: 0.,
                          pitch = lambda time, state: 0.,
                          yaw = lambda time, state: 0.):

        self.roll_control = roll
        self.pitch_control = pitch
        self.yaw_control = yaw

    def get_mass_props(self, time, state):

        return self.mass, self.Ixx, self.Iyy, self.Izz, self.Ixz

    def get_thrust(self, time, state):

        return self.Tx, self.Tz

    def get_aero_coeffs(self, time, state):

        # Get control values
        delta_roll = self.roll_control(time, state)
        delta_pitch = self.pitch_control(time, state)
        delta_yaw = self.yaw_control(time, state)

        # Get aerodynamic angles from the state vector
        u, v, w = state[:3]
        V = np.sqrt(u*u + v*v + w*w)
        beta = np.arcsin(v/V)
        alpha = np.arcsin(w/V)

        # Build up coefficients and return the array
        p, q, r = state[3:6]
        Cx = self.Cx0 + alpha*self.Cxa + delta_pitch*self.Cxdp
        Cy = beta*self.Cyb + delta_roll*self.Cyda + delta_yaw*self.Cydr
        Cz = self.Cz0 + alpha*self.Cza + delta_pitch*self.Czdp

        Cl = beta*self.Clb + delta_roll*self.Clda + delta_yaw*self.Cldr + (p*self.Clp + r*self.Clr)*self.bref/(2*u)
        Cm = self.Cm0 + alpha*self.Cma + delta_pitch*self.Cmdp + q*self.Cmq*self.cref/(2*u)
        Cn = beta*self.Cnb + delta_roll*self.Cnda + delta_yaw*self.Cndr + (p*self.Cnp + r*self.Cnr)*self.bref/(2*u)

        return Cx, Cy, Cz, Cl, Cm, Cn


def get_initial_state(airspeed, alpha, gamma = 0.0):
    """
    Returns aircraft state vector for straight flight based on input airspeed, angle of attack, and optionall flight path angle

    Parameters
    ----------
    airspeed : float
        True airspeed for trim
    trim_alpha : float
        Angle of attack in radians
    gamma : float, optional
        Initial flight path angle in radians. Default is 0.0

    Returns
    -------
    state : array
        6DOF state vector of (speeds, rates, angles, earth-coordinates)
    """

    theta = alpha + gamma
    return np.array([airspeed*np.cos(alpha), 0., airspeed*np.sin(alpha), # u, v, w
                     0., 0., 0.,                                    # p, q, r
                     0., theta, 0.,                                 # phi, theta, psi
                     0., 0., 0.])                                   # Xe, Ye, Ze 

def get_trim(aircraft, airspeed, rho = RHO_SL_SLUG_FT3, gamma = 0.0):
    """
    Obtain steady, level trim solution for a symmetric aircraft using pitch and thrust control. 

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft object which implements "get_thrust," "get_mass_props," and 
        "get_aero_coeffs" properties
    airspeed : float
        True airspeed for trim
    rho : float, optional
        Atmospheric density at trim altitude. Default value is sea level in slug/ft^3
    gamma : float, optional
        Initial flight path angle in radians. Default is 0.0

    Returns
    -------
    trim_alpha : float
        Trim angle of attack in radians
    trim_pitch : float
        Trim pitch command in radians
    trim_throttle : foat
        Trim throttle command, as a fraction of the return call to "get_thrust" 
    """

    # Due to the way controls are called, want to grab these placeholders to reset for lader
    roll0 = aircraft.roll_control
    pitch0 = aircraft.pitch_control
    yaw0 = aircraft.yaw_control
    Tx0 = aircraft.Tx
    Tz0 = aircraft.Tz


    def obj_fun(trim_params):

        alpha, delta_pitch, thrust_frac = trim_params

        Xtrim = get_initial_state(airspeed, alpha, gamma = gamma)
        
        aircraft.set_aero_controls(roll = lambda time, state: 0.,
                                   pitch = lambda time, state: delta_pitch,
                                   yaw = lambda time, state: 0.)
        aircraft.set_thrust(thrust_frac*Tx0, thrust_frac*Tz0)
    
        states_dot = eom_6dof(0., Xtrim, aircraft, rho = RHO_SL_SLUG_FT3)

        return states_dot[[0, 2, 4]]

    xout = fsolve(obj_fun, [0., 0., 1.])    

    aircraft.set_aero_controls(roll = roll0, pitch = pitch0, yaw = yaw0)
    aircraft.set_thrust(Tx0, Tz0)

    return xout

# Define an Aircraft for external use
def GET_F106():
    
    mass = 1000.
    weight = G_FT_S2*mass
    sref = 695
    bref = 38.13
    cref = 23.76
    
    alpha0 = 11.5/180*np.pi   # Documented
    Cx0 = -0.02
    Cz0 = -0.05
    Cxa = -0.6
    Cza = -2.80
    Cma = -0.16
    Cm0 = -0.05
    Cmdp = -0.30
    
    Czdp = -0.56
    Cxdp = Cxa*(Czdp/Cza)  # Scaled estimate
    
    # Trim aircraft at set alpha0
    # Fx = qinf*Sref*(Cx0 + Cxa*alpha + Cxdp*delta) + Tx - W*sin(theta0) = 0
    # Fz = qinf*Sref*(Cz0 + Cza*alpha + Czdp*delta) + W*cos(theta) = 0
    # My = qinf*Sref*cref*(Cm0 + Cma*alpha + Cmdp*delta) = 0
    # With alpha0 fixed, solve for delta, Tx, and Cm0 as missing info:
    #
    # qinf*sref*Cxdp*delta + Tx = -qinf*sref*(Cx0 + Cxa*alpha) + W*sin(theta0)
    # qinf*sref*Czdp*delta = -qinf*sref*(Cz0 + Cza*alpha) - W*cos(theta)
    # qinf*sref*cref*(Cm0 + Cmdp*delta) = -qinf*sref*cref*Cma*alpha
    #
    CL0 = -0.316
    qinf = -weight/(sref*CL0)
    
    theta0 = alpha0
    A = np.array([[qinf*sref*Cxdp, 1.0, 0.0], 
                  [qinf*sref*Czdp, 0.0, 0.0 ],
                  [qinf*sref*cref*Cmdp, 0.0, qinf*sref*cref]])
    
    b = np.array([-qinf*sref*(Cx0 + Cxa*alpha0) + weight*np.sin(theta0),
                  -qinf*sref*(Cz0 + Cza*alpha0) - weight*np.cos(theta0),
                 -qinf*sref*cref*Cma*alpha0])
    
    delta0, Tx0, Cm0 = np.linalg.solve(A, b)

    # sin_alpha = np.sin(alpha0)
    # cos_alpha = np.cos(alpha0)
    
    
    # CZ0 = Cz0 + Cza*alpha0 + Czdp*delta0
    # CX0 = Cxa*alpha0 + Cxdp*delta0
    # CL0 = -CZ0*cos_alpha + CX0*sin_alpha
    # CD0 = -CX0*cos_alpha - CZ0*sin_alpha

    # Lift = qinf*sref*CL ==> qinf = weight/(CL*sref)
    
    # TODO: Get rid of this
    airspeed = np.sqrt(2*qinf/RHO_SL_SLUG_FT3)

    f106 = AircraftLinear()
    f106.set_mass_props(32170/G_FT_S2, 19000, 185000, 200000, Ixz = 6000)
    f106.set_ref_dims(sref, bref, cref)
    f106.set_thrust(Tx0)
    
    f106.set_aero_coeffs(Cx0 = Cx0,
                         Cz0 = Cz0,
                         Cm0 = Cm0,
                         Cxa = Cxa,
                         Cza = Cza,
                         Cma = Cma,
                         Cxdp = Cxdp,
                         Czdp = Czdp,
                         Cmdp = Cmdp,
                         Cyb = -0.50,
                         Clb = -0.079,
                         Cnb = 0.069,
                         Cyda = 0.23,
                         Clda = 0.087,
                         Cnda = -0.055,
                         Cydr = 0.068,
                         Cldr = 0.015,
                         Cndr = -0.057,
                         Clp = -0.179,
                         Cnp = 0.0079,
                         Clr = -0.084, 
                         Cnr = -0.39)
    
    return f106, airspeed

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # Can we get the same trim solution using the function
    f106, veas = GET_F106()
    alpha0_rad, delta0_rad, throttle0 = get_trim(f106, veas)
    
    print('Trim alpha [function] %.1f deg' % (alpha0_rad*57.3))
    print('Trim delta [function] %.1f deg' % (delta0_rad*57.3))
    print('Trim thrust fraction [function] %.3f' % (throttle0))
    
    trim_X0 = get_initial_state(veas, alpha0_rad)
    
    
    # Output timesteps
    dt = 0.01
    T = 10.0
    time = np.arange(0, T, dt)
    
    #################
    # Pitch Doublet #
    #################
    pitch_start = 0.5
    Tpitch = 2.0
    dpitch = 2.5/57.3
    pitch_history = lambda t, x: delta0_rad - dpitch*np.sin(np.pi*(t - pitch_start)/Tpitch)*(t > pitch_start)*(t < (pitch_start + Tpitch))
    
    f106.set_aero_controls(pitch = pitch_history)
    
    eom = lambda t, x: eom_6dof(t, x, f106)
    sol = solve_ivp(eom, [0, T], trim_X0, t_eval = time)
    u, v, w, p, q, r, phi, theta, psi, Xe, Ye, Ze = sol.y
    outputs = np.array([eom_6dof(ti, yi, f106, output_mode = True) for ti, yi in zip(time, sol.y.T)]).T
    V, alpha, beta, p, q, r, phi, theta, psi, xe, ye, ze, ax, ay, az, roll_control, pitch_control, yaw_control = outputs
    
    lstyle = 'k-'
    fig, ax = plt.subplot_mosaic([['image', 'alpha [rad]'],
                                  ['image', 'q [rad/s]',],
                                  ['image', 'az [g]'],
                                  ['placehold', 'Pitch [rad]']],
                                figsize = (10, 10), 
                                constrained_layout = True)
    
    
    ax['alpha [rad]'].plot(sol.t, alpha, lstyle)
    ax['q [rad/s]'].plot(sol.t, q, lstyle)
    ax['az [g]'].plot(sol.t, az, lstyle)
    ax['Pitch [rad]'].plot(sol.t, pitch_control, lstyle)
    
    
    for key, a in ax.items():
        if key == 'image':
            continue
        a.set_xlabel('Time [s]', fontweight = 'bold')
        a.set_ylabel(key, fontweight = 'bold')
        a.grid()
    
    # Read & plot image + add annotations
    image = plt.imread('images/f106_pitch_doublet.png')
    ax['image'].imshow(image)
    ax['image'].axis('off')
    ax['placehold'].axis('off')
    
    fig.suptitle('Simulated vs. As-Flown F106 Pitch Doublet Time Histories; M0.6 ~11 Deg AoA', fontweight = 'bold') 
    fig.savefig('plots/f106_pitch_doublet.png', bbox_inches = 'tight')
    
    ###############
    # Rudder Kick #
    ###############
    rudder_start = 0.5
    Trudder = 1.0
    drudder = 20/57.3
    yaw_history = lambda t, x: drudder*np.sin(np.pi*(t - rudder_start)/Trudder)*(t > rudder_start)*(t < (rudder_start + Trudder))
    pitch_history = lambda t, x: delta0_rad
    
    f106.set_aero_controls(pitch = pitch_history, 
                           yaw = yaw_history)
    
    eom = lambda t, x: eom_6dof(t, x, f106)
    
    sol = solve_ivp(eom, [0, T], trim_X0, t_eval = time)
    u, v, w, p, q, r, phi, theta, psi, Xe, Ye, Ze = sol.y
    outputs = np.array([eom_6dof(ti, yi, f106, output_mode = True) for ti, yi in zip(time, sol.y.T)]).T
    V, alpha, beta, p, q, r, phi, theta, psi, xe, ye, ze,  ax, ay, az, roll_control, pitch_control, yaw_control = outputs
    
    lstyle = 'k-'
    fig, ax = plt.subplot_mosaic([['image', 'beta [rad]'],
                                  ['image', 'p [rad/s]',],
                                  ['image', 'r [rad/s]', ],
                                  ['image', 'ay [g]'],
                                  ['placehold', 'Rudder [rad]']],
                                figsize = (10, 10), 
                                constrained_layout = True)
    
    
    ax['beta [rad]'].plot(sol.t, beta, lstyle)
    ax['p [rad/s]'].plot(sol.t, p, lstyle)
    ax['r [rad/s]'].plot(sol.t, r, lstyle)
    ax['ay [g]'].plot(sol.t, ay, lstyle)
    ax['Rudder [rad]'].plot(sol.t, yaw_control, lstyle)
    
    
    for key, a in ax.items():
        if key == 'image':
            continue
        a.set_xlabel('Time [s]', fontweight = 'bold')
        a.set_ylabel(key, fontweight = 'bold')
        a.grid()
    
    # Read & plot image + add annotations
    image = plt.imread('images/f106_yaw_doublet.png')
    ax['image'].imshow(image)
    ax['image'].axis('off')
    ax['placehold'].axis('off')
    
    fig.suptitle('Simulated vs. As-Flown F106 Rudder Kick Time Histories; M0.6 ~11 Deg AoA', fontweight = 'bold') 
    fig.savefig('plots/f106_yaw_doublet.png', bbox_inches = 'tight')
    
    
    


