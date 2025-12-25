# -*- coding: utf-8 -*-
"""

Helper functions for linearization of the longitudinal and lateral/directional
equations of rigid body aircraft dynamics.

"""

import numpy as np
from scipy.signal import lti

from nonlinear_flight_dynamics import GET_F106, get_trim, get_initial_state, eom_6dof, RHO_SL_SLUG_FT3, G_FT_S2

RAD_DEG = 180/np.pi



def get_linearized_6dof(aircraft, airspeed, trim_X0, rho = RHO_SL_SLUG_FT3):
    """
    Obtains linearized state-space matrices for the supplied aircraft object
    at specified airspeed and trim conditions. Aircraft must be set at a control
    setting and throttle level that puts it at or near a valid trim state,
    or the linearization will be invalid

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft object which implements "get_thrust," "get_mass_props," and
        "get_aero_coeffs" properties
    airspeed : float
        True airspeed for trim
    trim_X0 : array
        Trim initial conditions for the 6DOF solver
    rho : float, optional
        Atmospheric density at trim altitude. Default value is sea level in slug/ft^3

    Returns
    -------
    lti_sys : LinearTimeInvariant
        Returns a state-space representation of a linear time invariant system
        obtained by the linearization. States are the 6DOF states, inputs are
        the control inputs, and outputs are the 6DOF outputs
    y_0 : Array
        Vector of linear system outputs at the trim state

    """

    # Get derivatives with respect to each state
    A = list()
    C = list()
    t_eval = 0.
    delta_state = 0.001  # Doesn't matter, coeffs are linear
    xdot_0 = eom_6dof(t_eval, trim_X0, aircraft)
    y_0 = eom_6dof(t_eval, trim_X0, aircraft, output_mode = True)
    roll_0 = aircraft.roll_control
    pitch_0 = aircraft.pitch_control
    yaw_0 = aircraft.yaw_control

    for istate in range(len(trim_X0)):

        delta_x = trim_X0.copy()
        delta_x[istate] += delta_state
        delta_xdot = eom_6dof(t_eval, delta_x, aircraft) - xdot_0
        delta_y = (eom_6dof(t_eval, delta_x, aircraft, output_mode = True) - y_0)

        A.append(delta_xdot/delta_state)
        C.append(delta_y/delta_state)

    # Get derivatives with respect to each input -- do manually since controls
    # are a little weird
    B = list()
    D = list()
    delta_control = 0.001  # Doesn't matter, coeffs are linear

    # Roll
    aircraft.set_aero_controls(roll = lambda t, x: roll_0(t, x) + delta_control,
                               pitch = pitch_0,
                               yaw = yaw_0)
    delta_xdot = eom_6dof(t_eval, trim_X0, aircraft) - xdot_0
    delta_y = (eom_6dof(t_eval, trim_X0, aircraft, output_mode = True) - y_0)
    B.append(delta_xdot/delta_control)
    D.append(delta_y/delta_control)

    # Pitch
    aircraft.set_aero_controls(roll = roll_0,
                               pitch = lambda t, x: pitch_0(t, x) + delta_control,
                               yaw = yaw_0)
    delta_xdot = eom_6dof(t_eval, trim_X0, aircraft) - xdot_0
    delta_y = (eom_6dof(t_eval, trim_X0, aircraft, output_mode = True) - y_0)
    B.append(delta_xdot/delta_control)
    D.append(delta_y/delta_control)

    # Yaw
    aircraft.set_aero_controls(roll = roll_0,
                               pitch = pitch_0,
                               yaw = lambda t, x: yaw_0(t, x) + delta_control)
    delta_xdot = eom_6dof(t_eval, trim_X0, aircraft) - xdot_0
    delta_y = (eom_6dof(t_eval, trim_X0, aircraft, output_mode = True) - y_0)
    B.append(delta_xdot/delta_control)
    D.append(delta_y/delta_control)

    # Reset controls to initial values
    aircraft.set_aero_controls(roll = roll_0,
                               pitch = pitch_0,
                               yaw = yaw_0)

    A = np.array(A).T
    B = np.array(B).T
    C = np.array(C).T
    D = np.array(D).T

    return lti(A, B, C, D), y_0



def get_linearized_long(aircraft, trim_X0, rho = RHO_SL_SLUG_FT3, g = G_FT_S2):
    """
    Obtains linearized state-space matrices for the supplied aircraft object
    at specified airspeed and trim conditions. Aircraft must be set at a control
    setting and throttle level that puts it at or near a valid trim state,
    or the linearization will be invalid

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft object which implements "get_thrust," "get_mass_props," and
        "get_aero_coeffs" properties
    airspeed : float
        True airspeed for trim
    trim_X0 : array
        Trim initial conditions for the 6DOF solver
    rho : float, optional
        Atmospheric density at trim altitude. Default value is sea level in slug/ft^3
    g : float, optional
        Gravitional acceleration value. Default is sea level value in ft/s^2

    Returns
    -------
    lti_sys : LinearTimeInvariant
        Returns a state-space representation of a linear time invariant system
        obtained by the linearization. States are the 6DOF states, inputs are
        the control inputs, and outputs are the 6DOF outputs
    y_0 : Array
        Vector of linear system outputs at the trim state

    """

    # Aerodynamic derivatives
    stab_coeffs, control_coeffs = get_aero_derivatives_long(aircraft, trim_X0)
    Cxu, Czu, Cmu, Cxa, Cza, Cma, Cmq = stab_coeffs
    Cxdp, Czdp, Cmdp = control_coeffs

    # Neglect any propulsion derivatives for now
    Txu = 0.
    Tzu = 0.
    Txdt = 0.
    Tzdt = 0.

    # Inertial, dynamic pressure, and reference quantities
    mass = aircraft.mass
    Iyy = aircraft.Iyy
    u0 = trim_X0[0]
    w0 = trim_X0[2]
    theta0 = trim_X0[7]
    VTAS = np.sqrt(u0*u0 + w0*w0)
    qinf = 0.5*rho*VTAS**2
    sref = aircraft.sref
    cref = aircraft.cref

    # Initial coefficient values where needed
    Cx0, Cz0, _ = np.array(aircraft.get_aero_coeffs(0, trim_X0))[[0, 2, 4]]

    # A matrix definition (state transition)
    A = np.array([[qinf*sref*(Cxu + 2*Cx0/u0)/mass + Txu/mass,
                   qinf*sref*Cxa/(VTAS*mass), -w0, -g*np.cos(theta0), 0., 0.], # u
                  [qinf*sref*(Czu + 2*Cz0/u0)/mass + Tzu/mass,
                  qinf*sref*Cza/(VTAS*mass), u0, -g*np.sin(theta0), 0., 0.], # w
                  [qinf*sref*cref*Cmu/Iyy, qinf*sref*cref*Cma/(VTAS*Iyy),
                  qinf*sref*cref**2/(2*VTAS)*Cmq, 0., 0., 0.], # q
                  [0., 0., 1., 0., 0., 0.], # theta
                  [np.cos(theta0), np.sin(theta0), 0., -u0*np.sin(theta0) + w0*np.cos(theta0), 0., 0.], # XE
                  [-np.sin(theta0), np.cos(theta0), 0., -u0*np.cos(theta0) - w0*np.sin(theta0), 0., 0.], # YE
                 ])

    # B matrix definition (input influence)
    B = np.array([[qinf*sref*Cxdp/mass, Txdt/mass], # u
                  [qinf*sref*Czdp/mass, Tzdt/mass], # w
                  [qinf*sref*cref*Cmdp/Iyy, 0.],   # q
                  [0., 0.,],   # theta
                  [0., 0.,],   # XE
                  [0., 0.,]])  # ZE

    # C matrix definition (state output)
    C = np.array([[u0/VTAS, w0/VTAS, 0., 0., 0., 0.], # VTAS
                  [0., 1/VTAS, 0., 0., 0., 0.],  # Alpha
                  [0., 0., 1., 0., 0., 0.],  # q
                  [0., 0., 0., 1., 0., 0.],  # theta
                  [(qinf*sref*(Czu + 2*Cz0/u0) + Tzu)/(mass*g),
                  qinf*sref*Cza/(mass*g*VTAS), 0., 0., 0., 0.],  # nz
                  [0., 0., 0., 0., 0., -1.],  # h
                  [0., 0., 0., 0., 0., 0.]]) # Delta pitch

    # D matrix definition (input feed-forward)
    D = np.array([[0., 0.],
                  [0., 0.],
                  [0., 0.],
                  [0., 0.],
                  [qinf*sref*Czdp/(mass*g), Tzdt/(mass*g)],
                  [0., 0.],
                  [1., 0.]])



    # Initial output values from indexed 6DOF EOM outputs
    # V, alpha, beta, [*0*, *1*, 2,
    # p, q, r,         3, *4*, 5,
    # phi, theta, psi, 6, *7*, 8
    # xe, ye, ze,      9, 10, *11*,
    # ax, ay, az,      12, 13, *14*,
    # roll_control, pitch_control, yaw_control    15, *16*, 17]
    y_0 = eom_6dof(0., trim_X0, aircraft, output_mode = True)[[0, 1, 4, 7, 14, 11, 16]]

    return lti(A, B, C, D), y_0

# Longitudinal dynamics comparison
def get_aero_derivatives_long(aircraft, trim_X0, delta_state = 0.01, delta_control = 0.01):
    """
    Obtains the aerodynamic derivatives associated with longitudinal stability
    & control via finite difference using the supplied aircraft object at specified
    trim conditions. Aircraft must be set at a control setting and throttle level
    that puts it at or near a valid trim state, or the linearization will be invalid.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft object which implements "get_thrust," "get_mass_props," and
        "get_aero_coeffs" properties
    airspeed : float
        True airspeed for trim
    trim_X0 : array
        Trim initial conditions for the 6DOF solver

    Returns
    -------
    stab_coeffs : array
        Array of stability coeffiicents in the order [Cx_u, Cz_u, Cm_u, Cx_alpha,
        Cz_alpha, Cm_alpha, Cm_q]
    control_coeffs : array
        Array of control coefficients in the order [Cx_delta_pitch, Cz_delta_pitch,
        Cm_delta_pitch]

    """

    # Initialize
    stab_coeffs = list()
    control_coeffs = list()
    roll_0 = aircraft.roll_control
    pitch_0 = aircraft.pitch_control
    yaw_0 = aircraft.yaw_control

    # Initial coeffs in order [Cx, Cy, Cz, Cmx, Cmy, Cmz]
    coeff_inds = [0, 2, 4]
    coeffs_0 = np.array(aircraft.get_aero_coeffs(0, trim_X0))[coeff_inds]

    # Change due to u at constant alpha
    u0 = trim_X0[0]
    w0 = trim_X0[2]
    VTAS = np.sqrt(u0*u0 + w0*w0)
    alpha0 = np.arcsin(trim_X0[2]/VTAS)
    VTAS1 = VTAS + delta_state
    delta_x = trim_X0.copy()
    delta_x[0] = VTAS1*np.cos(alpha0)
    delta_x[2] = VTAS1*np.sin(alpha0)
    coeffs_du = np.array(aircraft.get_aero_coeffs(0, delta_x))[coeff_inds]
    stab_coeffs.extend((coeffs_du - coeffs_0)/delta_state)

    # Change due to w
    delta_x = trim_X0.copy()
    alpha1 = alpha0 + delta_state
    # delta_x[0]
    delta_x[0] = VTAS*np.cos(alpha1)
    delta_x[2] = VTAS*np.sin(alpha1)
    VTAS_new = np.sqrt(sum(delta_x[:2]**2))
    alpha_new = np.arcsin(delta_x[2]/VTAS_new)
    coeffs_da = np.array(aircraft.get_aero_coeffs(0, delta_x))[coeff_inds]
    stab_coeffs.extend((coeffs_da - coeffs_0)/(delta_state))

    # Change due to q
    delta_x = trim_X0.copy()
    delta_x[4] += delta_state*2*VTAS/aircraft.cref
    coeffs_dq = np.array(aircraft.get_aero_coeffs(0, delta_x))[4]  # Cm only
    stab_coeffs.append((coeffs_dq - coeffs_0[-1])/delta_state)

    # Change due to control input
    aircraft.set_aero_controls(roll = roll_0,
                               pitch = lambda t, x: pitch_0(t, x) + delta_control,
                               yaw = yaw_0)
    coeffs_dpitch = np.array(aircraft.get_aero_coeffs(0, trim_X0))[coeff_inds]
    control_coeffs.extend((coeffs_dpitch - coeffs_0)/delta_control)

    # Reset controls to initial values
    aircraft.set_aero_controls(roll = roll_0,
                               pitch = pitch_0,
                               yaw = yaw_0)

    return np.array(stab_coeffs), np.array(control_coeffs)



def get_aero_derivatives_lat(aircraft, trim_X0, delta_state = 0.01, delta_control = 0.01):
    """
    Obtains the aerodynamic derivatives associated with lateral/directional stability
    & control via finite difference using the supplied aircraft object at specified
    trim conditions. Aircraft must be set at a control setting and throttle level
    that puts it at or near a valid trim state, or the linearization will be invalid.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft object which implements "get_thrust," "get_mass_props," and
        "get_aero_coeffs" properties
    trim_X0 : array
        Trim initial conditions for the 6DOF solver

    Returns
    -------
    stab_coeffs : array
        Array of stability coeffiicents in the order [Cy_beta, Cl_beta, Cn_beta, Cl_p, Cn_p,
        Cl_r, Cn_r]
    control_coeffs : array
        Array of control coefficients in the order [Cy_delta_aileron, Cy_delta_rudder,
        Cl_delta_aileron, Cl_delta_rudder, Cn_delta_aileron, Cn_delta_rudder]

    """

    # Initialize
    stab_coeffs = list()
    control_coeffs = list()
    roll_0 = aircraft.roll_control
    pitch_0 = aircraft.pitch_control
    yaw_0 = aircraft.yaw_control

    # Coeffs vector is in order [Cx, Cy, Cz, Cmx, Cmy, Cmz]
    coeff_inds = [1, 3, 5]
    coeffs_0 = np.array(aircraft.get_aero_coeffs(0, trim_X0))[coeff_inds]

    # Initial items
    u0 = trim_X0[0]
    w0 = trim_X0[2]
    VTAS = np.sqrt(u0*u0 + w0*w0)
    alpha0 = np.arcsin(trim_X0[2]/VTAS)

    # Change due to beta at constant VTAS & alpha
    # This requires a reduction in u to keep the
    # total velocity vector constant
    delta_x = trim_X0.copy()
    dbeta = delta_state

    delta_x[0] = VTAS*np.cos(alpha0)*np.cos(dbeta) # Decrease by cos(dbeta)
    delta_x[1] = VTAS*np.sin(dbeta)

    coeffs_db = np.array(aircraft.get_aero_coeffs(0, delta_x))[coeff_inds]
    stab_coeffs.extend((coeffs_db - coeffs_0)/delta_state)

    # Change due to p
    delta_x = trim_X0.copy()
    delta_x[3] += delta_state*2*VTAS/aircraft.bref
    coeffs_dp = np.array(aircraft.get_aero_coeffs(0, delta_x))[[3, 5]]  # Cl and Cn only
    stab_coeffs.extend((coeffs_dp - coeffs_0[1:])/(delta_state))

    # Change due to r
    delta_x = trim_X0.copy()
    delta_x[5] += delta_state*2*VTAS/aircraft.bref
    coeffs_dr = np.array(aircraft.get_aero_coeffs(0, delta_x))[[3, 5]]  # Cl and Cn only
    stab_coeffs.extend((coeffs_dr - coeffs_0[1:])/(delta_state))

    # Change due to roll input
    aircraft.set_aero_controls(roll = lambda t, x: roll_0(t, x) + delta_control,
                               pitch = pitch_0,
                               yaw = yaw_0)
    coeffs_droll = np.array(aircraft.get_aero_coeffs(0, trim_X0))[coeff_inds]
    control_coeffs.extend((coeffs_droll - coeffs_0)/delta_control)

    # Change due to rudder input
    aircraft.set_aero_controls(roll = roll_0,
                               pitch = pitch_0,
                               yaw = lambda t, x: yaw_0(t, x) + delta_control)
    coeffs_dyaw = np.array(aircraft.get_aero_coeffs(0, trim_X0))[coeff_inds]
    control_coeffs.extend((coeffs_dyaw - coeffs_0)/delta_control)

    # Reset controls to initial values
    aircraft.set_aero_controls(roll = roll_0,
                               pitch = pitch_0,
                               yaw = yaw_0)

    return np.array(stab_coeffs), np.array(control_coeffs)


def get_linearized_lat(aircraft, trim_X0, rho = RHO_SL_SLUG_FT3, g = G_FT_S2):
    """
    Obtains linearized lat/dir state-space matrices for the supplied aircraft object
    at specified airspeed and trim conditions. Aircraft must be set at a control
    setting and throttle level that puts it at or near a valid trim state,
    or the linearization will be invalid

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft object which implements "get_thrust," "get_mass_props," and
        "get_aero_coeffs" properties
    trim_X0 : array
        Trim initial conditions for the 6DOF solver
    rho : float, optional
        Atmospheric density at trim altitude. Default value is sea level in slug/ft^3
    g : float, optional
        Gravitional acceleration value. Default is sea level value in ft/s^2

    Returns
    -------
    lti_sys : LinearTimeInvariant
        Returns a state-space representation of a linear time invariant system
        obtained by the linearization. States are the lat-dir states, inputs are
        the control inputs, and outputs are the lat-dir outputs
    y_0 : Array
        Vector of lat-dir linear system outputs at the trim state

    """

    # Aerodynamic derivatives
    stab_coeffs, control_coeffs = get_aero_derivatives_lat(aircraft, trim_X0)
    Cyb, Clb, Cnb, Clp, Cnp, Clr, Cnr = stab_coeffs
    Cyda, Clda, Cnda, Cydr, Cldr, Cndr = control_coeffs

    # Inertial, dynamic pressure, and reference quantities
    mass = aircraft.mass
    Ixx = aircraft.Ixx
    Izz = aircraft.Izz
    Ixz = aircraft.Ixz
    Izzh = Izz/Ixx
    Ixzh = Ixz/Ixx
    Ih = Izzh - Ixzh*Ixzh

    u0 = trim_X0[0]
    w0 = trim_X0[2]
    theta0 = trim_X0[7]
    VTAS = np.sqrt(u0*u0 + w0*w0)
    qinf = 0.5*rho*VTAS**2
    sref = aircraft.sref
    bref = aircraft.bref
    Qbar = qinf*sref*bref

    # A matrix definition (state transition)
    A = np.array([[qinf*sref*Cyb/(mass*VTAS), w0, -u0, g*np.cos(theta0), 0., 0.], # v
                  [Qbar/VTAS*(Izzh/Ih*Clb + Ixzh/Ih*Cnb)/Ixx,
                   Qbar*bref/(2*VTAS)*(Izzh/Ih*Clp + Ixzh/Ih*Cnp)/Ixx,
                   Qbar*bref/(2*VTAS)*(Izzh/Ih*Clr + Ixzh/Ih*Cnr)/Ixx,
                   0, 0., 0.], # p
                  [Qbar/VTAS*(Ixzh/Ih*Clb + 1./Ih*Cnb)/Ixx,
                   Qbar*bref/(2*VTAS)*(Ixzh/Ih*Clp + 1./Ih*Cnp)/Ixx,
                   Qbar*bref/(2*VTAS)*(Ixzh/Ih*Clr + 1./Ih*Cnr)/Ixx,
                   0, 0., 0.], # r
                  [0., 1., np.tan(theta0), 0., 0., 0.], # phi
                  [0, 0, 1/np.cos(theta0), 0, 0., 0.], # psi
                  [1., 0., 0., -w0, u0*np.cos(theta0) + w0*np.sin(theta0), 0.], # YE
                 ])


    # B matrix definition (input influence)
    B = np.array([[qinf*sref*Cyda/mass, qinf*sref*Cydr/mass], # v
                  [Qbar*(Izzh/Ih*Clda + Ixzh/Ih*Cnda)/Ixx, Qbar*(Izzh/Ih*Cldr + Ixzh/Ih*Cndr)/Ixx], # p
                  [Qbar*(Ixzh/Ih*Clda + 1/Ih*Cnda)/Ixx, Qbar*(Ixzh/Ih*Cldr + 1/Ih*Cndr)/Ixx],   # r
                  [0., 0.,],   # phi
                  [0., 0.,],   # psi
                  [0., 0.,]])  # YE

    # C matrix definition (state output)
    C = np.array([[1/VTAS, 0., 0., 0., 0., 0.], # beta
                  [0., 1, 0., 0., 0., 0.],  # p
                  [0., 0., 1., 0., 0., 0.],  # r
                  [0., 0., 0., 1., 0., 0.],  # phi
                  [0., 0., 0., 0., 1., 0.],  # psi
                  [qinf*sref*Cyb/(mass*g*VTAS),
                  0., 0., 0., 0., 0.],  # ny
                  [0., 0., 0., 0., 0., 1],  # YE
                  [0., 0., 0., 0., 0., 0.], # Delta aileron
                  [0., 0., 0., 0., 0., 0.]]) # Delta rudder

    # D matrix definition (input feed-forward)
    D = np.array([[0., 0.],
                  [0., 0.],
                  [0., 0.],
                  [0., 0.],
                  [0., 0.],
                  [qinf*sref*Cyda/(mass*g), qinf*sref*Cydr/(mass*g)],
                  [0., 0.],
                  [1., 0.],
                  [0., 1.]])


    # Initial output values from indexed 6DOF EOM outputs
    # V, alpha, beta, [0, 1, *2*,
    # p, q, r,         *3*, 4, *5*,
    # phi, theta, psi, *6*, 7, *8*
    # xe, ye, ze,      9, *10*, 11,
    # ax, ay, az,      12, *13*, 14,
    # roll_control, pitch_control, yaw_control    *15*, 16, *17*]
    y_0 = eom_6dof(0., trim_X0, aircraft, output_mode = True)[[2, 3, 5, 6, 8, 13, 10, 15, 17]]

    return lti(A, B, C, D), y_0


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.signal import lsim

    f106, veas = GET_F106()
    alpha0_rad, delta0_rad, throttle0 = get_trim(f106, veas)
    trim_X0 = get_initial_state(veas, alpha0_rad)
    f106.set_aero_controls(pitch = lambda t, x: delta0_rad)

    linsys, y_0 = get_linearized_6dof(f106, veas, trim_X0)
    nl_eom = lambda t, x: eom_6dof(t, x, f106)

    # Run comparison simulation between NL and linearized 6DOF
    dt = 0.01
    T = 10.0
    time = np.arange(0, T, dt)
    lstyle_nl = '-'
    lstyle_lin = '--'

    pitch_start = 0.5
    Tpitch = 2.0
    dpitch = 2.5/57.3
    rudder_start = 0.5
    Trudder = 1.0
    drudder = 20/57.3

    zero = lambda t, x: 0.*t
    pitch_trim = lambda t, x: delta0_rad*(t >= 0.)
    pitch_rap = lambda t, x: delta0_rad - dpitch*np.sin(np.pi*(t - pitch_start)/Tpitch)*(t > pitch_start)*(t < (pitch_start + Tpitch))
    rudder_rap = lambda t, x: drudder*np.sin(np.pi*(t - rudder_start)/Trudder)*(t > rudder_start)*(t < (rudder_start + Trudder))

    for (roll_hist, pitch_hist, yaw_hist), title in zip([(zero, pitch_rap, zero),
                                                         (zero, pitch_trim, rudder_rap)],
                                                        ['Pitch Rap Comparison; Linear vs. Nonlinear Solution',
                                                         'Rudder Rap Comparison']):

        f106.set_aero_controls(roll = roll_hist,
                               pitch = pitch_hist,
                               yaw = yaw_hist)

        eom = lambda t, x: eom_6dof(t, x, f106)
        sol = solve_ivp(eom, [0, T], trim_X0, t_eval = time)
        # u, v, w, p, q, r, phi, theta, psi, Xe, Ye, Ze = sol.y
        outputs_nl = np.array([eom_6dof(ti, yi, f106, output_mode = True) for ti, yi in zip(time, sol.y.T)]).T

        U = np.vstack([roll_hist(sol.t, trim_X0),
                       pitch_hist(sol.t, trim_X0) - pitch_trim(sol.t, trim_X0),
                       yaw_hist(sol.t, trim_X0)]).T

        _, outputs_lin, _ = lsim(linsys, U, sol.t)

        outputs_lin += y_0

        fig, ax_dict = plt.subplot_mosaic([['V [ft/s]', 'alpha [deg]', 'beta [deg]'],
                                      ['p [deg/s]', 'q [deg/s]', 'r [deg/s]'],
                                      ['phi [deg]', 'theta [deg]', 'psi [deg]'],
                                      ['Aileron [deg]', 'Elevator [deg]', 'Rudder [deg]']],
                                    figsize = (10, 8),
                                    constrained_layout = True)


        for outputs, lstyle, label in zip([outputs_nl, outputs_lin.T],
                                           [lstyle_nl, lstyle_lin],
                                             ['Nonlinear', 'Linearized']):

            V, alpha, beta, p, q, r, phi, theta, psi, xe, ye, ze, ax, ay, az, roll_control, pitch_control, yaw_control = outputs
            ax_dict['V [ft/s]'].plot(sol.t, V, label = label)

            ax_dict['alpha [deg]'].plot(sol.t, alpha*RAD_DEG, lstyle_nl, label = label)
            ax_dict['q [deg/s]'].plot(sol.t, q*RAD_DEG, lstyle_nl, label = label)
            ax_dict['theta [deg]'].plot(sol.t, theta*RAD_DEG, lstyle_nl, label = label)
            ax_dict['Elevator [deg]'].plot(sol.t, pitch_control*RAD_DEG, lstyle_nl, label = label)

            ax_dict['beta [deg]'].plot(sol.t, beta*RAD_DEG, lstyle_nl, label = label)
            ax_dict['p [deg/s]'].plot(sol.t, p*RAD_DEG, lstyle_nl, label = label)
            ax_dict['r [deg/s]'].plot(sol.t, r*RAD_DEG, lstyle_nl, label = label)
            ax_dict['phi [deg]'].plot(sol.t, phi*RAD_DEG, lstyle_nl, label = label)
            ax_dict['psi [deg]'].plot(sol.t, psi*RAD_DEG, lstyle_nl, label = label)

            ax_dict['Aileron [deg]'].plot(sol.t, roll_control*RAD_DEG, lstyle_nl, label = label)
            ax_dict['Rudder [deg]'].plot(sol.t, yaw_control*RAD_DEG, lstyle_nl, label = label)

        ax_dict['alpha [deg]'].legend(loc = 'upper right', framealpha = 1.)
        for key, a in ax_dict.items():
            a.set_xlabel('Time [s]', fontweight = 'bold')
            a.set_ylabel(key, fontweight = 'bold')
            a.grid()

        fig.suptitle(title, fontweight = 'bold')

    # Check stab coeff calculation vs expected
    (Cx_u, Cz_u, Cm_u, Cx_alpha, Cz_alpha, Cm_alpha, Cm_q), (Cx_delta, Cz_delta, Cm_delta) = get_aero_derivatives_long(f106, trim_X0)

    print('Longitudinal coefficients; stored vs. expected')
    print('Cx_u', 0, Cx_u)
    print('Cz_u', 0, Cz_u)
    print('Cm_u', 0, Cm_u)
    print('Cx_a', f106.Cxa, Cx_alpha)
    print('Cz_a', f106.Cza, Cz_alpha)
    print('Cm_a', f106.Cma, Cm_alpha)
    print('Cm_q', f106.Cmq, Cm_q)
    print('Cx_delta', f106.Cxdp, Cx_delta)
    print('Cz_delta', f106.Czdp, Cz_delta)
    print('Cm_delta', f106.Cmdp, Cm_delta)

    # Checkout of linearization against previous results
    linsys_long, y_0_long = get_linearized_long(f106, trim_X0)

    # print(np.linalg.eigvals(linsys_long.A))
    # print(linsys_long.A)

    long_coeffs = [0, 2, 4, 7, 9, 11]
    A_check = np.array([s[long_coeffs] for s in linsys.A[long_coeffs]])

    # Run simulation
    roll_hist = lambda t, x: 0.
    yaw_hist = lambda t, x: 0.
    f106.set_aero_controls(roll = zero,
                           pitch = pitch_rap,
                           yaw = zero)

    eom = lambda t, x: eom_6dof(t, x, f106)
    sol = solve_ivp(eom, [0, T], trim_X0, t_eval = time)
    # u, v, w, p, q, r, phi, theta, psi, Xe, Ye, Ze = sol.y
    outputs_nl = np.array([eom_6dof(ti, yi, f106, output_mode = True) for ti, yi in zip(time, sol.y.T)]).T

    U = np.vstack([zero(sol.t, trim_X0),
                   pitch_rap(sol.t, trim_X0) - pitch_trim(sol.t, trim_X0),
                   zero(sol.t, trim_X0)]).T

    # Numerical linear system
    _, outputs_lin, _ = lsim(linsys, U, sol.t)
    outputs_lin += y_0

    # Explictly written longitudinal system
    U = np.vstack([pitch_rap(sol.t, trim_X0) - pitch_trim(sol.t, trim_X0),
                    zero(sol.t, trim_X0)]).T
    _, outputs_lin_long, _ = lsim(linsys_long, U, sol.t)
    outputs_lin_long += y_0_long

    fig, ax_dict = plt.subplot_mosaic([['V [ft/s]', 'alpha [deg]'],
                                      ['q [deg/s]', 'theta [deg]'],
                                      ['Altitude [ft]', 'nz [g]']],
                                    figsize = (10, 8),
                                    constrained_layout = True)


    for outputs, lstyle, label in zip([outputs_nl, outputs_lin.T, outputs_lin_long.T],
                                       [lstyle_nl, lstyle_lin, '-.'],
                                         ['Nonlinear', 'Linearized 6DOF', 'Linearized/Longitudinal']):
        try:
            V, alpha, beta, p, q, r, phi, theta, psi, xe, ye, ze, ax, ay, az, roll_control, pitch_control, yaw_control = outputs
            ax_dict['V [ft/s]'].plot(sol.t, V, lstyle, label = label)

            ax_dict['alpha [deg]'].plot(sol.t, alpha*RAD_DEG, lstyle, label = label)
            ax_dict['q [deg/s]'].plot(sol.t, q*RAD_DEG, lstyle, label = label)
            ax_dict['theta [deg]'].plot(sol.t, theta*RAD_DEG, lstyle, label = label)
            ax_dict['nz [g]'].plot(sol.t, az, lstyle, label = label)
            ax_dict['Altitude [ft]'].plot(sol.t, -ze, lstyle, label = label)


        except:
            V, alpha, q, theta, nz, h, pitch_control = outputs
            ax_dict['V [ft/s]'].plot(sol.t, V, lstyle, label = label)

            ax_dict['alpha [deg]'].plot(sol.t, alpha*RAD_DEG, lstyle, label = label)
            ax_dict['q [deg/s]'].plot(sol.t, q*RAD_DEG, lstyle, label = label)
            ax_dict['theta [deg]'].plot(sol.t, theta*RAD_DEG, lstyle, label = label)
            ax_dict['nz [g]'].plot(sol.t, nz, lstyle, label = label)
            ax_dict['Altitude [ft]'].plot(sol.t, h, lstyle, label = label)


    ax_dict['V [ft/s]'].legend(loc = 'upper right', framealpha = 1.)
    for key, a in ax_dict.items():
        a.set_xlabel('Time [s]', fontweight = 'bold')
        a.set_ylabel(key, fontweight = 'bold')
        a.grid()

    fig.suptitle('6DOF/Linearized 6DOF/Linear Longitudinal Pitch Rap Comparison', fontweight = 'bold')


    # Lateral/directional dynamics and comparison
    # Check stab coeff calculation vs expected
    (Cy_beta, Cl_beta, Cn_beta, Cl_p, Cn_p, Cl_r, Cn_r), (Cy_da, Cl_da, Cn_da, Cy_dr, Cl_dr, Cn_dr) = get_aero_derivatives_lat(f106, trim_X0)

    print('Lateral/directional coefficients; stored vs. expected')
    print('Cy_beta', f106.Cyb, Cy_beta)
    print('Cl_beta', f106.Clb, Cl_beta)
    print('Cn_beta', f106.Cnb, Cn_beta)
    print('Cl_p', f106.Clp, Cl_p)
    print('Cn_p', f106.Cnp, Cn_p)
    print('Cl_r', f106.Clr, Cl_r)
    print('Cn_r', f106.Cnr, Cn_r)

    print('Cy_da', f106.Cyda, Cy_da)
    print('Cl_da', f106.Clda, Cl_da)
    print('Cn_da', f106.Cnda, Cn_da)

    print('Cy_dr', f106.Cydr, Cy_dr)
    print('Cl_dr', f106.Cldr, Cl_dr)
    print('Cn_dr', f106.Cndr, Cn_dr)

    # Mess with matrix definitions
    aircraft = f106
    mass = aircraft.mass
    Ixx = aircraft.Ixx
    Izz = aircraft.Izz
    Ixz = aircraft.Ixz
    Izzh = Izz/Ixx
    Ixzh = Ixz/Ixx
    Ih = Izzh - Ixzh*Ixzh

    u0 = trim_X0[0]
    w0 = trim_X0[2]
    theta0 = trim_X0[7]
    VTAS = np.sqrt(u0*u0 + w0*w0)
    rho = RHO_SL_SLUG_FT3
    qinf = 0.5*rho*VTAS**2
    sref = aircraft.sref
    bref = aircraft.bref
    Qbar = qinf*sref*bref


    # Checkout of linearization against previous results
    linsys_lat, y_0_lat = get_linearized_lat(f106, trim_X0)

    # print(np.linalg.eigvals(linsys_lat.A))
    print(np.linalg.matrix_rank(linsys_lat.A.T))
    # print(linsys_lat.A)

    # lat_coeffs = [1, 3, 5, 6, 8, 10]
    # A_check = np.array([s[lat_coeffs] for s in linsys.A[lat_coeffs]])

    # Run simulation
    roll_hist = lambda t, x: 0.
    yaw_hist = lambda t, x: 0.
    f106.set_aero_controls(roll = zero,
                           pitch = pitch_trim,
                           yaw = rudder_rap)

    eom = lambda t, x: eom_6dof(t, x, f106)
    sol = solve_ivp(eom, [0, T], trim_X0, t_eval = time)
    # u, v, w, p, q, r, phi, theta, psi, Xe, Ye, Ze = sol.y
    outputs_nl = np.array([eom_6dof(ti, yi, f106, output_mode = True) for ti, yi in zip(time, sol.y.T)]).T

    U = np.vstack([zero(sol.t, trim_X0),
                   pitch_trim(sol.t, trim_X0),
                   rudder_rap(sol.t, trim_X0)]).T

    # Numerical linear system
    _, outputs_lin, _ = lsim(linsys, U, sol.t)
    outputs_lin += y_0

    # Explictly written longitudinal system
    U = np.vstack([zero(sol.t, trim_X0),
                   rudder_rap(sol.t, trim_X0)]).T
    _, outputs_lin_lat, _ = lsim(linsys_lat, U, sol.t)
    outputs_lin_lat += y_0_lat

    fig, ax_dict = plt.subplot_mosaic([['beta [deg]', 'phi [deg]', 'psi [deg]'],
                                       ['ny [g]',  'p [deg/s]', 'r [deg/s]'],
                                       ['YE [ft]', 'Aileron [deg]', 'Rudder [deg]']],
                                    figsize = (10, 8),
                                    constrained_layout = True)

    for outputs, lstyle, label in zip([outputs_nl, outputs_lin.T, outputs_lin_lat.T],
                                       [lstyle_nl, lstyle_lin, '-.'],
                                         ['NL 6DOF', 'Lin 6DOF', 'Lin/Lat']):
        try:
            V, alpha, beta, p, q, r, phi, theta, psi, xe, ye, ze, ax, ay, az, roll_control, pitch_control, yaw_control = outputs

            ax_dict['beta [deg]'].plot(sol.t, beta*RAD_DEG, lstyle, label = label)
            ax_dict['p [deg/s]'].plot(sol.t, p*RAD_DEG, lstyle, label = label)
            ax_dict['r [deg/s]'].plot(sol.t, r*RAD_DEG, lstyle, label = label)
            ax_dict['phi [deg]'].plot(sol.t, phi*RAD_DEG, lstyle, label = label)
            ax_dict['psi [deg]'].plot(sol.t, psi*RAD_DEG, lstyle, label = label)
            ax_dict['ny [g]'].plot(sol.t, ay, lstyle, label = label)
            ax_dict['YE [ft]'].plot(sol.t, ye, lstyle, label = label)
            ax_dict['Aileron [deg]'].plot(sol.t, roll_control*RAD_DEG, lstyle, label = label)
            ax_dict['Rudder [deg]'].plot(sol.t, yaw_control*RAD_DEG, lstyle, label = label)


        except:
            beta, p, r, phi, psi, ay, ye, da, dr = outputs

            ax_dict['beta [deg]'].plot(sol.t, beta*RAD_DEG, lstyle, label = label)
            ax_dict['p [deg/s]'].plot(sol.t, p*RAD_DEG, lstyle, label = label)
            ax_dict['r [deg/s]'].plot(sol.t, r*RAD_DEG, lstyle, label = label)
            ax_dict['phi [deg]'].plot(sol.t, phi*RAD_DEG, lstyle, label = label)
            ax_dict['psi [deg]'].plot(sol.t, psi*RAD_DEG, lstyle, label = label)
            ax_dict['ny [g]'].plot(sol.t, ay, lstyle, label = label)
            ax_dict['YE [ft]'].plot(sol.t, ye, lstyle, label = label)
            ax_dict['Aileron [deg]'].plot(sol.t, roll_control*RAD_DEG, lstyle, label = label)
            ax_dict['Rudder [deg]'].plot(sol.t, yaw_control*RAD_DEG, lstyle, label = label)


    ax_dict['beta [deg]'].legend(loc = 'upper right', framealpha = 1.)
    for key, a in ax_dict.items():
        a.set_xlabel('Time [s]', fontweight = 'bold')
        a.set_ylabel(key, fontweight = 'bold')
        a.grid()

    fig.suptitle('6DOF/Linearized 6DOF/Linear Longitudinal Rudder Rap Comparison', fontweight = 'bold')


