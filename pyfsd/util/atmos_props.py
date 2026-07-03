# -*- coding: utf-8 -*-
"""

Calculation and management of atmospheric and flight-associated properties

Source: https://www.pdas.com/programs/atmos.py

"""

import numpy as np

from convert import unit_convert  # Make global when ready

class AtmosProps:

    # Temperature gradients and pressure ratio values
    h_layers = np.array([0., 11., 20., 32., 47., 51., 71., 84.852, 200])
    temp_layers = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
    p_layers = np.array([1.0, 2.2336110E-1, 5.4032950E-2, 8.5666784e-3, 1.0945601E-3, 6.6063531E-4,
                    3.9046834E-5, 3.68501E-6])
    g_layers = np.array([-6.5,0.0, 1.0, 2.8, 0, -2.8, -2.0, 0.0 ])

    # Reference properties
    RHO_SL_SI = 1.225  # [kg/m^3]
    P_SL_SI = 101325.0 # [Pa]
    T_SL_SI = 288.15   # [K]

    RHO_SL_IM = 0.002377  # [slug/ft^3]
    P_SL_IM = 2116.21663  # [lbf/ft^2]
    T_SL_IM = 288.15*9/5  # [R]

    P_HG_REF = 29.9212

    GAMMA = 1.4
    R_SI = 287.0 # Specific gas constant [J/kg-K]
    R_IM = 1716.0 # [ft-lb/slug-R]

    def __init__(self, alt, dTambient = 0.0, Phg_SL = None, si_units = False):
        """

        Generates an object from a vector of altitudes, ambient temperatures,
        and barometer readings. Stored internal states are

        - alt : altitude, [ft] or [m]
        - rho : density, [slug/ft^3] or [kg/m^3]
        - p : pressure, [lbf-ft/^2] or [N/m^3]
        - T : temperature, [R] or [K]
        - ainf : speed of sound [ft/s] or [m/s]
        - si_units : SI flag


        Parameters
        ----------
        alt : float or array
            Geometric altitude above sea level. Units are [ft] for si_units
            False and [m] for si_units = True
        dTambient : float, optional
            Differential standard ambient temperature at altitude, if different
            from standard day. Units are [R] for si_units = False and [K] for
            si_units = True. The default is 0.0 in which case standard day
            temperatures are used.
        Phg_SL : float or array, optional
            Ambient barometric pressure at altitude. Units are [in-Hg].
            The default is None in which case standard day pressures are used.
        si_units : bool, optional
            Flag to return accept and return SI units. (For barometric pressure
            the unit is always [in-Hg]). The default is False

        Returns
        -------
        None.

        """

        self.alt = alt
        self.si_units = si_units

        if Phg_SL is None:
            Phg_SL = self.P_HG_REF
        self.rho, self.p, self.T = self.get_adjusted_props(alt, dTambient = dTambient,
                                                           Phg_SL = Phg_SL,
                                                           si_units = si_units)
        self.ainf = self.get_sound_speed(self.T, si_units = si_units)

    @staticmethod
    def get_adjusted_props(alt, dTambient = 0.0, Phg_SL = 29.9212,
                           si_units = False):
        """
        Return non-standard day properties based on ambient temperature input
        and barometric pressure input (adjusted to SL conditions per standard
        practice). The barometric pressure is used to scale the standard day
        pressure, and the ambient temperature is used to adjust the output
        density via the ideal gas law; p = R*rho*T

        Parameters
        ----------
        alt : float or array
            Geometric altitude above sea level. Units are [ft] for si_units
            False and [m] for si_units = True
        dTambient : float, optional
            Difference in ambient temperature from standard day at each given
            altitude, in [R] for si_units = False or [K] for si_units = True.
            The default is 0.0 for standard day temperatures.
        Phg_SL : float, optional
            Sea Level adjusted barometric pressure reading [inHG]. The default
            is 29.9212 for standard day pressures.
        si_units : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        rho : float or array
            Density value(s); [slug/ft^3] or [kg/m^3]
        p : float or array
            Pressure value(s); [lbf/ft^2] or [N/m^2]
        T : float or array
            Temperature value(s); [R] or [K]

        """

        # Standard day properties
        rho, p, T = AtmosProps.get_standard_props(alt, si_units = si_units)

        if not dTambient and Phg_SL == AtmosProps.P_HG_REF:
            return rho, p, T

        # Pressure adjustment
        p *= np.array(Phg_SL)/AtmosProps.P_HG_REF

        # Temperature adjustment
        T += dTambient

        # Density adjustment
        if si_units:
            rho = p/(AtmosProps.R_SI*np.array(T))
        else:
            rho = p/(AtmosProps.R_IM*np.array(T))

        return rho, p, T


    @staticmethod
    def get_sound_speed(temp, si_units = False):
        """
        Return speed of sound based on local temperature, in either Imperial
        [default] or SI units.

        Parameters
        ----------
        temp : float or array
            Input temperature, in [R] for si_units = False and [K] for
            si_units = True
        si_units : bool, optional
            Flag to indicate SI units as the input Temperature and return
            value. The default is False

        Returns
        -------
        ainf : float or array
            Calculated speed of sound, in [ft/s] for si_units = False or [m/s]
            for si_units = True

        """

        if si_units:
            return np.sqrt(AtmosProps.GAMMA*AtmosProps.R_SI*temp)
        else:
            return np.sqrt(AtmosProps.GAMMA*AtmosProps.R_IM*temp)


    @staticmethod
    def get_standard_props(alt, si_units = False):
        """

        Return standard-day atmospheric properties in either Imperial (default)
        or SI units, for an altitude input in either [ft] (default) or [m].

        Parameters
        ----------
        alt : float or array
            Geometric altitude above sea level. Units are [ft] for si_units
            False and [m] for si_units = True
        si_units : bool, optional
            Flag to indicate SI units as the input altitude and for the return
            values. The default is False

        Returns
        -------
        rho : float or array
            Density value(s); [slug/ft^3] or [kg/m^3]
        p : float or array
            Pressure value(s); [lbf/ft^2] or [N/m^2]
        T : float or array
            Temperature value(s); [R] or [K]

        """

        if not si_units:
            alt = unit_convert(np.array(alt), 'ft', 'm')

        sigma, delta, theta = AtmosProps.get_standard_ratios(alt)

        if si_units:
            return (AtmosProps.RHO_SL_SI*sigma,
                    AtmosProps.P_SL_SI*delta,
                    AtmosProps.T_SL_SI*theta)
        else:
            return (AtmosProps.RHO_SL_IM*sigma,
                    AtmosProps.P_SL_IM*delta,
                    AtmosProps.T_SL_IM*theta)

    @staticmethod
    def get_standard_ratios(alt):
        """
        Compute temperature, density, and pressure in standard atmosphere.
        Correct to 86 km.  Only approximate thereafter. Based on PDAS code
        available here:

        For a general description of the method, refer to Wikipedia:

            https://en.wikipedia.org/wiki/Barometric_formula

        Parameters
        ----------
        alt : float or array
            Geometric altitude [m]

        Returns
        -------
        sigma : density/sea-level standard density
        delta : pressure/sea-level standard pressure
        theta : temperature/sea-level std. temperature
        """

        alt = np.array(alt)/1000
        REARTH = 6369.0		# radius of the Earth (km)
        GMR = 34.163195


        h = alt*REARTH/(alt+REARTH)	# geometric to geopotential altitude

        # Get layer index based on geopotential altitude
        i = np.searchsorted(AtmosProps.h_layers, h, side = 'right') - 1
        deltah = h - AtmosProps.h_layers[i]  # height above local base

        # Temperature ratio from gradients and base temp
        tgrad = AtmosProps.g_layers[i]  # temp. gradient of local layer
        tbase = AtmosProps.temp_layers[i]  # base  temp. of local layer
        tlocal = tbase + tgrad*deltah  # local temperature
        theta = tlocal/AtmosProps.temp_layers[0]  # temperature ratio

        # Pressure ratio -- calculated using vectorization due to the logical
        # check on temp gradient
        delta = np.zeros(theta.shape)
        tg0inds = tgrad == 0.
        delta[tg0inds] = AtmosProps.p_layers[i[tg0inds]]*np.exp(-GMR*deltah[tg0inds]/tbase[tg0inds])
        delta[~tg0inds] = AtmosProps.p_layers[i[~tg0inds]]*(tbase[~tg0inds]/tlocal[~tg0inds])**(GMR/tgrad[~tg0inds])

        # delta multiplied by 1.0 to return as float if it is a length-1 array
        return delta/theta, delta*1., theta


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Original definition, available: https://www.pdas.com/programs/atmos.py
    def atmosphere(alt):
        """ Compute temperature, density, and pressure in standard atmosphere.
        Correct to 86 km.  Only approximate thereafter.
        Input:
        alt	geometric altitude, m.
        Return: (sigma, delta, theta)
        sigma	  density/sea-level standard density
        delta   pressure/sea-level standard pressure
        theta	   temperature/sea-level std. temperature
        """
        alt /= 1000.0
        REARTH = 6369.0		# radius of the Earth (km)
        GMR = 34.163195
        NTAB = 8			# length of tables

        htab = [ 0.0,  11.0, 20.0, 32.0, 47.0,
    	51.0, 71.0, 84.852]
        ttab = [ 288.15, 216.65, 216.65, 228.65, 270.65,
    	270.65, 214.65, 186.946 ]
        ptab = [ 1.0, 2.2336110E-1, 5.4032950E-2, 8.5666784E-3, 1.0945601E-3,
    	6.6063531E-4, 3.9046834E-5, 3.68501E-6 ]
        gtab = [ -6.5, 0.0, 1.0, 2.8, 0, -2.8, -2.0, 0.0 ]

        h = alt*REARTH/(alt+REARTH)	# geometric to geopotential altitude

        i=0; j=len(htab)
        while (j > i+1):
        	k = int((i+j)/2)
        	if h < htab[k]:
        	    j = k
        	else:
        	    i = k

        tgrad = gtab[i]		# temp. gradient of local layer
        tbase = ttab[i]		# base  temp. of local layer
        deltah=h-htab[i]		# height above local base
        tlocal=tbase+tgrad*deltah	# local temperature
        theta = tlocal/ttab[0]	# temperature ratio

        if 0.0 == tgrad:
            delta=ptab[i]*np.exp(-GMR*deltah/tbase)
        else:
            delta=ptab[i]*(tbase/tlocal)**(GMR/tgrad)
        sigma = delta/theta
        return ( sigma, delta, theta )

    # Actual property check
    hvec = np.linspace(0, 100*1000, 1000)

    sigma, delta, theta = np.array(AtmosProps.get_standard_ratios(hvec))
    sigma2, delta2, theta2 = np.array([atmosphere(h) for h in hvec]).T

    fig, ax = plt.subplots(1, 3, figsize = (12, 4), constrained_layout = True)

    ax[0].plot(hvec/1000, sigma)
    ax[0].plot(hvec/1000, sigma2, '--')
    ax[0].set_ylabel('Density Ratio $\mathbf{\sigma}$', fontweight = 'bold')

    ax[1].plot(hvec/1000, delta)
    ax[1].plot(hvec/1000, delta2, '--')
    ax[1].set_ylabel('Pressure Ratio $\mathbf{\delta}$', fontweight = 'bold')

    ax[2].plot(hvec/1000, theta)
    ax[2].plot(hvec/1000, theta2, '--')
    ax[2].set_ylabel(r'Temperature Ratio $\mathbf{\theta}$', fontweight = 'bold')

    for a in ax:
        a.grid()
        a.set_xlabel('Altitude [km]', fontweight = 'bold')

    rho_si, p_si, T_si = AtmosProps.get_standard_props(hvec, si_units = True)
    rho_im, p_im, T_im = AtmosProps.get_standard_props(unit_convert(hvec, 'm', 'ft'))

    print(AtmosProps.get_standard_props(hvec[0], si_units = True))

    ainf_si = AtmosProps.get_sound_speed(T_si, si_units = True)
    ainf_im = AtmosProps.get_sound_speed(T_im, si_units = False)

    print(AtmosProps.get_sound_speed(T_im[0]))

    print(AtmosProps.get_adjusted_props([hvec[0]]*2, [30.0, 30.], si_units = True))
    print(AtmosProps.get_adjusted_props([unit_convert(hvec[0], 'm', 'ft')]*2,
                                        0, si_units = False))

    atm_obj = AtmosProps(hvec, si_units = True)

    # What kind of sweet plot to make?
