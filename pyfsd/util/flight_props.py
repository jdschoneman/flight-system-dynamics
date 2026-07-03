# -*- coding: utf-8 -*-
"""

Flight property management class

"""

import numpy as np

from atmos_props import AtmosProps

from convert import unit_convert

class FlightProps:

    # Default units, which also correspond to units used for internal storage
    # Key for each sub-dictionary is self.si_units
    default_units = {'alt': {False: 'ft', True: 'm'},
                     'vtas': {False: 'fps', True: 'mps'},
                     'veas': {False: 'fps', True: 'mps'},
                     'qinf': {False: 'psf', True: 'Pa'}}

    def __init__(self, si_units = False,  **kwargs):

        self.alt = None
        self.vtas = None
        self.veas = None
        self.mach = None
        self.qinf = None

        self.si_units = si_units

        self.atmos_props = None



    def alt_vtas(self, alt, vtas, alt_unit = None, vtas_unit = None,
                 dTambient = 0.0, Phg_SL = AtmosProps.P_HG_REF):
        """
        Compute flight properties from altitude and true airspeed.

        Parameters
        ----------
        alt : float or array
            Geometric altitude above sea level. Units are [ft] for si_units
            False and [m] for si_units = True
        vtas : TYPE
            DESCRIPTION.
        alt_unit : TYPE, optional
            DESCRIPTION. The default is 'ft'.
        vtas_unit : TYPE, optional
            DESCRIPTION. The default is 'ft_s'.
        dTambient : float, optional
            Differential standard ambient temperature at altitude, if different
            from standard day. Units are [R] for si_units = False and [K] for
            si_units = True. The default is 0.0 in which case standard day
            temperatures are used.
        Phg_SL : float or array, optional
            Ambient barometric pressure at altitude. Units are [in-Hg].
            The default is AtmosProps.P_HG_REF in which case standard day
            pressures are used

        Returns
        -------
        None.

        """

        # Handle unit conversion
        if alt_unit is None:
            alt_unit = self.default_units['alt'][self.si_units]
        self.alt = unit_convert(alt, alt_unit, self.default_units['alt'][self.si_units])

        if vtas_unit is None:
            vtas_unit = self.default_units['vtas'][self.si_units]
        self.vtas = unit_convert(vtas, vtas_unit, self.default_units['vtas'][self.si_units])

        self.atmos_props = AtmosProps(self.alt, dTambient = dTambient,
                                      Phg_SL = Phg_SL, si_units = self.si_units)

        # Get remaining properties from true airspeed
        self.qinf = 0.5*self.vtas*self.vtas*self.atmos_props.rho
        self.veas = FlightProps.veas_from_qinf(self.qinf, self.si_units)
        self.mach = self.vtas/self.atmos_props.ainf

    def alt_veas(self, alt, veas, alt_unit = None, vtas_unit = None,
                 Phg_SL = AtmosProps.P_HG_REF, dTambient = 0.0):
        """
        Compute fligt properties from altitude and equivalent airspeed

        Parameters
        ----------
        alt : TYPE
            DESCRIPTION.
        veas : TYPE
            DESCRIPTION.
        alt_unit : TYPE, optional
            DESCRIPTION. The default is None.
        vtas_unit : TYPE, optional
            DESCRIPTION. The default is None.
        Phg_SL : TYPE, optional
            DESCRIPTION. The default is AtmosProps.P_HG_REF.
        dTambient : TYPE, optional
            DESCRIPTION. The default is 0.0.

        Returns
        -------
        None.

        """

    @staticmethod
    def qinf_from_veas(veas, si_units = False):
        """
        Obtain dynamic pressure from "equivalent" airspeed, based on the
        relation

            qinf = rho_sl*veas^2/2

        Parameters
        ----------
        veas : float or array
            Equivalent airspeed, in [ft/s] for si_units = True or [m/s] for
            si_units = False
        si_units : bool, optional
            Flag to return accept and return SI units. (For barometric pressure
            the unit is always [in-Hg]). The default is False

        Returns
        -------
        qinf : float or array
            Dynamic pressure, in [psf] for si_units = False or [Pa] for
            si_units = True

        """

        r0, _, _ = AtmosProps.get_standard_props(0, si_units = si_units)
        return np.sqrt(2*qinf/r0)

    @staticmethod
    def veas_from_qinf(qinf, si_units = False):
        """
        Obtain "equivalent" airspeed from dynamic pressure, based on the
        relation

            qinf = rho_sl*veas^2/2

            ==> veas = sqrt(2*qinf/rho_sl)

        Parameters
        ----------
        qinf : float or array
            Dynamic pressure, in [psf] for si_units = False or [Pa] for
            si_units = True
        si_units : bool, optional
            Flag to return accept and return SI units. (For barometric pressure
            the unit is always [in-Hg]). The default is False

        Returns
        -------
        veas : float or array
            Equivalent airspeed, in [ft/s] for si_units = True or [m/s] for
            si_units = False

        """

        r0, _, _ = AtmosProps.get_standard_props(0, si_units = si_units)
        return np.sqrt(2*qinf/r0)



    @staticmethod
    def alt_veas(alt, veas, alt_unit = 'ft', veas_unit = 'fps'):

        raise NotImplementedError

    @staticmethod
    def alt_mach(alt, mach, alt_unit = 'ft'):

        raise NotImplementedError

    @staticmethod
    def alt_qinf(alt, qinf, alt_unit = 'ft', qinf_unit = 'psf'):

        raise NotImplementedError

    @staticmethod
    def mach_qinf(mach, qinf, qinf_unit = 'psf'):

        raise NotImplementedError

    @staticmethod
    def mach_veas(mach, veas, veas_unit = 'ft_s'):

        raise NotImplementedError


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Basic altitude/airspeed data
    alts_ft = np.array([0, 5, 10, 15, 20, 40, 60])*1000
    vtas_fps = unit_convert(np.array([200, 250, 300, 350, 600, 700, 800]),
                            'kts', 'fps')

    fp1_im = FlightProps()
    fp1_si = FlightProps(si_units = True)

    fp1_im.alt_vtas(alts_ft, vtas_fps)
    fp1_si.alt_vtas(alts_ft, vtas_fps, alt_unit = 'ft', vtas_unit = 'fps')








