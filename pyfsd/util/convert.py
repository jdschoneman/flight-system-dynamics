# -*- coding: utf-8 -*-
"""

Created on Tue Jul 29 21:48:16 2025

@author: C2_sm

"""


class Convert:


    FT_M = 12*0.0254
    M_FT = 1/FT_M
    FPS_MPS = FT_M
    MPS_FPS = 1/FPS_MPS
    KTS_FPS = 1.688
    FPS_KTS = 1/1.688



    def __init__(self):

        pass


def unit_convert(val, unit_in, unit_out):
    """


    Parameters
    ----------
    val : TYPE
        DESCRIPTION.
    unit_in : str
        DESCRIPTION.
    unit_out : str
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if unit_in == unit_out:
        return val

    convert_str = '%s_%s' % (unit_in.upper(), unit_out.upper())

    return val*Convert.__dict__[convert_str]

if __name__ == '__main__':

    ft = 1.0
    m = ft*12*0.0254
    assert m == unit_convert(ft, 'ft', 'm')
    assert ft == unit_convert(m, 'm', 'ft')


