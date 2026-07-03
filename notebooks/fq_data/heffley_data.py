# -*- coding: utf-8 -*-
"""

Utilities for reading and processing the Heffley aircraft stability and
control derivative dataset.

This module provides functionality for importing aircraft stability and
control derivative data from CSV files derived from the Heffley flying
qualities database. In addition to loading the original dimensional
derivatives, the module computes the corresponding nondimensional body-axis
stability and control coefficients commonly used in flight dynamics
analysis.

The expected CSV format consists of a header row containing variable names,
a second row containing engineering units, and subsequent rows containing
flight condition data.

Classes
-------
StabilityDerivatives
    Container class for aircraft stability derivatives, reference geometry,
    and derived nondimensional coefficients.

"""

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


from scipy import ndimage
import itertools



class StabilityDerivatives:
    """
    Container for aircraft stability and control derivative data.

    The class stores the dimensional stability derivatives imported from a
    Heffley-format CSV file together with aircraft reference quantities and
    computed nondimensional aerodynamic coefficients.

    After loading a dataset with :meth:`read_csv`, the original data are
    accessible through the ``dataframe`` attribute while commonly used
    nondimensional derivatives are available as NumPy arrays.

    Parameters
    ----------
    name : str, optional
        User-defined name used to identify the aircraft or dataset.
        The default is ``''``.

    Attributes
    ----------
    name : str
        Aircraft or dataset name.

    dataframe : pandas.DataFrame or None
        Raw stability derivative data read from the CSV file.

    units : dict
        Dictionary mapping each dataframe column to its engineering units.

    g : float or None
        Gravitational acceleration corresponding to the dataset unit system.

    Sref : float
        Wing reference area.

    Bref : float
        Wing reference span.

    Cref : float
        Mean aerodynamic chord.
    """


    def __init__(self, name=''):
        """
        Initialize an empty stability derivative dataset.

        Parameters
        ----------
        name : str, optional
            Descriptive name for the aircraft or dataset.
            The default is ``''``.
        """

        self.name = name

        # Initialize empty attributes
        self.dataframe = None
        self.units = dict()
        self.g = None
        self.derivative_frame = None


    def read_csv(self, filename, build_coeffs = True, update_name = True):
        """
        Read a Heffley-format stability derivative CSV file.

        The CSV file is expected to contain variable names in the first row,
        engineering units in the second row, and flight condition data in all
        subsequent rows.

        Aircraft reference dimensions, unit information, and the appropriate
        value of gravitational acceleration are extracted automatically. If
        requested, nondimensional body-axis stability and control coefficients
        are computed immediately after loading.

        Parameters
        ----------
        filename : str or path-like
            Path to the CSV file.

        build_coeffs : bool, optional
            If True, compute nondimensional stability and control coefficients
            using :meth:`build_body_coeffs`. The default is True.

        update_name : bool, optional
            Updates the object name field with the aircraft name from the CSV,
            using index 0. The default is True

        Returns
        -------
        None

        Notes
        -----
        The current implementation assumes the CSV structure used throughout
        the Heffley dataset, with a dedicated units row immediately following
        the column headers.
        """

        self.dataframe = read_csv(filename, skiprows = (1, ))
        with open(filename) as fid:
            fid.readline()
            units = fid.readline().strip().split(',')

        for column, unit in zip(self.dataframe.columns, units):
            self.units[column] = unit

        # Reference dimensions
        self.Sref = self.dataframe.Ref_Area[0]
        self.Bref = self.dataframe.Ref_Span[0]
        self.Cref = self.dataframe.Ref_MAC[0]

        # Set gravity based on weight unit
        if self.units['Weight'] == 'lb':
            self.g = 32.174
        else:
            self.g = 9.81

        # Build coeffs if flag is true
        if build_coeffs:
            self.build_body_coeffs(self.dataframe)

        # Update name if flag is true
        if update_name:
            self.name = self.dataframe.Aircraft[0]

    def build_body_coeffs(self, df):
        """
        Compute nondimensional body-axis stability and control coefficients.

        Dimensional stability derivatives contained in the supplied dataframe
        are converted to the corresponding nondimensional aerodynamic
        coefficients using the aircraft reference geometry, inertia properties,
        dynamic pressure, and flight condition.

        The resulting coefficients are stored as NumPy arrays attached to the
        class instance.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing dimensional stability derivatives and aircraft
            reference quantities.

        Returns
        -------
        None

        Notes
        -----
        The current implementation assumes negligible products of inertia
        (Ixz ≈ 0) when recovering the lateral-directional rolling and yawing
        moment coefficients.

        Airspeed is assumed to be available from the flight condition data.
        """

        # Alpha derivatives
        u0 = df.VTAS
        force_alpha_scale = df.Qinf*self.Sref*self.g/(df.Weight*u0)
        pitch_alpha_scale = df.Qinf*self.Sref*self.Cref/(df.Iyy*u0)
        self.Cx_alpha = np.array(df.XW/force_alpha_scale)
        self.Cz_alpha = np.array(df.ZW/force_alpha_scale)
        self.Cm_alpha = np.array(df.MW/pitch_alpha_scale)


        # Alpha dot
        force_alpha_dot_scale = force_alpha_scale*self.Cref/(2*u0)
        pitch_alpha_dot_scale = df.Qinf*self.Sref*self.Cref**2/(2*df.Iyy*u0**2)
        self.Cz_alpha_dot = np.array(df.ZWD/force_alpha_dot_scale)
        self.Cm_alpha_dot = np.array(df.MWD/pitch_alpha_dot_scale)

        # Pitch rate
        force_q_scale = force_alpha_scale*self.Cref/2.0
        pitch_q_scale = df.Qinf*self.Sref*self.Cref**2/(2*df.Iyy*u0)
        self.Cz_q = np.array(df.ZQ/force_q_scale)
        self.Cm_q = np.array(df.MQ/pitch_q_scale)

        # Elevator control
        force_de_scale = df.Qinf*self.Sref*self.g/(df.Weight)
        pitch_de_scale = df.Qinf*self.Sref*self.Cref/(df.Iyy)
        self.Cx_de = np.array(df.XDE/force_de_scale)
        self.Cz_de = np.array(df.ZDE/force_de_scale)
        self.Cm_de = np.array(df.MDE/pitch_de_scale)

        self.calc_CL_CD()

        # TODO -- recover the exact L/N terms with non-negligible Ixz
        # Beta
        force_beta_scale = df.Qinf*self.Sref*self.g/df.Weight  # Has beta directly
        roll_yaw_beta_scale = df.Qinf*self.Sref*self.Bref   # Has beta directly
        self.Cy_beta = np.array(df.YB/force_beta_scale)
        Lb, Nb = self.unstar_roll_yaw(df.LB, df.NB, df.Ixx, df.Izz, df.Ixz)
        # Lb = df.LB*df.Ixx
        # Nb = df.NB*df.Izz
        self.Cl_beta = np.array(Lb/roll_yaw_beta_scale)
        self.Cn_beta = np.array(Nb/roll_yaw_beta_scale)

        # Roll rate
        # force_pr_scale = df.Qinf*self.Sref*self.g*self.Bref/(2*df.Weight*u0)
        roll_yaw_pr_scale = df.Qinf*self.Sref*self.Bref**2/(2*u0)
        Lp, Np = self.unstar_roll_yaw(df.LP, df.NP, df.Ixx, df.Izz, df.Ixz)
        self.Cl_p = np.array(Lp/roll_yaw_pr_scale)
        self.Cn_p = np.array(Np/roll_yaw_pr_scale)

        # Yaw rate
        Lr, Nr = self.unstar_roll_yaw(df.LR, df.NR, df.Ixx, df.Izz, df.Ixz)
        self.Cl_r = np.array(Lr/roll_yaw_pr_scale)
        self.Cn_r = np.array(Nr/roll_yaw_pr_scale)

        # Rudder/aileron
        force_dar_scale = df.Qinf*self.Sref*self.g/(df.Weight)
        roll_yaw_dar_scale = df.Qinf*self.Sref*self.Bref
        self.Cy_da = np.array(df.YDA/force_dar_scale)
        Lda, Nda = self.unstar_roll_yaw(df.LDA, df.NDA, df.Ixx, df.Izz, df.Ixz)
        self.Cl_da = np.array(Lda/roll_yaw_dar_scale)
        self.Cn_da = np.array(Nda/roll_yaw_dar_scale)

        self.Cy_dr = np.array(df.YDR/force_dar_scale)
        Ldr, Ndr = self.unstar_roll_yaw(df.LDR, df.NDR, df.Ixx, df.Izz, df.Ixz)
        self.Cl_dr = np.array(Ldr/roll_yaw_dar_scale)
        self.Cn_dr = np.array(Ndr/roll_yaw_dar_scale)

        self.derivative_frame = 'body'

    def body_to_stab(self, alpha_col="Alpha"):
        """
        Transform lateral-directional derivatives from body axes to stability axes.
        """

        if 'stab' in self.derivative_frame.lower():
            raise ValueError('Derivatives already in body frame for StabilityDerivatives %s'
                             % self.name)

        alpha = np.asarray(self.dataframe[alpha_col], dtype=float)

        if 'deg' in self.units[alpha_col].lower():
            alpha = np.deg2rad(alpha)

        ca = np.cos(alpha)
        sa = np.sin(alpha)

        def rotate_pair(Cl_body, Cn_body):
            Cl_stab = Cl_body*ca + Cn_body*sa
            Cn_stab = -Cl_body*sa + Cn_body*ca
            return Cl_stab, Cn_stab

        def rotate_rate_derivative_matrix(Cl_p_body, Cn_p_body,
                                          Cl_r_body, Cn_r_body):
            """
            Transform lateral-directional roll/yaw rate derivatives from body axes
            to stability axes.

            This accounts for both the transformation of the rolling/yawing moment
            components and the transformation of the roll/yaw rate components.
            """

            Cl_p_stab = (ca**2 * Cl_p_body
                         + ca*sa * Cn_p_body
                         + ca*sa * Cl_r_body
                         + sa**2 * Cn_r_body)

            Cl_r_stab = (-ca*sa * Cl_p_body
                         - sa**2 * Cn_p_body
                         + ca**2 * Cl_r_body
                         + ca*sa * Cn_r_body)

            Cn_p_stab = (-ca*sa * Cl_p_body
                         + ca**2 * Cn_p_body
                         - sa**2 * Cl_r_body
                         + ca*sa * Cn_r_body)

            Cn_r_stab = (sa**2 * Cl_p_body
                         - ca*sa * Cn_p_body
                         - ca*sa * Cl_r_body
                         + ca**2 * Cn_r_body)

            return Cl_p_stab, Cn_p_stab, Cl_r_stab, Cn_r_stab

        # Rotate the Cx and Cz terms, with Cx analogous to Cl and Cz analogous
        # to Cn
        self.Cx_alpha, self.Cz_alpha = rotate_pair(self.Cx_alpha,
                                                   self.Cz_alpha)

        # Side-force derivative: body Y and stability Y are normally identical
        # for this alpha-only rotation, so Cy terms do not need transformation.
        self.Cl_beta, self.Cn_beta = rotate_pair(self.Cl_beta, self.Cn_beta)
        # self.Cl_p, self.Cn_p = rotate_pair(self.Cl_p, self.Cn_p)
        # self.Cl_r, self.Cn_r = rotate_pair(self.Cl_r, self.Cn_r)
        self.Cl_da, self.Cn_da = rotate_pair(self.Cl_da, self.Cn_da)
        self.Cl_dr, self.Cn_dr = rotate_pair(self.Cl_dr, self.Cn_dr)

        self.Cl_p, self.Cn_p, self.Cl_r, self.Cn_r = rotate_rate_derivative_matrix(self.Cl_p,
                                                                                   self.Cn_p,
                                                                                   self.Cl_r,
                                                                                   self.Cn_r)

        self.derivative_frame = 'stability'

    def calc_CL_CD(self, alpha_col = 'Alpha'):
        """
        Calculate lift and drag coefficient derivatives with respect to alpha

        Returns
        -------
        None.

        """

        if self.derivative_frame == 'body':
            alpha = np.asarray(self.dataframe[alpha_col], dtype=float)

            if 'deg' in self.units[alpha_col].lower():
                alpha = np.deg2rad(alpha)

            ca = np.cos(alpha)
            sa = np.sin(alpha)

            def rotate_pair(Cx_body, Cz_body):
                Cx_stab = Cx_body*ca + Cz_body*sa
                Cz_stab = -Cx_body*sa + Cz_body*ca
                return Cx_stab, Cz_stab

            Cx_alpha, Cz_alpha = rotate_pair(self.Cx_alpha,
                                             self.Cz_alpha)
            Cx_de, Cz_de = rotate_pair(self.Cx_de,
                                       self.Cz_de)

        else:
            Cx_alpha = self.Cx_alpha
            Cz_alpha = self.Cz_alpha

            Cx_de = self.Cx_de
            Cz_de = self.Cz_de

        self.CL_alpha = -Cz_alpha
        self.CD_alpha = -Cx_alpha

        self.CL_de = -Cz_de
        self.CD_de = -Cx_de

    def plot_derivative(
        self,
        derivative,
        ax = None,
        xvar = "Mach",
        filters=None,
        label = None,
        marker = "o",
        linestyle = "-",
        grid = True,
        transform = None,
        **plot_kwargs):
        """
        Plot a stability or control derivative against a selected flight variable.

        Parameters
        ----------
        derivative : str
            Name of the derivative to plot. This may be either a computed class
            attribute, such as ``'Cm_alpha'`` or ``'Cl_beta'``, or a column name
            in ``dataframe``.

        ax : matplotlib.axes.Axes, optional
            Existing axes object on which to draw the plot. If None, a new figure
            and axes are created.

        xvar : str, optional
            Name of the dataframe column to use for the x-axis. The default is
            ``'Mach'``.

        label : str, optional
            Plot label. If None, ``self.name`` is used when available.

        marker : str, optional
            Marker style passed to ``Axes.plot``. The default is ``'o'``.

        linestyle : str, optional
            Line style passed to ``Axes.plot``. The default is ``'-'``.

        grid : bool, optional
            If True, enable grid lines on the axes. The default is True.

        transform : callable, optional
            Callable transform of (x, y) for plotting data in alternate spaces
            (e.g. pixel space on images). The default is None which applies no
            transform

        **plot_kwargs
            Additional keyword arguments passed directly to ``Axes.plot``.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the plotted derivative.

        Raises
        ------
        ValueError
            If no dataframe has been loaded.

        AttributeError
            If ``derivative`` is not found as either a class attribute or
            dataframe column.

        KeyError
            If ``xvar`` is not found in ``dataframe``.
        """

        if self.dataframe is None:
            raise ValueError("No data loaded. Call read_csv() before plotting.")

        mask = self._filter_index(filters)

        if xvar not in self.dataframe.columns:
            raise KeyError(f"x-axis variable '{xvar}' not found in dataframe.")

        if ax is None:
            _, ax = plt.subplots()

        x = self.dataframe.loc[mask, xvar]

        if hasattr(self, derivative):
            y = np.asarray(getattr(self, derivative))[mask]
        elif derivative in self.dataframe.columns:
            y = self.dataframe.loc[mask, derivative]
        else:
            raise AttributeError(
                f"Derivative '{derivative}' not found as an attribute "
                "or dataframe column.")

        if transform is not None:
            x, y = transform(x, y)

        if label is None:
            label = self.name if self.name else derivative

        ax.plot(x, y, marker=marker, linestyle=linestyle, label=label,
                **plot_kwargs,)

        ax.set_xlabel(self._axis_label(xvar))
        ax.set_ylabel(derivative)
        ax.set_title(f"{derivative} vs {xvar}")

        if grid:
            ax.grid(True)

        return ax

    def _filter_index(self, filters=None):
        """
        Build a boolean row mask from simple dataframe filter criteria.
        """

        if filters is None:
            return np.ones(len(self.dataframe), dtype=bool)

        mask = np.ones(len(self.dataframe), dtype=bool)

        for column, criterion in filters.items():
            if column not in self.dataframe.columns:
                raise KeyError(f"Filter column '{column}' not found.")

            values = self.dataframe[column]

            if isinstance(criterion, tuple) and len(criterion) == 2:
                lower, upper = criterion
                mask &= (values >= lower) & (values <= upper)

            elif isinstance(criterion, (list, set)):
                mask &= values.isin(criterion)

            else:
                mask &= values == criterion

        return mask

    def _axis_label(self, variable):
        """
        Build an axis label using dataframe units when available.

        Parameters
        ----------
        variable : str
            Variable name.

        Returns
        -------
        label : str
            Axis label containing the variable name and units, if available.
        """

        unit = self.units.get(variable, "")

        if unit:
            return f"{variable} ({unit})"

        return variable

    @staticmethod
    def unstar_roll_yaw(Lstar, Nstar, Ixx, Izz, Ixz):
        """
        Recover uncoupled dimensional rolling/yawing moment derivatives
        from starred equations-of-motion derivatives when Ixz is nonzero.

        For dimensional moment derivatives, Heffley’s L* / N* terms are
        already “starred” body-axis equations-of-motion terms:

            Lstar = (Izz*L + Ixz*N) / (Ixx*Izz - Ixz^2)
            Nstar = (Ixz*L + Ixx*N) / (Ixx*Izz - Ixz^2)

        So to recover the uncoupled dimensional moments:

            L = Ixx*Lstar - Ixz*Nstar
            N = Izz*Nstar - Ixz*Lstar

        Note that after “un-starring,” the nondimensional roll/yaw scales
        should not include Ixx or Izz; those were only appropriate when
        treating the stored values as angular acceleration derivatives directly.


        Parameters
        ----------
        Lstar : array
            Coupled roll rate angular acceleration vector
        Nstar : array
            Coupled yaw rate angular acceleration vector

        Returns
        -------
        L : array
            Uncoupled roll moment vector
        N : array
            Uncoupled yaw moment vector

        """

        L = Ixx*Lstar - Ixz*Nstar
        N = Izz*Nstar - Ixz*Lstar

        return L, N



def make_image_axis_transform(image, color, origin_value, x_axis_value,
                              y_axis_value, threshold=0.45, verbose = False):
    """
    Build a data-to-pixel transform from three colored calibration dots.

    Parameters
    ----------
    image : ndarray
        Image array as returned by ``matplotlib.pyplot.imread`` or similar.

    color : {'red', 'green', 'blue'}
        Dominant dot color to detect.

    origin_value : tuple of float
        Data-space ``(x, y)`` value corresponding to the dot at the plot
        origin.

    x_axis_value : tuple of float
        Data-space ``(x, y)`` value corresponding to the dot placed along
        the x-axis extent.

    y_axis_value : tuple of float
        Data-space ``(x, y)`` value corresponding to the dot placed along
        the y-axis extent.

    threshold : float, optional
        Dominance threshold used to identify colored pixels. The default is
        0.45.

    Returns
    -------
    transform : callable
        Function mapping data-space ``x`` and ``y`` arrays to image pixel
        coordinates ``x_pix`` and ``y_pix``.
    """

    img = np.asarray(image)

    if img.dtype.kind in "ui":
        img = img.astype(float) / np.iinfo(img.dtype).max

    if img.shape[-1] == 4:
        img = img[..., :3]

    color_index = {"red": 0, "green": 1, "blue": 2}[color.lower()]
    other_indices = [i for i in range(3) if i != color_index]

    dominant = img[..., color_index]
    others = np.maximum(img[..., other_indices[0]], img[..., other_indices[1]])

    mask = (dominant - others) > threshold

    labels, n_labels = ndimage.label(mask)

    if n_labels != 3:
        raise ValueError(f"Expected 3 {color} dots, found {n_labels} regions.")

    centers = np.array(
        ndimage.center_of_mass(mask, labels, range(1, n_labels + 1))
    )

    # center_of_mass returns (row, col), i.e. (y_pixel, x_pixel)
    pix = np.column_stack((centers[:, 1], centers[:, 0]))
    if verbose: print(pix)
    origin_value = np.asarray(origin_value, dtype=float)
    x_axis_value = np.asarray(x_axis_value, dtype=float)
    y_axis_value = np.asarray(y_axis_value, dtype=float)

    data_pts = np.vstack((origin_value, x_axis_value, y_axis_value))

    # Match detected dots to known roles using data-space structure:
    # origin shares y with x-axis point and x with y-axis point.
    # In pixel space, x-axis dot is farthest mostly in image x,
    # y-axis dot is farthest mostly in image y.
    #
    # Try all assignments and choose the one whose affine map has the
    # smallest residual.

    # best_error = np.inf
    # best_affine = None
    # best_perm = None

    data_aug = np.column_stack((data_pts, np.ones(3)))

    # Put in some logic to pull out the X/Y/origin points
    inds_available = {0, 1, 2}
    if x_axis_value[0] > origin_value[0]:  # X axis point is positive
        ind_xaxis = np.argmax(pix[:, 0])
    else:
        ind_xaxis = np.argmin(pix[:, 0])
    if y_axis_value[1] > origin_value[1]:  # Y axis point is positive
        ind_yaxis = np.argmin(pix[:, 1])  # Y axis increases DOWN
    else:
        ind_yaxis = np.argmax(pix[:, 1])
    inds_available.remove(ind_xaxis)
    inds_available.remove(ind_yaxis)
    ind_origin = inds_available.pop()

    pix_inds = [ind_origin, ind_xaxis, ind_yaxis]
    pix_ordered = pix[pix_inds, :]

    affine_x = np.linalg.solve(data_aug, pix_ordered[:, 0])
    affine_y = np.linalg.solve(data_aug, pix_ordered[:, 1])

    if verbose: print('Final Order:', pix_inds)

    def transform(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x_pix = affine_x[0]*x + affine_x[1]*y + affine_x[2]
        y_pix = affine_y[0]*x + affine_y[1]*y + affine_y[2]

        return x_pix, y_pix

    return transform


if __name__ == '__main__':

    aircraft = 'F104'
    filename = 'data/cr_2144/%s.csv' % aircraft

    stab_der = StabilityDerivatives()
    stab_der.read_csv(filename)

    stab_der.body_to_stab()

    # for inda, alt in enumerate(np.unique(stab_der.dataframe.Altitude)):
    #     if not inda:
    #         ax = stab_der.plot_derivative('Cm_alpha', filters = {'Altitude': alt},
    #                                       linestyle = '')
    #     else:
    #         stab_der.plot_derivative('Cm_alpha', ax = ax, filters = {'Altitude': alt},
    #                                  linestyle = '')

    # ax.legend()


    # Alternate appraoch of plotting items on
    image = plt.imread('data/cr_2144/screenshots/%s/cma_cmq.png' % aircraft)
    if aircraft == 'F4C':
        transform_cma = make_image_axis_transform(image, color="red",
                                                  origin_value=(0.0, 0.0),
                                                  x_axis_value=(2.2, 0.0),
                                                  y_axis_value=(0.0, -0.8))
        transform_cmq = make_image_axis_transform(image, color="blue",
                                                  origin_value=(0.0, 0.0),
                                                  x_axis_value=(2.2, 0.0),
                                                  y_axis_value=(0.0, -3))

    else:
        transform_cma = make_image_axis_transform(image, color="red",
                                                  origin_value=(0.0, 0.0),
                                                  x_axis_value=(2.0, 0.0),
                                                  y_axis_value=(0.0, -3.0))
        transform_cmq = make_image_axis_transform(image, color="blue",
                                                  origin_value=(0.0, 0.0),
                                                  x_axis_value=(2.0, 0.0),
                                                  y_axis_value=(0.0, -12))

    plt.figure()
    plt.imshow(image)

    for inda, alt in enumerate(np.unique(stab_der.dataframe.Altitude)):
        color = 'C%i' % inda
        stab_der.plot_derivative('Cm_alpha', ax = plt.gca(), filters = {'Altitude': alt},
                                 linestyle = '', transform = transform_cma, color = color)
        stab_der.plot_derivative('Cm_q', ax = plt.gca(), filters = {'Altitude': alt},
                                 linestyle = '', transform = transform_cmq, color = color)
        stab_der.plot_derivative('Cm_alpha_dot', ax = plt.gca(), filters = {'Altitude': alt},
                                 linestyle = '', transform = transform_cmq, color = color)


    plt.gca().axis('off')


