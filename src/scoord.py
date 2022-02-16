'''
Small library for converting between R,Z in the poloidal plane
of JET annd the wall poloidal S coordinate.

Written by Scott Silburn.
'''

import numpy as np
import os
from scipy.interpolate import interp1d


####   [ 1321, 1510, 'Tile 6' ]
def get_s_coord(R,Z,surface_tol=0.02,wall='ILW',plot=False):
    """
    Get the S coordinate for given R,Z coordinates.

    Parameters:

        R (float or array)  : R value(s) in metres. Must be the same shape as Z.

        Z (float or array)  : Z value(s) in metres. Must be the same shape as R.

        surface_tol (float) : Distance tolerance to the wall in metres. Any R,Z point \
                              further than this distance from the wall is assigned s = 0.

        wall (str)          : 'ILW' or 'C', which wall definition to use.

        plot (bool)         : Whether to show the wall contour and positions of given points.


    Returns:

        np.ndarray         : Array the same shape as input R,Z arrays containing the corresponding s coordinate \
                             in millimeters. Any points for which the s coordinate could not be determined or \
                             the provided point is not on the wall are assigned np.nan

    """
    if np.array(R).shape == ():
        R = np.array([R])
        Z = np.array([Z])
    else:
        R = np.array(R)
        Z = np.array(Z)

    # Load the S coordinate defition
    s,sR,sZ = get_s_definition(wall)

    sR = sR
    sZ = sZ
    s = s / 1e3

    out_shape = R.shape
    R = np.array(R).flatten()
    Z = np.array(Z).flatten()
    s_out = np.zeros(R.shape) + np.nan

    Rs = np.array( list(sR[1:] - sR[:-1]) + [sR[0] - sR[-1]] )
    Zs = np.array( list(sZ[1:] - sZ[:-1]) + [sZ[0] - sZ[-1]] )
    ds = np.array( list(s[1:] - s[:-1]) + [ np.sqrt( (Rs[-1] - Rs[0])**2 + (Zs[-1] - Zs[0])**2 ) ] )

    for i in range(R.size):

        # Vector from the start of each line segment to the given point.
        Rp = R[i] - sR
        Zp = Z[i] - sZ
        lp = np.sqrt(Rp**2 + Zp**2)

        # Length along the line segment is the dot product
        # of the vector from the start of the line segment to
        # the given point and the vector line segment, divided by the
        # segment length.
        l = (Rp*Rs + Zs*Zp) / ds

        # Numerical tolerance: any point which is within 0.1mm of the start
        # of the line segment is assumed to be on the start of the line segment.
        l[np.logical_and(l > -0.0001,l < 0)] = 0

        # Height of the given point from the line segment using Pythagoras.
        side_sq_diff = np.maximum(lp**2 - l**2,0)
        h = np.sqrt(side_sq_diff)

        # Normalise length to fraction of distance along line segment.
        l = l / ds

        # Line segments that the point could be on.
        seg_inds = np.argwhere( np.logical_and(l >= 0, l < 1) )

        if seg_inds.size == 0:
            continue

        # Select the line segment with the smallest height
        best_seg_ind = seg_inds[np.nanargmin(h[seg_inds])]
        h = h[best_seg_ind]
        l = l[best_seg_ind]
        s0 = s[best_seg_ind]

        # Accept the result if we're within the given height of the surface
        if h < surface_tol:
            s_out[i] = (s0 + l*ds[best_seg_ind]) * 1e3

    s_out = np.reshape(s_out,out_shape)


    if plot:
        import matplotlib.pyplot as plt
        plt.plot(sR,sZ,'k-x')
        plt.plot(R,Z,'ro')
        plt.axes().set_aspect('equal')

        for point_ind in range(R.size):
            if s_out[point_ind] > -1:
                rz_wall = get_R_Z(s_out[point_ind])
                plt.plot([R[point_ind], rz_wall[0][point_ind]],[Z[point_ind], rz_wall[1][point_ind]],'r--')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.show()

    return s_out


# Get R, Z corresponding to given s coordinate.
def get_R_Z(s_in,plot=False,wall='ILW'):

    if np.array(s_in).shape == ():
        s_in = np.array([s_in])
    else:
        s_in = np.array(s_in)

    # Load the S coordinate defition.
    s,sR,sZ = get_s_definition(wall)
    sR = interp1d(s,sR,bounds_error=False)
    sZ = interp1d(s,sZ,bounds_error=False)

    out_shape = s_in.shape

    s_in = np.array(s_in).flatten()
    R_out = np.zeros(s_in.shape)
    Z_out = np.zeros(s_in.shape)


    R_out = sR(s_in)
    Z_out = sZ(s_in)
    R_out = np.reshape(R_out,out_shape)
    Z_out = np.reshape(Z_out,out_shape)

    if plot:
        import matplotlib.pyplot as plt
        s,sR,sZ = get_s_definition(wall)
        plt.plot(sR,sZ,'k')
        plt.axes().set_aspect('equal')

        plt.plot(R_out,Z_out,'ro')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.show()

    return R_out,Z_out



'''
Get S coordinate definition.

Optional input: wall name: 'ILW' or 'C'

Returns 3 column vectors: s (mm), R(m), Z(m)
'''
def get_s_definition(wall='ILW'):

    if wall == 'ILW':
        s_def = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'s_coord_ILW_full.txt'))
    elif wall == 'C':
        s_def = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'sCoord_C.csv'),delimiter=',')

    return s_def[:,0],s_def[:,1],s_def[:,2]


def get_tile(s):
    tiles = np.zeros(s.shape)-1
    tiles[(s >= 162) & (s <= 414.9)] = 1
    tiles[(s >= 430.2) & (s <= 608.5)] = 3
    tiles[(s >= 711.6) & (s <= 925.5)] = 4
    tiles[(s >= 1061.8) & (s < 1125.5)] = 5.1
    tiles[(s >= 1125.5) & (s < 1190.1)] = 5.2
    tiles[(s >= 1190.1) & (s < 1254.3)] = 5.3
    tiles[(s >= 1254.3) & (s <= 1319)] = 5.4
    tiles[(s >= 1363.3) & (s <= 1552.6)] = 6
    tiles[(s >= 1622.2) & (s <= 1838.5)] = 7
    tiles[(s >= 1854.0) & (s <= 2133.9)] = 8
    return tiles


if __name__ == '__main__':

    print( get_s_coord(2.86517550966, -1.7112561243,plot=True) )
