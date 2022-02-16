import numpy as np
import sys
import matplotlib.pyplot as plt

import pandas as pd
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'CFC_composite_components.csv')

CFC_DATA = pd.read_csv(filename)

k_cfc = np.polyfit(CFC_DATA['degC'], CFC_DATA['Kx'],3)
C_cfc = np.polyfit(CFC_DATA['spec_heat_degC'].loc[range(
    12)], CFC_DATA['spec_heat'].loc[range(12)], 3)

K = lambda T: np.polyval(k_cfc,T)
C = lambda T: np.polyval(C_cfc,T)


def CFC_properties(T):

    k_a = 17.011
    k_b = 300.989
    k_T0 = 1442.6

    diff_a = 1.584e-5
    diff_b = 2.3015e-4
    diff_T0 = 405.5

    rho = 1800  # kg/m^3

    k = k_a+k_b/(1+T/k_T0)**2
    diff = diff_a+diff_b/(1+T/diff_T0)**2

    Cp = k/diff/rho

    return k, rho, Cp
#


def w_properties(T):

    k_a = 59.196
    k_b = 115.73
    k_T0 = 1989.97

    diff_a = 1.889e-5
    diff_b = 5.174e-5
    diff_T0 = 1526.67

    rho = 19300  # kg/m^3

    k = k_a+k_b/(1+T/k_T0)**2
    diff = diff_a+diff_b/(1+T/diff_T0)**2

    Cp = k/diff/rho

    return k, rho, Cp
#


def CFC(T):
    k = K(T)
    Cp = C(T)
    rho = 1800
    return k, rho, Cp


def get_material_properties(x): return CFC_properties(x)

def get_timestep(dx,T=80.):

    k,rho,cp = get_material_properties(T)
    alpha = k/(rho*cp)

    return dx**2/(4*alpha) / 2
#


def solve_2d(x,t,Q,depth=0.04,Tinit=80):
    # Square grid cells! Because I like easy things!
    dx = x[1] - x[0]

    k,rho,cp = get_material_properties(Tinit)
    alpha = k/(rho*cp)
    dt = t[1] - t[0]

    if 4*alpha*dt / dx**2 > 1:
        raise ValueError('Time step too big by factor {:.1f}, finite difference calculation will be unstable!'.format(4*alpha*dt / dx**2))

    Told = np.zeros([int(depth/dx)+2,x.size+2]) + Tinit
    Tnew = np.zeros(Told.shape)
    Tsurf = np.zeros([t.size,x.size])

    for Tind in range(t.size):

        for xind in range(1,Told.shape[1]-1):
            for depthind in range(1,Told.shape[0]-1):

                # Get the temperature dependent material properties
                k,rho,cp = get_material_properties(Told[depthind-1:depthind+2,xind-1:xind+2])
                alpha = (k/(rho*cp)).mean()

                # Apply some heat flux boundary conditions

                # Incoming heat flux from plasma
                if depthind == 1:
                    q = (Q[Tind,xind-1]- 0.1*5.67e-8*(Told[depthind,xind]+273.15)**4 ) * dx #-0.1*5.67e-8*(Told[depthind,xind]+273.15)**4 * dx
                # Allow radiative cooling from bottom & sides
                elif depthind == Told.shape[0]-2 or xind == 1 or xind == Told.shape[1]-2:
                    q =  -0.1*5.67e-8*(Told[depthind,xind]+273.15)**4 * dx
                else:
                    q = 0

                # Update node temperature
                Tnew[depthind,xind] = Told[depthind,xind] + (dt*alpha/dx**2) * (Told[depthind-1,xind] + Told[depthind+1,xind] + Told[depthind,xind-1] + Told[depthind,xind+1] - 4*Told[depthind,xind] + q/k.mean())

        # Boundary conditions at bottom and sides - insulated boundary
        Tnew[0,:] = Tnew[1,:]
        Tnew[-1,:] = Tnew[-2,:]
        Tnew[:,0] = Tnew[:,1]
        Tnew[:,-1] = Tnew[:,-2]

        # Get the surface temperature
        depth_t_grad = (Tnew[2,1:-1] - Tnew[1,1:-1])
        Tsurf[Tind,:] = Tnew[1,1:-1] - depth_t_grad

        Told = Tnew.copy()
    Told[0, 1:-1] = Tsurf[-1,:]
    return Tsurf, Told[:-1,:]
#


def solve_2d_updating(x,t,Q,Told,depth=0.04):

    dx = x[1] - x[0]
    dt = t[1] - t[0]

    Tnew = np.zeros(np.shape(Told))
    Tsurf = np.zeros([t.size,x.size])
    # Tsurf = np.zeros([t.size,x.size])

    for Tind in range(t.size):
        for xind in range(1,Told.shape[1]-1):
            for depthind in range(1,Told.shape[0]-1):

                # Get the temperature dependent material properties
                k,rho,cp = get_material_properties(Told[depthind-1:depthind+2,xind-1:xind+2])
                alpha = (k/(rho*cp)).mean()

                # Apply some heat flux boundary conditions

                # Incoming heat flux from plasma
                if depthind == 1:
                    q = ((Q[Tind, xind-1] - 0.1*5.67e-8*(Told[depthind, xind]+273.15)** 4) * dx) 
                # Allow radiative cooling from bottom & sides
                elif depthind == Told.shape[0]-2 or xind == 1 or xind == Told.shape[1]-2:
                    q =  -0.1*5.67e-8*(Told[depthind,xind]+273.15)**4 * dx
                else:
                    q = 0

                # Update node temperature
                Tnew[depthind,xind] = Told[depthind,xind] + (dt*alpha/dx**2) * (Told[depthind-1,xind] + Told[depthind+1,xind] + Told[depthind,xind-1] + Told[depthind,xind+1] - 4*Told[depthind,xind] + q/k.mean())

        # Boundary conditions at bottom and sides - insulated boundary
        Tnew[0,:] = Tnew[1,:]
        Tnew[-1,:] = Tnew[-2,:]
        Tnew[:,0] = Tnew[:,1]
        Tnew[:,-1] = Tnew[:,-2]

        # Get the surface temperature
        depth_t_grad = (Tnew[2,1:-1] - Tnew[1,1:-1])
        Tsurf[Tind,:] = Tnew[1,1:-1] - depth_t_grad

        Told = Tnew.copy()


    return Tsurf, Tnew
#


# x needs to be in metres.
def solve_to_tmax(x,t,Q,tstop,assertion_time,depth=0.04,Tinit=80.):

    get_material_properties = CFC_properties
    # Square grid cells! Because I like easy things!
    dx = x[1] - x[0]

    k,rho,cp = get_material_properties(Tinit)
    alpha = k/(rho*cp)
    dt = t[1] - t[0]
    t = [0]
    if 4*alpha*dt / dx**2 > 1:
        raise ValueError('Time step too big by factor {:.1f}, finite difference calculation will be unstable!'.format(4*alpha*dt / dx**2))

    Told = np.zeros([int(depth/dx)+2,x.size+2]) + Tinit
    Tnew = np.zeros(Told.shape)
    Tmax = []

    lud = 0
    Tind = -1
    asserted_time = 0.

    while asserted_time < assertion_time:
        Tind = Tind + 1
        t.append(t[-1] + dt)
        if Tind*dt > lud + 0.5:
            print('Time: {:.2f} s, Tmax = {:.0f} deg C (asserted for {:.0f}ms)...'.format(Tind * dt,Tmax[-1],max(0,asserted_time*1e3)))
            lud = Tind * dt
            sys.stdout.flush()

        for xind in range(1,Told.shape[1]-1):
            for depthind in range(1,Told.shape[0]-1):

                # Get the temperature dependent material properties
                k,rho,cp = get_material_properties(Told[depthind-1:depthind+2,xind-1:xind+2])
                alpha = (k/(rho*cp)).mean()

                # Apply some heat flux boundary conditions

                # Incoming heat flux from plasma
                if depthind == 1:
                    q = (Q[Tind % Q.shape[0],xind-1]- 0.1*5.67e-8*(Told[depthind,xind]+273.15)**4 ) * dx
                # Allow radiative cooling from bottom & sides
                elif depthind == Told.shape[0]-2 or xind == 1 or xind == Told.shape[1]-2:
                    q =  -0.1*5.67e-8*(Told[depthind,xind]+273.15)**4 * dx
                else:
                    q = 0

                # Update node temperature
                Tnew[depthind,xind] = Told[depthind,xind] + (dt*alpha/dx**2) * (Told[depthind-1,xind] + Told[depthind+1,xind] + Told[depthind,xind-1] + Told[depthind,xind+1] - 4*Told[depthind,xind] + q/k.mean())

        # Boundary conditions at bottom and sides - insulated boundary
        Tnew[0,:] = Tnew[1,:]
        Tnew[-1,:] = Tnew[-2,:]
        Tnew[:,0] = Tnew[:,1]
        Tnew[:,-1] = Tnew[:,-2]

        # Get the surface temperature
        depth_t_grad = (Tnew[2,1:-1] - Tnew[1,1:-1])
        Tsurf = Tnew[1,1:-1] - depth_t_grad/2.

        Told = Tnew.copy()
        Tmax.append(Tsurf.max())
        if Tmax[-1] > tstop:
            asserted_time = asserted_time + dt
        else:
            asserted_time = -dt

    print('Stopped at t = {:.2f}s'.format(max(t)))
    return np.array(t),np.array([Tinit] + Tmax)
#


if __name__ == '__main__':

    t = np.linspace(0,0.5,500)
    x = np.linspace(0,0.1,100)

    Q = np.zeros([t.size,x.size])
    Q[:,45:55] = 15e6

    Tsurf = solve_2d(x,t,Q)

    import matplotlib.pyplot as plt
    plt.imshow(Tsurf.T,aspect='auto')
    plt.colorbar()
    plt.show()
#


"""

"""

