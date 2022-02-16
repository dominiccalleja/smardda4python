
import AnalysisS4P as AS
import heatequation as ht
import smardda_updating as up
import TStools as TS

from fenics import *
import mshr as msh
from dolfin import *

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib as mpl

from scipy import signal
import pandas as pd
import pickle
import dill
import os
import math

def interpolate_solution(T, X, Y):
    Z = np.zeros([len(X), len(Y)])
    for i in range(len(X)):
        for j in range(len(Y)):
            Z[i, j] = T(Point(X[i], Y[j]))
    return Z


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'CFC_composite_components.csv')

CFC_DATA = pd.read_csv(filename)

k_cfc = np.polyfit(CFC_DATA['degC'], CFC_DATA['Kx'], 3)
C_cfc = np.polyfit(CFC_DATA['spec_heat_degC'].loc[range(
    12)], CFC_DATA['spec_heat'].loc[range(12)], 3)


def K(T): return np.polyval(k_cfc, T)
def C(T): return np.polyval(C_cfc, T)

"""
Material Properties
"""

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

"""
def Kappa(T):
    k_a = 17.011
    k_b = 300.989
    k_T0 = 1442.6
    return k_a+k_b/(1+T/k_T0)**2

def Kappa_re(T):
    k_a = 17.011
    k_b = 300.989
    k_T0 = 1442.6
    return 1/(k_a+k_b/((1+T/k_T0)**2))

def CP(T):
    k, rho, Cp = CFC_properties(T)
    rho = 1800

    return Cp*rho


"""

def Kappa(T):
    return K(T)
 
def Kappa_re(T):
    return 1/(Kappa(T))


def CP(T):
    rho = 1800
    return C(T)*rho


def diffu(T):
    rho=1800
    return K(T)/CP(T)


sigma = Constant(5.67E-8)

def epsilon(T):
    #eps = [0.1357, 0.529]
    #temp = [50., 3500.]
    eps= [0.18,0.27]
    temp =[400,1200]
    eps = np.polyfit(temp, eps, 1)
    return np.polyval(eps, T)*sigma



"""
Geometry Properties
"""

x_r = np.array([2.8041, 2.9868])
length = x_r[1]-x_r[0]
depth = 0.04

def construct_boundaries(mesh,length,depth):
    """
    Boundaries
    """
    tol = 1E-14

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0, tol)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], length, tol)

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0, tol)

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], depth, tol)

    # Initialize sub-domain instances
    left = Left()
    top = Top()
    right = Right()
    bottom = Bottom()

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)

    left.mark(boundaries, 1)
    top.mark(boundaries, 2)
    right.mark(boundaries, 3)
    bottom.mark(boundaries, 4)
    return boundaries

def build_solver(mesh_density,poly_degree,length=length,depth=depth):
    """
    Mesh Generation
    """
    domain = msh.Rectangle(Point(0, 0), Point(length, depth))
    mesh = msh.generate_mesh(domain, mesh_density)
    V = FunctionSpace(mesh, 'P', poly_degree)
    boundaries = construct_boundaries(mesh,length,depth)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    """
    Problem initialisation
    """

    T = TrialFunction(V)  # TrialFunction
    v = TestFunction(V)
    T_ = Function(V)
    
    return T, T_, v, V, ds, mesh

def recyle(Q):
    return r_a+r_b/(1+Q/r_T0)

def build_problem(Q, time, time_steps, T_init, T_ambient, mesh_density=30, poly_degree=2):

    dt = time / time_steps  # time step size

    T, T_, v, V, ds, mesh = build_solver(
        mesh_density, poly_degree, length=length, depth=depth)

    T_n = interpolate(Constant(T_init), V)
    
    """
    radiation = Kappa_re(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(0) + \
                Kappa_re(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(1) + \
                Kappa_re(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(2) + \
                Kappa_re(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(3)
    #radiation = Kappa(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(0) + \
    #    Kappa(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(1) + \
    #    Kappa(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(2) + \
    #    Kappa(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(3)

    F = -CP(T)*T*v*dx - dt*inner(Kappa(T)*nabla_grad(T), nabla_grad(v))*dx + \
        Q*v*dt*ds(2) + CP(T_n)*T_n*v*dx - radiation
    """
    Tvessel = 100
    radiation = Kappa(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4) * v*dt*ds(2) + \
                0.1E3 * (T - Tvessel)*v*dt*ds(4) + \
                0.1E3 * (T - Tvessel) * v*dt*ds(1) + \
                0.1E3 * (T - Tvessel)*v*dt*ds(3)

                #diffu(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4) * v*dt*ds(1) + \
                #diffu(T)*epsilon(T)*((T+273.15) ** 4 -(T_ambient+273.15)**4) * v*dt*ds(3)

    F = -CP(T)*T*v*dx - dt*inner(Kappa(T)*nabla_grad(T), nabla_grad(v))*dx + \
        Q*v*dt*ds(2) + CP(T)*T_n*v*dx - radiation

    #F = -T*v*dx + T_n*v*dx + dt * inner(Kappa(T)*nabla_grad(T), nabla_grad(v))*dx + \
    #     - dt* diffu(T)* Q*v*ds(2) -radiation


    T_ = Function(V)   # the most recently computed solution
    F = action(F, T_)
    J = derivative(F, T_, T)

    return T_, T_n, F, J, Q, mesh


def build_problem_noQ( time, time_steps, T_init, T_ambient, mesh_density=30, poly_degree=2):

    dt = time / time_steps  # time step size

    T, T_, v, V, ds, mesh = build_solver(
        mesh_density, poly_degree, length=length, depth=depth)

    T_n = interpolate(Constant(T_init), V)


    radiation = Kappa_re(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(0) + \
                Kappa_re(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(1) + \
                Kappa_re(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(2) + \
                Kappa_re(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(3)
    #radiation = Kappa(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(0) + \
    #    Kappa(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(1) + \
    #    Kappa(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(2) + \
    #    Kappa(T)*epsilon(T)*((T+273.15)**4 - (T_ambient+273.15)**4)*v*dt*ds(3)

    F = -CP(T)*T*v*dx - dt*inner(Kappa(T)*nabla_grad(T), nabla_grad(v))*dx + \
         CP(T_n)*T_n*v*dx - radiation

    a, L = lhs(F), rhs(F)



    return T, a, L

"""
Problem compute
"""


def heat_equation(Q, ts_time, time_steps, T_init=200,T_ambient=100, n_surface_interpolation=1000, mesh_density=30, poly_degree=2,VTK_OUTPUT=False,VTK_File=''):
    dt = ts_time / time_steps
    T_, T_n, F, J, Q, mesh = build_problem(
        Q, ts_time, time_steps, T_init, T_ambient, mesh_density=mesh_density, poly_degree=poly_degree)
    
    Rs = np.linspace(0, x_r.max()-x_r.min(), n_surface_interpolation)

    bcs = []
    c = 0
    T_surface = np.zeros([len(Rs), time_steps])
    t = dt
    times= []

    if VTK_OUTPUT:
        vtkfile = File(VTK_File)
        while t <= ts_time:
            Q.t = t
            problem = NonlinearVariationalProblem(F, T_, bcs, J)
            solver = NonlinearVariationalSolver(problem)
            solver.solve()
            prm = solver.parameters
            prm['newton_solver']['absolute_tolerance'] = 1E-9
            prm['newton_solver']['relative_tolerance'] = 1E-9
            prm['newton_solver']['maximum_iterations'] = 10000
            prm['newton_solver']['relaxation_parameter'] = 0.2
            T_n.assign(T_)
            T_surface[:, c] = np.squeeze(interpolate_solution(T_, Rs, [0.04]))
            times.append(t)
            c = c+1
            t += dt
            vtkfile << (T_, t)

    else:
        toolbar_width = 100
        # setup toolbar
        sys.stdout.write("[{}]".format(" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (1)) # return to start of line, after '['
        prog = 0
        prog_lim = int(time_steps/100)
        progress_c =0
        while t <= ts_time:
            Q.t = t
            problem = NonlinearVariationalProblem(F, T_, bcs, J)
            solver = NonlinearVariationalSolver(problem)
            solver.solve()
            prm = solver.parameters
            prm['newton_solver']['absolute_tolerance'] = 1E-9
            prm['newton_solver']['relative_tolerance'] = 1E-9
            prm['newton_solver']['maximum_iterations'] = 20000
            prm['newton_solver']['relaxation_parameter'] = 0.2

            T_n.assign(T_)
            T_surface[:, c] = np.squeeze(interpolate_solution(T_, Rs, [0.04]))
            times.append(t)
            c = c+1
            t += dt
            prog = prog +1
            if prog == prog_lim:
                progress_c += 1
                sys.stdout.write(
                    '\r'+'[{} - {}%'.format(progress_c*'#', progress_c))
                sys.stdout.flush()
                prog = 0
        sys.stdout.write("]\n")
    return T_, T_surface, Rs, np.array(times), mesh



def heat_equation_withInternal_HeatFlux(Q, ts_time, time_steps, T_init=200, n_surface_interpolation=1000, mesh_density=30, poly_degree=2,VTK_OUTPUT=False,VTK_File=''):
    dt = ts_time / time_steps
    T_, T_n, F, J, Q, mesh = build_problem(
        Q, ts_time, time_steps, T_init, 100, mesh_density=mesh_density, poly_degree=poly_degree)
    
    W = VectorFunctionSpace(mesh, 'P', poly_degree)
    Rs = np.linspace(0, x_r.max()-x_r.min(), n_surface_interpolation)


    bcs = []
    c = 0
    T_surface = np.zeros([len(Rs), time_steps])
    t = dt
    times= []

    toolbar_width = 100
    # setup toolbar
    sys.stdout.write("[{}]".format(" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (1)) # return to start of line, after '['
    prog = 0
    prog_lim = int(time_steps/100)
    progress_c =0
    flux = {}
    while t <= ts_time:
        Q.t = t
        problem = NonlinearVariationalProblem(F, T_, bcs, J)
        solver = NonlinearVariationalSolver(problem)
        solver.solve()
        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-9
        prm['newton_solver']['relative_tolerance'] = 1E-9
        prm['newton_solver']['maximum_iterations'] = 10000
        prm['newton_solver']['relaxation_parameter'] = 0.2
                    
        flux_u = project(-Kappa(T_)*grad(T_), W)
        flux[c] = flux_u
        T_n.assign(T_)
        T_surface[:, c] = np.squeeze(interpolate_solution(T_, Rs, [0.04]))
        times.append(t)
        c = c+1
        t += dt
        prog = prog +1
        if prog == prog_lim:
            progress_c += 1
            sys.stdout.write(
                '\r'+'[{} - {}%'.format(progress_c*'#', progress_c))
            sys.stdout.flush()
            prog = 0
    sys.stdout.write("]\n")
    return T_, T_surface, Rs, np.array(times), flux

class Q_bdry_matrix(UserExpression):
    def set_Q_field(self, Q, x_r, times):
        self.Q = Q
        self.x_r = x_r
        self.times = times
        self.t = 0

    def eval_cell(self, value, x, cell):
        ind = np.argmin(abs(self.times - self.t))
        value[0] = np.interp(x[0], self.x_r, self.Q[ind, :])





def basic_test():
    time = 6
    time_steps = 300
    Pmax = 1E7
    mu = length/2
    sig = 0.01

    ts = np.linspace(0, 6, 400)

    def Qbdry(x, t): return Pmax * \
        np.exp(-(x-(mu-(np.sin(t*12)/40)))**2 / (2*sig**2))

    Q = Expression('Pmax * exp(-pow(x[0]-(mu-sin(t*12)/40),2) / (2*pow(sig,2)))',
                degree=3, Pmax=Pmax, mu=mu, sig=sig, dx=length, t=0)  # *dx

    print('computing with analytical expression boundary...')
    T_, T_surface, Rs, times ,mesh = heat_equation(Q, time, time_steps)

    AS.plot_heatmap(T_surface.T, times, Rs, 'T (degC)',
                    'FEM - analytical boundary')

    print('computing with scalar field boundary...')
    ts = np.linspace(0, time, 600)
    X_R = np.linspace(0, x_r.max()-x_r.min(), 70)
    XX, YY = np.meshgrid(X_R, ts)
    Q_bdry = Qbdry(XX, YY)

    #Q = Q_bdry_matrix()
    #Q.set_Q_field(Q_bdry, X_R, ts)

    #T_, T_surfaceS, Rs, times = heat_equation(Q, time, time_steps)

    #AS.plot_heatmap(T_surfaceS.T, times, Rs, 'T (degC)',
    #                'FEM - scalar field boundary')

    """
    Verify solution
    """

    Tsurf, To = ht.solve_2d(X_R, ts, Q_bdry*.8,depth=0.04, Tinit=200)

    map0=AS.plot_heatmap(T_surface.T, times, Rs, 'T (degC)','Finite Element Model', return_fig=True)
    map1=AS.plot_heatmap(Tsurf, ts, X_R, 'T (degC)', 'Finite Difference Model', return_fig=True)

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(13.5, 4.5)
    ax[0] = map0.axes
    ax[1] = map1.axes
    map1(ax[1])

    map0
    map1
    plt.figure(figsize=[10.9,4.5])
    plt.plot(times, np.max(T_surface, axis=0),label='FEM-heat_equation')
    #plt.plot(times, np.max(T_surfaceS, axis=0), label='Scalar')
    plt.plot(ts, np.max(Tsurf, axis=1), label='FD-heat_equation')
    plt.xlim(0,6)
    plt.ylim(225, 680)
    plt.ylabel('Tmax (degC)', size=12)
    plt.xlabel('Time (sec)', size=12)
    plt.legend()
        
def SP_sweeping():
    with open('/Users/dominiccalleja/smardda_workflow/P90271/test_elm_data.pickle', 'rb') as handle:
        test_elm_data = pickle.load(handle)

    Temp_data = test_elm_data['T_prof']

    sim_time = 7.5

    ts = test_elm_data['time']
    ts = ts[range(0, len(ts), 5)]

    X_R = test_elm_data['x_r']-test_elm_data['x_r'].min()
    Q_bdry = test_elm_data['Qbdry']
    num_steps = len(ts[ts < sim_time])

    #Baseline
    Pmax = 6.21E6
    mu = 0.11
    sig = 0.008


    def Qbdry_base(x, t): return Pmax * \
        np.exp(-(x-(mu-(np.sin(t*20)/100)))**2 / (2*sig**2))


    simulation_times = ts[ts < sim_time]
    XX, YY = np.meshgrid(X_R, simulation_times)
    Q_bdry_sweep4 = Qbdry_base(XX, YY)
    AS.plot_heatmap(Q_bdry_sweep4, ts, X_R, '', '')


    def Qbdry_base(x, t): return Pmax * \
        np.exp(-(x-(mu-(np.sin(t*20)/200)))**2 / (2*sig**2))


    Q_bdry_sweep2 = Qbdry_base(XX, YY)
    AS.plot_heatmap(Q_bdry_sweep2, ts, X_R, '', '')


    def Qbdry_base(x, t): return Pmax * \
        np.exp(-(x-(mu-(np.zeros(np.shape(t)))))**2 / (2*sig**2))


    Q_bdry_sweep0 = Qbdry_base(XX, YY)
    AS.plot_heatmap(Q_bdry_sweep0, ts, X_R, '', '')


    #Q_a = Q_bdry[:, ts < sim_time].T

    #Q_apply = np.maximum(Q_a*.95,Q_bdry_sweep)

    Q = Q_bdry_matrix()
    Q.set_Q_field(Q_bdry_sweep4, X_R, simulation_times)

    VTK = '/Users/dominiccalleja/smardda_workflow/P90271/test_sweep_4.pvd'
    T_, T_surfaceS4, Rs, times , mesh= heat_equation(
        Q, sim_time, num_steps, mesh_density=140, poly_degree=4, VTK_OUTPUT=False, VTK_File=VTK)

    Q = Q_bdry_matrix()
    Q.set_Q_field(Q_bdry_sweep2, X_R, simulation_times)
    VTK = '/Users/dominiccalleja/smardda_workflow/P90271/test_sweep_2.pvd'
    T_, T_surfaceS2, Rs, times, mesh = heat_equation(
        Q, sim_time, num_steps, mesh_density=140, poly_degree=4, VTK_OUTPUT=False, VTK_File=VTK)

    Q = Q_bdry_matrix()
    Q.set_Q_field(Q_bdry_sweep0, X_R, simulation_times)
    VTK = '/Users/dominiccalleja/smardda_workflow/P90271/test_sweep_0.pvd'
    T_, T_surfaceS0, Rs, times, mesh = heat_equation(
        Q, sim_time, num_steps, mesh_density=140, poly_degree=4, VTK_OUTPUT=False, VTK_File=VTK)

    plt.figure(figsize=[15, 7])
    plt.plot(times, np.max(T_surfaceS4, axis=0), label='4cm Sweep')
    plt.plot(times, np.max(T_surfaceS2, axis=0), label='2cm Sweep')
    plt.plot(times, np.max(T_surfaceS0, axis=0), label='Stationary strike point')
    plt.xlim(0, sim_time)
    plt.xlabel('Time (s)')
    plt.ylabel('Max T (degC)')
    plt.legend()

def elm_test():
    with open('/Users/dominiccalleja/smardda_workflow/P90271/test_elm_data.pickle', 'rb') as handle:
        test_elm_data = pickle.load(handle)

    Temp_data = test_elm_data['T_prof']

    sim_time = 7.5

    ts = test_elm_data['time']
    #ts = ts[range(0,len(ts),5)]

    X_R = test_elm_data['x_r']-test_elm_data['x_r'].min()
    Q_bdry = test_elm_data['Qbdry']
    num_steps = len(ts[ts < sim_time])

    #Baseline 
    Pmax = 6.21E6
    mu = 0.11
    sig = 0.008
    def Qbdry_base(x, t): return Pmax * \
        np.exp(-(x-(mu-(np.sin(t*20)/100)))**2 / (2*sig**2))

    simulation_times = ts[ts < sim_time]
    XX, YY = np.meshgrid(X_R, simulation_times)
    Q_bdry_sweep = Qbdry_base(XX, YY)


    Q = Q_bdry_matrix()
    Q.set_Q_field(Q_bdry_sweep, X_R, simulation_times)
    VTK = '/Users/dominiccalleja/smardda_workflow/P90271/test_sweep_0.pvd'
    T_, T_surfaceS, Rs, times, mesh = heat_equation(
        Q, sim_time, num_steps, mesh_density=140, poly_degree=4, VTK_OUTPUT=False, VTK_File=VTK)

    T_times = Temp_data.index.values-Temp_data.index.values.min()

    plt.figure(figsize=[15, 7])
    plt.plot(T_times, np.max(Temp_data.values, axis=1), label='Experemental')
    plt.plot(times, np.max(T_surfaceS, axis=0), label='Model')
    plt.xlim(0, sim_time)
    plt.legend()


    plt.figure(figsize=[15,7])
    plt.plot(T_times, np.max(Temp_data.values, axis=1), label='Experemental')
    plt.plot(times, np.max(T_surfaceS, axis=0), label='Model')
    plt.xlim(2.7, 3.1)
    plt.legend()



    AS.plot_heatmap(Q_bdry_sweep, simulation_times, X_R, 'Q (W)', 'Q Baseline')
    Q_app = np.maximum(Q_a, Q_bdry_sweep)
    AS.plot_heatmap(Q_app, simulation_times, X_R, 'Q (W)', 'ELM Q - scalar_field')


    plt.figure(figsize=[15, 7])
    plt.plot(times, np.max(Q_app, axis=1), label='Q')
    plt.xlim(2.7, 3.1)
    plt.legend()

    AS.plot_heatmap(T_surfaceS.T, simulation_times, X_R, 'T (degC)','FEM - Model_field')
    AS.plot_heatmap(Temp_data.values, Temp_data.index.values, Temp_data.columns.values,
                    'T (degC)', 'IR Field')


def P90271_test_elms():

    """
    Lower the time res, its too high and i adjusted it figuring out what was wrong before 
    """
    print('evaluating test elms')

    top_dir = '/Users/dominiccalleja/smardda_workflow/Testing_ELM_temp_sim'
    #with open('/Users/dominiccalleja/smardda_workflow/P90271/test_elm_data_2_med_res.pickle', 'rb') as handle:
    #    test_elm_data = pickle.load(handle)
    
    with open('/Users/dominiccalleja/smardda_workflow/P90271/test_elm_data_2_med_VHIGHres_17500.pickle', 'rb') as handle:
        test_elm_data = pickle.load(handle)
    
    sim_time = 7.5

    Temp_data = test_elm_data['T_prof']
    ts = test_elm_data['time']
    X_R = test_elm_data['x_r']-test_elm_data['x_r'].min()
    Q_bdry = test_elm_data['ELM_Qprof_fitted']  # test_elm_data['Qbdry']
    num_steps = len(ts[ts < sim_time])
    times = ts[ts < sim_time]

    #Baseline
    Pmax = .95E7
    mu = 0.12
    sig = 0.006

    def Qbdry_base(x, t): return Pmax * \
        np.exp(-(x-(mu-(np.sin((t-0.1)*25)/50)))**2 / (2*sig**2))

    simulation_times = ts[ts < sim_time]
    XX, YY = np.meshgrid(X_R, simulation_times)
    Q_bdry_sweep = Qbdry_base(XX, YY)
    #AS.plot_heatmap(Q_bdry_sweep, simulation_times, X_R, 'Q (W)', 'Q Baseline')

    #Q_a = Q_bdry[:, ts < sim_time].T
    Q_a = Q_bdry[ts < sim_time, :].T

    elm_scale = .98
    Q_apply = np.maximum(Q_a.T*elm_scale, Q_bdry_sweep)

    Q = Q_bdry_matrix()
    Q.set_Q_field(Q_apply, X_R, simulation_times)

    simulation_name = 'SS{:.1e}_ELM{:.2f}_withRepKrad'.format(
        Pmax, elm_scale)
    
    print('Making output directory ... {}'.format(simulation_name))
    PATH = top_dir+'/'+simulation_name
    os.mkdir(PATH)
    os.mkdir(PATH+'/'+'PARAVIEW')

    VTK = PATH+'/'+'PARAVIEW'+'/test_FEM_model_{}.pvd'.format(simulation_name)
    T_, T_surfaceS, Rs, times, mesh = heat_equation(
        Q, sim_time, num_steps, mesh_density=240, poly_degree=4, VTK_OUTPUT=True, VTK_File=VTK,T_init=70)

    T_times = Temp_data.index.values-Temp_data.index.values.min()

    try:
        plt.figure(figsize=[15, 7])
        plt.plot(T_times, np.max(Temp_data.values, axis=1), label='Experemental')
        plt.plot(times, np.max(T_surfaceS, axis=0), label='Model')
        plt.xlim(0, sim_time)
        plt.legend()
        plt.savefig(PATH+'/Tmax.png')

        plt.figure(figsize=[15, 7])
        plt.plot(T_times, np.max(Temp_data.values, axis=1), label='Experemental')
        plt.plot(times, np.max(T_surfaceS, axis=0), label='Model')
        plt.xlim(2.7, 3.1)
        plt.legend()
        plt.savefig(PATH+'/Tmax_window.png')

        plt.figure(figsize=[15, 7])
        plt.plot(times, np.max(Q_apply, axis=1), label='MaxQ_model')
        plt.plot(test_elm_data['Q_prof'].index.values-np.min(test_elm_data['Q_prof'].index.values),
                np.max(test_elm_data['Q_prof'].values, axis=1), label='Max Q IR cam')
        plt.xlim(2.7, 3.1)
        plt.legend()
        plt.savefig(PATH+'/Qmax_window.png')
    except:
        plt.figure(figsize=[15, 7])
        plt.plot(T_times, np.max(Temp_data.values, axis=1), label='Experemental')
        plt.plot(times, np.max(T_surfaceS, axis=0)[:-1], label='Model')
        plt.xlim(0, sim_time)
        plt.legend()
        plt.savefig(PATH+'/Tmax.png')
        plt.figure(figsize=[15, 7])

        plt.plot(T_times, np.max(Temp_data.values, axis=1), label='Experemental')
        plt.plot(times, np.max(T_surfaceS, axis=0)[:-1], label='Model')
        plt.xlim(2.7, 3.1)
        plt.legend()
        plt.savefig(PATH+'/Tmax_window.png')
        plt.figure(figsize=[15, 7])
    
        plt.plot(times, np.max(Q_apply, axis=1)[:-1], label='MaxQ_model')
        plt.plot(test_elm_data['Q_prof'].index.values-np.min(test_elm_data['Q_prof'].index.values),
                np.max(test_elm_data['Q_prof'].values, axis=1), label='Max Q IR cam')
        plt.xlim(2.7, 3.1)
        plt.legend()
        plt.savefig(PATH+'/Qmax_window.png')



    AS.plot_heatmap(Q_bdry_sweep, simulation_times, X_R, 'Q (W)', 'Q Baseline',save_address=PATH+'/Q_SS.png')
    AS.plot_heatmap(Q_apply, simulation_times, X_R,'Q (W)', 'ELM Q - scalar_field',save_address=PATH+'/Q_SS+ELM.png')
    AS.plot_heatmap(test_elm_data['Q_prof'].values, test_elm_data['Q_prof'].index.values, test_elm_data['Q_prof'].columns.values,
                    'Q (W)', 'KL9a Q PPF', save_address=PATH+'/IR_CAM_Q.png')

    AS.plot_heatmap(T_surfaceS.T, simulation_times, X_R,
                    'T (degC)', 'FEM - Model_field', save_address=PATH+'/Tmodel.png')
    AS.plot_heatmap(Temp_data.values, Temp_data.index.values, Temp_data.columns.values,
                    'T (degC)', 'IR Field',save_address=PATH+'/TProf_data.png')


    Data_save = {}
    Data_save['simulation_input_data'] = test_elm_data
    Data_save['sim_time'] = sim_time
    Data_save['Pmax'] = Pmax
    Data_save['mu'] = mu
    Data_save['sig'] = sig
    Data_save['elm_scale'] = elm_scale
    Data_save['Q_apply'] = Q_apply
    #Data_save['T_'] = T_
    Data_save['T_surfaceS'] = T_surfaceS
    Data_save['Rs'] = Rs
    Data_save['times'] = times
    with open(PATH+'/Data_for_run.pickle', 'wb') as handle:
        pickle.dump(Data_save,handle)


def P90271_test_elms_2():
    """
    RUN WITH THE SMOOTHED Q real data 


    Lower the time res, its too high and i adjusted it figuring out what was wrong before 
    """
    print('evaluating test elms')

    top_dir = '/Users/dominiccalleja/smardda_workflow/CorrectKappaEq_init200_amb300_ves100'
    #with open('/Users/dominiccalleja/smardda_workflow/P90271/test_elm_data_2_med_res.pickle', 'rb') as handle:
    #    test_elm_data = pickle.load(handle)

    with open('/Users/dominiccalleja/smardda_workflow/P90271/test_elm_data_2_med_VHIGHres_17500.pickle', 'rb') as handle:
        test_elm_data = pickle.load(handle)

    sim_time = 7.5


    reduced_Q_ppf = test_elm_data['Q_prof'].iloc[::3,:]

    Temp_data = test_elm_data['T_prof']
    ts = reduced_Q_ppf.index.values-np.min(reduced_Q_ppf.index.values)
    # test_elm_data['time']
    X_R = reduced_Q_ppf.columns.values-np.min(reduced_Q_ppf.columns.values)
    #test_elm_data['x_r']-test_elm_data['x_r'].min()
    # test_elm_data['Qbdry']
    Q_bdry = reduced_Q_ppf.values

    #test_elm_data['ELM_Qprof_fitted']
    num_steps = len(ts[ts < sim_time])
    times = ts[ts < sim_time]
    simulation_times = ts[ts < sim_time]
    # np.maximum(Q_a.T*elm_scale, Q_bdry_sweep)
    Q_apply = Q_bdry[ts < sim_time,:]
    Q = Q_bdry_matrix()
    Q.set_Q_field(Q_apply, X_R, simulation_times)

    simulation_name = 'Full_Rreform_2'

    try:
        print('Making output directory ... {}'.format(simulation_name))
        PATH = top_dir+'/'+simulation_name
        os.mkdir(PATH)
        os.mkdir(PATH+'/'+'PARAVIEW')
    except:
        print('File already made')

    VTK = PATH+'/'+'PARAVIEW'+'/test_FEM_model_{}.pvd'.format(simulation_name)
    T_, T_surfaceS, Rs, times, mesh = heat_equation(
        Q, sim_time, num_steps, mesh_density=30, poly_degree=2, VTK_OUTPUT=False, VTK_File=[],T_init=160,T_ambient=300)

    print('Saving Result Cross Section!')
    file_s = File(PATH+'/'+"solution_cross_section.xml")
    file_s << T_
    file_m = File(PATH+'/'+"solution_mesh.xml")
    file_m << mesh


    T_times = Temp_data.index.values-Temp_data.index.values.min()

    Data_save = {}
    Data_save['simulation_input_data'] = test_elm_data
    Data_save['sim_time'] = sim_time
    Data_save['reduced_Q'] = reduced_Q_ppf
    #Data_save['mu'] = mu
    #Data_save['Pmax'] = Pmax
    #Data_save['sig'] = sig
    #Data_save['elm_scale'] = elm_scale
    Data_save['Q_apply'] = Q_apply
    #Data_save['T_'] = T_
    Data_save['T_surfaceS'] = T_surfaceS
    Data_save['Rs'] = Rs
    Data_save['times'] = times
    with open(PATH+'/Data_for_run.pickle', 'wb') as handle:
        pickle.dump(Data_save, handle)

    try:
        plt.figure(figsize=[15, 7])
        plt.plot(T_times, np.max(Temp_data.values, axis=1),
                    label='Experemental')
        plt.plot(times, np.max(T_surfaceS, axis=0), label='Model')
        plt.xlim(0, sim_time)
        plt.legend()
        plt.savefig(PATH+'/Tmax.png')

        plt.figure(figsize=[15, 7])
        plt.plot(T_times, np.max(Temp_data.values, axis=1),
                    label='Experemental')
        plt.plot(times, np.max(T_surfaceS, axis=0), label='Model')
        plt.xlim(2.7, 3.1)
        plt.legend()
        plt.savefig(PATH+'/Tmax_window.png')

        plt.figure(figsize=[15, 7])
        plt.plot(times, np.max(Q_apply, axis=1), label='MaxQ_model')
        plt.plot(test_elm_data['Q_prof'].index.values-np.min(test_elm_data['Q_prof'].index.values),
                    np.max(test_elm_data['Q_prof'].values, axis=1), label='Max Q IR cam full res')
        plt.plot(reduced_Q_ppf.index.values-np.min(reduced_Q_ppf.index.values),
                    np.max(reduced_Q_ppf.values, axis=1), label='Max Q IR cam red res')
        plt.xlim(2.7, 3.1)
        plt.legend()
        plt.savefig(PATH+'/Qmax_window.png')
    except:
        plt.figure(figsize=[15, 7])
        plt.plot(T_times, np.max(Temp_data.values, axis=1),
                    label='Experemental')
        plt.plot(times, np.max(T_surfaceS, axis=0)[:-1], label='Model')
        plt.xlim(0, sim_time)
        plt.legend()
        plt.savefig(PATH+'/Tmax.png')
        plt.figure(figsize=[15, 7])

        plt.plot(T_times, np.max(Temp_data.values, axis=1),
                    label='Experemental')
        plt.plot(times, np.max(T_surfaceS, axis=0)[:-1], label='Model')
        plt.xlim(2.7, 3.1)
        plt.legend()
        plt.savefig(PATH+'/Tmax_window.png')
        plt.figure(figsize=[15, 7])

        plt.plot(times, np.max(Q_apply, axis=1)[:-1], label='MaxQ_model')
        plt.plot(test_elm_data['Q_prof'].index.values-np.min(test_elm_data['Q_prof'].index.values),
                    np.max(test_elm_data['Q_prof'].values, axis=1), label='Max Q IR cam full res')
        plt.plot(reduced_Q_ppf.index.values-np.min(reduced_Q_ppf.index.values),
                 np.max(reduced_Q_ppf.values, axis=1), label='Max Q IR cam red res')
        plt.xlim(2.7, 3.1)
        plt.legend()
        plt.savefig(PATH+'/Qmax_window.png')

    #AS.plot_heatmap(reduced_Q_ppf.values, reduced_Q_ppf.index.values, reduced_Q_ppf.columns.values, 'Q (W)',
    #                'Q Baseline', save_address=PATH+'/Q_SS.png')
    AS.plot_heatmap(Q_apply, simulation_times, X_R, 'Q (W)',
                    'ELM Q - scalar_field', save_address=PATH+'/Q_SS+ELM.png')
    AS.plot_heatmap(test_elm_data['Q_prof'].values, test_elm_data['Q_prof'].index.values, test_elm_data['Q_prof'].columns.values,
                    'Q (W)', 'KL9a Q PPF', save_address=PATH+'/IR_CAM_Q.png')

    AS.plot_heatmap(T_surfaceS.T, simulation_times, X_R,
                    'T (degC)', 'FEM - Model_field', save_address=PATH+'/Tmodel.png')
    AS.plot_heatmap(Temp_data.values, Temp_data.index.values, Temp_data.columns.values,
                    'T (degC)', 'IR Field', save_address=PATH+'/TProf_data.png')

    """
    prepare figures for thesis 
    """
def thesis_figures():
    with open('/Users/dominiccalleja/smardda_workflow/Testing_ELM_temp_sim/simulation_Q_PPF_boundary_2/Data_for_run.pickle', 'rb') as handle:
        test_elm_data = pickle.load(handle)

    with open('/Users/dominiccalleja/smardda_workflow/P90271/test_elm_data_2_med_VHIGHres_17500.pickle', 'rb') as handle:
        Data = pickle.load(handle)

    map0 = AS.plot_heatmap(test_elm_data['reduced_Q'].values, test_elm_data['reduced_Q'].index.values,
                        test_elm_data['reduced_Q'].columns.values, 'Q (W)', 'T6 90271 KL9 radial heat flux', return_fig=True)

    map1 = AS.plot_heatmap(Data['T_prof'].values, Data['T_prof'].index.values,
                        Data['T_prof'].columns.values, 'T (degC)', 'T6 90271 radial KL9 temperature field', return_fig=True)
    map2 = AS.plot_heatmap(test_elm_data['T_surfaceS'].T, test_elm_data['times']+Data['T_prof'].index.values.min(), test_elm_data['Rs']+Data['T_prof'].columns.values.min(), 'T (degC)',
                        'T6 90271 model temperature field', return_fig=True)

    plt.figure(figsize=[10.9,4.5])
    plt.plot(Data['T_prof'].index.values, np.max(
        Data['T_prof'].values, axis=1), label='P90271 Tmax')
    plt.plot(test_elm_data['times']+Data['T_prof'].index.values.min(),
            np.max(test_elm_data['T_surfaceS'].T, axis=1)[:-1], label='model')

    plt.xlim(Data['T_prof'].index.values.min(), Data['T_prof'].index.values.min()+7.5)
    plt.ylabel('Tmax (degC)', size=12)
    plt.xlabel('Time (sec)', size=12)
    plt.legend()


    reduced_Q_ppf = Data['Q_prof'].iloc[::4, :]
    ts = reduced_Q_ppf.index.values-np.min(reduced_Q_ppf.index.values)
    # test_elm_data['time']
    X_R = reduced_Q_ppf.columns.values-np.min(reduced_Q_ppf.columns.values)
    #test_elm_data['x_r']-test_elm_data['x_r'].min()
    # test_elm_data['Qbdry']
    Q_bdry = reduced_Q_ppf.values

    #test_elm_data['ELM_Qprof_fitted']
    num_steps = len(ts[ts < sim_time])
    times = ts[ts < sim_time]
    simulation_times = ts[ts < sim_time]
    # np.maximum(Q_a.T*elm_scale, Q_bdry_sweep)
    Q_apply = Q_bdry[ts < sim_time,:]

    fig, ax = plt.subplots(nrows=2,sharex=True)
    fig.set_size_inches([10.9, 8])
    fig.subplots_adjust(wspace=0, hspace=0)
    ax[0].plot(Data['Q_prof'].index.values,
            np.max(Data['Q_prof'].values, axis=1), label='KL9 Data')
    ax[0].plot(times+Data['Q_prof'].index.values.min(),
            np.max(Q_apply, axis=1), label='model')
    ax[0].set_ylabel('Qmax (W)')
    ax[0].legend()
    ax[1].plot(Data['T_prof'].index.values, np.max(
        Data['T_prof'].values, axis=1), label='KL9 Data')
    ax[1].plot(test_elm_data['times']+Data['T_prof'].index.values.min(),
            np.max(test_elm_data['T_surfaceS'].T, axis=1)[:-1], label='model')
    ax[1].set_xlim([49,49.4])
    ax[1].set_ylim([500,850])
    ax[1].set_xlabel('Time (sec)')
    ax[1].set_ylabel('Tmax (degC)')

def SS_ELM_MODEL_COMPUTE():
    top_dir = '/Users/dominiccalleja/smardda_workflow/Testing_ELM_temp_sim'
    # '/users/sgdcalle/smardda4python/'
    home = '/Users/dominiccalleja/smardda_workflow'
    pickle_path = home + '/P90271/P90271_updating_pickle.pickle'
    with open(pickle_path, 'rb') as handle:
        P90271_data = pickle.load(handle)

    #with open('/Users/dominiccalleja/smardda_workflow/Testing_ELM_temp_sim/simulation_Q_PPF_boundary_2/Data_for_run.pickle', 'rb') as handle:
    #    Other_Data = pickle.load(handle)

    #with open('/Users/dominiccalleja/smardda_workflow/P90271/test_elm_data_2_med_VHIGHres_17500.pickle', 'rb') as handle:
    #    ELM_Data = pickle.load(handle)

    # Good results - but freq too low, and the sweep amplitude buggared
    # with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_ELM.pickle', 'rb') as handle:
    #    ELM_Data = pickle.load(handle)

    #with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_ELM_mod_sweep_amp.pickle', 'rb') as handle:
    #    ELM_Data = pickle.load(handle)

    #with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_ELM_mod_sweep_amp_2_freq70_2.pickle', 'rb') as handle:
    #    ELM_Data = pickle.load(handle)

    #with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_100Samp.dill', 'rb') as handle:
    #    ELM_Data = dill.load(handle)
    
    #with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_100Samp.dill', 'rb') as handle:
    #    ELM_Data = dill.load(handle)

    #with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_20Samp_ELMpushed_outboard.dill', 'rb') as handle:
    #    ELM_Data = dill.load(handle)

    #with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_20_Samp_f58-70.dill', 'rb') as handle:
    #    ELM_Data = dill.load(handle)

    #with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_20_Samp_f53-65.dill', 'rb') as handle:
    #    ELM_Data = dill.load(handle)

    with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_20_Samp_f49-51.dill', 'rb') as handle:
        ELM_Data = dill.load(handle)

    l_q = [8, 8, 8]
    s = [4.5, 4.5, 4.5]

    elm_fac=.9
    for sampI in range(0,9):
        print('\n\n\n\n\n\n ===================================RUN {} =================================== \n\n\n\n\n\n'.format(sampI))
        simulation_name = 'red-7sec_{}_q_{}_s_{}_elm_{}_samp-nelm{}_cumELME_{:.2e}'.format(
            sampI, l_q, s, elm_fac, ELM_Data['N_ELM'][sampI],ELM_Data['ELM_Energy'][sampI])

        print('Making output directory ... {}'.format(simulation_name))
        PATH = top_dir+'/'+simulation_name
        os.mkdir(PATH)
        os.mkdir(PATH+'/'+'PARAVIEW')
        VTK = PATH+'/'+'PARAVIEW'+'/test_FEM_model_{}.pvd'.format(simulation_name)

        P_sol_mu = savgol_filter(ELM_Data['PSol'].mean(axis=1),1301, 2)
        times = ELM_Data['time']
        t = ELM_Data['PSol'].index.values-ELM_Data['PSol'].index.values.min()

        Psol = np.interp(times,t,P_sol_mu)
        Psol[times<0.45]=8.5E6

        time_threshold = 8
        t_ind = times < time_threshold

        plt.figure()
        plt.plot(t, P_sol_mu,label='PSol_mu')
        plt.plot(times, Psol,label='PSol')
        plt.savefig(PATH+'/PSol.png')

        #Qbdry1 = up.construct_boundary_model(l_q, s, Psol[t_ind],
        #                                           P90271_data['GP'], 14E6, times[t_ind], ELM_Data['x_r'], 4)

        Qbdry = up.construct_boundary_model_smooth(l_q, s, Psol[t_ind],
                                    P90271_data['GP'], 14E6, times[t_ind], ELM_Data['x_r'], 4)

        #AS.plot_heatmap(Qbdry.T, times, ELM_Data['x_r'], 'Q (W)', 'Steady State Model Q')

        Q = np.maximum(Qbdry, ELM_Data['Qbdry_mat'][sampI][:, t_ind]*elm_fac)
        Q = Q.T
        Q_a = Q_bdry_matrix()
        Q_a.set_Q_field(Q, ELM_Data['x_r']-ELM_Data['x_r'].min(), times[t_ind])

        AS.plot_heatmap(Q, times[times < time_threshold], ELM_Data['x_r'], 'Q (W)',
                        'Combined Model Q', save_address=PATH+'/Q_SS+ELM.png')
        AS.plot_heatmap(ELM_Data['Qbdry_mat'][sampI].T, times, ELM_Data['x_r'],
                        'Q (W)', 'ELM Model Q', save_address=PATH+'/Q_ELM.png')
        AS.plot_heatmap(Qbdry.T, times[times < time_threshold], ELM_Data['x_r'],
                        'Q (W)', 'Steady State Model Q', save_address=PATH+'/Q_SS.png')

        AS.plot_heatmap(ELM_Data['Q_prof'].values, ELM_Data['Q_prof'].index.values, ELM_Data['Q_prof'].columns.values,
                        'Q (W)', 'KL9a Q PPF', save_address=PATH+'/IR_CAM_Q.png')
        
        Temp_data = ELM_Data['T_prof']

        plt.figure(figsize=[15, 7])
        plt.plot(times[times < time_threshold]+(Temp_data.index.values.min()),
                np.max(Q, axis=1), label='MaxQ_model')
        plt.savefig(PATH+'/model_Qmax.png')
        
        plt.figure(figsize=[15, 7])
        plt.plot(ELM_Data['Q_prof'].index.values,
                np.max(ELM_Data['Q_prof'].values, axis=1), label='Max Q IR')
        plt.savefig(PATH+'/data_Qmax.png')

        plt.figure(figsize=[15, 7])
        plt.plot(ELM_Data['Q_prof'].index.values,
                np.max(ELM_Data['Q_prof'].values, axis=1), label='Max Q IR')
        plt.plot(times[times < time_threshold]+(Temp_data.index.values.min()),
                np.max(Q, axis=1), label='MaxQ_model')
        plt.savefig(PATH+'/Qmax_2.png')

        T_, T_surfaceS, Rs, timess, mesh = heat_equation(
            Q_a, times[t_ind][-1], len(times[t_ind]), mesh_density=160, poly_degree=2, VTK_OUTPUT=False, VTK_File=VTK)


        AS.plot_heatmap(T_surfaceS.T, timess+(Temp_data.index.values.min()), ELM_Data['x_r'],
                        'T (degC)', 'FEM - Model_field', save_address=PATH+'/Tmodel.png')
        AS.plot_heatmap(Temp_data.values, Temp_data.index.values, Temp_data.columns.values,
                        'T (degC)', 'IR Field', save_address=PATH+'/TProf_data.png')


        Data_save = {}
        Data_save['sim_time'] = times[-1]
        Data_save['model_Q'] = Q
        Data_save['model_ELM_Q'] = ELM_Data['Qbdry_mat'][sampI]
        Data_save['model_SS_Q'] = Qbdry
        Data_save['lambda_q'] = l_q
        Data_save['S'] = s
        #Data_save['Pmax'] = Pmax
        #Data_save['sig'] = sig
        #Data_save['elm_scale'] = elm_scale
        #Data_save['T_'] = T_
        Data_save['T_surfaceS'] = T_surfaceS
        Data_save['Rs'] = Rs
        Data_save['times'] = times[times<time_threshold]
        Data_save['ELM_Data4Run'] = ELM_Data
        
        with open(PATH+'/Data_for_run.pickle', 'wb') as handle:
            pickle.dump(Data_save, handle)


        plt.figure(figsize = [15, 7])
        plt.plot(Temp_data.index.values, np.max(Temp_data.values, axis=1),
                    label = 'Experemental')
        plt.plot(timess+(Temp_data.index.values.min()), np.max(T_surfaceS, axis=0), label = 'Model')
        plt.xlim(Temp_data.index.values.min(), Temp_data.index.values.max())
        plt.legend()
        plt.savefig(PATH+'/Tmax.png')
        plt.figure(figsize = [15, 7])

        plt.plot(Temp_data.index.values, np.max(Temp_data.values, axis=1),
                    label = 'Experemental')
        plt.plot(timess+(Temp_data.index.values.min()),
                np.max(T_surfaceS, axis=0), label='Model')
        plt.xlim(Temp_data.index.values.min()+2, Temp_data.index.values.min()+2.4)
        plt.legend()
        plt.savefig(PATH+'/Tmax_window.png')

        plt.figure(figsize = [15, 7])
        plt.plot(timess+(Temp_data.index.values.min()),
                    np.max(Q, axis=1), label='MaxQ_model')
        plt.plot(ELM_Data['Q_prof'].index.values,
                    np.max(ELM_Data['Q_prof'].values, axis=1), label = 'Max Q IR cam full res')
        plt.savefig(PATH+'/Qmax.png')

        plt.figure(figsize=[15, 7])
        plt.plot(timess+(Temp_data.index.values.min()),np.max(Q, axis=1), label='MaxQ_model')
        plt.plot(ELM_Data['Q_prof'].index.values,
                    np.max(ELM_Data['Q_prof'].values, axis=1), label = 'Max Q IR cam full res')
        plt.xlim(Temp_data.index.values.min()+2, Temp_data.index.values.min()+2.4)
        plt.legend()
        plt.savefig(PATH+'/Qmax_window.png')


def test_stochastic_model():

    results_directory = '/Users/dominiccalleja/smardda_workflow/P90271/TIME_SERIES_FINAL'

    with open(results_directory+'/test_0_full_stochastic_boundary.pickle', 'rb') as handle:
        Data = pickle.load(handle)

    with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_20_Samp_f49-51.dill', 'rb') as handle:
        ELM_Data = dill.load(handle)

    Q = Data['Qbdry']
    Temp_data = ELM_Data['T_prof']

    times = Data['Times']
    Q_a = Q_bdry_matrix()
    Q_a.set_Q_field(Q, Data['x_r']-Data['x_r'].min(), times)

    PATH = results_directory+'/FEM_run_9'

    os.mkdir(PATH)

    # Plot Q profiles
    AS.plot_heatmap(ELM_Data['Q_prof'].values, ELM_Data['Q_prof'].index.values,
                    ELM_Data['Q_prof'].columns.values, 'Q (W)', 'IR KL9a Q - JPN 90271', save_address=PATH+'/raw_q_prof.png')

    AS.plot_heatmap(Q, times+ELM_Data['Q_prof'].index.values[0], Data['x_r'], 'Q (W)',
                    'Combined Model Q', save_address=PATH+'/full_stochastic_mod_q.png')

    T_, T_surfaceS, Rs, timess, mesh = heat_equation(
        Q_a, times[-1000], len(times[:-1000]), mesh_density=30, poly_degree=2, VTK_OUTPUT=False, VTK_File='')

    #plot temperatures
    AS.plot_heatmap(T_surfaceS.T, timess+ELM_Data['Q_prof'].index.values[0],
                    Data['x_r'], 'T (degC)', 'FEM - Model_field', save_address=PATH+'/Tmodel.png')

    AS.plot_heatmap(Temp_data.values, Temp_data.index.values, Temp_data.columns.values,
                    'T (degC)', 'IR Field', save_address=PATH+'/TProf_data.png')
    """
        plt.figure(figsize= [15, 7])
        plt.plot(Temp_data.index.values, np.max(Temp_data.values, axis=1),
                    label = 'Experemental')
        plt.plot(timess+(Temp_data.index.values.min()), np.max(T_surfaceS, axis=0), label = 'Model')
        plt.xlim(Temp_data.index.values.min(), Temp_data.index.values.max())
        plt.legend()
        plt.savefig(PATH+'/Tmax.png')
        plt.figure(figsize = [15, 7])

        plt.plot(Temp_data.index.values, np.max(Temp_data.values, axis=1),
                    label = 'Experemental')
        plt.plot(timess+(Temp_data.index.values.min()),
                np.max(T_surfaceS, axis=0), label='Model')
        plt.xlim(Temp_data.index.values.min()+2, Temp_data.index.values.min()+2.4)
        plt.legend()
        plt.savefig(PATH+'/Tmax_window.png')
    """
    save_data = {}
    save_data['Q'] = Q
    save_data['times'] = times
    save_data['x_r'] = Data['x_r']
    #save_data['T_'] = T_
    save_data['T_surfaceS'] = T_surfaceS
    #save_data['mesh']= mesh
    save_data['Rs'] = Rs
    save_data['timess'] = timess
    with open(PATH+'/simulation_data.pickle', 'wb') as handle:
        pickle.dump(save_data, handle)


def sim_temp_boundary():
    import pathos.multiprocessing as mp
    import time
    #p = mp.Pool(2)

    results_directory = '/Users/dominiccalleja/smardda_workflow/P90271/TIME_SERIES_FINAL'

    with open(results_directory+'/FullRun_FEM_ANALYSIS.pickle', 'rb') as handle:
        RESULTS = pickle.load(handle)
    """    
    with open(results_directory+'/FullRun_1_good.pickle', 'rb') as handle:
        RESULTS = pickle.load(handle)
    """
    #with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_20_Samp_f49-51.dill', 'rb') as handle:
    #    ELM_Data = dill.load(handle)
    res_dir = results_directory+'/bounddary_sim1'
    os.mkdir(res_dir)

    time_0 = 47.00009536743164
    times   = RESULTS['times']
    x_r = RESULTS['x_r']

    def multi_process( Qbdry, times, time_0, x_r, res_dir):
        t0 = time.time()
        print('Begin Q bdry')
        Q = Qbdry
        Q_a = Q_bdry_matrix()
        Q_a.set_Q_field(Q, x_r-x_r.min(), times)

        PATH = res_dir+'/FEM_run_{}'.format(np.random.randint(0,1000,1))
        os.mkdir(PATH)

        AS.plot_heatmap(Q, times+time_0, x_r, 'Q (W)',
                        'Combined Model Q', save_address=PATH+'/full_stochastic_mod_q.png')

        T_, T_surfaceS, Rs, timess, mesh = heat_equation(
            Q_a, times[-1000], len(times[:-1000]), mesh_density=30, poly_degree=2, VTK_OUTPUT=False, VTK_File='')

        AS.plot_heatmap(T_surfaceS.T, timess+time_0, x_r, 'T (degC)', 'FEM - Model_field', save_address=PATH+'/Tmodel.png')

        t1 = time.time()
        print('Evaluation time 1 ind : {}'.format(t1-t0))

        save_data = {}
        save_data['Q'] = Q
        save_data['times'] = times
        save_data['x_r'] = x_r
        #save_data['T_'] = T_
        save_data['T_surfaceS'] = T_surfaceS
        #save_data['mesh']= mesh
        save_data['Rs'] = Rs
        save_data['timess'] = timess
        with open(PATH+'/simulation_data.pickle', 'wb') as handle:
            pickle.dump(save_data, handle)

    def tmp_multi(Qbdry): return multi_process( Qbdry, times, time_0, x_r, res_dir)

    #J = range(len(RESULTS['Qbdry']))
    for i in range(len(RESULTS['Qbdry'])):
        print(RESULTS['Qbdry'][i])
        tmp_multi(RESULTS['Qbdry'][i])
    #p.map(tmp_multi,  RESULTS['Qbdry'][:4])
    #p.close()


def sim_temp_boundary_2():
    import pathos.multiprocessing as mp
    import time
    #p = mp.Pool(2)

    results_directory = '/Users/dominiccalleja/smardda_workflow/P90271/TIME_SERIES_FINAL'

    with open(results_directory+'/FullRun_FEM_ANALYSIS.pickle', 'rb') as handle:
        RESULTS = pickle.load(handle)
    """    
    with open(results_directory+'/FullRun_1_good.pickle', 'rb') as handle:
        RESULTS = pickle.load(handle)
    """
    #with open('/Users/dominiccalleja/smardda_workflow/P90271/EVAL_ELM_Sweep_SP_20_Samp_f49-51.dill', 'rb') as handle:
    #    ELM_Data = dill.load(handle)
    res_dir = results_directory+'/bounddary_sim1_ru2lowres'
    #os.mkdir(res_dir)

    time_0 = 47.00009536743164
    times = RESULTS['times'][::3]
    x_r = RESULTS['x_r'][::2]
    time_threshold = 6
    t_ind = times < time_threshold
    
    def multi_process(Qbdry, times, time_0, x_r, res_dir, t_ind):
        t0 = time.time()
        print('Begin Q bdry')
        Q = Qbdry[::3, ::2]
        Q_a = Q_bdry_matrix()
        Q_a.set_Q_field(Q, x_r-x_r.min(), times)

        PATH = res_dir+'/FEM_run_{}'.format(np.random.randint(1000, 1400, 1)[0])
        os.mkdir(PATH)

        AS.plot_heatmap(Q, times+time_0, x_r, 'Q (W)',
                        'Combined Model Q', save_address=PATH+'/full_stochastic_mod_q.png')

        T_, T_surfaceS, Rs, timess, mesh = heat_equation(
            Q_a, times[t_ind][-1], len(times[t_ind]), mesh_density=30, poly_degree=2, VTK_OUTPUT=False, VTK_File='')

        AS.plot_heatmap(T_surfaceS.T, timess+time_0, x_r, 'T (degC)',
                        'FEM - Model_field', save_address=PATH+'/Tmodel.png')

        t1 = time.time()
        print('Evaluation time 1 ind : {}'.format(t1-t0))

        save_data = {}
        save_data['Q'] = Q
        save_data['times'] = times
        save_data['x_r'] = x_r
        #save_data['T_'] = T_
        save_data['T_surfaceS'] = T_surfaceS
        #save_data['mesh']= mesh
        save_data['Rs'] = Rs
        save_data['timess'] = timess
        with open(PATH+'/simulation_data.pickle', 'wb') as handle:
            pickle.dump(save_data, handle)

    def tmp_multi(Qbdry): return multi_process(Qbdry, times, time_0, x_r, res_dir, t_ind)

    #J = range(len(RESULTS['Qbdry']))
    for i in range(26,31): 
        print(RESULTS['Qbdry'][i])
        tmp_multi(RESULTS['Qbdry'][i])
    #p.map(tmp_multi,  RESULTS['Qbdry'][:4])
    #p.close()
"""
def merge_sweep(a, b, time):
    n = len(time)
    Q = np.zeros([len(a), n])
    for i in range(n):
        Q[:, i] = ((a*(n-i)) + (b*i))/n
    return Q


def construct_boundary_model_smooth(lambda_q, sigma, Psol, GP, p_baseline, times, R, sweep_frequency):
    # lambda_q = shape(n_slices,n_gps)
    #
    #
    symmetry = 0.5
    dx = signal.sawtooth(2 * np.pi * times *
                            sweep_frequency, symmetry)*((len(GP)-1)/2)  #
    #Dx = np.round(dx-np.min(dx))

    Qbdry = np.zeros([len(R), len(times)])
    Qb = []

    Q = up.sweep_boundary_model(lambda_q, sigma, R, GP, sweep_frequency)

    NUM_COND = (dx[1]-dx[0])*.5

    k = []
    for i in range(len(GP)):  #
        print(i)
        if i > round(min(dx)) + abs(round(min(dx))) & i < round(max(dx)) + abs(round(min(dx))):
            k.append(np.where(np.logical_and(dx + abs(round(min(dx)))
                                                > i-NUM_COND, dx + abs(round(min(dx))) < i+NUM_COND))[0])
        elif i == round(max(dx)) + abs(round(min(dx))):
            k.append(signal.find_peaks(dx)[0])
        elif i == round(min(dx)) + abs(round(min(dx))):
            k.append(signal.find_peaks(-dx)[0])

    k[0] = np.insert(k[0], [0, len(k[0])], [0, len(dx)-1])
    K = np.sort(np.concatenate(k))
    ind = dx[K].round()

    for i in range(len(K)-1):
        A = int(ind[i]+abs(min(ind)))
        B = int(ind[i+1]+abs(min(ind)))
        Qbdry[:, K[i]:K[i+1]] = merge_sweep(Q[A], Q[B], times[K[i]:K[i+1]])/p_baseline

    Qbdry = Qbdry*[Psol]
    return Qbdry
"""
"""
Psol = Psol[t_ind]
GP = P90271_data['GP']
p_baseline = 14E6
times = times[t_ind]
R = ELM_Data['x_r']
sweep_frequency = 4

Qbdry1 = up.construct_boundary_model(lambda_q, sigma, Psol[t_ind],
                                    P90271_data['GP'], 14E6, times[t_ind], ELM_Data['x_r'], 4)

Qbdry = construct_boundary_model_smooth(lambda_q, sigma, Psol[t_ind],
                                    P90271_data['GP'], 14E6, times[t_ind], ELM_Data['x_r'], 4)

Q = np.maximum(Qbdry, ELM_Data['Qbdry_mat'][sampI][:, t_ind]*elm_fac)
Q = Q.T
Q_a = Q_bdry_matrix()
Q_a.set_Q_field(Q, ELM_Data['x_r']-ELM_Data['x_r'].min(), times[t_ind])
T_, T_surfaceS, Rs, timess = heat_equation(
    Q_a, times[t_ind][-1], len(times[t_ind]), mesh_density=160, poly_degree=2, VTK_OUTPUT=False)






AS.plot_heatmap(Qbdry1.T, times[t_ind], ELM_Data['x_r'],
                        'Q (W)', 'Steady State Model Q')

AS.plot_heatmap(Qbdry.T, times[t_ind], ELM_Data['x_r'],
                        'Q (W)', 'Steady State Model Q')
AS.plot_heatmap(Q, times[t_ind], ELM_Data['x_r'],
                'Q (W)', 'Steady State Model Q')













def merge_sweep(A,B,time):
    n = len(time)
    Q = np.zeros([len(A),n])
    for i in range(n):
        Q[:,i]=((i*A)+((n/-i)*B))/n
    return Q

def find_nearest(arrayA, arrayB):
    arrayA = np.asarray(arrayA)
    arrayB = np.asarray(arrayB)

    array = np.zeros(np.shape(arrayB))
    for ind,X in enumerate(arrayB):
        idx = (np.abs(arrayA - X)).argmin()
        array[ind] = arrayA[idx]
    return array

def find_consec(a, step=1):
    vals = []
    for i, x in enumerate(a):
        if i == 0:
            diff = a[i + 1] - x
            if diff == step:
                vals.append(x)
        elif i < a.size-1:
            diff = a[i + 1] - x
            if diff > step:
                vals.append(a[i + 1])
    return np.array(vals)



AS.plot_heatmap(Q_bdry[:,:1500].T,times[:1500],R,'','')


qr = merge_sweep(Q[A], Q[B], times[K[i]:K[i+1]])


time = times[K[i]:K[i+1]]
a = Q[A]
b = Q[B]





lambda_q = [lambda_q, lambda_q, lambda_q]
sigma = [s, s, s]
Psol = Psol[t_ind]
GP = P90271_data['GP']
p_baseline = 14E6
times = times[t_ind]
R = ELM_Data['x_r']
sweep_frequency = 4

Qtest = construct_boundary_model([lambda_q, lambda_q, lambda_q], [s, s, s], Psol[t_ind],
                                    P90271_data['GP'], 14E6, times[t_ind], ELM_Data['x_r'], 4)


def constuct_sweep_Q(Q_prof,times,sweep_frequency):


AS.plot_heatmap(Qbdry.T, times[times < time_threshold], ELM_Data['x_r'],
                'Q (W)', 'Steady State Model Q')

Q_prof=[ Qbdry[:, C],Qbdry[:, A], Qbdry[:, B]]

ts = np.linspace(times[0],1,int((1-times[0])*sweep_frequency*4),endpoint=False)

c=0
idx = np.zeros(np.shape(ts))
for i in range(len(ts)):
    idx[i] = 2*math.mod(i/3 - i/3+0.5)



ts1 = find_nearest(times,ts)

dx = signal.sawtooth(2 * np.pi * (times-(0.25/2)) * sweep_frequency,
                        .5)*((3-1)/2)
Dx = dx-np.min(dx)

plt.plot(times, Dx)
plt.xlim([0,1])

eps = 1E-3
k = (times[-1]-times[0])*sweep_frequency
a=[]
while len(a) < k+1:
    eps+=1E-4
    print(eps)
    a = np.where(Dx > 2-eps)[0]
    b = np.where(np.logical_and(Dx>1-eps,Dx<1+eps))[0]
    c = np.where(Dx < 0+eps)[0]

Qbdry = np.zeros()
for i in range(len(a)):


np.sort(np.concatenate([a,b,c]))

    n = len(times)
    k = (times[-1]-times[0])*sweep_frequency
    Q = np.zeros([len(A), n])
    for i in range(k-1):        
        Q[:, i] = ((i*A)+((n-i)*B))/n
        
plt.figure(figsize=[15,5])
plt.scatter(times[a], Dx[a])
plt.scatter(times[b], Dx[b])
plt.scatter(times[c], Dx[c])


plt.figure(figsize=[15, 5])
plt.scatter(times[A], Dx[A])
plt.scatter(times[B], Dx[B])
plt.scatter(times[C], Dx[C])


sweep_frequency=4

  # np.round(dx-np.min(dx))

plt.plot(times,Dx)

A = np.where(np.logical_and(times > 0.25, times < 0.251))[0][0]
B = np.where(np.logical_and(times > 0.3, times < 0.301))[0][0]
C = np.where(np.logical_and(times > 0.35, times < 0.351))[0][0]

ti_me = times[np.logical_and(times > 0.249, times < 0.25+(2*0.1251))]
ti_me[-1]-ti_me[0]


Q = merge_sweep( Qbdry[:, C],Qbdry[:, A], Qbdry[:, B],ti_me)

AS.plot_heatmap(Q.T,ti_me,ELM_Data['x_r'],'','')

Qb = np.concatenate(Q)
"""
if __name__ == '__main__':

    #basic_test()
    #elm_test()
    #P90271_test_elms()
    #P90271_test_elms_2()
    #P90271_test_elms_2()
    #SS_ELM_MODEL_COMPUTE()
    
    
    #test_stochastic_model()
    #sim_temp_boundary()
    sim_temp_boundary_2()
