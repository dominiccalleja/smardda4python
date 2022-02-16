#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 7 15:33:03 2019

Probably consider a complete rewrite to include these functions in the smardda for python libraries.

This is likely to increase tthe performance of the codes and reduce challenges with the execution of other scripts we may be able to optimise
most of the functions can be reliably included into the other working libraries.

This library had been depricated and is no longer essential for the completion of the pulse reconstruction workflow.
"""

__author__ = "Dominic Calleja"
__copyright__ = "Copyright 2019"
__credits__ = ["Wayne Arter"]
__license__ = "MIT"
__version__ = "0.1"
__date__ = '10/01/2019'
__maintainer__ = "Dominic Calleja"
__email__ = "d.calleja@liverpoo.ac.uk"
__status__ = "Draft"
from scipy.signal import savgol_filter
import smardda4python as sm
import AnalysisS4P as AS
from scipy import signal
import matplotlib.pyplot as plt
import time
import copy
import os
import numpy as np
import pandas as pd
import glob
"import data"


print('+====================================+')
print('|              TStools               |')
print('| Tools to compute JET time series   |')
print('| plasma wall loading.               |')
print('|                                    |')
print('|           Version: '+__version__ + 12*' '+' |')
print('|                                    |')
print('| '+__copyright__+' (c) ' + __author__+' |')
print('+====================================+')
print(' ')


def Read_JET_Power_data(PulseNumber):
    from jet.data import sal
    PSol = sal.get('/pulse/'+str(PulseNumber)+'/ppf/signal/jetppf/scal/psol')
    PRad = sal.get('/pulse/'+str(PulseNumber)+'/ppf/signal/jetppf/scal/prad')
    PTot = sal.get('/pulse/'+str(PulseNumber)+'/ppf/signal/jetppf/scal/plth')
    return PSol, PRad, PTot

def Read_JET_SP_data(PulseNumber):
    from jet.data import sal
    OB_SP_EFIT = sal.get('/pulse/'+str(PulseNumber)+'/ppf/signal/jetppf/efit/rsom')
    #OB_SP_EFTF = sal.get('/pulse/'+str(PulseNumber)+'/ppf/signal/jetppf/eftf/rsom')
    return OB_SP_EFIT,# OB_SP_EFTF

def format_JET_data(Data, Ylabel):

    Dict = {'Times': Data.dimensions[0].data, Ylabel: Data.data}
    Output = pd.DataFrame(Dict, columns=['Times', Ylabel])
    return Output


def Sweeping_Sawtooth(x, phase, length, amplitude):
    alpha = (amplitude)/(length/2)
    return -amplitude/2+amplitude*((x-phase) % length == length/2) \
        + alpha*((x-phase) % (length/2))*((x-phase) % length <= length/2) \
        + (amplitude-alpha*((x-phase) % (length/2)))*((x-phase) % length > length/2)


def MeshMapping1(Cell_centres, Ansys_mesh_file):
    Data = pd.read_csv(Ansys_mesh_file)
    DATA = np.asarray(Data.iloc[:, [1, 2, 3]])
    QId = []
    for i in range(len(Data.iloc[:, 1])):
        QId.append(np.argmin(np.sum((DATA[i, :]-Cell_centres)**2, axis=1)))
#
    OutputDict = {'Node_Coordinates': DATA,
                  'Node2Node_map': QId}

    return OutputDict


def resample_time_series(Data, column = 'Tmax',sample_rate = '.5ms'):
    """
    function to resample ppf data to get a uniform sample
    """
    Data['ts']=Data['Times']-Data['Times'].min()
    Data.index = pd.TimedeltaIndex(Data['ts'],unit='s')
    Data.drop(columns='Times')

    resampled_data = Data.resample(sample_rate).agg({column : 'mean','Times': 'median'})
    #resampled_data=resampled_data.drop(columns='ts')
    return resampled_data

def signal_trend(Data,order):
    """
    Fit Trend to Data
    """
    import statsmodels.api as sm
    residuals = sm.tsa.tsatools.detrend(Data, order=order)
    trend = Data-residuals
    return trend, residuals

def rm_peaks(data,rise_window,fall_window,peaks=[],fill_value=0,height=[40,120],distance=100):
    """
    Remove peaks in y by specifying a rise and fall window (indicies valued)
    """
    y = data.copy(deep=True)
    if not peaks:
        peaks = signal.find_peaks(y,height=height,distance=distance)[0]
    low_pass = []
    for i,p in enumerate(peaks):
        if p+fall_window > len(y):
            break
        low_pass.append([x for x in range(p-rise_window,p+fall_window)])
    low_pass = np.array(low_pass)
    low_pass = np.reshape(low_pass,[np.size(low_pass)])
    y[low_pass] = fill_value
    return y

def rm_peaks_iterative(data, window=901, distance=100, height_0 = [110,120]):
    """
    Remove peaks in y by specifying a rise and fall window (indicies valued)
    """
    y = data.copy(deep=True)
    c = 0
    peaks = np.argmax(y.values)
    mu = savgol_filter(y, window, 1)
    boarder = mu + \
        np.concatenate([np.std(y)*np.ones(window-1),
                        y.rolling(window).std().dropna()])
    while peaks.any():

        #mu = np.concatenate(
        #   [np.mean(y)*np.ones(window-1), mu.dropna()])
        mu = savgol_filter(y, window, 1)
        y[peaks] = mu[peaks]
        peaks = signal.find_peaks(
            y, height=[boarder, height_0[1]], distance=distance)[0]
        c =c+1
    print('Executed {} passes'.format(c))
    return y

def inverse_sample(x_s,ps,Nsamples):
    sample = np.random.rand(Nsamples)
    return  np.interp(sample,ps,x_s)

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def fit_residual_distibution(Data, Nsamples, expected=[], kde = True):
    from scipy.stats import gaussian_kde
    from scipy.optimize import curve_fit
    import statsmodels.api as sm

    y,x   = np.histogram(Data , 70)

    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    # Fit multi modal_distributions

    delta = 0.2*((np.max(Data)-np.min(Data))/2)
    x_s = np.linspace(np.min(Data)-delta,np.max(Data)+delta,Nsamples)
    if kde:
        kde = gaussian_kde([Data])
        y_k = kde.evaluate(x_s)
        ps = ((np.cumsum(y_k)- np.min(np.cumsum(y_k))) / (np.max(np.cumsum(y_k)) - np.min(np.cumsum(y_k))))
    else:
        if len(expected)==3:
            params,cov=curve_fit(gauss,x,y,expected)
            y_k = gauss(x_s,*params)
            ps = ((np.cumsum(y_k)- np.min(np.cumsum(y_k))) / (np.max(np.cumsum(y_k)) - np.min(np.cumsum(y_k))))
        elif len(expected)==6:
            params,cov=curve_fit(bimodal,x,y,expected)
            y_k = bimodal(x_s,*params)
            ps = ((np.cumsum(y_k)- np.min(np.cumsum(y_k))) / (np.max(np.cumsum(y_k)) - np.min(np.cumsum(y_k))))
        else:
            print('ERROR: you entered {} parameters in expected variable \n expected variable is the guess at mean, std, and A of gaussians \n for bimodal or unimodal you must enter 6 or 3 resepctively'.format(len(expected)))
            return
    return x_s , ps, y_k

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

def inv_difference(last_ob, value):
	return value + last_ob

def inverse_difference(x0, dataset):
    inv_dif = []
    inv_dif.append(inv_difference(x0, dataset[0]))
    for i in range(1, len(dataset)):
        inv_dif.append(inv_difference(inv_dif[-1], dataset[i]))
    return pd.Series(inv_dif)

def plot_lag_scatter(Data,Nlag,nrows,lowess=False,title=[]):
    import statsmodels.api as sm
    ncols = int(Nlag)/int(nrows)
    if not ncols.is_integer():
        return print('ERROR: nrows must be a factor of Nlag -- Entered nrows: {} Nlag: {}'.format(nrows,Nlag))

    fig, axs = plt.subplots(int(nrows),int(ncols))
    fig.set_size_inches(int(ncols*4),int(nrows*4))
    if title:
        fig.suptitle(title)
    c = 0
    j = 0
    for i in range(1, Nlag+1):

        if c == ncols:
            c = 0
            j = j+1
        tau = Data[range(0,len(Data)-i)]
        tau_1 = Data[range(i,len(Data))]
        cor = np.corrcoef(tau,tau_1)[0][1]
        axs[j,c].plot(tau,tau_1,'+')
        axs[j,c].set_title(r'Res $\tau$ vs $\tau$+{}   ($\rho$ {:.3})'.format(i,cor))
        if lowess:
            lowess_p = sm.nonparametric.lowess(tau,tau_1)
            axs[j,c].plot(lowess_p[:,0],lowess_p[:,1],'red',label='lowess')
        axs[j,c].legend()
        c = c+1

    for ax in fig.get_axes():
        ax.label_outer()


"""
Smardda evaluation tools
"""

def ScaleQ(Index, Q_map, PSOL, Pbase):
    Name = 'Ansys_Flux_'+str(Index)
    Q_out = []
    for i in range(len(Q_map)):
        Q_out.append((Q_map[i]/Pbase)*PSOL)
    return Q_out, Name


def Ansys_Input_Generation(Simulation_Input_Table, Baseline_Q_Results_Dictionary, Baseline_PSol):
    Q = []
    Qout = pd.DataFrame()
    for i in Simulation_Input_Table.index:
        KEYS = list(Baseline_Q_Results_Dictionary.keys())
        sim = Simulation_Input_Table.loc[i]
        Q, Name = ScaleQ(i, Baseline_Q_Results_Dictionary['Eq_'+str(sim['Equilibrium_Number'])]
                         ['Results']['Q'], sim['Mean_Psol'], 1E+6)
        Qout.insert(loc=i, column=Name, value=Q)
    return Qout

#  need to change the extraction of JET power data to allow you not to use sal!
class PrepareSimulationObject:
    def __init__(self, GEOMpath, CTLpath, EQpath, Working_Directory, PulseNumber=None, Read_JET_Data=True,OB_SP=[]):
        self.GEOMpath = GEOMpath
        self.CTLpath = CTLpath
        self.EQpath = EQpath
        self.PulseNo = PulseNumber
        self.Working_Directory = Working_Directory

        self.strikepoint_sweep_frequency = 4

        self.plot = False

        if PulseNumber == None:
            print('PulseNumber missing, cannot compute series, enter OB_SP vector')

        if not OB_SP:
            Read_JET_Data == True
        else:
            self.OB_SP = OB_SP

        if Read_JET_Data == True:
            self.jet_ppf_data = data_holder()
            print('Extracting JET PPF data, may take a couple of minutes')

            PSol, PRad, PTot = Read_JET_Power_data(self.PulseNo)
            OB_SP_EFIT  = Read_JET_SP_data(self.PulseNo)
            OB_SP_EFTF = Read_JET_SP_data(self.PulseNo)

            setattr(self.jet_ppf_data,'PSol',PSol)
            setattr(self.jet_ppf_data,'PRad',PRad)
            setattr(self.jet_ppf_data,'PTot',PTot)
            setattr(self.jet_ppf_data,'OB_SP_EFIT',OB_SP_EFIT[0])
            setattr(self.jet_ppf_data,'OB_SP_EFTF',OB_SP_EFTF[0])

            print('Completed retrieval of JET Data')


    def Geometry(self, Target, Shadow, RESpath=[], SHADpath=[]):
        # Target and Shadow vtk files input as strings
        # Only specify Target or Shad path if they are in different directories
        if RESpath == []:
            RESpath = self.GEOMpath
        if SHADpath == []:
            SHADpath = self.GEOMpath

        self.Targetfullpath = os.path.join(RESpath, Target)
        self.Shadowfullpath = os.path.join(SHADpath, Shadow)

        self.Target = Target
        self.Shadow = Shadow

    def Equilibrium(self, ListEq=None, EquilibriumTimes=None, EquilIds=None):
        """
        NB. Needs new features to load Equils from other sources but for now the simple implimentation of get_equil_times will do
        """

        equil = get_equil_times(self.EQpath,self.PulseNo)
        eq_pd = get_equil_SP(equil, self.jet_ppf_data.OB_SP_EFIT, label = 'EFIT_SP')

        self.EquilibriumData = eq_pd

    def pulse_construction(self, tmin, tmax, sim_type='SP_track',sweep_symmetry=0.5):
        """
        tmin and tmax  = pulse times
        """
        sim_types = ['SP_track', 'Periodic', 'Static']
        if sim_type not in sim_types:
            raise ValueError("Invalid sim type: [%s] \n choose from: %s" % (sim_type, sim_types))

        if sim_type == sim_types[0]:
            sim_times = np.linspace(tmax, tmin, num=100)
            simulation_array = sp_track_sweep_reconstruction(sim_times, tmin, tmax, self.EquilibriumData,  self.jet_ppf_data.OB_SP_EFIT, label = 'EFIT_SP_nearest')

        if sim_times == sim_types[1]:
            print('default strikepoint sweep frequency %f has been selected, modify .strikepoint_sweep_frequency if incorrect ' % (self.strikepoint_sweep_frequency))
            simulation_array = stepwise_sp_sweep_reconstruction(self.EquilibriumData,tmin, tmax, self.strikepoint_sweep_frequency , sweep_symmetry=0.5, plot=self.plot)

        if sim_times == sim_types[2]:
            print('Not yet defined')

        self.Simulation_Input_Table = simulation_array
#
        # ReconDict = {'Model_Times': self.Sim_times,
                     # 'Simulation_Times': self.Sim_times+tmin,
                     # 'Equilibrium_Number': self.EqList,
                     # 'Eq_ind':self.SP_ind,
                     # 'Strike_Point_Efit': self.SP['Strike_Point'][self.EqList],
                     # 'Eqdsk_StrikePoint': EqSP,
                     # 'Eqdsk_filename': self.EquilibriumData.Equil[self.EqList]}
#
           # self.Simulation_Input_Table =pd.DataFrame(ReconDict,columns=['Model_Times',
                                                                        # 'Simulation_Times',
                                                                        # 'Equilibrium_Number',
                                                                        # 'Strike_Point_Efit',
                                                                        # 'Eqdsk_StrikePoint'])
    # self.Simulation_type = sim_type
        # self.Simulation_Input_Table = pd.DataFrame(ReconDict, columns=['Model_Times',
                                                                       # 'Simulation_Times',
                                                                       # 'Equilibrium_Number',
                                                                       # 'Eq_ind',
                                                                       # 'Strike_Point_Efit',
                                                                       # 'Eqdsk_StrikePoint',
                                                                       # 'Eqdsk_filename'])
        # self.Simulation_Input_Table = self.Simulation_Input_Table.reset_index(drop=True)
#
    def Simulation_Construction(self, UserTopSimDir=[], run_all_timesteps=False):

        if UserTopSimDir == []:
            TopSimDir = 'P_' + str(self.PulseNo) + '_' + self.Simulation_type
            Sdir = os.path.join(self.Working_Directory, TopSimDir)
        else:
            TopSimDir = UserTopSimDir
            if self.Working_Directory in TopSimDir:
                Sdir = TopSimDir
            else:
                Sdir = os.path.join(self.Working_Directory, TopSimDir)
        try:
            os.mkdir(Sdir)
        except OSError as error:
            print(error)

        if run_all_timesteps == False:
            Model_runs = self.Simulation_Input_Table.Equilibrium_Number.unique()
            print("You are executing %s Smardda runs \n The Q profile will then be normalised and scaled by the input PSol to complete the full pulse for each equilibrium" % (
                str(len(Model_runs))))

        else:
            Model_runs = self.Simulation_Input_Table.Equilibrium_Number
            print("Warning: You are attempting to execute Smardda %s times! \n only recomended if footprint parameters vary significantly" % (
                str(len(Model_runs))))

        sim_name = []
        self.sim_dir = []
        for i in range(len(Model_runs)):
            sim_name.append('Equil_'+str(int(Model_runs[i])))

            self.sim_dir.append(os.path.join(Sdir, sim_name[i]))
            try:
                os.mkdir(os.path.join(Sdir, self.sim_dir[i]))
            except OSError as error:
                print(error)

        sim = sm.smardda(self.Target, self.Shadow, sim_name[0],
                         self.EquilibriumData.Equil[Model_runs[0]],
                         self.sim_dir[0])
        self.sim = sim
        Sim_List = [copy.deepcopy(sim) for i in range(len(Model_runs))]
        print("Copied simulation objects")

        self.run_all_timesteps = run_all_timesteps
        self.sim_name = sim_name
        self.Model_runs = Model_runs
        self.simulation_list = Sim_List
        self.Top_simulation_dir = TopSimDir

    def Simulation_Initialisation(self, PSol=1E6, Lamda_q=0.0012, Sigma=0.0001):

        if self.run_all_timesteps == False:
            PSol = np.ones([len(self.Model_runs), 1])*PSol
            Lamda_q = np.ones([len(self.Model_runs), 1])*Lamda_q
            Sigma = np.ones([len(self.Model_runs), 1])*Sigma
            psiref = np.ones([len(self.Model_runs), 1])*1.7760963
            self.RunParams = {'PSol': 1E6,
                              'Lamda_q': 0.0012,
                              'Sigma': 0.0001,
                              'psiref': 1.7760963}
        for i in range(len(self.Model_runs)):
            self.simulation_list[i].Equil = self.sim_name[i]
            self.simulation_list[i].Equid = self.EquilibriumData.Equil[self.Model_runs[i]]
            self.simulation_list[i].SimDir = self.sim_dir[i]
            self.simulation_list[i].update_object()

            self.simulation_list[i].plasma_parameters(
                float(PSol[i]), float(Lamda_q[i]), float(Sigma[i]))
            self.simulation_list[i].vessel_parameters(psiref=float(psiref[i]))

            print("Modified copied simulation object (%s of %s) : %s" %
                  (str(i+1), str(len(self.Model_runs)), self.sim_name[i]))
            self.simulation_list[i].resultsCTL()
            self.simulation_list[i].shadowCTL()
            self.simulation_list[i].HDSCTL()
            self.simulation_list[i].powcalCTL()
            print("Saved smardda CTL files in %s:  (%s of %s) " %
                  (self.sim_dir, str(i+1), str(len(self.Model_runs))))

            os.system('cp {} {}'.format(os.path.join(self.GEOMpath, self.Target), self.sim_dir[i]))
            os.system('cp {} {}'.format(os.path.join(self.GEOMpath, self.Shadow), self.sim_dir[i]))
            os.system('cp {} {}'.format(os.path.join(
                self.EQpath, self.simulation_list[i].Equid), self.sim_dir[i]))

    def execute_in_parallel(self):
        import multiprocessing
        Max_process = multiprocessing.cpu_count()
        pool = multiprocessing.Pool()
        self.simulation_list[0].executefullsimulation()

    def Simulation_Execution(self):
        for i in range(len(self.Model_runs)):
            print('Processing {} in {}'.format(
                self.simulation_list[i].SimName, self.simulation_list[i].SimDir))
            tic = time.clock()
            self.simulation_list[i].executefullsimulation()
            toc = time.clock()
            print('Completed processing {} in {}'.format(
                self.simulation_list[i].SimName, self.simulation_list[i].SimDir))
            print('Time to complete: %.2f (sec)' % (toc-tic))

    def Extract_HeatFlux(self):

        """
        Needs to be modified to utilise the new features of the Analysis functions
        """
        Dict = {}
        Dict2 = {}
        GDict = {}
        for i in range(len(self.Model_runs)):
            Results = []
            BodyData = []
            Results = AS.Analysis(
                self.Target, self.simulation_list[i].ResultsVTK, self.simulation_list[i].SimDir, 'res_log.txt')
            Results.ExtractGeom()
            Results.ExtractStats()
            Footprint_Q, Radial_location,Coords = Results.extract_profile(10)
            BodyData = Results.Individual_Body_Data

            Dict['Equilibrium_'+str(i)] = {'Model_Times': self.sim_name,
                                           'Simulation_Directory': self.simulation_list[i].SimDir,
                                           'Results_VTK': self.simulation_list[i].ResultsVTK,
                                           'Equilibrium_Number': self.EquilibriumData.Equil[self.Model_runs[i]],
                                           'Run_Parameters': self.RunParams,
                                           'Heat_Flux': Results.Q,
                                           'Intergrated_Q': Results.Total_IntQ,
                                           'Footprint_Q': Footprint_Q,
                                           'Radial_location':Radial_location,
                                           'Footprint_Coords':Coords}

            Dict2['Equilibrium_'+str(i)] = {'Tile6_a': BodyData['BodyNumber_21.0'],
                                            'Tile6_b': BodyData['BodyNumber_22.0'],
                                            'Tile6_c': BodyData['BodyNumber_23.0'],
                                            'Tile6_d': BodyData['BodyNumber_24.0']}

        self.Results = Dict
        self.Tile6_Results = Dict2
        GDict = {'Target': self.Target,
                 'Cell_Area': Results.Area,
                 'Cell_Centres': Results.cell_centre,
                 'Elements': Results.Cells,
                 'Nodes': Results.Points,
                 'Body_Numbers': Results.Body_Numbers}
        self.Results_Geometry = GDict
 #   def Map2AnsysMesh(self,Ansys_Mesh )
 #

    def SpecifyEquilibriumSequence(self, EquilSequence, EquilTimes):
        self.EquilibriumSequence = EquilSequence
        self.EquilibriumSequenceTimes = EquilTimes


def copyR(logfile):
    """Print copyright information to file."""
    outputf = open(logfile, 'w')
    outputf.write('+'+'='*77+'+ \n')
    tl = 'TStools'
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    tl = 'Time Series compute tools TStools'
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    tl = ' '
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    tl = ' Version: '+__version__
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    outputf.write('|'+' '*77+'| \n')
    tl = __copyright__+' (c) ' + __author__
    sp1 = (77-len(tl))//2
    sp2 = 77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|' + '\n')
    outputf.write('+'+'='*77+'+' + '\n')
    outputf.write('\n')
    outputf.close()
    return


class data_holder():
    pass


""" Functions to deal with the Equilibria and Strike point """
def rename_eqilibrium(path,pulse_no,old_string,new_string):
    import os
    import re
    import glob
    list_eq = []
    rep_eq = []
    equilibrium_times = []
    equil_files = glob.glob(path+'/*'+str(pulse_no)+'*.eqdsk')
    for i in range(len(equil_files)):
        list_eq.append(equil_files[i].split('/')[-1])
        rep_eq.append(list_eq[i].replace(old_string,new_string))
        os.rename(path+'/'+list_eq[i],path+'/'+rep_eq[i])

def get_equil_times(path,pulse_no):
    """
    Function extracts the times from the Eqdsk file names in a directory defined by path, containing a substring that matches the pulse number.

    """
    import re
    list_eq = []
    equilibrium_times = []
    equil_files = glob.glob(path+'/*'+str(pulse_no)+'*.eqdsk')
    for i in range(len(equil_files)):
        list_eq.append(equil_files[i].split('/')[-1])
        equilibrium_times.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", list_eq[i])[1]))

    Data = {'Times': equilibrium_times, 'Equil': list_eq}
    equil = pd.DataFrame(Data, columns=['Times', 'Equil'])
    equil = equil.sort_values(by='Times')
    equil = equil.reset_index(drop=True)
    return equil


def get_equil_SP(equil_pd, ppf_sp, label = 'EFIT_SP'):
    """
    Use EFIT or EFTF JET ppf to find the strike point of a pandas array of eqdsk files.
    pd format corresponding to the output of the get_equil_times funtions.

    i.e equil = pd.DataFrame('Times': ~, 'Equil': ~)
    """
    ppf_r_nearest =[]
    ppf_r_interp =[]
    d_interp_nearest = []

    for i in range(len(equil_pd.Times)):
        inp = abs(ppf_sp.dimensions[0].data-equil_pd.Times[i]).argmin()
        ppf_r_nearest.append(ppf_sp.data[inp])
        ppf_r_interp.append(np.interp(equil_pd.Times[i],ppf_sp.dimensions[0].data,ppf_sp.data))
        d_interp_nearest.append(ppf_r_nearest[i] - ppf_r_interp[i])

    eq_pd = equil_pd.copy(deep=True)
    eq_pd.insert(loc=len(eq_pd.columns),column=label+'_nearest',value = ppf_r_nearest)
    eq_pd.insert(loc=len(eq_pd.columns),column=label+'_interp',value = ppf_r_interp)
    eq_pd.insert(loc=len(eq_pd.columns),column=label+'_differece',value = d_interp_nearest)
    return eq_pd


def stepwise_sp_sweep_reconstruction(equil_pd,tmin,tmax,sweep_frequency,symmetry=0.5,numerical_accuracy=20,plot=False):
    """
    INPUTS :
    Eq_list         = a list where the indicies correspond to a point in the sweep cycle.
    t_min and t_max = The corresponding pulse times
    sweep_frequency = The frequency of strike point sweeping  (if equil list is a complete sweep sweep_frequency=half JET sweep frequency )
    symmetry        = Symmetry of the sawtooth wave form (t_fall/t_rise)

    numerical_accuracy = Number of sample points in each sweep
    plot               = true to plot the wave form and the stepwise approximation

    OUTPUT :
    pandas array : columns = ['Times' 'Equilibrium_number' 'Equilibrium_file']

    N.B this function assumes uniformly spaced equilibrium. Need to re-write in the future for irregular spaced values.
    """
    l_eqlist = len(equil_pd)

    if (l_eqlist % 1):
        le = l_eqlist
    else:
        le = l_eqlist-1

    t = np.linspace(tmin,tmax,num=(l_eqlist/2)*sweep_frequency*(tmax-tmin)*numerical_accuracy)
    dx = signal.sawtooth(2 * np.pi * t * sweep_frequency, symmetry)*((le)/2)
    Dx = np.round(dx-np.min(dx))

    Data = {'Times':t,'Equilibrium_number':Dx,'Equilibrium_file':equil_pd['Equil'][Dx]}
    output = pd.DataFrame(Data, columns=['Times', 'Equilibrium_number','Equilibrium_file'])

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(t,dx-np.min(dx))
        plt.plot(t,Dx)
        plt.xlabel('Time (s)')
        plt.ylabel('Equilibrium filename')
        plt.yticks(ticks = range(0,l_eqlist),labels = equil_pd['Equil'])

    return output


def sp_track_sweep_reconstruction(sim_times, t_min, t_max, equil_pd, ppf_sp, label = 'EFIT_SP_nearest'):
    """
    INPUTS :
    sim_times       = A vector begining at 0 with the simulation time steps
    t_min and t_max = The corresponding pulse times
    equil_pd        = The pandas formatted table of eqdisk files produced by the get_equil_SP function
    ppf_sp          = PPF signal of the EFIT or EFTF sp data
    label           = The label of the target value from the ppf_sp pd array

    OUTPUT :
    output          = pd array  ['Times','pulse_times','Equilibrium_file','Equilibrium_SP']
    """
    EqList = []
    ind = []
    EqSP = []
    for i in range(len(sim_times)):
        ind.append(abs(ppf_sp.dimensions[0].data-(sim_times[i]+t_min)).argmin())
        EqList.append(
            abs(equil_pd[label] - ppf_sp.data[ind[i]]).argmin())
        EqSP.append(equil_pd[label][EqList[i]])

    labs = ['Times','pulse_times','Equilibrium_file',label]
    dict = {labs[0]:sim_times, labs[1]:sim_times+t_min, labs[2]:EqList, labs[3]:EqSP}
    output = pd.DataFrame(dict, columns=labs)

    return output


def check_user_specified_sp(user_specified_sp,ppf_sp,label = 'EFIT_SP'):
    """
    user_specified_sp = should be a numpy array with the first column corresponding to a time and the second column corresponding to a radial location of the strikepoint
    user_specified_sp = np.array([times, radial distance])

    if using EFTF change lable to EFTF_SP
    """
    d_r = []
    interp_d_r = []
    d_interp_r =[]
    for i,t,r in enumerate(ppf_sp):
        inp = abs(ppf_sp.dimensions[0].data-t).argmin()
        d_r.append(ppf_sp.data[inp] - r)
        interp_d_r.append(np.interp(t,ppf_sp.dimensions[0].data,ppf_sp.data))
        d_interp_r.append(interp_d_r[i]-r)\

    string = ['Times',label+'_dif_nearest',label+'_dif_interp']
    Data = {string[0]: user_specified_sp[0], string[1]: d_r, string[2]:d_interp_r }

    strike_point_descrepancy = pd.DataFrame(Data, columns=string)

    return strike_point_descrepancy


""" Functions to deal with PSol """

def psol_step_trend(psol_pd,sim_times,tmin,tmax,time_split=[],power_split=[],window_size = 21, plot=False,poly_order=[2,5]):

    """
    time_split = sim time not pulse time i.e pulse time - min(pulse time)
    """
    print('''---------------------- most cleaned PSol -----------------------''')
    print('''-------------------- Extracting step chage ---------------------''')


    I_SimWindow  = psol_pd['Times'].between(tmin,tmax)
    psol_pd = psol_pd[I_SimWindow]
    psol_pd['Times'] = psol_pd['Times']-tmin
    psol_pd = psol_pd.reset_index(drop = True)
    # Rise
    T1 = psol_pd['Times'].idxmin()
    if not (time_split or power_split):
        print ('Must enter a time or power pulse value to enable the trend fitting')
    elif not time_split:
        T2 = psol_pd['Times'][psol_pd['Psol']>power_split].idxmin()
    elif not power_split:
        T2 = abs(psol_pd['Times']-time_split).argmin()

    # flat_top
    T3 = psol_pd['Times'].idxmax()
    """ Polyfit and the evaluations """
    P01 = poly_order[0]
    P02 = poly_order[1]


    poly1 = np.polyfit(psol_pd['Times'][T1:T2],psol_pd['Psol'][T1:T2],P01)
    poly2 = np.polyfit(psol_pd['Times'][T2:T3],psol_pd['Psol'][T2:T3],P02)
    y1 = np.polyval(poly1,sim_times[sim_times<psol_pd['Times'][T2]])
    y2 = np.polyval(poly2,sim_times[sim_times>psol_pd['Times'][T2]])

    Psol_hat = savgol_filter(psol_pd['Psol'], window_size, 1)

    tr1 = np.polyval(poly1,psol_pd['Times'][0:T2])
    tr2 = np.polyval(poly2,psol_pd['Times'][T2:])
    trend = np.concatenate([tr1,tr2],axis=0)

    psol_tr = psol_pd.copy(deep=True)
    psol_tr['Psol'] = psol_tr['Psol']-trend
    Psol_mu = (psol_tr).rolling(window = window_size).mean()
    Psol_std= (psol_tr).rolling(window = window_size).std()

    p_smooth = np.concatenate([y1,y2],axis=0)
    mu = np.interp(sim_times,psol_pd['Times'],Psol_mu['Psol'])
    std = np.interp(sim_times,psol_pd['Times'],Psol_std['Psol'])

    string = ['Times','PPF_times','PSol_smooth','PSol_savgol','PSol_mu','PSol_std','PSol_raw_interp']
    Data = {string[0]:sim_times,
            string[1]:sim_times+tmin,
            string[2]:p_smooth,
            string[3]:np.interp(sim_times,psol_pd['Times'],Psol_hat),
            string[4]:mu,
            string[5]:std,
            string[6]:np.interp(sim_times,psol_pd['Times'],psol_pd['Psol'])}

    output = pd.DataFrame(Data, columns=string)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(output['Times'],output['PSol_raw_interp'],label='Raw Psol')
        plt.plot(output['Times'],output['PSol_smooth'],label='PSol_smooth')
        #plt.plot(output['Times'],output['PSol_savgol'],label='PSol_savgol')
        #plt.plot(output['Times'],output['PSol_mu'],label='Psol_mu')
        plt.plot(output['Times'],output['PSol_smooth']+ (output['PSol_mu']+2*output['PSol_std']),label='Psol_+2std')
        plt.plot(output['Times'],output['PSol_smooth']+ (output['PSol_mu']-2*output['PSol_std']),label='Psol_-2std')
        plt.ylabel('Power (W)')
        plt.ylim(0,1.1*output['PSol_smooth'].max())
        plt.xlabel('simulation times (s)')
        plt.legend()

    return output

#
#
# def time_step_calculation(Sweep_Frequency):
#
        # self.SweepFrequency = 4
        # self.EquilibriumFrequency = 0.25/4
#
        # calc_EqFreq = (max(self.EquilibriumData.Times)-min(self.EquilibriumData.Times)
                       # )/(self.EquilibriumData.shape[0]-1)
#
        # self.Eq_time_error = calc_EqFreq
        # self.Eq_time_erPercent = (calc_EqFreq - self.EquilibriumFrequency)/self.EquilibriumFrequency
        # self.tmin = tmin
        # self.tmax = tmax
#
        # self.SimLength = self.tmax-self.tmin
        # N_sim = self.SimLength / self.EquilibriumFrequency
        # N_sim = int(N_sim)
#
        # self.Sim_times = np.array(range(N_sim))*self.EquilibriumFrequency
#
        # EQ_SP = []
        # self.SP = format_JET_data(self.OB_SP, 'Strike_Point')
        # for i in range(len(self.EquilibriumData.Times)):
            # EQ_SP.append(self.SP['Strike_Point']
                         # [abs(self.SP['Times'] - self.EquilibriumData.Times[i]).idxmin()])
#
        # if 'Strike_Point' not in self.EquilibriumData:
            # self.EquilibriumData.insert(2, 'Strike_Point', EQ_SP)
#
#
#
#
#
        # EqList = []
        # ind = []
        # EqSP = []
    # if sim_type == 'SP_track':
        # for i in range(len(self.Sim_times)):
            # ind.append(abs(self.SP['Times']-(self.Sim_times[i]+self.tmin)).idxmin())
            # EqList.append(
                # abs(self.EquilibriumData['Strike_Point'] - self.SP['Strike_Point'][ind[i]]).idxmin())
            # EqSP.append(self.EquilibriumData.Strike_Point[EqList[i]])
        # self.EqList = EqList
          # ReconDict = {'Model_Times':self.Sim_times,
                       # 'Simulation_Times':self.Sim_times+tmin,
                       # 'Strike_Point_Efit':self.SP['Strike_Point'][ind],
                       # 'Equilibrium_Number':EqList,
                       # 'Eqdsk_StrikePoint':EqSP,
                       # 'Eqdsk_filename':self.EquilibriumData.Equil[EqList]}

    # if sim_type == 'Periodic':
        # symmetry = 0.5
#
        # SP = []
        # SPL = self.EquilibriumData.Strike_Point.idxmin()
        # SPU = self.EquilibriumData.Strike_Point.idxmax()
        # mid = (self.EquilibriumData.Strike_Point.max() -
               # self.EquilibriumData.Strike_Point.min())/2+self.EquilibriumData.Strike_Point.min()
        # SPM = abs(self.EquilibriumData.Strike_Point-mid).idxmin()
#
        # self.SP_i = [SPL, SPM, SPU]
#
        # self.SP_ind = signal.sawtooth(2 * np.pi*self.Sim_times *
                                      # self.SweepFrequency, symmetry)+1
#
        # for i in range(len(self.Sim_times)):
#
            # EqList.append(self.SP_i[int(self.SP_ind[i])])
            # EqSP.append(self.EquilibriumData.Strike_Point[EqList[i]])
        # self.EqList = EqList
        # EqList = np.round(EqList, 0)



#
# def find_equilibrium_strike_point(user_specified = False, ob_sp=[], check_user_specified=False, efit_obsp=[], eftf_obsp=[], PulseNo=[]):
#
    # if user_specified:
#
        # OB_SP = ob_sp
        # if check_user_specified:
            # if not efit_obsp:
                # if not PulseNo:
                    # print('No Pulse Number has been defined. Either suppy an efit_obsp series or supply a valid JET pulse number (or check user)')
                    # return
                # else:
                    # PSol, PRad, PTot, OB_SP = Read_JET_Power_data(PulseNo)
#
            # for i in len(OB_SP):
                # inp = abs(object.OB_SP.dimensions[0].data-OB_SP).argmin()
#
    # else:
        # if not PulseNo:
            # print('No Pulse Number has been defined. Either suppy an efit_obsp series or supply a valid JET pulse number')
            # return
        # else:
            # PSol, PRad, PTot, OB_SP = Read_JET_Power_data(PulseNo)




# def TimeSeries_Smoothing()


# def MeshMapping(Cell_centres, Ansys_mesh_file):
# if '.csv' in Ansys_mesh_file:
# Data = pd.read_csv(Ansys_mesh_file)
#
# print("Read Ansys Mesh : %s" % (Ansys_mesh_file.split('/')[-1]))
# Header = Data.columns.values
# X = [i for i, s in enumerate(Header) if 'X' in s]
# Y = [i for i, s in enumerate(Header) if 'Y' in s]
# Z = [i for i, s in enumerate(Header) if 'Z' in s]
# DATA = np.asarray(Data.iloc[:, [X[0], Y[0], Z[0]]])
# else:
# raise ValueError("Invalid file type enter CSV")
# if '.csv' in Cell_centres:
# Cell_centres = pd.read_csv(Cell_centres)
# print("Read Cell Centre csv : %s" % (Cell_centres.split('/')[-1]))
# Header2 = Cell_centres.columns.values
# X = [i for i, s in enumerate(Header2) if 'X' in s]
# Y = [i for i, s in enumerate(Header2) if 'Y' in s]
# Z = [i for i, s in enumerate(Header2) if 'Z' in s]
#
# Cell_centres = np.asarray(Cell_centres.ix[:, [X[0], Y[0], Z[0]]])
# else:
# Cell_centres = np.array(Cell_centres)
#
# if abs(np.mean(Cell_centres)) > 900:
# print("Order of magnitude of Cell Centres not the same as Ansys mesh")
# Cell_centres = Cell_centres/1000
# print("Read in smardda Results and Target ansys mesh: \n Results mesh contains %s Nodes \n Target mesh contains %s Nodes) " % (
# str(len(Cell_centres)), str(len(Data.ix[:, X]))))
#
# QId = []
# for i in range(len(Data.iloc[:, 1])):
# QId.append(np.argmin(np.sum((DATA[i, :]-Cell_centres)**2, axis=1)))
# print("Completed mapping nodes, the index in :'Node2Node_map gives the mapping")
#
# OutputDict = {'Node_Coordinates': DATA,
# 'Node2Node_map': QId}
