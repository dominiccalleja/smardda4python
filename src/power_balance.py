
"""
Copyright (c) 2019

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""
#Created on Tue Oct 16 14:10:16 2017


import shutil
__author__ = "Dominic Calleja"
__copyright__ = "Copyright 2019"
__credits__ = ["Wayne Arter", "Dominic Calleja"]
__license__ = "MIT"
__version__ = "0.4"
__date__ = '10/12/2019'
__maintainer__ = "Dominic Calleja"
__status__ = "Draft"
__email__ = "dominiccalleja@yahoo.co.uk, Wayne.Arter@ukaea.uk"

"""
Version history:
power_balance  0.2: Included new functions to extract the cell and body numbers from
            VTK files and a function to combine VTK files. This currently uses
            vtktfm module of Wayne Arters SMARDDA library. With the new atribute
            extraction functions this should be modified to execute entierly in
            this module. (Development intended to be independent of device
            - currently compatible with JET and DEMO)
ELM_Model  0.1: First attempt.
"""

"""
Class list:
- ELM_Model()
- Power_Balance()
"""


import smardda4python as sm
import smardda_updating as up
import PPF_Data as PPF  # exec.
import AnalysisS4P as AS
import heatequation as ht
import TStools as TS
from PPF_Data import PPF_Data

try:
	from jet.data import sal
except:
    print('No sal availiable, must use python3.5 install')


from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 15})

import pathos.multiprocessing as mp
import scipy
from scipy import integrate
from scipy.signal import savgol_filter
from sklearn import mixture
import scipy.optimize as op
import math
import copy
import pandas as pd
import numpy as np
import pickle
import sys
import dill


class ELM_Model():
    def __init__(self, simulation_times, pulse_times, model='full'):
        """ default description
        Class : ELM_Model

        Description : This class is used to develop a stochasatic ELM deposition model. This class is used to 
        fit a Gaussian Mixture model to the coefficients of the ELM fit model to develop a stochastic ELM 
        deposition model for tile 6 analyses. 

        Inputs : simulation_times = [0, lengthOfPulse]
                 pulse_times = JET recorded pulse times

        Returns : The init initialises a class that is used to store, train, sample and plot stochastic ELM models.
        """

        self._GMM_model_ = model
        self.sim_times = simulation_times
        self.pulse_times = pulse_times

        if not hasattr(self,'ELM_wetted_fration'):
            self.ELM_wetted_fration = 0.86
        if not hasattr(self,'ELM_power_split'):
            self.ELM_power_split    = 1
        if not hasattr(self,'ELM_Q_Bounds'):
            self.ELM_Q_Bounds = [0, 2.5E8]
        if not hasattr(self,'ELM_E_Bounds'):
            self.ELM_E_Bounds = [0,.45E6]

        self._elm_energy_constraint = False

                      
        print('ELM_Model initialised')
        # ref for power split during elms : 10.1088/0741-3335/49/5/002

    def load_ELM_pickle(self, ELM_pickle):
        """
        Method : load_ELM_pickle()

        Description: load_ELM_pickle() is used to load an ELM model from a pickle. This can be used to save the effort 
        of re-evaluating the full class every time one wishes to generate a stochastic ELM model. 

        Inputs : Expecting a pickle with attribuste: 

        """
        if isinstance(ELM_pickle, str):
            print('Extracting ELM Data : {}'.format(ELM_pickle))
            try:
                with open(ELM_pickle, 'rb') as handle:
                    self.ELMS = pickle.load(handle)
            except:
                print('Error {} looks like a path to the ELM pickle but doesnt look like its there'.format(
                    ELM_pickle))
        else:
            self.ELMS = ELM_pickle

        self.x_r = self.ELMS['x_r']
        self.ELM_pd = self.ELMS['ELMS_out_smooth']
        self.Qprof = self.ELMS['Q_pd_smooth']

        if self._GMM_model_ == 'full':
            self.GMM = self.ELMS['gmm']
            self.adjust = self.ELMS['adjust_gmm']
        else:
            self.GMM = self.ELMS['gmm_reduced']
            self.adjust = self.ELMS['adjust_gmm_reduced']

        return self.GMM

    def train_GMM(self, ELM_PD=[], QPROF=[], plot=False, test_model=True, n_components=6):
        """
        Method : train_GMM()

        Description:Used to train a GMM model for the stochastic ELM model. 

        Inputs : No inputs required. 
            Optional:
                    ELM_PD = A pandas table 
                    QPROF  = JET IR Camera field
                    n_components = Number of gaussians in the GMM
        """
        #CHECKING YOU HAVE EVERYTHING YOU NEED
        ################################################################################################################
        
        # Overwritting ELM_PD and QPROF if one is in the class but a new one has been supplied to this method
        if isinstance(ELM_PD,pd.DataFrame):
            if not hasattr(self, 'ELM_pd'):
                print('Overwriting self.ELM_pd')
            self.ELM_pd = ELM_PD

        if isinstance(QPROF,pd.DataFrame):
            if not hasattr(self, 'ELM_pd'):
                print('Overwriting self.ELM_pd')
            self.Qprof = QPROF
        
        if not (hasattr(self, 'ELM_pd') and hasattr(self, 'Qprof')):
            print('Missing ELM_pd or Q_Profile\n Either pass to this function or evaluate workflow with PPF.Power_Balance.compute_power_balance')

                        #if not hasattr(self, 'P_model'):
                #
                        #    self.GMM, self.adjust, X_train, X_test, Y_pd, Energy_model = train_GMM(
                        #        self._evaluate_elm_model,  self.ELM_pd, self.Qprof, n_components=n_components)
                        #else:
        # Apply null condition to GMM model constraints if none have been supplied
        if not hasattr(self,'conditions'):
            self.conditions=[]
        ################################################################################################################

        # Generate training data object 
        self._training_data(self.ELM_pd)

        # Main function inside this method to actually train the GMM
        self.GMM, self.adjust, self.X_train, self.X_test, self.Y_train, self.Y_test = train_GMM(
            self._evaluate_elm_model, self.ELM_training_data, self.Qprof, conditions=self.conditions, P_model=self.P_model, n_components=n_components)

        if test_model:
            # Not currently doing anything important
            #TODO add some general model tests 

            P_E_sim, X_E_sim = up.ecdf(self.Y_test)
            P_E_reg, X_E_reg = up.ecdf(self.Y_train.values)

        if plot:
            #Plotting the GMM 
            self.plot_GMM()

        return self.X_train, self.X_test
    
    def clean_ELM_pd(self):
        """
        This method applies the user specified constraints to the GMM training data and removes any points that 
        violate the conditions. 
        """
        X_train, drop = apply_conditions(self.ELM_pd[['model_area', 'model_Qmax', 'model_S_decay',
                                                      'model_ELM_duration', 'model_strike_point', 'model_q_baseline', 'model_Energy']], self.conditions)
        self.ELM_pd_clean = self.ELM_pd.drop(index=drop)


    def plot_GMM(self,confidence=True,alpha_level=0.95,xlim=[]):
        """
        This method is used to plot
            - the pairwise GMM coeficients comparing a sample of the GMM to the emperical fit coefcients.  
            - The cumulative emperical Energy of the QProf and the GMM test samples evaluated through the ELM shape model 
        """
        
        PPF.plot_mixed_gaussian_model(self.X_test, self.X_train)
        if not hasattr(self,'ELM_pd_clean'):
            self.clean_ELM_pd()
        plot_elm_model_fit(self.ELM_pd_clean)
        if confidence:
            # Y_train
            plot_model_confidence(self.ELM_pd['IR_Energy'], self.Y_test, alpha_level, xlim=xlim)

    def _training_data(self,ELM_pd = []):
        """
        This method can be used to add a pandas table for the ELM coeficients needed to train the GMM model. 
        """
        if isinstance(ELM_pd,pd.DataFrame):
            self.ELM_training_data = ELM_pd[['model_area', 'model_Qmax', 'model_S_decay',
                             'model_ELM_duration', 'model_strike_point', 'model_q_baseline', 'model_Energy']].copy(deep=True)
        elif hasattr(self, 'ELM_pd'):
            self.ELM_training_data = self.ELM_pd[['model_area', 'model_Qmax', 'model_S_decay',
                                             'model_ELM_duration', 'model_strike_point', 'model_q_baseline', 'model_Energy']].copy(deep=True)
        else:
            print('ERROR: No ELM training data has been supplied. \n\tRecomend you use the power_balance class to generate training data!')      
        return self.ELM_training_data

    def _get_training_data(self):
        return self._training_data(self.ELM_training_data)

    def _get_training_labels(self):
        if hasattr(self,'ELM_training_data'):
            self._training_data()
        return self._training_data(self.ELM_training_data).columns.values

    def init_model_conditions(self,**kwargs):
        """
        Method used to input user specified model conditions. 

        Required Key Value pairs. 
            Where the key corresponds to a key in the ELM training data. 
            Value must be a two sided interval. 
                aka :
                    self.init_model_conditions(model_area = [0.5,10])

        Pass as many comma deliminated key value pairs as you wish.
        """

        if not hasattr(self,'ELM_training_data'):
            print('Missing ELM_Training_Data evaluating  self._training_data()')
            self._training_data()

        print('generating conditions for GMM model')
        labels = self._get_training_labels()
        conditions = {}
        c = 0
        for k, V in kwargs.items():

            if not k in labels:
                print('Warning {} not in {}'.format(k, labels))

            input_ind = np.where([x in k for x in labels])[0][0]
            print('Condition {} : {} in range[{} {}]'.format(c, k, V[0], V[1]))
            def cond(x, input_ind=input_ind, v=V): return np.invert(
                np.logical_and(x[:, input_ind] > v[0], x[:, input_ind] < v[1]))
            conditions[c] = cond
            c += 1
        self.conditions = conditions
        return conditions

    def P_model(self, N, condition=None, GMM=None, adjust=None):
        """
        This method is used to generate samples of the GMM whilst applying the specified conditions. 
        It is used by a number of other methods also, but it is acessible to the user if they wish to 
        generate samples of the GMM too.

        """
        if GMM:
            self.GMM = GMM
            if adjust.any():
                self.adjust = adjust
        X, _ = self.GMM.sample(N)

        X = X * self.adjust
        c = 0

        if isinstance(self.conditions,dict):
            #cond = condition

            def cond(x, conditions=self.conditions):
                for i, k in enumerate(conditions.keys()):
                    if conditions[k](x).any():
                        return True
                return False
            
            while cond(X):
                c = c+1
                for i, k in enumerate(self.conditions.keys()):
                    if np.sum(self.conditions[k](X))>0:
                        s1, _ = self.GMM.sample(np.sum(self.conditions[k](X)))
                        X[self.conditions[k](X), :] = s1 * self.adjust  # S constraint
        return X

    def _tau_to_taufall(self, tau):
        pv = np.polyfit(self.ELM_pd.model_ELM_duration,
                        self.ELM_pd.Tau_fall, 1)
        return np.polyval(pv, tau)

    def SET_ELM_ENERGY_CONSTRAINT(self):
        print('setting Energy constraint on ELMs instead of power')
        self._elm_energy_constraint = True

    def _evaluate_elm_model(self, x, Q_out=False, t_interp=50):
        A = x[0]
        Q = x[1]
        S = x[2]
        t = x[3]
        sp = x[4]
        q0 = x[5]

        tau_fall = self._tau_to_taufall(t)
        tau_rise = tau_fall - t

        times = np.linspace(tau_rise, tau_fall, t_interp)
        Q_bdry, model_QR, model_Q_int, model_Energy = PPF.simulation_ELM_integrate(
            times, self.x_r, q0, Q, S, sp, A, wetted_fraction=self.ELM_wetted_fration, power_split=self.ELM_power_split)

        #print('Qmax {:.2e} \n IntQ_max : {:.2e} \n Energy : {:.2e}'.format(
        #        Q_bdry.max().max(), model_Q_int.max(),model_Energy))
        # or self.ELM_Q_Bounds[0] > model_Q_int.min():
        c = 0
        if not self._elm_energy_constraint:
            def Constrain(Q,E):
                return (self.ELM_Q_Bounds[1] < Q.max().max() or self.ELM_Q_Bounds[0] > Q.min().min())
        else:
            def Constrain(Q, E):
                return (self.ELM_E_Bounds[1] < E or self.ELM_E_Bounds[0] > E)

        while Constrain(Q_bdry, model_Energy):
            c +=1
            X = self.P_model(1)
            Q_bdry, model_QR, model_Q_int, model_Energy = PPF.simulation_ELM_integrate(
                times, self.x_r, X[0, 5], X[0, 1], X[0, 2], X[0, 4], X[0, 0], wetted_fraction=self.ELM_wetted_fration, power_split=self.ELM_power_split)
            if math.isnan(model_Energy):
                print('WARNGING: Unexpected NaN')
                X = self.P_model(1)
                Q_bdry, model_QR, model_Q_int, model_Energy = PPF.simulation_ELM_integrate(
                    times, self.x_r, X[0, 5], X[0, 1], X[0, 2], X[0, 4], X[0, 0], wetted_fraction=self.ELM_wetted_fration, power_split=self.ELM_power_split)

        #print('Resampled {} times for breaching Q threshold [{:.2e},{:.2e}] W'.format(c,self.ELM_Q_Bounds[0],self.ELM_Q_Bounds[1]))
        if Q_out:
            return model_Energy, Q_bdry, model_Q_int
        return model_Energy

    def set_frequency(self, fq=60):
        """
        setting frequency to a sample generating function allows you to add uncertainty to the sample
        """
        if callable(fq):
            def f(x): return fq(x)
            self.freq = f
        else:
            def f(x): return fq
            self.freq = f


    def generate_synthetic_timeseries(self, n_timeseries, time_points, n_cores=10, Q_bdry=False):
        #TODO: fix the parallelisation bug!! 
        # |--> COMPLETE. NEED TODO more tests

        if not hasattr(self, 'freq'):
            self.set_frequency()
        print('Generating {} synthetic ELMS: dT_n {}'.format(
            n_timeseries, time_points))
        print('Executing self.generate_synthetic_timeseries() on {} cores'.format(n_cores))
        p = mp.Pool(n_cores)


        times = np.linspace(self.sim_times[0], self.sim_times[-1], time_points)
        
        P_model= copy.deepcopy(self.P_model)
        eval_elm_model= copy.deepcopy(self._evaluate_elm_model)
        def evaluate_elm_model(x):
            return eval_elm_model(x,Q_out=True)
        freq = copy.deepcopy(self.freq)
        
        elm_lag = self.elm_lag

        def func(seed): return multi_process_func(seed,elm_lag = elm_lag,times=times,P_model=P_model,evaluate_elm_model=evaluate_elm_model,freq = freq,x_r=self.x_r)
        
        results = p.map(func, np.random.randint(1, 1E6, n_timeseries))
        p.close()
        self.sinthetic_elm_times = times
        Energy = np.array([results[i][1] for i in range(n_timeseries)])
        sinthetic_ELM_power = np.concatenate(
            [[results[i][0]] for i in range(n_timeseries)], axis=0)

        self.results = results
        if Q_bdry:
            Q_ELM_bdry = []
            for i in range(n_timeseries):
                Q_ELM_bdry.append(results[i][2])
            return sinthetic_ELM_power, Energy, Q_ELM_bdry
        return sinthetic_ELM_power, Energy

    def keys(self):
        return self.__dict__.keys()

    def pickle_dump(self, save_address):

        with open(save_address, 'wb') as handle:
            dill.dump(self, handle)

def load_object(save_address):
    with open(save_address, 'rb') as handle:
        res = dill.load(handle)
    return res

class Power_Balance(PPF_Data, ELM_Model):
    def __init__(self, pulse_number, pulse_times, sample_rate='.12ms', smoothing_window=51, ELM_power_split=[], smoothing_poly_order=3, load_generated_data=False):
        
        PPF_Data.__init__(self, pulseNo=pulse_number)
        self.pulse_number = pulse_number
        self.pulse_times = pulse_times
        self.sim_times = np.array(pulse_times)-pulse_times[0]
        self.sample_rate = sample_rate
        self.smoothing_window = smoothing_window
        self.smoothing_poly_order = smoothing_poly_order
        self._ppf_data = False
        self.number_of_cores = 10
        self._n_synthetic_elm_counter = 0

        self.grad_elm_detect = [2,-10]
        self.elm_lag = 0.3
        self.strikepoint_spreading_factor = 0.7
        self.ELM_wetted_fration = 0.94
        self.ELM_Q_Bounds = [0, 2.5E8]
        self.ELM_E_Bounds = [0,.45E6]
        self.ELM_power_split = ELM_power_split
        
        if not load_generated_data:
            try:
                self._load_JET_DATA()
            except:
                print('No JET data loaded!\n use load_generated_data')
        else:
            print(
            	'You have chosen to load generated data! \n load_generated_data=True \n Please use self.load_generated_data(generated_data)')


    def generated_data(self):
        return extract_generated_data(self)

    def load_generated_data(self, generated_data):
        input_generated_data(self, generated_data)
        print('Generated attributes loaded into self')

    def _load_JET_DATA(self, dW_method = 'savgol', dW_window=201, dw_interp=2, dW_partitions = 100, dw_order=4, dw_smoothing_order=2, finite_difference_step=0.8):  # dw_interp = 5, dw_smoothing_order = 3,
        #ELM_Model.__init__(self, self.sim_times, self.pulse_times)
        print('Loading JET power data ...')
        power_pd = PPF.get_JET_power_data(self.pulse_number)

        print('Formatting JET power data ...')
        input_power, power_pd = PPF.format_JET_input_power_data(
            self.pulse_times, power_data=power_pd, resample_rate=self.sample_rate)
        self.resampled_power_data = input_power
        P_rad_resample, PRad = PPF.format_JET_radiation_data(
            self.pulse_times, power_pd, resample_rate=self.sample_rate)

        input_power[input_power.columns[input_power.columns != 'Time']] = savgol_filter(input_power.drop(
            columns='Time'), window_length=self.smoothing_window, polyorder=self.smoothing_poly_order, axis=0)

        if hasattr(input_power, 'W_mhd') and hasattr(input_power, 'W_dia'):
            W = PPF.stored_energy(input_power['W_dia'], input_power['W_mhd'])
            input_power['W'] = W
        elif hasattr(input_power, 'W_mhd') and not hasattr(input_power, 'W_dia'):
            input_power['W'] = input_power['W_mhd']
        elif hasattr(input_power, 'W_dia') and not hasattr(input_power, 'W_mhd'):
            input_power['W'] = input_power['W_dia']
        else:
            print('No plasma storred Energy found!')
        try:
            if dW_method=='savgol':
                self.compute_dWdt_savgol_deriv(
                    dW_window=dW_window, dw_order=dw_order)
            else:
                self.compute_dWdt_finite_difference(dW_partitions=dW_partitions, dw_interp=dw_interp, dw_order=dw_order,
                                                dw_smoothing_order=dw_smoothing_order, finite_difference_step=finite_difference_step)
            flag = True
        except:
            flag = False
            print('WARNING: No stored plasma enegry used in calculations')

        try: 
            psol_ppf = PPF.sal.get('/pulse/'+str(self.pulse_number)+'/ppf/signal/jetppf/scal/psol')
            psol_ppf = TS.format_JET_data(psol_ppf,'PSOL')
            self.psol_ppf = psol_ppf.drop(index = np.where(np.logical_or(psol_ppf['Times']>self.pulse_times[1],psol_ppf['Times']<self.pulse_times[0]))[0])
        except:
            print('No JET PSOL ppf found')
            print('TODO: fill with standard PSOL calc. P_out-P_rad')

        try:
            self.EFIT()
            SP_Data = TS.format_JET_data(self.efit.EFIT_SPr, 'SP_R')
            SP_Data = SP_Data.drop(index=np.where(np.logical_or(
            	SP_Data['Times'] > self.pulse_times[1], SP_Data['Times'] < self.pulse_times[0]))[0])
            self.SP_Data = SP_Data
        except:
            print('No JET EFIT strike point ppf found')

        print('Computing P_OUT ...')
        print('\t Computing P_IN ...')
        self.P_IN = input_power[list(np.array(list(input_power.columns.values))[[x in ['P_ohm', 'P_ich', 'P_nbi'] for x in input_power.columns.values]])].sum(axis=1)

        print('\t Smoothing P_Rad ...')
        P_rad_resample['P_rad'] = savgol_filter(P_rad_resample['P_rad'], window_length=self.smoothing_window, polyorder=self.smoothing_poly_order)
        
        self.P_RAD = P_rad_resample['P_rad'].reindex(index=input_power.index, method='nearest')
        self.P_IN.index = input_power['Time']
        self.P_RAD.index = input_power['Time']
        self.resampled_power_data['P_rad'] = self.P_RAD.values

        print('\t P_OUT complete')
        self.P_OUT = self.P_IN.copy(deep=True)
        if flag:
            self.P_OUT = self.P_OUT - self.P_RAD.values - np.squeeze(self.dW.values)
        else:
            self.P_OUT = self.P_IN.values - self.P_RAD.values
        self.P_OUT.index = input_power['Time']


        print('Attempting to prepare Tile 5 power data...')
        try:
            P_T5_resample, PT5 = PPF.format_JET_tile5_data(
                self.pulse_times, power_pd, resample_rate=self.sample_rate)
            P_T5 = P_T5_resample.copy(deep=True)
            P_T5['P_tile5'] = savgol_filter(
                P_T5['P_tile5'], window_length=self.smoothing_window, polyorder=self.smoothing_poly_order)
            self.P_T5 = P_T5['P_tile5'].reindex(
                index=input_power.index, method='nearest')
            self.P_T5.index = input_power['Time']
            print('\t Completed Tile 5 power prep')
        except:
            print('\t No JET Tile 5 data availiable ... skipping')

    def compute_dWdt_finite_difference(self, dW_partitions= 200,dw_interp= 2,dw_order= 1,dw_smoothing_order= 2,finite_difference_step=0.1, smoothing_window =[]):
        if not smoothing_window:
            smoothing_window = self.smoothing_window
        
        print('Computing dWdt with finite difference \n \t step size {}  \n \t signal partitions {} \n \t smoothing window {} \n \t smoothing time {:.3f} (sec) \n'.format(
            finite_difference_step, dW_partitions, smoothing_window, (self.resampled_power_data['Time'].values[1]-self.resampled_power_data['Time'].values[0])*smoothing_window))

        W = self.resampled_power_data['W'].rolling(smoothing_window*2).mean().fillna(method='bfill')
        W.index = self.resampled_power_data['Time']
        tW, dWdt = calculate_psol_1D(W, partitions=dW_partitions, interp=dw_interp, order=dw_order, smoothing_order=dw_smoothing_order, res=finite_difference_step, ELM_MEAN=True)
        dWdt_interp = np.interp( self.resampled_power_data['Time'].values, tW, dWdt)
        self.dW = pd.DataFrame(
            dWdt_interp, index=self.resampled_power_data['Time'].values, columns=['dWdt'])
        self.resampled_power_data['dWdt'] = dWdt_interp
        return self.dW

    def compute_dWdt_savgol_deriv(self, dW_window=100, dw_order=2):
        print('Computing dWdt with Savitzky-Golay \n \t smoothing window {} \n \t smoothing time {:.3f} (sec)\n'.format(
             dW_window, (self.resampled_power_data['Time'].values[1]-self.resampled_power_data['Time'].values[0])*dW_window))

        dW = savgol_filter(self.resampled_power_data['W'], window_length=dW_window,
                           polyorder=dw_order, deriv=1, axis=0, delta=self.resampled_power_data['Time'][1]-self.resampled_power_data['Time'][0])
        self.dW = pd.DataFrame(
            dW, index=self.resampled_power_data['Time'].values, columns=['dWdt'])
        return self.dW


    def _return_smooth_power(self): return self.P_IN, self.P_OUT


    def get_PPF_data(self):
        self._ppf_data = True
        print('\t Loading JET IR camera data ...')
        self.IRcameraQ()
        print('\t Loading JET ELM data ...')
        self.ELMS()

    def process_ELMS(self, inter_ELM_period=0.90, IR_smoothing_window=9, IR_smoothing_polyorder=3, t_rise=0.003, t_fall=0.004, skip_elm=[]):
        
        if not self._ppf_data:  # not hasattr(self, 'IRcam'):
            print('Missing JET PPF Data : Executing self.get_PPF_data()')
            self.get_PPF_data()
            self._load_JET_DATA()
        if not hasattr(self, 'ELM_wetted_fration'):
            ELM_Model.__init__(self, np.array(
                self.pulse_times)-self.pulse_times[0], self.pulse_times, model='full')
        if not hasattr(self, 'SP_Data'):
            self.SP_Data = []

        print('Processing IR camera data ...')
        Q_pd, Q_pds, R, t = PPF.slice_Q_profile(
            self.IRcam.T6_QProf, sim_times=self.pulse_times)
        ELMS_pd = PPF.Inter_ELM_times(
            self.elms, filter_window=self.pulse_times, inter_ELM_period=inter_ELM_period)
        ELMS_out_smooth, Q_pd_smooth, Qprofile_smooth = PPF.smooth_and_extract_ELMS(
            Q_pd, ELMS_pd, R[0], self.SP_Data, window_length=IR_smoothing_window, polyorder=IR_smoothing_polyorder, wetted_fraction=self.ELM_wetted_fration, power_split=self.ELM_power_split, t_rise=t_rise, t_fall=t_fall, grad_rise=self.grad_elm_detect[0], grad_fall=self.grad_elm_detect[1], PN=self.pulse_number, skip_elm=skip_elm)

        self.x_r = R[0]
        self.ELM_pd = ELMS_out_smooth
        self.Qprof = Q_pd_smooth
        self.full_ELM_data = Qprofile_smooth

    def calculate_P_IR(self, wetted_fraction=[], power_split=[], power_split_bounds=[]):

        if not hasattr(self, 'Qprof'):
            print('Missing Q_profile Data : Executing self.process_ELMS()')
            self.process_ELMS()


        if not power_split:
            try:
                print('Calculating Avg O/I power split ...')
                BeII, split = PPF.calc_power_split(self.pulse_number, self.pulse_times)
                X = np.histogram(BeII['BeOut']/(BeII['BeIn']+BeII['BeOut']), 100)
                print('\t O/I power split calculated from BeII: {}'.format(split))
                def q_hat(k, mx, mu, sig): return  mx * PPF.gaussian(k, mu, sig)
                S = op.curve_fit(q_hat, X[1][1:], X[0], p0=[np.max(X[0]),split, 1], maxfev=int(1E5))
                print('\t O/I power split fitted gaussian std : {}'.format(S[0][2]))
                self.power_split = [split-abs(S[0][2]),split,split+abs(S[0][2])]
                print('\t O/I power split : {}'.format(self.power_split))

            except:
                power_split = .60
                power_split_bounds = [.55, .65]
                self.power_split = [power_split_bounds[0],power_split, power_split_bounds[1]]
                print('No O/I power split specified.')
                print(
                    'No Be-II data found. Out In power split set to {}'.format(self.power_split))
        else:
            self.power_split = power_split

        if not wetted_fraction:
            wf_label = ['JET', 'SMARDDA', 'Richiusa', 'Eich']
            wetted_fraction = []
            try:
                WF6 = PPF.sal.get('/pulse/'+str(self.pulse_number) +
                                  '/ppf/signal/kl9ppf/9aqp/wft6').data[0]
            except:
                WF6 = .94

            # modify to calculate from AS_smardda
            wetted_fraction.append(.8669)
            wetted_fraction.append(WF6)
            wetted_fraction.append(.7)
            wetted_fraction.append(.8)
            print('Using SMARDDA / JET / Richiusa, M/ Eich T.  toroidally wetted fractions: {}'.format(wetted_fraction))
            print('Richiusa, M : Maximizing JET divertor mechanical performance for the upcoming Deuterium-Tritium experiments (2018)')
            print(
                'Eich T.: Type-I ELM power deposition profile width and temporal shape in JET (2011)')
        else:
            wf_label = ['WF{:2f}'.format(x) for x in wetted_fraction]

        if hasattr(wetted_fraction, 'tolist'):
            wetted_fraction = wetted_fraction.tolist()

        # or not hasattr(wetted_fraction, 'shape')
        if not isinstance(wetted_fraction, list):
            wetted_fraction = [wetted_fraction]

        self.wetted_fraction = wetted_fraction

        print(
            'self.P_IR - P_IR  -- WF : {} - PS: {}'.format(wetted_fraction[0], self.power_split[1]))
        self.P_IR = PPF.IR_Q_to_Power_robust(
            self.Qprof, torr_wet=wetted_fraction[0])*(1/self.power_split[1])
        self.P_IR = self.P_IR.rename(columns={'IntQ': 'P_IR'})
        self.P_IR = self.P_IR.reindex(index=self.P_IN.index, method='nearest')
        self.P_IR_UQ = self.P_IR.copy(deep=True)

        for i, wf in enumerate(wetted_fraction):
            for j, ps in enumerate(self.power_split):
                if wf_label:
                    IR = (PPF.IR_Q_to_Power_robust(
                        self.Qprof, torr_wet=wf))*(1/ps)
                    self.P_IR_UQ['P_IR-WF:{}-PS:{}'.format(wf_label[i], ps)] = IR.reindex(
                        index=self.P_IN.index, method='nearest')
                else:
                    IR = (PPF.IR_Q_to_Power_robust(
                        self.Qprof, torr_wet=wf))*(1/ps)
                    self.P_IR_UQ['P_IR-WF:{:2f}-PS:{}'.format(wf, ps)] = IR.reindex(
                        index=self.P_IN.index, method='nearest')

        return self.P_IR_UQ

    def UQ_PSOL(self, ERROR=.05):
        """
        Error[0] =  Error on Input
        Error[1] =  Error on P_RAD
        """
        if not hasattr(ERROR, 'len'):
            ERROR = [ERROR,ERROR]

        if hasattr(self, 'dW'):
            self.P_dWdT_UQ = percentage_error(
                self.dW['dWdt'], ERROR[1], 'dWdt')

        self.P_IN_UQ = percentage_error(self.P_IN, ERROR[0], 'P_IN')
        self.P_RAD_UQ = percentage_error(self.P_RAD, ERROR[1], 'P_RAD')
        data = {}
        data['P_OUT'] = self.P_OUT.values

        if hasattr(self, 'dW'):
            print('P_OUT = P_IN + P_RAD + dW/dt')
            data['P_OUT_LB'] = self.P_IN_UQ['P_IN_LB'] - \
                self.P_RAD_UQ['P_RAD_UB'] - self.P_dWdT_UQ['dWdt_UB']
            data['P_OUT_UB'] = self.P_IN_UQ['P_IN_UB'] - \
                self.P_RAD_UQ['P_RAD_LB'] - self.P_dWdT_UQ['dWdt_LB']
        else:
            data['P_OUT_LB'] = self.P_IN_UQ['P_IN_LB'] - \
                self.P_RAD_UQ['P_RAD_UB']
            data['P_OUT_UB'] = self.P_IN_UQ['P_IN_UB'] - \
                self.P_RAD_UQ['P_RAD_LB']

        self.P_OUT_UQ = pd.DataFrame(data, index=self.P_OUT.index.values, columns=[
                                     'P_OUT', 'P_OUT_LB', 'P_OUT_UB'])
        return self.P_OUT_UQ

    def init_ELM_model_from_pickle(self, ELM_pickle, model='full'):
        ELM_Model.__init__(self, np.array(self.pulse_times) -
                           self.pulse_times[0], self.pulse_times, model='full')
        self.load_ELM_pickle(ELM_pickle)
        self._n_synthetic_elm_counter = 0

    def init_ELM_model(self, plot=True):
        if not hasattr(self, 'ELM_pd'):
            print('Missing ELM Data: Executing self.process_ELMS()')
            self.process_ELMS()
        if not hasattr(self, 'Qprof'):
            print('Missing QProf Data: Executing self.process_ELMS()')
            self.process_ELMS()
        ELM_Model.__init__(self, np.array(self.pulse_times) -
                           self.pulse_times[0], self.pulse_times, model='full')

        self.train_GMM(self, plot=plot)

    def generate_synthetic_ELMs(self, n_timeseries, time_resolution, frequency=65, frequency_bounds=4, Q_bdry=False):

        if not hasattr(self, '_n_synthetic_elm_counter') or not hasattr(self, 'elm_lag'):
            print('Missing ELM Data.\n')
            # <<<< replace this with automatic generation of GMM!
            print('Running ELM_Model._init() ...\n no pickle supplied training GMM ...')
            self.init_ELM_model()

        def f(x): return frequency + \
            np.random.uniform(-frequency_bounds, frequency_bounds, 1)#[0]
        self.set_frequency(fq=f)

        description = 'Generated synthetic ELM data set \n ELM frequency : {}(Hz) - [{},{}] \
                    \n Time resolution : {} \n N timeseries : {}'.format(frequency, frequency-frequency_bounds, frequency+frequency_bounds, time_resolution, n_timeseries)
        print(description)

        if hasattr(self, 'Synthetic_ELMs'):
            if not hasattr(self, Multi_synth_elm):
                self.Multi_synth_elm = PPF.data_holder()
            setattr(self.Multi_synth_elm, str(
                self._n_synthetic_elm_counter), self.Synthetic_ELM_data)

        self.Synthetic_ELM_data = PPF.data_holder()

        if Q_bdry:
            sinthetic_ELM_power, Energy, Q_bdry = self.generate_synthetic_timeseries(
                n_timeseries, time_resolution, n_cores=self.number_of_cores, Q_bdry=True)
            setattr(self.Synthetic_ELM_data, 'Q_bdry', Q_bdry)
            keys = ['Description', 'Power', 'Power_full_divertor', 'Energy_tot_GMM', 'Q_bdry','Time']
        else:
            sinthetic_ELM_power, Energy = self.generate_synthetic_timeseries(
                n_timeseries, time_resolution, n_cores=self.number_of_cores)
            keys = ['Description', 'Power', 'Power_full_divertor', 'Energy_tot_GMM', 'Time']
        setattr(self.Synthetic_ELM_data, 'Description', description)
        setattr(self.Synthetic_ELM_data, 'Power', sinthetic_ELM_power)
        setattr(self.Synthetic_ELM_data,
                'Power_full_divertor', sinthetic_ELM_power)
        setattr(self.Synthetic_ELM_data, 'Energy_tot_GMM', Energy)
        setattr(self.Synthetic_ELM_data, 'Time', self.sinthetic_elm_times)
        setattr(self.Synthetic_ELM_data, 'keys', keys)

        self._n_synthetic_elm_counter += 1
        return self.Synthetic_ELM_data

    def compute_energy_balance(self, n_elm_series=10, UQ=True, plot=True, verbose=True):
        if not hasattr(self, 'P_IR'):
            print('Missing P_IR Data : Executing self.calculate_P_IR()')
            self.calculate_P_IR()
        if not hasattr(self, 'GMM'):
            print('Missing GMM model: Executing self.init_ELM_model()')
            self.init_ELM_model()

        if UQ == True:
            if not hasattr(self, 'Power_in_pd'):
                print('Missing Power Bounds : Evaluating self.UQ_PSOL()')
                self.UQ_PSOL()
        else:
            self.UQ_PSOL(ERROR=0)

        if not hasattr(self, 'Synthetic_ELM_data'):
            self.Synthetic_ELM_data = self.generate_synthetic_ELMs(
                n_elm_series, len(self.P_IR.index))

        self.P_ELM = pd.DataFrame(self.Synthetic_ELM_data.Power.T, index=self.Synthetic_ELM_data.Time+self.pulse_times[0], columns=[
                                  'P_ELM_{}'.format(i) for i in range(np.shape(self.Synthetic_ELM_data.Power.T)[1])])
        label = []
        if hasattr(self, 'P_T5'):
            print('Tile 5 Data availiable so E_IR = E_IR + E_IR_T5')
            self.t5_toroidal_factor = ((2*np.pi*2.2*self.wetted_fraction[0]))
            self.P_IR_UQ_with_T5 = self.P_IR_UQ.copy(deep=True)
            self.P_IR_UQ_with_T5[self.P_IR_UQ_with_T5.columns.values] = (
                self.P_IR_UQ.values.T + (self.P_T5.values*self.t5_toroidal_factor)).T
            self.E_IR_pd_with_T5, lab = calculate_energy(
                self.P_IR_UQ_with_T5, 1)
        else:
            print('No Tile 5 data')

        self.E_IR_pd, lab = calculate_energy(self.P_IR_UQ, 1)
        label.append(lab)
        self.E_IN_pd, lab = calculate_energy(self.P_IN_UQ, 1)
        label.append(lab)
        self.E_OUT_pd, lab = calculate_energy(self.P_OUT_UQ, 1)
        label.append(lab)
        self.E_ELM_pd, lab = calculate_energy(self.P_ELM, 1)
        label.append(lab)
        label = [x.rstrip('_') for i, x in enumerate(label)]
        if verbose:
            print('Completed power and energy calculation :')
            mu = []
            mx = []
            mi = []
            arr = []
            names = []
            for i, lab in enumerate(label):
                names.append(lab)
                E = getattr(self, lab+'_pd')
                mu.append(E.iloc[-1].mean())
                mx.append(E.iloc[-1].max())
                mi.append(E.iloc[-1].min())
                arr.append(E.iloc[-1])
                print(
                    ' '*7 + '{}'.format('\t'.join(['{}'.format(x) for x in E.columns.values])))
                print('{} : {}(J)'.format(lab, '\t'.join(
                    ['{:.3e} (J)'.format(x) for x in E.iloc[-1]])))

            print(
                '   ' + '\n {} : \n'.format('\t'.join(['{}'.format(x for x in names)])))
            print('Mean Energy : {}'.format(
                '\t'.join(['{:.3e} (J)'.format(x) for x in mu])))
            print('Min Energy : {}'.format(
                '\t'.join(['{:.3e} (J)'.format(x) for x in mi])))
            print('Max Energy : {}'.format(
                '\t'.join(['{:.3e} (J)'.format(x) for x in mx])))

    def compute_psol(self, rolling_window=200, n_dt=100, piecewise_order=3, smoothing_order=1, finite_difference_step=3):

        if not hasattr(self, 'E_OUT_pd'):
            print('No Energy data found. Evaluating: self.compute_energy_balance()')
            self.compute_energy_balance()

        print('Evaluating P_SOL \n Energy smoothing window : {}(sec) \n Signal disection : {} \n finite difference step size : {} '.format(
            ((self.E_OUT_pd.index[1]-self.E_OUT_pd.index[0])*rolling_window), n_dt, finite_difference_step))
        print('Evaluating P_SOL for {} synthetic ELM series'.format(
            len(self.E_ELM_pd.columns)))
        print('E_OUT computations : {}'.format(self.E_OUT_pd.columns))

        "calculate baseline"

        T, P = calculate_psol(self.E_OUT_pd['E_OUT'], self.E_ELM_pd, elm_lag=self.elm_lag,
                              smoothing_window=rolling_window, interp=n_dt, order=piecewise_order, fitting_order_Energy=smoothing_order)
        Psol_calc = pd.DataFrame(P, index=T)
        Psol_calc.columns = ['baseline']

        label_list = ['baseline']
        data = np.zeros([len(Psol_calc.index.values), len(
            self.E_OUT_pd.columns)*len(self.E_ELM_pd.columns)])
        len_elm = len(self.E_ELM_pd.columns)
        for i, label_out in enumerate(self.E_OUT_pd.columns.values):
            for j, label_elm in enumerate(self.E_ELM_pd.columns):
                out = self.E_OUT_pd[label_out].rolling(rolling_window).mean().fillna(method='bfill')
                elm = self.E_ELM_pd[label_elm].rolling(rolling_window).mean().fillna(method='bfill')
                T, data[:, (i*len_elm)+j] = calculate_psol(out, elm, elm_lag=self.elm_lag,
                                                           smoothing_window=rolling_window, interp=n_dt, order=piecewise_order, fitting_order_Energy=smoothing_order)

            label_list = label_list + [label_out + '_' + x.replace(commonprefix(
                self.E_ELM_pd.columns.values.tolist()), '') for x in self.E_ELM_pd.columns.values.tolist()]

        PSOL = pd.DataFrame(
            data, columns=label_list[1:], index=Psol_calc.index)
        PSOL = pd.concat([Psol_calc, PSOL], axis=1)

        self.P_SOL_pd = PSOL

    def plot_psol_calculation(self, plot_smoothing_window=201):

        if not hasattr(self, 'P_SOL_pd'):
            print('Missing P_SOL_pd : Executing self.compute_psol()')
            self.compute_psol()

        fig, axs = plt.subplots(2, 1, figsize=(17, 12), sharex=True)
        fig.subplots_adjust(wspace=.2, hspace=0)

        axs[0].plot(self.E_OUT_pd.index.values, np.min(
            self.E_OUT_pd.values, axis=1)/1E6, c='red', alpha=0.3)
        axs[0].plot(self.E_OUT_pd.index.values, np.max(
            self.E_OUT_pd.values, axis=1)/1E6, c='red', alpha=0.3)
        axs[0].fill_between(self.E_OUT_pd.index.values, np.max(
            self.E_OUT_pd.values, axis=1)/1E6, np.min(self.E_OUT_pd.values, axis=1)/1E6, color='red', alpha=0.3)
        axs[0].plot(self.E_OUT_pd.index.values, self.E_OUT_pd['E_OUT'].values /
                    1E6, c='red', alpha=0.9, label='E_SOL + E_ELM')
        axs[0].plot(self.E_ELM_pd.index, np.max(
            self.E_ELM_pd, axis=1)/1E6, c='C0', alpha=0.3)
        axs[0].plot(self.E_ELM_pd.index, np.min(
            self.E_ELM_pd, axis=1)/1E6, c='C0', alpha=0.3)
        axs[0].fill_between(self.E_ELM_pd.index, np.max(
            self.E_ELM_pd, axis=1)/1E6, np.min(self.E_ELM_pd, axis=1)/1E6, color='C0', alpha=0.3)
        axs[0].plot(self.E_ELM_pd.index, np.mean(
            self.E_ELM_pd, axis=1)/1E6, c='C0', alpha=0.9, label='Synthetic E_ELM')
        axs[0].set_ylabel('Energy (MJ)')
        axs[0].set_xlim([self.E_ELM_pd.index.values[0],
                         self.E_ELM_pd.index.values[-1]])
        axs[0].legend()

        try : 
            self.psol_ppf
            axs[1].plot(self.psol_ppf['Times'],self.psol_ppf['PSOL']/1E6,label='P_SOL JET PPF',c='green')

        except:
            print('No PSOL PPF')
        axs[1].plot(self.P_SOL_pd.index.values, savgol_filter(
            self.P_SOL_pd.values, plot_smoothing_window, 3, axis=0)/1E6, c='C0', alpha=0.08)

        axs[1].plot(self.P_SOL_pd.index.values, savgol_filter(
            self.P_SOL_pd.values, plot_smoothing_window, 3, axis=0)/1E6, c='C0', alpha=0.01)
        axs[1].plot(self.P_SOL_pd.index.values, savgol_filter(
            self.P_SOL_pd.values, plot_smoothing_window, 3, axis=0)/1E6, c='C0', alpha=0.08)
        axs[1].plot(self.P_SOL_pd.index.values, savgol_filter(np.min(
            self.P_SOL_pd.values, axis=1)/1E6, plot_smoothing_window, 3), label='PSOL_LB')
        axs[1].plot(self.P_SOL_pd.index.values, savgol_filter(np.max(
            self.P_SOL_pd.values, axis=1)/1E6, plot_smoothing_window, 3), label='PSOL_UB')
        axs[1].plot(self.P_SOL_pd.index.values, savgol_filter(np.mean(self.P_SOL_pd, axis=1),
                                                              plot_smoothing_window*2+1, 3)/1E6, label='PSOL Mean', c='blue', linewidth=2, linestyle='--')
        axs[1].set_ylabel('Power (MW)')
        axs[1].legend()
        plt.show()

    def plot_power_workflow(self,legend=False):

        if hasattr(self,'dW'):
            fig, axs = plt.subplots(5, figsize=(17, 14), sharex=True)
        else:
            fig, axs = plt.subplots(4, figsize=(17, 14), sharex=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        try:
            axs[0].plot(self.resampled_power_data['Time'],
                        self.resampled_power_data['P_nbi']/1E6, c='orange', label='P_nbi')
        except:
            print('no P_nbi data')
        try:
            axs[0].plot(self.resampled_power_data['Time'],
                        self.resampled_power_data['P_ohm']/1E6, c='purple', label='P_ohm')
        except:
            print('no P_ohm data')
        try:
            axs[0].plot(self.resampled_power_data['Time'],
                        self.resampled_power_data['P_ich']/1E6, c='green', label='P_ich')
        except:
            print('no P_ich data')
        try:
            axs[0].plot(self.P_IN.index, self.P_IN /
                        1E6, label='P_tot', c='blue')
        except:
            print('no P_tot data')
        axs[0].set_xlim(self.pulse_times)
        axs[0].set_ylabel('Power (MW)')
        axs[0].legend()
        #axs[1].plot(PRad_re['Time'], PRad_re['P_rad'],alpha=0.2,c='red')
        axs[1].plot(self.P_RAD.index, self.P_RAD.values /
                    1E6, alpha=0.7, c='red', label='P_RAD')
        axs[1].set_ylabel('Power (MW)')
        axs[1].legend()

        if hasattr(self,'dW'):
            axs[2].plot(self.P_OUT.index, self.P_OUT.values /
                        1E6, alpha=0.7, label='P_sol + P_elm')
            axs[2].plot(self.resampled_power_data['Time'],
                        self.dW['dWdt']/1E6, label='dW/dt', c='green')
            axs[2].set_ylabel('Power (MW)')
            axs[2].legend()
            ind_IR = 3
        else:
            axs[1].plot(self.P_OUT.index, self.P_OUT.values /
                        1E6, alpha=0.7, label='P_sol + P_elm')
            ind_IR = 2
        
        axs[ind_IR].plot(self.P_IR.index.values, self.P_IR/1E6,
                    label='P_IR * {:.3f}'.format(1/self.power_split[0]), c='red')
        try:
            axs[ind_IR].plot(self.P_T5.index, (self.P_T5 *
                                          self.t5_toroidal_factor) / 1E6, label='P_T5', c='green')
            rect = [0.48, 0.001, 0.28, 0.4]
            subax1 = PPF.add_subplot_axes(axs[ind_IR], rect)
            tindle = np.logical_and(self.P_T5.index > (self.pulse_times[0] + (0.5*(self.pulse_times[1]-self.pulse_times[0]))), self.P_T5.index < (
                self.pulse_times[0] + 0.75*(self.pulse_times[1]-self.pulse_times[0])))
            subax1.plot(self.P_T5.index[tindle], (self.P_T5[tindle] *
                                                  self.t5_toroidal_factor) / 1E6, label='P_IR_T5', c='green')
            subax1.set_xlim((self.pulse_times[0] + (0.5*(self.pulse_times[1]-self.pulse_times[0])), (
                self.pulse_times[0] + 0.75*(self.pulse_times[1]-self.pulse_times[0]))))
            subax1.axes.get_xaxis().set_visible(False)
        except:
            print('no P_T5 data')

        axs[ind_IR].legend()
        axs[ind_IR].set_ylabel('Power (MW)')
        for i, lab in enumerate(list(self.E_IR_pd.columns)):
            axs[ind_IR + 1].plot(self.E_IR_pd.index.values, self.E_IR_pd[lab].values /
                        1E6, alpha=0.9, label=lab, linestyle='--')

        #if hasattr(self,'P_IR_UQ_with_T5'):
        #    axs[3].fill_between(self.P_IR_UQ_with_T5.index.values, np.min(self.P_IR_UQ_with_T5.values,axis=1)/1E6, np.max(self.P_IR_UQ_with_T5.values,axis=1) / 1E6, color='red', alpha=0.08)
        #else:
        axs[ind_IR + 1].fill_between(self.E_IR_pd.index.values, np.min(
            self.E_IR_pd.values, axis=1)/1E6, np.max(self.E_IR_pd.values, axis=1)/1E6, color='red', alpha=0.08)

        axs[ind_IR + 1].fill_between(self.E_OUT_pd.index.values, self.E_OUT_pd['E_OUT_UB'] /
                            1E6, self.E_OUT_pd['E_OUT_LB'] / 1E6, color='C0', alpha=0.2)

        axs[ind_IR + 1].plot(self.E_IR_pd.index.values, np.max(
            self.E_IR_pd.values, axis=1)/1E6, c='red', alpha=1)  # ,label='E_IR_UB')
        axs[ind_IR + 1].plot(self.E_IR_pd.index.values, np.min(
            self.E_IR_pd.values, axis=1)/1E6, c='red', alpha=0.7)  # ,label='E_IR_LB')

        axs[ind_IR + 1].plot(self.E_OUT_pd.index.values,
                    self.E_OUT_pd['E_OUT_LB'] / 1E6, c='C0', alpha=0.6)
        axs[ind_IR + 1].plot(self.E_OUT_pd.index.values,
                    self.E_OUT_pd['E_OUT_UB'] / 1E6, c='C0', alpha=0.6)

        #if hasattr(self,'P_IR_UQ_with_T5'):
        #    axs[3].plot(self.E_IR_pd.index.values , np.mean(self.E_IR_pd.values,axis=1)/1E6, label='Energy IR (5+6)', c='red', linewidth=2)
        #else:
        axs[ind_IR + 1].plot(self.E_IR_pd.index.values, np.mean(
            self.E_IR_pd.values, axis=1)/1E6, label='Energy IR', c='red', linewidth=2)

        axs[ind_IR + 1].plot(self.E_OUT_pd.index.values, self.E_OUT_pd['E_OUT'] /
                    1E6, label='Energy (SOL + ELM)', c='C0', linewidth=2)
        if legend:
            axs[ind_IR + 1].legend()
        axs[ind_IR + 1].set_xlabel('Time (s)')
        axs[ind_IR + 1].set_ylabel('Energy (MJ)')
        plt.show()


def extract_generated_data(OBJ):
    generated_data = {}
    for key in OBJ.keys():
        generated_data[key] = getattr(OBJ, key)
    return generated_data


def input_generated_data(OBJ, generated_data):
    for key in generated_data.keys():
        generated_data[key] = setattr(OBJ, key, generated_data[key])
    return OBJ


def isitInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def calculate_energy(Pow_pd, calibration_factor):
    names = Pow_pd.columns.values.tolist()
    prefix = commonprefix(names)
    prefix = prefix.replace('P', 'E')
    re_names = [names[i].replace('P', 'E') for i in range(len(names))]

    print('Integrating {} time series'.format(prefix))

    E_pd = Pow_pd.copy(deep=True)
    E_pd = E_pd.drop(index=E_pd.index[0])
    for i, name in enumerate(names):
        E_pd[name] = integrate.cumtrapz(
            Pow_pd[name].values * calibration_factor, Pow_pd.index.values)

    E_pd.columns = re_names
    return E_pd, prefix


def percentage_error(signal, error, label):

    LB = signal.values*(1-error)
    UB = signal.values*(1+error)

    data = {}
    data[label] = signal.values
    data[label+'_LB'] = LB
    data[label+'_UB'] = UB
    return pd.DataFrame(data, index=signal.index.values, columns=[label, label+'_LB', label+'_UB'])


def commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m:
        return ''
    prefix = m[0]
    for item in m:
        for i in range(len(prefix)):
            if prefix[:i+1] != item[:i+1]:
                prefix = prefix[:i]
                if i == 0:
                    return ''
                break
    return prefix


def apply_conditions(X_train, conditions):
    
    drop = []
    for i, k in enumerate(conditions.keys()):
        if np.sum((conditions[k](X_train.values))) > 0:
            d = np.where((conditions[k](X_train.values)))[0]
            if len(d)==0:
                break
            else:
                drop.append(d)
                print(drop[-1])
    if drop:
        drop = np.unique(np.concatenate(drop))
    
    old_index_x_train = X_train.index.values
    X_train = X_train.drop(index=drop)
    X_train = X_train.reset_index(drop=True)
    return X_train, drop

def train_GMM(elm_model,  ELM_pd, Qprof, conditions=[],N_model_samples=1000, n_components=6, P_model=[]):

    X_train = ELM_pd #.rename(columns={'model_area': 'Width (m)', 'model_Qmax': 'Q_max (W)', 'model_q_baseline': 'Q_0 (W)',
                     #                 'model_S_decay': 'S', 'model_ELM_duration': 'tau (s)', 'model_strike_point': 'Major Rad (m)', 'model_Energy': 'Energy (J)'})

    if isinstance(conditions, dict):
        X_train, drop = apply_conditions(X_train, conditions)
    Y_pd = X_train['model_Energy']  # ['Energy (J)']
    X_train = X_train.drop(columns='model_Energy')  # 'Energy (J)')

    # [1, X_train['Q_max (W)'].max(), 1, X_train['tau (s)'].min(), 1, X_train['Q_0 (W)'].max()]
    adjust = X_train.max().values
    X_pd = X_train #[['Width (m)', 'Q_max (W)', 'S'	, 'tau (s)', 'Major Rad (m)',	'Q_0 (W)']]

    GMM = mixture.GaussianMixture(
        n_components=n_components, covariance_type='full').fit(X_pd/adjust)


    X_test = P_model(N_model_samples, GMM=GMM, adjust=adjust)
    X_test = pd.DataFrame(data=X_test, columns=X_pd.columns.values)

    Y_test = np.zeros(len(X_test.index.values))
    for i in range(len(X_test.index.values)):
        Y_test[i] = elm_model(X_test.iloc[i].values)
        while Y_test[i] > np.mean(Y_pd.values) + (6*np.std(Y_pd.values)) or Y_test[i] < np.mean(Y_pd.values) - (6*np.std(Y_pd.values)):
            X_test.iloc[i] = P_model(1, GMM=GMM, adjust=adjust)
            Y_test[i] = elm_model(X_test.iloc[i].values)

    return GMM, adjust, X_train, X_test, Y_pd, Y_test



def multi_process_func(seed,elm_lag=[],times=[],P_model=[],evaluate_elm_model=[],freq = [],x_r=[],strike_point=[]):
    # strikepoint[0] = Times strikepoint[1] = SP position 
    np.random.seed(seed)
    N_elm = int((times[-1]-times[0])*freq(1))
    print('N_elms in syntetic time series: {}'.format(N_elm))
    time_ELM = np.linspace(times.min() + elm_lag, times.max(), N_elm)
    print('Time ELM shape : {}'.format(np.shape(time_ELM)))
    theta = P_model(N_elm)
    theta = theta[np.random.permutation(N_elm), :]
    Energy = []
    #x_r = np.linspace(2.8041, 2.9868, 50)

    Q_bdry_ELM = np.zeros([len(x_r), len(times)])
    ELM_power = np.zeros(len(times))
    Energy = 0
    for k, j in enumerate(np.array(range(N_elm))):
        x = theta[k, :]
        t_adj = times[np.argmin(abs(times-time_ELM[k]))]

        if len(strike_point):
            SP = np.interp(time_ELM[k], strike_point[0], strike_point[1])
            x[4] = SP

        e_elm, q_elm, q_int = evaluate_elm_model(x)

        elms_times = q_int.index.values + t_adj
        ind_ELM = np.logical_and(
            times > elms_times.min(), times < elms_times.max())
        fit = np.interp(times[ind_ELM], elms_times, q_int['IntQ'])
        ELM_power[ind_ELM] = fit
        Energy = Energy+e_elm

        #if Q_bdry:
        t_tmp = t_adj+q_elm.index.values
        # np.argmin(abs(times - time_ELM[j]))
        ind_ELM = np.logical_and(t_tmp[0] < times, t_tmp[-1] > times)
        Q_E = np.zeros([len(x_r), len(times[ind_ELM])])
        for i, t in enumerate(times[ind_ELM]):
            Q_E[:, i] = np.interp(x_r, q_elm.columns.values, q_elm.iloc[np.argmin(
                abs(t_tmp-t)), :])  # fit(times[ind_ELM])
        Q_bdry_ELM[:, ind_ELM] = Q_E

    return ELM_power, Energy, Q_bdry_ELM


def calculate_psol(E_OUT, E_ELM, elm_lag=[], smoothing_window=101, interp=100, order=3, fitting_order_Energy=2,spline=True):

    
    #ind = np.argmin(abs(E_ELM.index.values[0]+elm_lag - E_ELM.index.values))
    #print(ind)
    Time = np.linspace(E_OUT.index.values[0], E_OUT.index.values[-1], interp)
    if hasattr(E_ELM,'columns'):
        elm_y = np.mean(E_ELM.values, axis=1)
    else:
        elm_y = E_ELM.values
    if not spline:
        pval_0 = np.polyfit(E_OUT.index.values, E_OUT.values, fitting_order_Energy)
        pval_E0 = np.polyfit(E_ELM.index.values, elm_y, fitting_order_Energy)
        #print(pval_E0)
        #pval_E1 = np.polyfit(E_ELM.index.values[ind:], np.mean(E_ELM.values[ind:, :], axis=1), fitting_order_Energy)
        #print(pval_E1)

        #pval_E1 = np.polyfit(E_ELM.index.values[ind:], E_ELM.values[ind:], fitting_order_Energy)
        #intp0 = int(round((ind/len(E_OUT.index.values))*interp))
        #intp1 = int(round((1-(ind/len(E_OUT.index.values)))*interp))
        Y0 = np.polyval(pval_0, np.linspace(E_OUT.index.values[0], E_OUT.index.values[-1], interp)) - np.polyval(
            pval_E0, np.linspace(E_OUT.index.values[0], E_OUT.index.values[-1], interp))
        #Y1 = np.polyval(pval_0, np.linspace(E_OUT.index.values[0]+elm_lag, E_OUT.index.values[-1], intp1)) - np.polyval(
        #    pval_E1, np.linspace(E_OUT.index.values[0]+elm_lag, E_OUT.index.values[-1], intp1)) 
        P_SOL = savgol_filter(Y0, window_length=smoothing_window,
                              polyorder=order, deriv=1, axis=0, delta=Time[1]-Time[0])
    else:
        # https://stats.stackexchange.com/questions/226553/why-use-regularisation-in-polynomial-regression-instead-of-lowering-the-degree/226566#226566
        f_elm = scipy.interpolate.CubicSpline(E_ELM.index.values, elm_y)#, kind='cubic')
        f_out = scipy.interpolate.CubicSpline(E_OUT.index.values, E_OUT.values)#, kind='cubic')
        P_SOL = savgol_filter(f_out(Time)-f_elm(Time), window_length=smoothing_window,polyorder=order, deriv=1, axis=0, delta=Time[1]-Time[0])

    return Time, P_SOL

"""
def calculate_psol(E_OUT, E_ELM, elm_lag=[], smoothing_window = 101,interp=100, order=3, fitting_order_Energy=2, finite_difference_step=0.02, ELM_MEAN=True):

    #if not elm_lag:
    #    slices = np.linspace(
    #        E_OUT.index.values[0], E_OUT.index.values[-1], partitions)
    #else:
    #    slices = np.concatenate([[np.array(E_OUT.index.values[0])], np.linspace(
    #        E_OUT.index.values[0]+elm_lag, E_OUT.index.values[-1], partitions-1)])

    pval_0 = np.polyfit(E_OUT.index.values, E_OUT.values, fitting_order_Energy)
    
    print('INDEX: {}'.format(E_OUT.index.values))
    print('BREAK: {}'.format(E_OUT.index.values[0]+elm_lag))
    ind = np.argmin(abs(E_OUT.index.values[0]+elm_lag - E_OUT.index.values))
    print(ind)
    pval_E0 = np.polyfit(E_ELM.index.values[:ind], np.mean(E_ELM.values[:ind,:], axis=1), fitting_order_Energy)
    pval_E1 = np.polyfit(E_ELM.index.values[ind:], np.mean(E_ELM.values[ind:,:], axis=1), fitting_order_Energy)

    Y0 = np.polyval(pval_0, np.linspace(E_OUT.index.values[0], E_OUT.index.values[0]+elm_lag, interp)) - np.polyval(
        pval_E0, np.linspace(E_OUT.index.values[0], E_OUT.index.values[0]+elm_lag, interp))
    Y1 = np.polyval(pval_0, np.linspace(E_OUT.index.values[0]+elm_lag, E_OUT.index.values[-1], interp)) - np.polyval(
        pval_E1, np.linspace(E_OUT.index.values[0]+elm_lag,E_OUT.index.values[-1],interp))

    def ESOL_CALC(X):
        _Y = np.zeros(np.shape(X))
        # np.polyval(np.polyfit(np.linspace(E_OUT.index.values[0],E_OUT.index.values[ind], interp), Y0, order), X[X < E_OUT.index.values[0]+elm_lag])
        _Y[X < E_OUT.index.values[0]+elm_lag] = 0
        _Y[X > E_OUT.index.values[0]+elm_lag] = np.polyval(np.polyfit(np.linspace(E_OUT.index.values[ind],E_OUT.index.values[-1], interp), Y1, order), X[[X > E_OUT.index.values[0]+elm_lag]])
        return _Y

    #
    P_SOL = savgol_filter(ESOL_CALC(E_OUT.index.values), window_length=smoothing_window,
                      polyorder=order, deriv=1, axis=0, delta=finite_difference_step)

    
    for i in range(partitions-1):

        ind_O = np.logical_and(E_OUT.index.values >
                               slices[i], E_OUT.index.values < slices[i+1])
        ind_E = np.logical_and(E_ELM.index.values >
                               slices[i], E_ELM.index.values < slices[i+1])

        pval_0 = np.polyfit(
            E_OUT.index.values[ind_O], E_OUT.values[ind_O], smoothing_order)
        if ELM_MEAN:
            pval_E = np.polyfit(E_ELM.index.values[ind_E], np.mean(
                E_ELM.values[ind_E, :], axis=1), smoothing_order)
        else:
            pval_E = np.polyfit(
                E_ELM.index.values[ind_E], E_ELM.values[ind_E], smoothing_order)
        Y = np.polyval(pval_0, np.linspace(slices[i], slices[i+1], interp)) - np.polyval(
            pval_E, np.linspace(slices[i], slices[i+1], interp))

        def ESOL_CALC(X):
            return np.polyval(np.polyfit(np.linspace(slices[i], slices[i+1], interp), Y, order), X)

        if i == 0:
            ts = np.linspace(slices[i], slices[i+1], interp)
            P_SOL = PPF.derivative(ESOL_CALC, np.linspace(
                slices[i], slices[i+1], interp), h=res)
        else:
            ts = np.concatenate(
                [ts, np.linspace(slices[i], slices[i+1], interp)])
            P_SOL = np.concatenate([P_SOL, PPF.derivative(
                ESOL_CALC, np.linspace(slices[i], slices[i+1], interp), h=40)])
    return E_OUT.index.values, P_SOL
"""

def calculate_psol_1D(E_OUT, elm_lag=[], partitions=10, interp=100, order=3, smoothing_order=2, res=2, ELM_MEAN=True):

    if not elm_lag:
        slices = np.linspace(
            E_OUT.index.values[0], E_OUT.index.values[-1], partitions)
    else:
        slices = np.concatenate([[np.array(E_OUT.index.values[0])], np.linspace(
            E_OUT.index.values[0]+elm_lag, E_OUT.index.values[-1], partitions-1)])

    for i in range(partitions-1):

        ind_O = np.logical_and(E_OUT.index.values >
                               slices[i], E_OUT.index.values < slices[i+1])

        pval_0 = np.polyfit(
            E_OUT.index.values[ind_O], E_OUT.values[ind_O], smoothing_order)

        Y = np.polyval(pval_0, np.linspace(slices[i], slices[i+1], interp))

        def ESOL_CALC(X):
            return np.polyval(np.polyfit(np.linspace(slices[i], slices[i+1], interp), Y, order), X)

        if i == 0:
            ts = np.linspace(slices[i], slices[i+1], interp)
            P_SOL = PPF.derivative(ESOL_CALC, np.linspace(
                slices[i], slices[i+1], interp), h=res)
        else:
            ts = np.concatenate(
                [ts, np.linspace(slices[i], slices[i+1], interp)])
            P_SOL = np.concatenate([P_SOL, PPF.derivative(
                ESOL_CALC, np.linspace(slices[i], slices[i+1], interp), h=40)])
    return ts[1:], P_SOL[1:]


def UQ_advanced(power_pd, ERROR=.05, window_length=2001, polyorder=3):
    """
    Error[0] =  Error on Input
    Error[1] =  Error on P_RAD
    """
    inputs = list(np.array(list(power_pd.columns.values))[
                  [x in ['P_ohm', 'P_ich', 'P_nbi'] for x in power_pd.columns.values]])
    outputs = list(np.array(list(power_pd.columns.values))[
                   [x in ['dWdt', 'P_rad'] for x in power_pd.columns.values]])

    inp = {}
    for i, label in enumerate(inputs):
        inp[label] = savgol_filter(
            power_pd[label], window_length=window_length, polyorder=3)
        inp[label+'_std'] = power_pd[label].rolling(
            window_length).std().fillna(method='bfill').values
    labels = [item for sublist in [[x, x+'_std']
                                   for x in inputs] for item in sublist]
    input_pd = pd.DataFrame(inp, columns=labels)
    input_pd.index = power_pd.index

    oup = {}
    for i, label in enumerate(outputs):
        oup[label] = savgol_filter(
            power_pd[label], window_length=window_length, polyorder=3)
        oup[label+'_std'] = power_pd[label].rolling(
            window_length).std().fillna(method='bfill').values
    labels = [item for sublist in [[x, x+'_std']
                                   for x in outputs] for item in sublist]
    output_pd = pd.DataFrame(oup, columns=labels)
    output_pd.index = power_pd.index

    print('calculating input uncertainties: assuming independence')
    IN_mu = input_pd[inputs].sum(axis=1)
    IN_std = ((input_pd[[x+'_std' for x in inputs]
                        ].pow(2, axis=1)).sum(axis=1)).pow(0.5)
    print('calculating output uncertainties: assuming independence')
    OUT_mu = output_pd[outputs].sum(axis=1)
    OUT_std = ((output_pd[[x+'_std' for x in outputs]
                          ].pow(2, axis=1)).sum(axis=1)).pow(0.5)

    P_IN_UQ = {}
    P_IN_UQ['P_IN'] = IN_mu
    P_IN_UQ['P_IN_LB'] = IN_mu * (1-ERROR)
    P_IN_UQ['P_IN_UB'] = IN_mu * (1+ERROR)
    P_IN_UQ['P_IN_STD'] = IN_std
    P_IN_UQ['P_IN_STD_LB'] = (IN_mu*(1-ERROR)) - IN_std
    P_IN_UQ['P_IN_STD_UB'] = (IN_mu*(1+ERROR)) + IN_std
    p_in = pd.DataFrame(P_IN_UQ, index=power_pd.index.values, columns=[
                        'P_IN', 'P_IN_LB', 'P_IN_UB', 'P_IN_STD', 'P_IN_STD_LB', 'P_IN_STD_UB'])
    p_in.index = p_in.index.values.astype('float')/1E9

    P_OUT_UQ = {}
    P_OUT_UQ['P_OUT'] = OUT_mu
    P_OUT_UQ['P_OUT_LB'] = OUT_mu * (1-ERROR)
    P_OUT_UQ['P_OUT_UB'] = OUT_mu * (1+ERROR)
    P_OUT_UQ['P_OUT_STD'] = OUT_std
    P_OUT_UQ['P_OUT_STD_LB'] = (OUT_mu*(1-ERROR)) - OUT_std
    P_OUT_UQ['P_OUT_STD_UB'] = (OUT_mu*(1+ERROR)) + OUT_std
    p_out = pd.DataFrame(P_OUT_UQ, index=power_pd.index.values, columns=[
                         'P_OUT', 'P_OUT_LB', 'P_OUT_UB', 'P_OUT_STD', 'P_OUT_STD_LB', 'P_OUT_STD_UB'])
    p_out.index = p_out.index.values.astype('float')/1E9
    return p_in, p_out

def plot_model_confidence(Y_train,Y_test,alpha_level,xlim=[]):
    mpl.rcParams.update({'font.size': 12})
    P_E_sim, X_E_sim = up.ecdf(Y_test)
    P_E_reg, X_E_reg = up.ecdf(Y_train)#.values)
    """
    PPF.plot_mixed_gaussian_model(self.X_test, self.X_train)
    if not hasattr(self,'ELM_pd_clean'):
        self.clean_ELM_pd()
    plot_elm_model_fit(self.ELM_pd_clean)
    """
    fig, axs = plt.subplots(1, figsize=(8, 8), sharex=True)
    fig.subplots_adjust(wspace=0, hspace=0)


    L, R, x_i = up.confidence_limits_distribution(Y_train, alpha_level, plot=False)
    axs.step(x_i/1E6, L, c='green', alpha=0.4,label='{:.2f} KS-conf LB'.format(alpha_level))
    axs.step(x_i/1E6, R, c='orange',alpha=0.4, label='{:.2f} KS-conf RB'.format(alpha_level))
    axs.fill_between(x_i/1E6,L,R,color='red',step="pre",alpha=0.09)
    if not xlim:
        xlim = [np.min(x_i/1E6), np.max(x_i/1E6)]

    axs.set_xlim(xlim)
    
    axs.step(X_E_reg/1E6, P_E_reg, c='red', linewidth=2, label='ELM Energy')
    axs.step(X_E_sim/1E6, P_E_sim, c='C0', linewidth=2,label='Synthetic ELM Energy')
    axs.set_xlabel('Energy (MJ)')
    axs.set_ylabel('P(x)')
    axs.legend()
    plt.show()

def plot_elm_model_fit(ELMS_out):

    fig, axs = plt.subplots(2, 2, figsize=(17, 12))
    fig.subplots_adjust(wspace=.2, hspace=.2)
    axs[0, 0].errorbar(ELMS_out['model_Energy'], ELMS_out['IR_Energy'],
                       yerr=None, xerr=ELMS_out['Prof_rmse']/3E2, ls='', zorder=0)
    axs[0, 0].scatter(ELMS_out['model_Energy'],
                      ELMS_out['IR_Energy'], linewidth=3, marker='+', color='red')
    #axs[0, 0].set_xlim([ELMS_out['model_Energy'].min()-1000, ELMS_out['model_Energy'].max()+1000])
    #axs[0, 0].set_ylim([ELMS_out['IR_Energy'].min()-1000,
    #                    ELMS_out['IR_Energy'].max()+1000])
    axs[0, 0].set_aspect('equal', 'box')
    axs[0, 0].set_ylabel('Energy IR (J)')
    axs[0, 0].set_xlabel('Energy Model (J)')
    axs[0, 0].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    axs[0, 1].hist(ELMS_out['IR_Energy']-ELMS_out['model_Energy'], 80)
    axs[0, 1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axs[0, 1].set_xlabel('Energy residuals (J)')
    #axs[0, 1].set_xlim([-.9E5, 0.3E5])
    #plt.figure()
    #axs[2,0].errorbar(markersize=10)
    axs[1, 0].errorbar(ELMS_out['model_Qmax'], ELMS_out['IR_max_q'],
                       yerr=None, xerr=ELMS_out['error_qmax'], ls='', zorder=0)
    axs[1, 0].scatter(ELMS_out['model_Qmax'], ELMS_out['IR_max_q'],
                      linewidth=3, marker='+', color='red')
    axs[1, 0].set_aspect('equal', 'box')
    axs[1, 0].set_ylabel('Qmax IR (W)')
    axs[1, 0].set_xlabel('Qmax Model (W)')
    axs[1, 0].set_xlim([ELMS_out['model_Qmax'].min()-5000, ELMS_out['model_Qmax'].max()+5000])
    axs[1, 0].set_ylim([ELMS_out['IR_max_q'].min()-5000,
                        ELMS_out['IR_max_q'].max()+5000])
    axs[1, 1].hist(ELMS_out['IR_max_q']-ELMS_out['model_Qmax'], 70)
    axs[1, 1].set_xlabel('Qmax residuals (W)')
    #axs[1, 1].set_xlim([-2E7, 4.5E7])
    plt.show()


"""def func(seed):
            np.random.seed(seed)
            N_elm = int((times[-1]-times[0])*freq(1))
            print('N_elms in syntetic time series: {}'.format(N_elm))
            time_ELM = np.linspace(elm_lag, times.max(), N_elm)
            theta = P_model(N_elm)
            #theta = theta[np.random.permutation(N_elm), :]
            Energy = []
            x_r = np.linspace(2.8041, 2.9868, 50)

            Q_bdry_ELM = np.zeros([len(x_r), len(times)])
            ELM_power = np.zeros(len(times))
            Energy = 0
            for k, j in enumerate(0-np.array(range(N_elm))):
                x = theta[k, :]
                e_elm, q_elm, q_int = evaluate_elm_model(x)

                t_adj = times[np.argmin(abs(times-time_ELM[k]))]
                elms_times = q_int.index.values + t_adj
                ind_ELM = np.logical_and(
                    times > elms_times.min(), times < elms_times.max())
                fit = np.interp(times[ind_ELM], elms_times, q_int['IntQ'])
                ELM_power[ind_ELM] = fit
                Energy = Energy+e_elm

                #if Q_bdry:
                t_tmp = t_adj+q_elm.index.values
                # np.argmin(abs(times - time_ELM[j]))
                ind_ELM = np.logical_and(t_tmp[0] < times, t_tmp[-1] > times)
                Q_E = np.zeros([len(x_r), len(times[ind_ELM])])
                for i, t in enumerate(times[ind_ELM]):
                    Q_E[:, i] = np.interp(x_r, q_elm.columns.values, q_elm.iloc[np.argmin(
                        abs(t_tmp-t)), :])  # fit(times[ind_ELM])
                Q_bdry_ELM[:, ind_ELM] = Q_E

            return ELM_power, Energy, Q_bdry_ELM
"""

