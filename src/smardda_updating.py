#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function wraps the required updating functions.
Updating4Smardda
"""
import pandas as pd
__author__ = "Dominic Calleja"
__copyright__ = "Copyright 2019"
__credits__ = "Ander Gray"
__license__ = "MIT"
__version__ = "0.1"
__date__ = '01/04/2019'
__maintainer__ = "Dominic Calleja"
__email__ = "d.calleja@liverpool.ac.uk"
__status__ = "Draft"

print('+===================================+')
print('|         Updating4Smardda          |')
print('|Bayesian updating tools for Smardda|')
print('| Credit: '+__credits__+' adapted from |')
print('| previous implimentation of TMCMC  |')
print('|          Version: '+__version__ + 12*' '+' |')
print('|                                   |')
print('| '+__copyright__+' (c) ' + __author__+'|')
print('+====================================+')
print('new break condition implimented ')

import random
import numpy as np
from scipy.linalg import eigh, cholesky
from scipy import stats
from scipy import optimize
from scipy import signal
import time
import pickle
import os
import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import GPy
import ray
import sys

import heatequation as ht

#@ray.remote

class smardda_history():
    def __init__(self, seq_ind, save_address):
        print('initialising history')
        self.seq_ind = seq_ind
        self.save_address = save_address
        if seq_ind > 0:
            self._load_results()
        else:
            self.results_sequence = {}

    def pass_intermediary_results(self, Th, Like, Th_j, Lp_j):
        self.Th   = Th
        self.Like = Like
        self.Th_j = Th_j
        self.Lp_j = Lp_j
        self.intermediary_results = {'posterior': self.Th,
                                     'likelihood': self.Like,
                                     'intermediary_TH': self.Th_j,
                                     'intermediary_Lp': self.Lp_j,
                                     'seq_ind': self.seq_ind}

        self.results_sequence[self.seq_ind] = self.intermediary_results
        self._save_results()

    def return_intermediary_results(self):
        return self.results_sequence

    def _save_results(self):
        with open(self.save_address, 'wb') as handle:
            pickle.dump(self.results_sequence, handle, protocol=4)
        print('Stored {} results'.format(self.seq_ind))

    def _load_results(self):
        with open(self.save_address, 'rb') as handle:
            self.results_sequence = pickle.load(handle)
        print('Loaded {} results'.format(self.seq_ind))

    def _store_results(self):
        if not hasattr(self, 'intermediary_results'):
            self.intermediary_results = {'posterior': self.Th,
                                         'likelihood': self.Like,
                                         'seq_ind': self.seq_ind}
        self.results_sequence[self.seq_ind] = self.intermediary_results
        return print('Stored {} results'.format(self.seq_ind))

class smardda_model(smardda_history):
    def __init__(self, seq_ind, save_address, number_of_model_evaluation=1, model_radial_resolution=50, pulse_slices=7, simulation_time_steps=220, resolution=1):
        self.seq_ind = seq_ind
        print('seq_ind in smardda_model : {}'.format(self.seq_ind))
        super().__init__(seq_ind, save_address)

        self.number_of_model_evaluation = number_of_model_evaluation
        self.model_radial_resolution = model_radial_resolution
        self.pulse_slices = pulse_slices
        self.simulation_time_steps = simulation_time_steps
        self.evaluation_points_per_slice = resolution

        self.target_data_points = 80

        print('seq ind : {}'.format(self.seq_ind))
        print('save address : {}'.format(self.save_address))
        s1 = 'N model evaluations : {}'.format(self.number_of_model_evaluation )
        s2 = 'Resolution x Time : {} x {}'.format(self.model_radial_resolution,self.simulation_time_steps)
        s3 = 'Evaluation points per slice : {}'.format(self.evaluation_points_per_slice)
        s4 = 'Time series partitions : {}'.format(self.pulse_slices)
        self._model_string = s1+'\n'+s2+'\n'+s3+'\n'+s4
        print(self._model_string+'\n')


    def load_timeseries_from_pulse(self, pickle_path):
        inputs = ['Target_T', 'Times', 'Psol_pd',
                  'radial_locations', 'Target_residuals']
        print('The pickle class requires: {}'.format(inputs))

        print('loading pickle ...')
        with open(pickle_path, 'rb') as handle:
            self.execution_data = pickle.load(handle)

        print('Formatting pulse data ...')
        temperature = self.execution_data['Target_T']
        self.times = np.linspace(
            0, temperature['Times'].max(), self.simulation_time_steps)
        self.residuals = self.execution_data['Target_residuals']

        self.Psol = np.interp(
            self.times, self.execution_data['Psol_pd']['Times'], self.execution_data['Psol_pd']['PSol_smooth'])

        "simulation parameters"
        self.radial_locations = np.linspace((self.execution_data['radial_locations']).min(
        ), (self.execution_data['radial_locations']).max(), self.model_radial_resolution)
        self.slices = np.linspace(
            0, self.simulation_time_steps, num=self.pulse_slices, dtype=int)
        print('load_timestep_seq_ind : {}'.format(self.seq_ind))
        print('slices : {}'.format(self.slices))
        print('evaluation slice : {}'.format(self.slices[self.seq_ind]))

        self.slices = np.concatenate([[0], np.array(
            (self.slices[1:] + self.slices[:-1]) / 2).astype(int), [self.simulation_time_steps-1]])

        "Target Data"
        self.T_target_mu = noise(
            self.slices, self.times, temperature, self.residuals, self.target_data_points)
        print('shape T target : {}'.format(np.shape(self.T_target_mu)))
        print('T_target_mu : {}'.format(self.T_target_mu))
        #np.interp(
        #    self.times[self.slices], temperature['Times'], temperature['Tmax'])

        if self.evaluation_points_per_slice >1:
            self.subs = [np.linspace(self.slices[i],self.slices[i+1],self.evaluation_points_per_slice,dtype=int) for i in range(self.pulse_slices)]
        else:
            self.subs = [self.slices[i+1] for i in range(self.pulse_slices)]

        self._sub_t_mu = [
            np.interp(self.times[self.subs[i]], temperature['Times'], temperature['Tmax']) for i in range(self.pulse_slices)]
        print('Complete.')
        print('+'+'='*77+'+ \n\n')

    def _posterior_history(self, Th_posterior, L_posterior, n_histories, seq_ind, replace=True, sample=True):
        if sample:
            weight = np.abs(1/L_posterior)/np.sum(np.abs(1/L_posterior))
            self.history_idx = np.random.choice(
                np.shape(L_posterior)[0], n_histories, replace=replace, p=weight)
        else:
            srix = np.argsort(L_posterior)
            self.history_idx = srix[range(n_histories)]
        return self.history_idx

    def _sample_series_history(self, n_samples, mean=False):
        lambda_history = np.zeros(
            [self.number_of_model_evaluation, len(self.times[range(self.slices[self.seq_ind])])])
        sigma_history = np.zeros(
            [self.number_of_model_evaluation, len(self.times[range(self.slices[self.seq_ind])])])

        for i in range(self.seq_ind):
            TH = self.results_sequence[i]['posterior']
            L = self.results_sequence[i]['likelihood']
            self._posterior_history(TH, L, n_samples, i)
            idx = np.random.choice(np.shape(self.history_idx)[
                                   0], 1, replace=True)
            #print('test_resample')
                                   #can i change?
            #l_h, s_h = probabilistic_model(
            #    TH[:, idx], self.number_of_model_evaluation, self.evaluation_points_per_slice)
            # probabilistic_model_t(TH[:, idx], self.number_of_model_evaluation)
            l_h, s_h = TH[:, idx]

            l_q, s = construct_history(
                l_h, s_h, self.number_of_model_evaluation, self.times, i, self.slices, self.evaluation_points_per_slice)
            lambda_history[:, range(self.slices[i], self.slices[i+1])] = l_q
            sigma_history[:, range(self.slices[i], self.slices[i+1])] = s

        if mean:
            # model so you can just pick the mean of the time series realisations to reduce compute sampling
            #either mean of the whole set (probs), or mean for each sample maybe
            print('not implimented')
        self.lambda_history = lambda_history
        self.sigma_history = sigma_history
        return self.lambda_history, self.sigma_history

    def _model(self, theta, resample_history=30):
        if self.seq_ind > 0:
            print('re-samples of posterior currently set to {}, better coverage \n with increase, better convergence with lower'.format(resample_history))
            lambda_history, sigma_history = self._sample_series_history(resample_history)
            #inp, kni, impute = up.q_slice(self.times, self.seq_ind, self.slices, res)
        else:
            lambda_history = []
            sigma_history = []
            #updating_assumed_density
        return updating_assumed_density_t(np.reshape(theta, [len(theta), 1]), self.number_of_model_evaluation, self.seq_ind, self.slices, self.evaluation_points_per_slice,
                                           self.times, self.Psol, self.execution_data, self.radial_locations, lq_0=5, sig_0=3, lq_hist=lambda_history,
                                           sig_hist=sigma_history)




class smardda_model_swept(smardda_history):
    def __init__(self, seq_ind, save_address, number_of_model_evaluation=1, model_radial_resolution=40, pulse_slices=7, simulation_time_steps=200, resolution=1):
        self.seq_ind = seq_ind
        print('seq_ind in smardda_model : {}'.format(self.seq_ind))
        super().__init__(seq_ind, save_address)

        self.number_of_model_evaluation = number_of_model_evaluation
        self.model_radial_resolution = model_radial_resolution
        self.pulse_slices = pulse_slices
        self.simulation_time_steps = simulation_time_steps
        self.evaluation_points_per_slice = resolution

        self.target_data_points = 60

        print('seq ind : {}'.format(self.seq_ind))
        print('save address : {}'.format(self.save_address))
        s1 = 'N model evaluations : {}'.format(self.number_of_model_evaluation )
        s2 = 'Resolution x Time : {} x {}'.format(self.model_radial_resolution,self.simulation_time_steps)
        s3 = 'Evaluation points per slice : {}'.format(self.evaluation_points_per_slice)
        s4 = 'Time series partitions : {}'.format(self.pulse_slices)
        self._model_string = s1+'\n'+s2+'\n'+s3+'\n'+s4
        print(self._model_string+'\n')


    def load_timeseries_from_pulse(self, pickle_path):
        inputs = ['Target_T', 'Times', 'Psol_pd',
                  'radial_locations', 'Target_residuals']
        print('The pickle class requires: {}'.format(inputs))

        print('loading pickle ...')
        with open(pickle_path, 'rb') as handle:
            self.execution_data = pickle.load(handle)

        print('Formatting pulse data ...')
        temperature = self.execution_data['Target_T']
        self.times = np.linspace(
            0, temperature['Times'].max(), self.simulation_time_steps)
        self.residuals = self.execution_data['Target_residuals']

        self.Psol = np.interp(
            self.times, self.execution_data['Psol_pd']['Times'], self.execution_data['Psol_pd']['PSol_smooth'])

        "simulation parameters"
        self.radial_locations = np.linspace((self.execution_data['radial_locations']).min(
        ), (self.execution_data['radial_locations']).max(), self.model_radial_resolution)
        self.slices = np.linspace(
            0, self.simulation_time_steps, num=self.pulse_slices, dtype=int)
        print('load_timestep_seq_ind : {}'.format(self.seq_ind))
        print('slices : {}'.format(self.slices))
        print('evaluation slice : {}'.format(self.slices[self.seq_ind]))

        #self.slices = np.concatenate([[0], np.array(
         #   (self.slices[1:] + self.slices[:-1]) / 2).astype(int), [self.simulation_time_steps-1]])

        "Target Data"
        self.T_target_mu = noise(
            self.slices, self.times, temperature, self.residuals, self.target_data_points)
        print('shape T target : {}'.format(np.shape(self.T_target_mu)))
        print('T_target_mu : {}'.format(self.T_target_mu))
        #np.interp(
        #    self.times[self.slices], temperature['Times'], temperature['Tmax'])

        if self.evaluation_points_per_slice >1:
            self.subs = [np.linspace(self.slices[i],self.slices[i+1],self.evaluation_points_per_slice,dtype=int) for i in range(self.pulse_slices)]
        else:
            self.subs = [self.slices[i+1] for i in range(self.pulse_slices-1)]

        #self._sub_t_mu = [
        #    np.interp(self.times[self.subs[i]], temperature['Times'], temperature['Tmax']) for i in range(self.pulse_slices)]
        print('Complete.')
        print('+'+'='*77+'+ \n\n')

    def _posterior_history(self, Th_posterior, L_posterior, n_histories, seq_ind, replace=True, sample=True):
        if sample:
            weight = np.abs(1/L_posterior)/np.sum(np.abs(1/L_posterior))
            self.history_idx = np.random.choice(
                np.shape(L_posterior)[0], n_histories, replace=replace, p=weight)
        else:
            srix = np.argsort(L_posterior)
            self.history_idx = srix[range(n_histories)]
        return self.history_idx

    def _sample_series_history(self, n_samples, mean=False):
        #lambda_history = np.zeros(
        #    [self.number_of_model_evaluation, len(self.times[range(self.slices[self.seq_ind])])])
        #sigma_history = np.zeros(
        #    [self.number_of_model_evaluation, len(self.times[range(self.slices[self.seq_ind])])])

        lambda_history =[]
        sigma_history = []
        for i in range(self.seq_ind):
            TH = self.results_sequence[i]['posterior']
            L = self.results_sequence[i]['likelihood']
            self._posterior_history(TH, L, n_samples, i)
            idx = np.random.choice(np.shape(self.history_idx)[
                                   0], n_samples, replace=True)
            #print('test_resample')
                                   #can i change?
            #l_h, s_h = probabilistic_model(
            #    TH[:, idx], self.number_of_model_evaluation, self.evaluation_points_per_slice)
            # probabilistic_model_t(TH[:, idx], self.number_of_model_evaluation)
            t_tmp = TH[:, idx]
            l_h = t_tmp[:3]
            s_h = t_tmp[3:]

            #l_q, s = construct_history(
            #    l_h, s_h, self.number_of_model_evaluation, self.times, i, self.slices, self.evaluation_points_per_slice)
            lambda_history.append(l_h)#[:, range(self.slices[i], self.slices[i+1])] = l_q
            sigma_history.append(s_h)#[:, range(self.slices[i], self.slices[i+1])] = s

        if mean:
            # model so you can just pick the mean of the time series realisations to reduce compute sampling
            #either mean of the whole set (probs), or mean for each sample maybe
            print('not implimented')
        self.lambda_history = lambda_history
        self.sigma_history = sigma_history
        return self.lambda_history, self.sigma_history

    def _model(self, theta, resample_history=1):
        if self.seq_ind > 0:
            #print('re-samples of posterior currently set to {}, better coverage \n with increase, better convergence with lower'.format(resample_history))
            lambda_history, sigma_history = self._sample_series_history(resample_history)
            #inp, kni, impute = up.q_slice(self.times, self.seq_ind, self.slices, res)
        else:
            lambda_history = []
            sigma_history = []
            #updating_assumed_density
        return update_assumed_density_s(np.reshape(theta, [len(theta), 1]), self.number_of_model_evaluation, self.seq_ind, self.slices, self.evaluation_points_per_slice,
                                        self.times, self.Psol, self.execution_data['GP'], self.radial_locations, self.execution_data['T_init'], lq_0=5, sig_0=3, lq_hist=lambda_history, sig_hist=sigma_history)

# updating_assumed_density_t(np.reshape(theta, [len(theta), 1]), self.number_of_model_evaluation, self.seq_ind, self.slices, self.evaluation_points_per_slice,
#                                    self.times, self.Psol, self.execution_data, self.radial_locations, lq_0=5, sig_0=3, lq_hist=lambda_history,
#                                    sig_hist=sigma_history)


class smardda_tmcmc_temperature(smardda_model_swept):
    def __init__(self, i, prior, prior_sample, likelihood, n_per_layer, n_batches, n_perameters, pickle_path, sequence_ind, save_address,extra_string='skylake'):
        self.seq_ind = sequence_ind
        self.sequence_ind = sequence_ind
        print('seq_ind in TMCMC : {}'.format(sequence_ind))
        super().__init__(sequence_ind, save_address)
        self.prior = prior
        self.prior_sample = prior_sample
        self.n_per_layer = n_per_layer
        self.n_batches = n_batches
        self.n_perameters = n_perameters

        self.epsilon = 0.4

        self._worker_ind = i
        self._worker_string = 'Worker_sf_{}_id_{}-'.format(self._worker_ind,np.random.randint(0,100,1)[0])
        self.logfile = '{}logfile{}.txt'.format(self._worker_string,extra_string)

        copyR(self.logfile)
        outputf=open(self.logfile, 'a')
        outputf.write('Model parameters : \n')
        outputf.write('{}\n'.format(self._model_string))
        outputf.write('+'+'='*77+'+ \n')
        outputf.write('{} Loading data from pickle...\n'.format(self._worker_string))

        start = time.time()
        self.load_timeseries_from_pulse(pickle_path)
        outputf.write('{} data loading complete. Time Elapsed {}\n'.format(self._worker_string, timestamp_format(start, time.time())))
        self.likelihood = {}
        self.evaluation_times = {}
        outputf.close()

    def _likelihood(self, theta):
        ll, m = euclid_mu(theta, self._model,
                          self.T_target_mu[self.sequence_ind+1], self.epsilon, self.slices[self.sequence_ind+1])
        return ll

    def _batch_likelihood(self, theta):
        outputf=open(self.logfile, 'a')
        outputf.write('{} Begin likelihood evaluation ...\n'.format(self._worker_string))
        outputf.close()
        self._t_like_start = time.time()
        #change for Psol too!
        len_theta = 2 #4*self.evaluation_points_per_slice + n_p_cor(self.evaluation_points_per_slice)

        dim = np.asarray(np.shape(theta)) != len_theta
        batch_len = np.shape(theta)[np.where(dim)[0][0]]

        likelihood = []
        model_output = []
        evaluation_times = []
        for i in range(batch_len):
            outputf = open(self.logfile, 'a')
            outputf.write('+'+'-'*77+'+ \n')
            print('{} evaluating likelihood {} of {} '.format(self._worker_string, i+1, batch_len))
            outputf.write('{} evaluating likelihood {} of {} \n'.format(self._worker_string, i+1, batch_len))

            t = time.time()
            ll, m = euclid_mu(theta[:, i], self._model, self.T_target_mu[self.sequence_ind+1],
                              self.epsilon, self.slices[self.sequence_ind+1])
            print('Target ind : {}'.format(self.slices[self.sequence_ind+1]))
            #print('Model Output : \n {}'.format(['sim {} : {}'.format(
            #    i, m) in range(self.number_of_model_evaluation)]))
            print('Model Tmu : {} (degC)'.format(m))
            print('Data Tmu : {} (degC)'.format(
                self.T_target_mu[self.sequence_ind+1]))
            self._string5 = 'Target ind: {}\npulse slice: {} \npulse time: {} (sec) \nTHJ: {} \nModel Tmu: {}(degC) \nData Tmu: {}(degC)\n'.format(
                self.sequence_ind, self.slices[self.sequence_ind+1], self.times[self.slices[self.sequence_ind+1]], theta[:, i], m, self.T_target_mu[self.sequence_ind+1])
            likelihood.append(ll)
            model_output.append(m)
            evaluation_times.append(time.time()-t)
            outputf.write(self._string5)
            outputf.close()

        self.likelihood['worker_{}'.format(
            self._worker_ind)] = np.array(likelihood)
        # don't save model output in a dict because of memory issues.
        self.model_output = model_output
        self.evaluation_times['worker_{}'.format(
            self._worker_ind)] = np.array(evaluation_times)

        self._t4 = time.time()
        self._string4 = '{} Completed likelihood evaluation : Time Elapsed {}\n'.format(
            self._worker_string, timestamp_format(self._t_like_start,self._t4))


        print(self._string4)
        print('+'+'='*77+'+ \n')
        outputf = open(self.logfile, 'a')
        outputf.write(self._string4)
        outputf.write('+'+'='*77+'+ \n')
        outputf.close()
        return likelihood

    def markov_chain(self, SIGMA_j, THJ, LPJ, a_j1, length, burnin):
        outputf=open(self.logfile, 'a')
        batch_len = len(LPJ)
        print('{} Initialising Markov Chains ...'.format(self._worker_string))
        print('{} Evaluating {} chains ...'.format(self._worker_string, batch_len))
        outputf.write('{} Initialising Markov Chains ...\n'.format(self._worker_string))
        outputf.write('{} Evaluating {} chains ...\n'.format(self._worker_string, batch_len))
        outputf.close()

        Theta_new =[]
        Likelihood_new = []
        accept_proportion =[]
        t = time.time()

        for i in range(batch_len):
            outputf=open(self.logfile, 'a')
            #print('{} evaluating chain {} of {}'.format(self._worker_string, i+1, batch_len))
            outputf.write('+'+'-'*77+'+ \n')
            outputf.write('{} evaluating chain {} of {}\n'.format(self._worker_string, i+1, batch_len))
            #print('THJ :{}'.format(THJ[:, i]))
            Th, L, a = parallel_markov_chain(
                self.prior, self._likelihood, SIGMA_j, THJ[:, i], LPJ[i], a_j1, length[i], burnin=burnin)
            Theta_new.append(Th)
            Likelihood_new.append(L)
            accept_proportion.append(a)
            self._string6 = 'seed THJ :{} \nupdate THJ :{} \nLikelihod :{}\n'.format(
                THJ[:, i], Th, L)
            outputf.write(self._string6)
            outputf.close()

        #Theta_new = np.concatenate(Theta_new,axis=0)
        #print(Theta_new)
        #Likelihood_new = np.array(Likelihood_new)
        #accept_proportion = np.array(accept_proportion)
        ap = np.array(accept_proportion)
        print('{} Completed MC evaluation : Time Elapsed {}'.format(self._worker_string, timestamp_format(t, time.time())))
        print('{} mean/min/max accpetance rate : {:.3f}/{:.3f}/{:.3f}'.format(
            self._worker_string,np.mean(ap),np.max(ap),np.min(ap)))
        outputf=open(self.logfile, 'a')
        outputf.write('{} Completed MC evaluation : Time Elapsed {} \n'.format(
            self._worker_string, timestamp_format(t, time.time())))
        outputf.write('{} mean/min/max accpetance rate : {:.3f}/{:.3f}/{:.3f}\n'.format(
            self._worker_string, np.mean(ap), np.max(ap), np.min(ap)))
        outputf.write('+'+'='*77+'+ \n')
        outputf.close()
        print('+'+'='*77+'+ \n')
        return Theta_new, np.asarray(Likelihood_new), accept_proportion



class smardda_inter_elm():
    def __init__(self, prior, prior_sample, likelihood, n_per_layer, n_batches, n_perameters, pickle_path):

        #with open(pickle_path, 'rb') as handle:
        #    GP_data = pickle.load(handle)

        #self.GP = GP_data['GP']
        #self.data_table = data_table
        #self.data = data
        self.likelihood = likelihood
        self.prior = prior
        self.prior_sample = prior_sample
        self.n_perameters = n_perameters

    def _likelihood(self, theta):
        return self.likelihood(theta)

    def _batch_likelihood(self, theta):
        print(theta)
        len_theta = self.n_perameters
        dim = np.asarray(np.shape(theta)) != len_theta
        batch_len = np.shape(theta)[np.where(dim)[0][0]]
        logL = []
        for i in range(batch_len):
            logL.append(self.likelihood(theta[:, i]))
        return logL

    def markov_chain(self, SIGMA_j, THJ, LPJ, a_j1, length, burnin):
        #outputf = open(self.logfile, 'a')
        batch_len = len(LPJ)

        Theta_new = []
        Likelihood_new = []
        accept_proportion = []
        t = time.time()
        for i in range(batch_len):

            Th, L, a = parallel_markov_chain(
                self.prior, self._likelihood, SIGMA_j, THJ[:, i], LPJ[i], a_j1, length[i], burnin=burnin)
            Theta_new.append(Th)
            Likelihood_new.append(L)
            accept_proportion.append(a)
        ap = np.array(accept_proportion)
        return Theta_new, np.asarray(Likelihood_new), accept_proportion

@ray.remote
class smardda_tmcmc(smardda_inter_elm): # smardda_tmcmc_temperature
    def __init__(self, i, fT, sample_fT, log_fD_T, n_per_layer, Ncores, n_parameters, pickle_path, seq_ind, save_address):
        self.i = i
        super().__init__(fT, sample_fT, log_fD_T,n_per_layer, Ncores, n_parameters, pickle_path)

def euclid_mu(theta, model, data, epsilon_r, slice1):  # theta_i,
    #temp
    w_model = model(theta)
    #d_mu = np.array(data)
    #m_mu = np.mean(w_model, axis=0)
    #res = np.linalg.norm(d_mu-m_mu[sub_ind])
    #sigma = np.std(m_mu)
    # gauss  -(0.5/epsilon_r**2) * np.sum(res**2)#

    #Area = area_metric_robust(w_model[:,slice1], data)
    #log = -(Area**2) / (epsilon_r**2)

    log = -0.5 * (1 / epsilon_r)**2 * np.matrix(w_model[:, slice1] - data)* np.matrix(w_model[:, slice1] - data).T
    #log = -res**2 / epsilon_r**2  # 1/(sigma*np.sqrt(2*np.pi)) *

    return log, w_model[:, slice1]

def noise(slices, model_times, temperature, residuals, window):
    T_target_mu = np.interp(
        model_times[slices[:-1]], temperature['Times'], temperature['Tmax'])
    T_target_mu = np.concatenate([T_target_mu,np.array([temperature['Tmax'][-1]])])
    res = []
    target = []
    for i, s in enumerate(slices):
        A = max(0, int(s-window/2))
        B = min(int(s+window/2), len(model_times)-1)
        ind = residuals['Times'].between(model_times[A], model_times[B])
        res.append(residuals['Tmax'][ind].values)
        target.append(res[-1]+T_target_mu[i])
    return target


"""
sample_prior
"""

def prior_pdf_smardda(x,res):
    L_q_mu = stats.norm.pdf(x[range(res)], loc=6, scale=2)  # normal
    if any(x[range(res)]) > 10:
        L_q_mu = np.zeros(res)
    L_q_sig = stats.uniform.pdf(x[range(res, 2*res)],loc=0.01,scale=2)
    Sig_mu = stats.lognorm.pdf(x[range(2*res, 3*res)], 0.7 , loc=0.4, scale=2)
    if any(x[range(2*res, 3*res)])>7:
        Sig_mu = np.zeros(res)
    Sig_sig = stats.uniform.pdf(x[range(3*res, 4*res)], loc=0.01, scale=2)
    Corr = stats.uniform.pdf(
        x[range(4*res, 4*res+n_p_cor(res))], loc=-1, scale=2)  # uniform
    return np.prod(np.prod([L_q_mu, L_q_sig, Sig_mu, Sig_sig, Corr]))


def prior_rnd_smardda(N, res):
    Corr = stats.uniform.rvs(
        loc=-1,    scale=2, size=[n_p_cor(res), N])  # uniform
    L_q_mu = stats.norm.rvs(loc=6,     scale=2, size=[res, N])  # normal
    L_q_sig = stats.uniform.rvs(loc=0.01,  scale=2, size=[res, N])
    Sig_mu = stats.lognorm.rvs(0.7, loc=0.4,   scale=2, size=[res, N])
    Sig_sig = stats.uniform.rvs(loc=0.01,  scale=2, size=[res, N])
    return np.concatenate([L_q_mu, L_q_sig, Sig_mu, Sig_sig, Corr], axis=0)


def n_p_cor(res):
    k = res-1
    for i in range(k-1, 0, -1):
        k += i
    return int(k)


def inclusive_range(start, stop, step):
    return range(start, (stop + 1) if step >= 0 else (stop - 1), step)


def generate_corr(d, partial_corrs):

    #   The scripts generates a correlation matrix from partial
    #   correlations, as described in:
    #
    #       Joe, Harry, "Generating random correlation matrices based on partial correlations."
    #           Journal of Multivariate Analysis 97.10 (2006): 2177-2189.
    #
    #   The partial correlations may vary between[-1, 1] and we need
    #   nchoosek(d, 2) of them. When these parameters are randomised, a
    #   random correlation matrix with (nearly) uniformly distributed
    #   correlations is produced
    #
    #   It is the partialCorrs which will be calibrated in the bayesian
    #   updating
    #
    #   D == dimension of the correlation matrix
    #   partialCorr == the partial correlations, nchoosek(d, 2) of them

    P = np.zeros([d, d])
    S = np.eye(d)
    ll = 0

    for k in range(d-1):
        for i in range(k+1, d):
            P[k, i] = partial_corrs[ll]
            p = P[k, i]
            for l in inclusive_range(k-1, 0, -1):
                p = p * np.sqrt((1-P[l, i]**2) *
                                (1-P[l, k]**2))+(P[l, i]*P[l, k])
            S[k, i] = p
            S[i, k] = p
            ll = ll+1
    return S

"""
Updating models
"""

def sample_history(Th_posterior, Lp_posterior, n_histories):
    idx = np.random.choice(np.shape(Lp_posterior)[0], n_histories, replace=True,
                           p=np.abs(Lp_posterior)/np.sum(np.abs(Lp_posterior)))

    histories = Th_posterior[idx]
    return idx, histories


def populate_history(histories, n):
    # need to fix the dimesnsions to make it work with an output
    theta_0 = []
    for i in range(n):
        theta_0.append(histories[np.random.randint(0,)])
    return

def q_slice(times, s, ti_s, res):
    k = np.zeros(len(times))

    delt = int((ti_s[s+1] - ti_s[s])/res/2)
    ind = np.linspace(ti_s[s]+delt, ti_s[s+1]-delt, res, dtype=int)
    #ind = ind[1:]
    k[ind] = 1
    kni = ti_s[s] + np.where(k[range(ti_s[s], ti_s[s+1])] < 1)[0]
    impute = [ind[np.abs(x-ind).argmin()] for i, x in enumerate(kni)]
    k = k > 0
    return k, kni, impute

def fill_boundary(l_q,sig,R,GP,gp_scale,p_baseline,Q_bdry):

    pred = np.concatenate([[np.ones(np.size(R)) * l_q],
                           [np.ones(np.size(R)) * sig],
                           [R]], axis=0)
    m, s = GP.predict(pred.T)
    mask = (Q_bdry == 0).all(0)
    ind = np.where(mask)[0]
    Q_bdry[:, ind] = B = np.array((m * gp_scale) / p_baseline)
    return Q_bdry


def probabilistic_model_t(theta, N, burnin=True):
    # TODO Can we do this with the marginal kernel density instead of being constrained to the distribution???
    # or with cointergration by a linear cointergration function
    # theta---------------------------------------
    # range(rs) = partial correlations
    # range(rs,4*rs) = lambda_q
    # range(4*rs,8*rs) = sigma
    d = 2
    l_q_mu = theta[0]
    l_q_sig = theta[1]
    s_mu = theta[2]
    s_sig = theta[3]

    r = generate_corr(d, theta[4])
    try:
        c = cholesky(r)
    except:
        evals, evecs = eigh(r)
        c = np.dot(evecs, np.diag(np.sqrt(evals)))
        print('WARNING: Theta does not permit cholesky, reverting to eigh. If this happens a lot consider regression implimentation. \n  \
            Your covariance matrix is ill-conditioned/nearly singular/has a high condition number.\n \
            check the Lambda_q and Sigma values')

    z = np.random.normal(size=[N, d])  # sample from the gaussian copula
    x = np.matrix(z) * np.matrix(c).T       # project correlation
    x = np.array(x)
    # inverse transorm sample copula
    u = stats.norm.cdf(x)

    lambda_model = stats.norm.ppf(u[:, 0], loc=l_q_mu, scale=l_q_sig)
    sigma_model = stats.norm.ppf(u[:, 1], loc=s_mu, scale=s_sig)
    #TO DO : epsilon_Psol needs to be implimented
    return lambda_model, sigma_model

def probabilistic_model(theta, N, res, burnin=True):
    # TODO Can we do this with the marginal kernel density instead of being constrained to the distribution???
    # or with cointergration by a linear cointergration function
    # theta---------------------------------------
    # range(rs) = partial correlations
    # range(rs,4*rs) = lambda_q
    # range(4*rs,8*rs) = sigma
    l_q_mu = theta[range(res)]
    l_q_sig = theta[range(res, 2*res)]
    s_mu = theta[range(2*res, 3*res)]
    s_sig = theta[range(3*res, 4*res)]
    if res > 1:
        d = res
    else:
        d = 2
    r = generate_corr(d, theta[range(4*d, (4*d)+n_p_cor(d))])
    try:
        c = cholesky(r)
    except:
        evals, evecs = eigh(r)
        c = np.dot(evecs, np.diag(np.sqrt(evals)))
        print('WARNING: Theta does not permit cholesky, reverting to eigh. If this happens a lot consider regression implimentation. \n  \
            Your covariance matrix is ill-conditioned/nearly singular/has a high condition number.\n \
            check the Lambda_q and Sigma values')


    z = np.random.normal(size=[N, d])  # sample from the gaussian copula
    x = np.matrix(z) * np.matrix(c).T       # project correlation
    x = np.array(x)
    # inverse transorm sample copula
    u = stats.norm.cdf(x)

    if res >1:
        print('evaluating with autocorrelation')
        lambda_model = []
        sigma_model = []
        for i in range(res):
            lambda_model.append(stats.norm.ppf(
                u[:, i], loc=l_q_mu[i], scale=l_q_sig[i]))
            sigma_model.append(stats.norm.ppf(
                u[:, i], loc=s_mu[i], scale=s_sig[i]))
        lambda_model = np.array(lambda_model)
        sigma_model = np.array(sigma_model)
    else:
        lambda_model = stats.norm.ppf(u[:, 0], loc=l_q_mu, scale=l_q_sig)
        sigma_model = stats.norm.ppf(u[:, 1], loc=s_mu, scale=s_sig)
    #TO DO : epsilon_Psol needs to be implimented
    return lambda_model, sigma_model


def boundary_condion(lambda_q, sigma_mid, Psol, GP, p_baseline, gp_scale, times, seq_ind, slices, R):
    Q_bdry = np.zeros([len(R), len(times)])
    Qb = []
    ar = np.concatenate([[lambda_q], [sigma_mid]], axis=0)
    uq, u_idx, c_uq = np.unique(ar, axis=1, return_index=True, return_counts=True)

    sort_ix = np.argsort(u_idx)
    uq = uq[:,sort_ix]
    u_idx = u_idx[sort_ix]
    c_uq = c_uq[sort_ix]
    idx_repeated = np.where(c_uq > 1)[0]
    index_sets = [np.logical_and(ar[0]==uq[0,i],ar[1]==uq[1,i]) for i in idx_repeated]

    #u_idx = np.sort(u_idx)
    for i, ix in enumerate(u_idx[:-1]):  #
        pred = np.concatenate([[np.ones(np.size(R))*lambda_q[ix]],
                               [np.ones(np.size(R))*sigma_mid[ix]],
                               [R]], axis=0)
        m, s = GP.predict(pred.T)
        Q_bdry[:, index_sets[i]] = np.array((m * gp_scale)/p_baseline)
        Qb.append(m*gp_scale)
    #Q_bdry = fill_boundary(
    #    lambda_q[slices[seq_ind+1]], sigma_mid[slices[seq_ind+1]], R, GP, gp_scale, p_baseline, Q_bdry)
    Qb = np.array(Qb)
    Qb = np.squeeze(Qb)
    Q_bdry = Q_bdry*[Psol]
    return Q_bdry

def execute_model(lambda_q, sigma_mid, Psol, GP, q_scale_gp, Tinit, times, seq_ind, slices, R, Told=[], first=True, verbose=False):

    t0 = time.time()
    Q_bdry = boundary_condion(
        lambda_q, sigma_mid, Psol, GP, 8E6, q_scale_gp, times, seq_ind, slices, R)
    t1 = time.time()
    Tsurf, To = ht.solve_2d(R-min(R), times, Q_bdry.T*0.95, depth=0.036, Tinit=Tinit)
    t2 = time.time()
    if verbose:
        print('Time Elapsed: Boundary condition {} - Heat Equation {}'.format(timestamp_format(t0,t1),timestamp_format(t1,t2)))
    #T_target = np.interp(
    #    run_data['Target_times'], run_data['Target_T']['Times'], run_data['Target_T']['Tmax'])
    T_out = np.max(Tsurf, axis=1)
    return T_out, Tsurf, To

def construct_history(lq_hist, sig_hist, n_model, times, seq_ind, slices, res):

    lambda_q = np.ones(slices[seq_ind+1] - slices[seq_ind])
    sigma_mid = np.ones(slices[seq_ind+1] - slices[seq_ind])
    inp_h, kni_h, impute_h = q_slice(
        times[range(slices[seq_ind], slices[seq_ind+1])], 0, [slices[seq_ind], slices[seq_ind+1]]-slices[seq_ind], res)
    l_q = []
    sig = []
    for i in range(n_model):
        lambda_q[inp_h] = lq_hist[:, i]
        sigma_mid[inp_h] = sig_hist[:, i]
        lambda_q[kni_h] = lambda_q[impute_h]
        sigma_mid[kni_h] = sigma_mid[impute_h]
        l_q.append(lambda_q)
        sig.append(sigma_mid)
        lambda_q = np.ones(slices[seq_ind+1] - slices[seq_ind])
        sigma_mid = np.ones(slices[seq_ind+1] - slices[seq_ind])
    return l_q, sig

#def updating_assumed_density(theta, N, seq_ind, slices, res, times, Psol, run_data, R, lq_0=5, sig_0=3, lq_hist=[], sig_hist=[] ,nburn=5, burnin=False):
#    if burnin:
#        N = nburn
#
#    GP = run_data['GP']
#    l_q, sig = probabilistic_model(theta, N, res)
#    inp, kni, impute = q_slice(times, seq_ind, slices, res)
#    lambda_q = np.ones(len(times))*lq_0
#    sigma_mid = np.ones(len(times))*sig_0
#
#    temperature_prediction = []
#    for i in range(N):
#        #print('Executing {} sample'.format(i))
#        if seq_ind > 0:
#            lambda_q[range(slices[seq_ind+1])] = lq_hist[i,:]
#            sigma_mid[range(slices[seq_ind+1])] = sig_hist[i,:]
#
#        lambda_q[inp] = l_q[:,i]
#        sigma_mid[inp] = sig[:,i]
#        lambda_q[kni] = lambda_q[impute]
#        sigma_mid[kni] = sigma_mid[impute]
#        T_out, Tsurf, To = execute_model(
#            lambda_q, sigma_mid, Psol, GP, run_data['q_scale_gp'], run_data['T_init'], times, seq_ind, slices, R)
#        temperature_prediction.append(T_out)
#    temperature_prediction = np.array(temperature_prediction)
#    return temperature_prediction


def construct_history(lq_hist, sig_hist, n_model, times, seq_ind, slices, res):

    lambda_q = np.ones(slices[seq_ind+1] - slices[seq_ind])
    sigma_mid = np.ones(slices[seq_ind+1] - slices[seq_ind])
    inp_h, kni_h, impute_h = q_slice(
        times[range(slices[seq_ind], slices[seq_ind+1])], 0, [slices[seq_ind], slices[seq_ind+1]]-slices[seq_ind], res)
    l_q = []
    sig = []
    for i in range(n_model):
        lambda_q[inp_h] = lq_hist[i]
        sigma_mid[inp_h] = sig_hist[i]
        lambda_q[kni_h] = lambda_q[impute_h]
        sigma_mid[kni_h] = sigma_mid[impute_h]
        l_q.append(lambda_q)
        sig.append(sigma_mid)
        lambda_q = np.ones(slices[seq_ind+1] - slices[seq_ind])
        sigma_mid = np.ones(slices[seq_ind+1] - slices[seq_ind])
    return l_q, sig


def updating_assumed_density_t(theta, N, seq_ind, slices, res, times, Psol, run_data, R, lq_0=5, sig_0=3, lq_hist=[], sig_hist=[], nburn=5, burnin=False):
    if burnin:
        N = nburn

    GP = run_data['GP']
    l_q, sig = theta[0],theta[1] #probabilistic_model_t(np.reshape(theta,[5,1]), N)
    inp, kni, impute = q_slice(times, seq_ind, slices, 1)
    lambda_q = np.ones(len(times))*lq_0
    sigma_mid = np.ones(len(times))*sig_0

    temperature_prediction = []
    for i in range(N):
        #print('Executing {} sample'.format(i))
        if np.logical_and(seq_ind > 0,seq_ind<np.max(slices)):
            print('non inclusive seq_ind : {}'.format(seq_ind))
            lambda_q[range(slices[seq_ind])] = np.squeeze(lq_hist[i])
            sigma_mid[range(slices[seq_ind])] = np.squeeze(sig_hist[i])
        elif seq_ind == np.max(slices):
            print('inclusive seq_ind : {}'.format(seq_ind))
            lambda_q[inclusive_range(0,slices[seq_ind],1)] = np.squeeze(lq_hist[i])
            sigma_mid[inclusive_range(0,slices[seq_ind],1)] = np.squeeze(sig_hist[i])

        lambda_q[inp] = l_q#[i]
        sigma_mid[inp] = sig#[i]
        lambda_q[kni] = lambda_q[impute]
        sigma_mid[kni] = sigma_mid[impute]
        T_out, Tsurf, To = execute_model(
            lambda_q, sigma_mid, Psol, GP, run_data['q_scale_gp'], run_data['T_init'], times, seq_ind, slices, R)
        temperature_prediction.append(T_out)
    temperature_prediction = np.array(temperature_prediction)
    return temperature_prediction


"""
Sweeps
"""

def sweep_boundary_model(lambda_q,sigma,x_r,GP,sweep_frequency):
    length = 1
    c = -1
    Q =[]
    reset = len(GP)

    steps = len(GP)
    for i in range(steps):
        c = c + 1
        pred = np.concatenate([[np.ones(len(x_r))*lambda_q[c]],[np.ones(len(x_r))*sigma[c]],[x_r]], axis=0)
        Q_pred, Q_std = GP[c]['GP'].predict(pred.T)
        Q_pred = np.reshape(Q_pred, [len(Q_pred)])
        Q.append(Q_pred*GP[c]['q_scale_gp'])
        if c == reset-1:
            c = -1
    return Q


def construct_boundary_model(lambda_q, sigma, Psol, GP, p_baseline, times, R, sweep_frequency):
    # lambda_q = shape(n_slices,n_gps)
    #
    #
    symmetry = 0.5
    dx = signal.sawtooth(2 * np.pi * times * sweep_frequency, symmetry)*((len(GP)-1)/2)
    Dx = np.round(dx-np.min(dx))

    Q_bdry = np.zeros([len(R), len(times)])
    Qb = []

    Q = sweep_boundary_model(lambda_q, sigma, R, GP, sweep_frequency)

    for i in range(len(np.unique(Dx))):
        Q_bdry[:, Dx == i] = np.reshape(np.repeat(Q[i], sum(Dx == i)), [len(Q[i]), sum(Dx == i)])/p_baseline

    Q_bdry = Q_bdry*[Psol]
    return Q_bdry


def updating_sweep_boundary(theta, N, seq_ind, slices, res, times, Psol, GP, R, lq_hist=[], sig_hist=[]):
    t0 = theta[0]*np.ones(3)
    t1 = theta[1]*np.ones(3)
    theta = np.concatenate([t0,t1])
    if seq_ind == 0:
        Q_bdry = construct_boundary_model(
            theta[:3], theta[3:], Psol, GP, 14E6, times, R, 4)
    else:
        Q = []
        for i in range(seq_ind):
            q_bd = construct_boundary_model(lq_hist[i], sig_hist[i], Psol[range(
                slices[i], slices[i+1])], GP, 14E6, times[range(slices[i], slices[i+1])], R, 4)
            Q.append(q_bd)

        q_ed = construct_boundary_model(theta[:3], theta[3:], Psol[range(
            slices[seq_ind], slices[seq_ind+1])], GP, 14E6, times[range(slices[seq_ind], slices[seq_ind+1])], R, 4)
        Q.append(q_ed)
        
        q_ed = construct_boundary_model(lq_hist[-1], sig_hist[-1], Psol[range(
            slices[seq_ind+1], slices[-1])], GP, 14E6, times[slices[seq_ind+1]:], R, 4)
        Q.append(q_ed)

        Q_bdry = np.concatenate(Q, axis=1)
    return Q_bdry


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

    Q = sweep_boundary_model(lambda_q, sigma, R, GP, sweep_frequency)

    NUM_COND = (dx[1]-dx[0])*.5

    k = []
    for i in range(len(GP)):  #
        print(i)
        if (i > round(min(dx)) + abs(round(min(dx))) )& (i < round(max(dx)) + abs(round(min(dx)))):
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
        Qbdry[:, K[i]:K[i+1]
              ] = merge_sweep(Q[A], Q[B], times[K[i]:K[i+1]])/p_baseline

    Qbdry = Qbdry*[Psol]
    return Qbdry

def execute_model_sweep(Q_bdry, Tinit, times,  R):
    #t0 = time.time()
    Tsurf, To = ht.solve_2d(R-min(R), times, Q_bdry *
                            0.95, depth=0.036, Tinit=Tinit)
    #t1 = time.time()
    T_out = np.max(Tsurf, axis=1)
    return T_out, Tsurf, To


def update_assumed_density_s(theta, N, seq_ind, slices, res, times, Psol, GP, R, T_init,lq_0=5, sig_0=3, lq_hist=[], sig_hist=[]):
    temperature_prediction = []
    for i in range(N):
        if seq_ind == 0:
            lq_hist = np.zeros(10)
            sig_hist = np.zeros(10)

        Q_bdry = updating_sweep_boundary(
            theta, [], seq_ind, slices, res, times, Psol, GP, R, lq_hist=lq_hist, sig_hist=sig_hist)
        T_out, Tsurf, To = execute_model_sweep(Q_bdry.T, T_init, times,  R)
        temperature_prediction.append(T_out)
    temperature_prediction = np.array(temperature_prediction)
    return temperature_prediction



"""
METRICS =========================================================================
"""
"""
Confidence interval
- Frequentist confidence interval is used to characterise the sampling uncertainty in
    given data
- Gosset recognised the need for the
"""

def confidence_interval_mean(data,confidence_interval = 99,plot=True):
    " This is only appropriate for one d data array"

    data = data.squeeze()
    n = np.shape(data)
    if len(n)>1:
        print('ERROR: Data array must be 1 dimensional- dimensions = {}'.format(len(n)))

    n = n[0]
    x_bar = np.mean(data)
    s = np.std(data)
    t_alpha = stats.t.ppf(1- ((100-confidence_interval)/2/100), n-1)

    l_alpha = x_bar - t_alpha * (s/np.sqrt(n))
    h_alpha = x_bar + t_alpha * (s/np.sqrt(n))

    if plot:
        py,d = ecdf(data)
        plt.step(d,py,c = 'blue',label='Data')
        plt.plot(np.ones(2)*h_alpha,[0,1],c='red',alpha=90,label='{} mean ci '.format(1- ((100-confidence_interval)/2/100)))
        plt.plot(np.ones(2)*l_alpha,[0,1],c='red',alpha=70,label='{} mean ci '.format(((100-confidence_interval)/2/100)))
        plt.title('sample mean {} % ci'.format(confidence_interval))
        plt.legend()
    return [l_alpha,h_alpha]

def s_ln(x,data):
    n = np.size(data)
    l = np.sum(data<=x)
    return l/n

def smirnov_critical_value(alpha,n):
    # a = np.array([0.20,0.15,0.10,0.05,0.025,0.01,0.005,0.001])
    # c_a = np.array([1.073,1.138,1.224,1.358,1.48,1.628,1.731,1.949])
#
    # if any(np.isclose(0.0049,a,2e-2)):
        # c_alpha = c_a[np.where(np.isclose(0.0049,a,2e-2))[0]]
    # else:
    c_alpha = np.sqrt(-np.log(alpha/2)*(1/2))
    return (1/np.sqrt(n))*c_alpha

def confidence_limits_distribution(x,alpha,interval=False,n_interp = 100,plot=False,x_lim=[-10,10],label=''):
    """
    The confidence limits of F(x) is an inversion of the well known KS-test.
    KS test is usually used to test whether a given F(x) is the underlying probability distribution of Fn(x).

    See      : Experimental uncertainty estimation and statistics for data having interval uncertainty. Ferson et al.
               for this implimentation. Here interval valued array is valid.
    """

    if not interval:
        data = np.zeros([2,np.size(x)])
        data[0] = x
        data[1] = x
    else:
        data = x

    x_i = np.linspace(np.min(data[0])+x_lim[0],np.max(data[1])+x_lim[1], n_interp)

    N = np.size(data[0])

    if N < 50 :
        print('Dont trust me! I really struggle with small sample sizes\n')
        print('TO DO: Impliment the statistical conversion table for Z score with lower sample size')

    b_l = lambda x: min(1 ,s_ln(x,data[0])+smirnov_critical_value(round((1-alpha)/2,3),N))
    b_r = lambda x: max(0 ,s_ln(x,data[1])-smirnov_critical_value(round((1-alpha)/2,3),N))

    L = []
    R = []
    for i,xi in enumerate(x_i):
        L.append(b_l(xi))
        R.append(b_r(xi))

    if plot:
        pl,xl = ecdf(data[0])
        pr,xr = ecdf(data[1])
        plt.step(xl,pl,color='blue',label='data',alpha =0.3)
        plt.step(xr,pr,color='blue',alpha =0.7)
        plt.step(x_i,L,color='red',label='data',alpha =0.7)
        plt.step(x_i,R,color='red',alpha =0.7,label = 'KS confidence limits {}%'.format(alpha))
        plt.xlabel(label)
        plt.ylabel('P(x)')
    return L, R, x_i


"""
Area metric
"""

def area_metric_robust(D1,D2):
    """
    #   Returns the stochastic distance between two data
    #   sets, using the area metric (horizontal integral between their ecdfs)
    #
    #   As described in: "Validation of imprecise probability models" by S.
    #   Ferson et al. Computes the area between two ECDFs
    #
    #                  By Marco De Angelis, (adapted for python Dominic Calleja)
    #                     University of Liverpool by The Liverpool Git Pushers
    """

    if np.size(D1)>np.size(D2):
        d1 = D2
        d2 = D1
    else:
        d1 = D1
        d2 = D2      # D1 will always be the larger data set

    Pxs,xs = ecdf(d1)            # Compute the ecdf of the data sets
    Pys,ys = ecdf(d2)

    Pys_eqx = Pxs
    Pys_pure = Pys[0:-1]  # this does not work with a single datum
    Pall = np.sort(np.append(Pys_eqx,Pys_pure))

    ys_eq_all = np.zeros(len(Pall))
    ys_eq_all[0]=ys[0]
    ys_eq_all[-1]=ys[-1]
    for k in range(1,len(Pall)-1):
        ys_eq_all[k] = interpCDF_2(ys,Pys,Pall[k])

    xs_eq_all = np.zeros(len(Pall))
    xs_eq_all[0]=xs[0]
    xs_eq_all[-1]=xs[-1]
    for k in range(1,len(Pall)-1):
        xs_eq_all[k] = interpCDF_2(xs,Pxs,Pall[k])

    diff_all_s = abs(ys_eq_all-xs_eq_all)
    diff_all_s = diff_all_s[range(1,len(diff_all_s))]
    diff_all_p = np.diff(Pall)
    area = np.matrix(diff_all_p) * np.matrix(diff_all_s).T

    return np.array(area)[0]


def area_metric_robust(D1,D2):
    """
    #   Returns the stochastic distance between two data
    #   sets, using the area metric (horizontal integral between their ecdfs)
    #
    #   As described in: "Validation of imprecise probability models" by S.
    #   Ferson et al. Computes the area between two ECDFs
    #
    #                  By Marco De Angelis, (adapted for python Dominic Calleja)
    #                     University of Liverpool by The Liverpool Git Pushers
    """

    if np.size(D1)>np.size(D2):
        d1 = D2
        d2 = D1
    else:
        d1 = D1
        d2 = D2      # D1 will always be the larger data set

    Pxs,xs = ecdf(d1)            # Compute the ecdf of the data sets
    Pys,ys = ecdf(d2)

    Pys_eqx = Pxs
    Pys_pure = Pys[0:-1]  # this does not work with a single datum
    Pall = np.sort(np.append(Pys_eqx,Pys_pure))

    ys_eq_all = np.zeros(len(Pall))
    ys_eq_all[0]=ys[0]
    ys_eq_all[-1]=ys[-1]
    for k in range(1,len(Pall)-1):
        ys_eq_all[k] = interpCDF_2(ys,Pys,Pall[k])

    xs_eq_all = np.zeros(len(Pall))
    xs_eq_all[0]=xs[0]
    xs_eq_all[-1]=xs[-1]
    for k in range(1,len(Pall)-1):
        xs_eq_all[k] = interpCDF_2(xs,Pxs,Pall[k])

    diff_all_s = abs(ys_eq_all-xs_eq_all)
    diff_all_s = diff_all_s[range(1,len(diff_all_s))]
    diff_all_p = np.diff(Pall)
    area = np.matrix(diff_all_p) * np.matrix(diff_all_s).T

    return np.array(area)[0]

def area_metric_simple(D1, D2):
    """
    #   Returns the stochastic distance between two data
    #   sets, using the area metric (horizontal integral between their ecdfs)
    #
    #   As described in: "Validation of imprecise probability models" by S.
    #   Ferson et al.
    #
    #   Considered to be a stochastic generalisation of euclidean distance
    #   For example:
    #               ->  AreaMetric(1,2) == 2
    #
    #               ->  a = randn(1,10000);
    #                   b = randn(1,10000) + 2;
    #
    #                   AreaMetric(a,b) ~= 2;
    #                               (aproaches 2 as sampling increases)
    #
    #                        By Ander Gray, (adapted for python Dominic Calleja)
    #                           University of Liverpool
    #
    """
    if np.size(D1)>np.size(D2):
        d1 = D2
        d2 = D1
    else:
        d1 = D1
        d2 = D2      # D1 will always be the larger data set

    Pxs,xs = ecdf(d1);            # Compute the ecdf of the data sets
    Pys,ys = ecdf(d2);

    subs = np.zeros(np.size(d1));         # vector of differences


    # -1 because the final step in ecdf is flat (no area)
    for i in range(len(Pxs)-1):
        ind = np.where(Pxs[i] <= Pys)[0]                  # Find intersect of ps
        subs[i] = abs(xs[i] - ys[ind[0]])          # Subtract physical vals

    # Integral (area) is the sum of the heights * the base
    area = np.sum(subs) * 1/len(d1);
    return area


def interpCDF_2(xd,yd,pvalue):
    """
    %INTERPCDF Summary of this function goes here
    %   Detailed explanation goes here
    %
    % .
    % . by The Liverpool Git Pushers
    """
    # [yd,xd]=ecdf(data)
    beforr = np.zeros(len(yd))
    beforr = np.diff(pvalue <= yd) ==1
    beforrr = np.append(0,beforr[:])
    if pvalue==0:
        xvalue = xd[1]
    else:
        xvalue = xd[beforrr==1]

    outputArg1 = xvalue

    return outputArg1


def ecdf(x):
    xs = np.sort(x)
    #xs = np.append(xs,xs[-1])
    n = xs.size
    y = np.linspace(0,1,n)
    #np.arange(1, n+1) / n
    #xs = np.append(xs[0],xs)
    #ps =
    return [y,xs]

"""
U-Pool
"""


def u_pool2(data, mod, n_dimensions, compare='uniform', plot=False):
    # WARNING: only implimented for a single sample of the model output.
    # u-pooling
    #
    # u-pooling, proposed by Ferson and Oberkampf, utilities the inverse probability transform to aggregated disperate dimensions in
    # the probability dimension in order to gain a aggregated metric for a multidimensional output simulation.
    # first used in the context of model validation this is the first attempt to utilise the approach for model updating.
    # ============================== ============================== ============================== ============================== ==========
    # Comment : Very useful in this context as we hav a high dimension time series output. We have a full distribution of noise in the data
    # at each time point, and our heat equation model outputs a single vector of maxT at each time point.
    # The metric is constrained to [0,.5] interval making the design of the loss function trivial and flexible.
    # ============================== ============================== ============================== ============================== ==========
    # Ferson, S., and Oberkampf, W. L., 2009, “Validation of Imprecise Probability Models, ”
    #                   By Dominic Calleja (The liverpool Git Pushers)

    #n_data_samples = np.array(np.shape(d))[np.asarray(np.shape(d))!=n_dimensions]

    ki = np.size(data)
    data = np.reshape(data, [n_dimensions, int(ki/n_dimensions)])
    ki = int(ki/n_dimensions)

    mod = np.reshape(mod, [n_dimensions, 1])
    pi = 1

    #if pi != n_dimensions:
    #    print('Something wrong! Either you have too many samples in your model, or you dont have enough dimensions! Currently method only works for a single model evaluation')
    #   return
    p_k = []
    x_k = []
    ui = []
    for k in range(n_dimensions):
       p, x = ecdf(data[k])
       p_k.append(p)
       x_k.append(x)
       ui.append(np.interp(mod[k], x, p).item())

    Pu, Xu = ecdf(ui)
    # Need to impliment different comparison distributions!
    comparison = np.linspace(0, 1, 2000)
    Pc, Xc = ecdf(comparison)

    if plot:
        plt.figure()
        plt.step(Xu, Pu, color='red', label='model inverse interpolant')
        plt.step(Xc, Pc, color='blue',
                 label='comaparison distribution : ' + compare)
        plt.show()
    area_mismatch = area_metric_robust(comparison, ui)
    discrepancy = ui
    return area_mismatch, discrepancy

"""
Gaussian metric
"""

def p_x_theta_pdf(data, w_model, epsilon_r,dim_mod): #theta_i,
    # x       = set of observations         nobs x dim_x
    # theta_i = point in epistemic space    nsamples x dim_theta

    # Estimate the PDF p_x_theta_pdf(x | theta)
    # compute likelihood accordind to the GOCE paper
    # p(i) = p_x_theta_pdf(x(i,:) | theta)

    diff = (w_model - data)
    diff = np.matrix(diff)
    p = -0.5*(1/epsilon_r)**2 * (diff * diff.T)

#    p =[]
#    for dim in range(int(dim_mod)):
#        p.append(-1/2*np.nansum(((1-data[dim]**2 /w_model[dim]**2)/epsilon_r[dim])**2,axis = 0))
    return np.array(p)

def p_x_theta_pdf2(data, w_model, epsilon_r,dim_mod): #theta_i,
    # x       = set of observations         nobs x dim_x
    # theta_i = point in epistemic space    nsamples x dim_theta

    # Estimate the PDF p_x_theta_pdf(x | theta)
    # compute likelihood accordind to the GOCE paper
    # p(i) = p_x_theta_pdf(x(i,:) | theta)

    p =[]
    for dim in range(int(dim_mod[0])):
        diff = (w_model[dim] - data[dim])
        diff = np.matrix(diff)
        p.append( -0.5*(1/epsilon_r[dim])**2 * (diff * diff.T))
    return np.array(p)

"""
Transitional Models
"""
#The tranistion model defines how to move from sigma_current to sigma_new
transition_model = lambda x: [x[0],np.random.normal(x[1],0.5,(1,))]


def proppdf(x, mu, covmat,box):
    # Proposal PDF for the Markov Chain.
    y = np.multiply(stats.multivariate_normal.pdf(x, mu, np.matrix(covmat), allow_singular=False),box(x))     #q(x,y) = q(x|y).
    return y

def proprnd(mu, covmat, box):

    while True:
        t = stats.multivariate_normal.rvs(mu,covmat)
        if not box(t)==0:
            break
    return t

def proppdf_independent(x, mu,box):
    # Proposal PDF for the Markov Chain.
    y = np.multiply(stats.multivariate_normal.pdf(x, mu),box(x))     #q(x,y) = q(x|y).
    return y

def proprnd_independent(mu, box):
    while True:
        t = stats.multivariate_normal.rvs(mu)
        if not box(t)==0:
            break
    return t

"""
Log likelihood
"""

def area_likelihood(model, theta, Data, n_samples=1, epsilon = [], epsilon_r =[]): #simple=True,

    y = model(theta)

    dim = np.array(np.shape(y))[np.asarray(np.shape(y))!=n_samples]
    if dim.size==0:
        dim = np.ones(1)

    if dim != 1:
        if not any(np.asarray(np.shape(Data))==dim):
            print('Number of dimensions in the model output {} does not match any dimension of data {}'.format(dim,np.shape(Data)))
            return

    n_data = int(np.size(Data)/dim[0])
    y = np.reshape(np.asarray(y),[int(dim[0]),n_samples])
    Data = np.reshape(Data, [int(dim[0]),int(np.size(Data)/dim[0])])

    if not epsilon and not epsilon_r:
        epsilon_r = np.std(Data,axis =1)
    elif epsilon_r:
        epsilon_r = epsilon_r
    else:
        epsilon_r = np.ones(int(dim[0]))* epsilon

    LogL =[]
    if dim[0] ==1:
        LogL = -(1/np.sum(epsilon_r))**2 * area_metric_robust(Data.T, y.T)**2
    else:
        for i in range(dim[0]):
            LogL.append(-(1 / epsilon_r[i])**2 * area_metric_robust(Data[i].T, y[i].T)**2)
    return np.sum(LogL)

def area_likelihood_simple(model, theta, Data, n_samples=1, epsilon = [], epsilon_r =[]): #simple=True,

    y = model(theta)

    dim = np.array(np.shape(y))[np.asarray(np.shape(y))!=n_samples]
    if dim.size==0:
        dim = np.ones(1)

    if dim != 1:
        if not any(np.asarray(np.shape(Data))==dim):
            print('Number of dimensions in the model output {} does not match any dimension of data {}'.format(dim,np.shape(Data)))
            return

    n_data = int(np.size(Data)/dim[0])
    y = np.reshape(np.asarray(y),[int(dim[0]),n_samples])
    Data = np.reshape(Data, [int(dim[0]),int(np.size(Data)/dim[0])])

    if not epsilon and not epsilon_r:
        epsilon_r = np.std(Data,axis =1)
    elif epsilon_r:
        epsilon_r = epsilon_r
    else:
        epsilon_r = np.ones(int(dim[0]))* epsilon

    LogL =[]
    if dim[0] ==1:
        LogL = -(1/np.sum(epsilon_r))**2 * area_metric_simple(Data.T, y.T)**2
    else:
        for i in range(n_data):
            LogL.append(-(1 / epsilon_r[i])**2 * area_metric_simple(Data[i].T, y[i].T)**2)

    return np.sum(LogL)



# def emperical_likelihood(model,theta,Data,axis_dimension=0,epsilon = 0.1):
    # y = model(theta)
    # LogL =[]
#
    # ex,ey = ecdf(y)
    # for i in range(np.shape(y)[0]):
        # stochastic_distance = stats.kstest(y[i], Data[i])
        # LogL.append( -(1/epsilon)**2 * stochastic_distance**2)
    # LogL = np.sum(np.array(LogL))
    # return LogL
#
# def gaussian_likelihood(model,theta,Data,n_model=1,epsilon = 0.1):
    # y = model(theta)
#
    # dim = np.array(np.shape(y))[np.asarray(np.shape(y))!=n_model]
    # if dim.size==0:
        # dim = np.ones(1)
#
    # if dim != 1:
        # if not any(np.asarray(np.shape(Data))==dim):
            # print('Number of dimensions in the model output {} does not match any dimension of data {}'.format(dim,np.shape(Data)))
            # return
#
    # y = np.reshape(np.asarray(y),[int(dim[0]),n_model])
    # Data = np.reshape(Data, [int(dim[0]),int(np.size(Data)/dim[0])])
#
    # LogL = []
    # for i in range(int(dim[0])):
        #diff = np.matrix(y[i]-Data[i,j])
        #ll = stats.normal(y[i],1).pdf(Data[i]).prod()
        # sigma = np.std(Data[i])
        # mu = np.mean(Data[i])
        # logl = np.nansum(-np.log(sigma * np.sqrt(2* np.pi) )-((Data[i]-mu)**2) / (2*sigma**2))
        #-0.5*(1/epsilon)**2 * diff*diff.T
        # LogL.append(logl)
    # LogL = np.array(logl)

def gaussian_likelihood2(theta,Data,model=[], n_model=1,epsilon = [], epsilon_r =[]):
    if model:
        y = model(theta)

    dim = np.array(np.shape(y))[np.asarray(np.shape(y))!=n_model]
    if dim.size==0:
        dim = np.ones(1)

    if dim != 1:
        if not any(np.asarray(np.shape(Data))==dim):
            print('Number of dimensions in the model output {} does not match any dimension of data {}'.format(dim,np.shape(Data)))
            return
    n_data = int(np.size(Data)/dim[0])
    y = np.reshape(np.asarray(y),[int(dim[0]),n_model])
    Data = np.reshape(Data, [int(dim[0]),int(np.size(Data)/dim[0])])

    if not epsilon and not epsilon_r:
        epsilon_r = np.std(Data,axis =1)
    elif epsilon_r:
        epsilon_r = epsilon_r
    else:
        epsilon_r = np.ones(int(dim[0]))* epsilon

    LogL =[]
    if dim[0] ==1:
        for i in range(n_data):
            LogL.append(np.sum(( p_x_theta_pdf(Data[:,i], y, epsilon_r, dim[0]))))
    else:
        for i in range(n_data):
            LogL.append(np.sum(( p_x_theta_pdf2(Data[:,i], y, epsilon_r, dim))))
    #LogL = np.array(LogL)
    if np.isinf(LogL).any():
        LogL = -1e10
    return np.sum(LogL)
"""
Acceptance rule
"""
#Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        # Expenciate to compensate for log of likelihood
        return (accept < (np.exp(x_new-x)))

def acceptance_rate(n_accept,nsamples,thin,burnin):
    return n_accept/(nsamples*thin+burnin)

def rejection_rate(n_rej,nsamples,thin,burnin):
    return n_rej/(nsamples*thin+burnin)
"""
Log Tool
"""
def mylog(x):
    from math import inf
    # Define to avoid the warnings.
    x = np.array(x)
    x = np.reshape(x,[np.size(x)])
    y = - inf * np.ones(np.size(x))
    y[x>0] = np.log(x[x>0])
    return y

"""
Metropolis Hastings
"""
def metropolis_hastings_simple(likelihood_computer,prior, transition_model, param_init,iterations,data,acceptance_rule):
    """
    # likelihood_computer(x,data): returns the likelihood that these parameters generated the data
    # transition_model(x)        : a function that draws a sample from a symmetric distribution and returns it
    # param_init                 : a starting sample
    # iterations                 : number of accepted to generated
    # data                       : the data that we wish to model
    # acceptance_rule(x,x_new)   : decides whether to accept or reject the new sample

           By Dominic Calleja
          University of Liverpool
    """

    x = param_init
    accepted = []
    rejected = []
    for i in range(iterations):
        x_new =  transition_model(x)
        x_lik = likelihood_computer(x,data)
        x_new_lik = likelihood_computer(x_new,data)
        if (acceptance_rule(x_lik + np.log(prior(x)),x_new_lik+np.log(prior(x_new)))):
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)

    return np.array(accepted), np.array(rejected)

#logpdf = log_fj1, proppdf = propdf, proprnd = prop_rnd
def metropolis_hastings(log_pdf, start, proprnd, proppdf,nsamples=1, nchain = 1,burnin=0 ,thin=1):
    """
    # log_pdf(x,data): returns the likelihood that these parameters generated the data
    # start          : a starting sample
    # fT(x)          : a function that draws a sample from a symmetric distribution and returns it
    # prop_rnd       : proposal sampler
    # prop_pdf       : proposal pdf (check prior)

           By Dominic Calleja
          University of Liverpool
    """
    x0 = start
    distnDims = np.shape(start)[0]

    U = np.log(np.random.rand(nchain,nsamples*thin+burnin))
    accepted = []
    rejected = []
    a = 0
    smpl =[]
    for i in range(0-burnin,nsamples*thin):
        #proposal distribution--------------------------------------------------------------------------
        #prop_pdf = lambda x,y : proppdf(x, y, cov_gauss, fT)  #q(x,y) = q(x|y).
        #prop_rnd = lambda x : proprnd(x, cov_gauss, fT)    #mvnrnd(x, cov_gauss, 1)
        y = proprnd(x0)#, cov_gauss, fT)   #prop_rnd(x0)
        #if i>=0:
        #print('loop-{} x0:{}\n'.format(i,x0))
            #print('cov-{} \n'.format(cov_gauss))
        q1 = proppdf(x0, y)#, cov_gauss, fT) # log_prop_pdf(x0,y)
        q2 = proppdf(y, x0)#, cov_gauss, fT) # log_prop_pdf(y,x0)
        q1 = mylog(q1)
        q2 = mylog(q2)

        x_lik = log_pdf(y)
        x_new_lik = log_pdf(x0)

        rho = (q1+x_lik) - (q2+x_new_lik)
    #    else:
    #        rho = log_pdf(y) - log_pdf(x0)
#        Ui = U[:,i+burnin]
#        acc = Ui<=min(rho,0)
#        if acc :
#            x0 = y
#        accepted = accepted + acc
        if U[:,i+burnin] <=min(rho,0):
            x0 = np.array(y)
            accepted.append(y)
            a = a + 1
        else:
            rejected.append(y)

        if np.logical_and(i>=0,np.mod(i,thin)==0):
            smpl.append(x0)
        # Accept rate can be used to optimize the choice of scale parameters in
        # random walk MH sampler. See for example Roberts, Gelman and Gilks (1997)

    acceptance_rate = a/(nsamples*thin+burnin)
    return smpl, acceptance_rate

"""
markov for par_tmcmc
"""


def parallel_markov_chain(fT, log_fD_T, SIGMA_j, THJ, LPJ, a_j1, length, burnin=None):
    Th_lead = THJ
    Lp_lead = LPJ
    Th_new = []
    Lp_new = []
    a = 0
    time_stopper = time.time()
    for l in range(0-burnin, length.astype(int)):
        #------------------------------------------------------------------
        # Candidate sample generation (normal over feasible space)
        #------------------------------------------------------------------
        while True:
            Th_cand = stats.multivariate_normal.rvs(Th_lead, SIGMA_j)

            t_delta = time.time()-time_stopper

            if t_delta/60 > 45:
                print('\n\n\n WARNING: GOT VERY STUCK!!!!!!!! \n\n\n')
                
                Th_lead = np.ones(np.shape(Th_lead))*np.random.uniform(2,8)
                print('moving sample to {}'.format(Th_lead))
                
            if not fT(Th_cand) == 0:
                break
        #------------------------------------------------------------------
        # Log-likelihood of candidate sample
        #------------------------------------------------------------------
        if fT(Th_cand) == 0:
            GAMMA = 0
            Lp_cand = Lp_lead
        else:
            Lp_cand = log_fD_T(Th_cand)
            GAMMA = np.exp(a_j1*(Lp_cand - Lp_lead))*fT(Th_cand)/fT(Th_lead)
        #------------------------------------------------------------------
        # Rejection step
        #------------------------------------------------------------------
        thresh = np.random.rand()
        if thresh <= min(1, GAMMA):
            Th_lead = Th_cand
            Lp_lead = Lp_cand
            a = a+1
        if l >= 0:
            if thresh <= min(1, GAMMA):
                Th_new.append(Th_cand)
                Lp_new.append(Lp_cand)
            else:
                Th_new.append(Th_lead)
                Lp_new.append(Lp_lead)
    return np.array(Th_new), np.reshape(np.asarray(Lp_new), [length.astype(int)]), np.array(a / (burnin+length))


def markov_chain_seed(wn_j_csum, Nm):
    # Definition of Markov chains: seed sample
    mkchain_ind = np.zeros(Nm)
    for i_mc in range(Nm):
        #while True:
        mkchain_ind[i_mc] = np.argwhere(np.random.rand() < wn_j_csum)[0]

    seed_index = np.unique(mkchain_ind).astype(int)
    N_Mc = np.size(seed_index)
    return N_Mc, seed_index, mkchain_ind
"""
Tempering parameter
"""


def calculate_pj_alpha(log_fD_T_thetaj, alpha_j):
    #----------------------------------------------------------------------
    # choose pj    (Bisection method)
    #----------------------------------------------------------------------
    low_alpha = alpha_j
    up_alpha = 2
    Lp_adjust = np.max(log_fD_T_thetaj)

    while ((up_alpha - low_alpha)/((up_alpha + low_alpha)/2)) > 1e-6:
        x1 = (up_alpha + low_alpha)/2
        wj_test = np.exp((x1-alpha_j)*(log_fD_T_thetaj-Lp_adjust))
        cov_w = np.std(wj_test)/np.mean(wj_test)
        if cov_w > 1:
             up_alpha = x1
        else:
            low_alpha = x1
    alpha_j = min(1, x1)
    return alpha_j, Lp_adjust
"""
def calculate_pj_alpha(log_fD_T_thetaj,alpha_j):

    #----------------------------------------------------------------------
    # choose pj    (Bisection method)
    #----------------------------------------------------------------------

    low_alpha = alpha_j;
    up_alpha = 2;
    Lp_adjust = max(log_fD_T_thetaj);

    while ((up_alpha - low_alpha)/((up_alpha + low_alpha)/2)) > 1e-6:
        x1 = (up_alpha + low_alpha)/2
        wj_test = np.exp((x1-alpha_j)*(log_fD_T_thetaj-Lp_adjust))
        cov_w   = np.std(wj_test)/np.mean(wj_test);
        if cov_w > 1:
             up_alpha = x1
        else:
            low_alpha = x1
    alpha_j = min(1,x1)
    return alpha_j
"""
def calculate_pj1(log_fD_T_thetaj, pj):
    """
    #----------------------------------------------------------------------
    # choose pj    (minimization method)
    #----------------------------------------------------------------------
    """
    #  std(wj)
    # --------- <= threshold
    #  mean(wj)
    threshold = 1
    wj = lambda e: np.exp(abs(e)*log_fD_T_thetaj)
    fmin = lambda e: np.std(wj(e)) - threshold * np.mean(wj(e)) +  np.finfo(np.double).tiny

    e = abs(optimize.fsolve(fmin, 0))

    if np.isnan(e):
        print('There is an error finding e')
        return

    pj1 = np.min([1, pj + e])
    return pj1


"""
TMCMC algorithm
"""

def tmcmc_par(log_fD_T, fT, sample_fT, n_per_layer=[], n_batches=[], n_parameters=[], max_n_stages=100, burnin=0, lastburnin=0, beta2=0.01, beta2_b=[], alpha_threshold=[], log_fd_T2=[], process='ray', pickle_path=[],redis_address=[], redis_password=[],num_cpus=[], seq_ind=[], save_address=[],launch_remote_workers=False,logfile='tmcmc_updating_sf_logfile.txt'):
    # Transitional Markov Chain Monte Carlo
    #
    # Usage:
    # [samples_fT_D, fD] = tmcmc(fD_T, fT, sample_from_fT, N)

    # inputs:
    # log_fD_T       = function handle of log(fD_T(t))
    # fT             = function handle of fT(t)
    # sample_from_fT = handle to a function that samples from of fT(t)
    # N              = number of samples of fT_D to generate
    # n_batches      = doubles as n_cores if n_batches>n_cores
    # parallel       = True (default)
    # burnin         = uniform burn in on each iteration for metropolis_hastings
    # lastburnin     = burnin on final iteration to ensure sampling from posterior
    # thining        = intermitten acceptance criterion
    # beta           = is a control parameter that is chosen to balance the
    #                   potential for large MCMC moves while mantaining a
    #                   reasonable rejection rate
    # outputs:
    # samples_fT_D   = samples of fT_D (N x D)
    # log_fD         = log(evidence) = log(normalization constant)
    #
    # This program implements a method described in:
    # Ching, J. and Chen, Y. (2007). "Transitional Markov Chain Monte Carlo
    # Method for Bayesian Model Updating, Model Class Selection, and Model
    # Averaging." J. Eng. Mech., 133(7), 816-832.

  # Square of scaling parameter(MH algorithm)
    #Preallocation of number of stages

    # PRE_ALLOCATION
    Th_j = []  # cell(max_n_stages, 1)
    alpha_j = np.zeros(max_n_stages)
    Lp_j = []
    Log_S_j = np.zeros(max_n_stages)
    wn_j = []
    n_per_batch = n_per_layer/n_batches
    acceptance = []
    t = time.time()
    #logfile = 'tmcmc_updating_sf_logfile.txt'
    if seq_ind == 0 or launch_remote_workers:
        copyR(logfile)
    outputf=open(logfile, 'a')

    # import libraries
    # multiprocessing setup
    #if parallel:
    import pathos.multiprocessing as mp

    Ncores = min(mp.cpu_count(), n_batches)
    print('TMCMC is running on {} cores'.format(Ncores))
    outputf.write('TMCMC is running on {} cores\n'.format(Ncores))
    outputf.write('seq ind : {}\n'.format(seq_ind))

    j = 0
    alpha_j[0] = 0
    # sample from prior
    Th_j.append(sample_fT(n_per_layer))
    Th0 = Th_j[0]

    if process == 'pathos':
    #par likelihood
        p = mp.Pool(Ncores)
        print('Executing processing with pathon. WARNING: if model is heavy the initialisation time may be very long!')
        def par_like(theta): return log_fD_T(Th0[:, theta])
        Lp0 = p.map(par_like, range(n_per_layer))
        Lp0 = np.array(Lp0)
        Lp0 = np.reshape(Lp0, [n_per_layer])
    if process == 'ray':
        if seq_ind ==0 or launch_remote_workers:
            if redis_password:
                ray.init(address="auto") #, _redis_password=redis_password)#os.environ["ip_head"]
                #ray.init(address='auto', _redis_password=redis_password)
                Ncores = num_cpus
            else:
                ray.init(num_cpus=Ncores, lru_evict=True)
        print('Executing processing with Ray. WARNING: if model is heavy there may be memory issues!')
        t_like = time.time()
        print("RAY NODES: {}".format(ray.nodes))
        actors = [smardda_tmcmc.remote(
            i, fT, sample_fT, log_fD_T, n_per_layer, Ncores, n_parameters, pickle_path, seq_ind, save_address) for i in range(Ncores)]
        theta_b = np.split(Th0, Ncores, axis=1)
        Lp0 = ray.get([actor._batch_likelihood.remote(
            theta_b[a][:, range(np.shape(theta_b[0])[1])]) for a, actor in enumerate(actors)])
        
        try:
            Lp0 = np.reshape(Lp0,[n_per_layer])
        except:
            Lp0 = np.concatenate(Lp0)
        print('Completed likelihood evaluation. Time Elapsed : {}'.format(timestamp_format(t_like,time.time())))
        outputf.write('Completed likelihood evaluation. Time Elapsed : {}\n'.format(timestamp_format(t_like,time.time())))
        

    Lp0[Lp0 == -np.inf] = -1E5
    Lp_j.append(Lp0)
    outputf.close()
    while alpha_j[j] < 1:
        outputf=open(logfile, 'a')
        if j+1 == max_n_stages:
            print('Reached limit of stages {}. Terminating run without convergence'.format(max_n_stages))
            outputf.write('Reached limit of stages {}. Terminating run without convergence'.format(max_n_stages))
            break

        t1 = time.time()
        print('TMCMC: Iteration j = {}'.format(j))
        outputf.write('TMCMC: Iteration j = {}\n'.format(j))
        # Find tempering parameter
        print('Computing the tempering ...')
        outputf.write('Computing the tempering ...\n')

        alpha_j[j+1], Lp_adjust = calculate_pj_alpha(Lp_j[j], alpha_j[j])
        print('TMCMC: Iteration j = {}, pj1 = {}'.format(j, alpha_j[j+1]))
        print('Computing the weights ...')
        outputf.write('TMCMC: Iteration j = {}, pj1 = {}\n'.format(j, alpha_j[j+1]))
        outputf.write('Computing the weights ...\n')
        #Adjusted weights
        w_j = np.exp((alpha_j[j+1]-alpha_j[j])*(Lp_j[j]-Lp_adjust))

        print('Computing the evidence ...')
        outputf.write('Computing the evidence ...\n')
        #Log-evidence of j-th intermediate distribution
        Log_S_j[j] = np.log(np.mean(np.exp(
            (Lp_j[j]-Lp_adjust)*(alpha_j[j+1]-alpha_j[j]))))+(alpha_j[j+1]-alpha_j[j])*Lp_adjust

        #Normalized weights
        wn_j.append(w_j/(np.sum(w_j)))
        print('Computing the covariance ...')
        outputf.write('Computing the covariance ...\n')

        # Weighted mean and coviariance
        if alpha_j[j+1] > alpha_threshold:
            beta2 = beta2_b
            #log_fD_T = log_fd_T2
            print('Adaptive likelihood: Switching Beta')

        Th_wm = np.matrix(Th_j[j]) * np.matrix(wn_j[j]).T
        SIGMA_j = np.zeros([n_parameters, n_parameters])
        for l in range(n_per_layer):
            SIGMA_j = SIGMA_j + beta2 * \
                wn_j[j][l] * (Th_j[j][:, l] - Th_wm.T).T * \
                (Th_j[j][:, l] - Th_wm.T)
        SIGMA_j = (SIGMA_j.T+ SIGMA_j)/2
        # Metropolis Hastings
        print('Inititialising Metropolis Hastings ...')
        outputf.write('Inititialising Metropolis Hastings ...\n')
        wn_j_csum = np.cumsum(wn_j[j])
        n_mc, seed_index, mkchain_ind = markov_chain_seed(
            wn_j_csum, n_per_layer)
        #print(seed_index)
        lengths = np.zeros(np.shape(seed_index))

        for i_mc in range(lengths.size):
            lengths[i_mc] = np.sum(seed_index[i_mc] == mkchain_ind)

        # Improve posterior sampling:
        if alpha_j[j+1] == 1:
            burnin = lastburnin

        # Preallocation
        a_j1 = alpha_j[j+1]
        THJ = Th_j[j][:, seed_index]
        LPJ = np.array(Lp_j[j])[seed_index]
        results = []

        print('Markov chains ...')
        outputf.write('Markov chains ...\n')
        outputf.close()
        outputf = open(logfile, 'a')
        if process == 'pathos':
            print('Executing mc with pathos')
            def func(t): return parallel_markov_chain(fT, log_fD_T, SIGMA_j,
                                                    THJ[:, t], LPJ[t], a_j1, lengths[t], burnin=burnin)
            results = p.map(func, range(n_mc))
            print('Formatting outputs ...')
            outputf.write('Formatting outputs ...\n')

            Th_j_tmp = results[0][0]
            Lp_j_tmp = results[0][1]
            acc_rate = []
            acc_rate.append(results[0][2])
            for i in range(1, n_mc):
                Th_j_tmp = np.concatenate((Th_j_tmp, results[i][0]), axis=0)
                Lp_j_tmp = np.concatenate((np.reshape(Lp_j_tmp, [len(Lp_j_tmp), 1]), np.reshape(results[i][1], [len(results[i][1]), 1])), axis=0)
                acc_rate.append(results[i][2])
            Th_j.append(Th_j_tmp.T)
            Lp_j.append(np.squeeze(np.array(Lp_j_tmp,dtype=float)))
            acceptance.append(acc_rate)
        if process == 'ray':
            print('Executing mc with ray')
            #SIGMA_j, THJ, LPJ, a_j1, length, burnin
            ind_mc = np.asarray(range(n_mc))
            batch_ind = np.array_split(ind_mc,Ncores)
            #return SIGMA_j, THJ, LPJ, batch_ind, actors
            results = ray.get([actor.markov_chain.remote(SIGMA_j,THJ[:,batch_ind[a]],
                                                        LPJ[batch_ind[a]],a_j1,lengths[batch_ind[a]],burnin) for a, actor in enumerate(actors)])

            print('Formatting outputs ...')
            outputf.write('Formatting outputs ...\n')
            r = []
            [r.append(np.concatenate(results[i][0], axis=0)) for i in range(len(results))]
            Th_j_tmp = np.concatenate(r, axis=0)
            l = []
            [l.append(np.hstack(results[i][1])) for i in range(len(results))]
            Lp_j_tmp = np.concatenate(l)
            Lp_j_tmp = np.array(Lp_j_tmp, dtype=float)
            a = []
            [a.append(results[i][2]) for i in range(len(results))]
            acc_rate = np.concatenate(a, axis=0)

            Th_j.append(Th_j_tmp.T)
            Lp_j.append(Lp_j_tmp)
            acceptance.append(acc_rate)

        print('TMCMC: Iteration j = {} complete. Time Elapsed : {} \n\n'.format(
            j, timestamp_format(t1,time.time())))
        outputf.write('TMCMC: Iteration j = {} complete. Time Elapsed : {} \n\n'.format(
            j, timestamp_format(t1, time.time())))
        outputf.write('+'+'='*77+'+ \n')
        j = j+1
        outputf.close()
    outputf=open(logfile, 'a')
    m = j
    print('TMCMC Complete: Evaluated posterior in {} iterations. Time Elapsed {} '.format(
        m, timestamp_format(t,time.time())))
    outputf.write('TMCMC Complete: Evaluated posterior in {} iterations. Time Elapsed {} '.format(
        m, timestamp_format(t,time.time())))
    outputf.write('+'+'='*77+'+ \n')

    Th_posterior = Th_j[m]
    Lp_posterior = Lp_j[m]
    #ray.shutdown()
    outputf.close()
    return Th_posterior,  Lp_posterior, acceptance, Th_j, Lp_j


def tmcmc(log_fD_T, fT, sample_from_fT, N, n_batches=1, parallel=True, burnin = 0, lastburnin = 100, thinning = 1 , beta = 0.2, store=False, plot=False,**kwargs):
    """
    # Transitional Markov Chain Monte Carlo
    #
    # This program implements a method described in:
    # Ching, J. and Chen, Y. (2007). "Transitional Markov Chain Monte Carlo
    # Method for Bayesian Model Updating, Model Class Selection, and Model
    # Averaging." J. Eng. Mech., 133(7), 816-832.
    #
    # Usage:
    # [samples_fT_D, fD] = tmcmc(fD_T, fT, sample_from_fT, N)

    # inputs:
    # log_fD_T       = function handle of log(fD_T(t))
    # fT             = function handle of fT(t)
    # sample_from_fT = handle to a function that samples from of fT(t)
    # N              = number of samples of fT_D to generate
    # n_batches      = doubles as n_cores if n_batches>n_cores
    # parallel       = True (default)
    # burnin         = uniform burn in on each iteration for metropolis_hastings
    # lastburnin     = burnin on final iteration to ensure sampling from posterior
    # thining        = intermitten acceptance criterion
    # beta           = is a control parameter that is chosen to balance the
                       potential for large MCMC moves while mantaining a
                       reasonable rejection rate
    # outputs:
    # samples_fT_D   = samples of fT_D (N x D)
    # log_fD         = log(evidence) = log(normalization constant)

    """
    t = time.time()
    # import libraries
    # multiprocessing setup
    if parallel:
        import pathos.multiprocessing as mp
        Ncores = min(mp.cpu_count(),n_batches)
        p = mp.Pool(Ncores)
        print('TMCMC is running on {} cores'.format(Ncores))

    """
    Constants
    """
    if store:
        if 'filename' in kwargs:
            filename = kwargs['filename']
        else:
            if not os.path.isdir(os.getcwd()+'/tmp_stored_pickles/'):
                os.mkdir(os.getcwd()+'/tmp_stored_pickles/')
            filename = os.getcwd()+'/tmp_stored_pickles/'

    S    = np.ones(100)     #
    with_replacement = True  # DO NOT CHANGE!!!
    plot_graphics    = False

    # Obtain N samples from the prior pdf f(T)
    j      = 0
    thetaj = sample_from_fT(N)  # theta0 = N x D
    pj     = 0                  # p0 = 0 (initial tempering parameter)
    D      = np.shape(thetaj)[0]    # size of the vector theta

    #plot list (it takes a long time to plot so its worth not plotting each step)
    plotlist = range(0,100,5)
    # Initialization of matrices and vectors
    thetaj1   = np.zeros([D,N])
    log_fD_T_thetaj = np.zeros(np.shape(thetaj))
    summary_data = {}
    # Main loop
    while pj < 1:
        t1 = time.time()
        j = j+1

        # Calculate the tempering parameter p(j+1):
        #This part can be parallelised
        #n_batches = 40
        n_per_batch = N/n_batches  # Must be an integer, and partition N exactly
        if not n_per_batch == int(n_per_batch):
            print('Error: n_per_batch must be int! its {}'.format(n_per_batch))
            return
        n_per_batch = int(n_per_batch)
        thetajIn = np.reshape(thetaj,(D,n_per_batch,n_batches))
        log_fD_T_thetajIn = np.zeros([n_per_batch, n_batches])

        """
        parallel execution of the likelihood
        """

        if parallel:
            print('Evaluating the likelihood in {} batches on {} cores.'.format(n_batches,Ncores))
            func = lambda t: execute_function(n_per_batch,log_fD_T,thetajIn[:,:,t])
            #job_args = [(for i in ]
            #[(log_fj1, thetaj[:,idx[i]], cov_gauss, fT) for i in range(N)]
            results = p.map(func, range(n_batches))
            log_fD_T_thetaj = np.reshape(results,[N])
        else:
            log_fD_T_thetajIn =[]
            for jk in range(n_batches):
                log_fD_list =[]
                for kk in range(n_per_batch):
                    log_fD_list.append(log_fD_T(thetajIn[:,kk,jk]))
                log_fD_T_thetajIn.append(log_fD_list)
            log_fD_T_thetaj = np.reshape(log_fD_T_thetajIn,[N])

        if plot:
            if j in plotlist:
                plt.figure()
                scatterplot_matrix(np.reshape(thetajIn,[D,N]), c=log_fD_T_thetaj)
                plt.show()

        likelihood_time_elapsed = time.time() - t1
        if np.any(np.isinf(log_fD_T_thetaj)):
            print('The prior distribution is too far from the true region')
            return
        t2 = time.time()

        pj1 = calculate_pj_alpha(log_fD_T_thetaj, pj)
        print('TMCMC: Iteration j = {}, pj1 = {}'.format(j, pj1))
        print('Completed likelihood evaluation. Time elapsed {:.4f} (sec)'.format(likelihood_time_elapsed))
        #Compute the plausibility weight for each sample wrt f_{j+1}
        print('Computing the weights ...')
        # wj     = fD_T(thetaj).^(pj1-pj)         % N x 1 (eq 12)
        a       = (pj1-pj)*log_fD_T_thetaj
        wj      = np.exp(a)
        wj_norm = wj/np.sum(wj)                # normalization of the weights

        # Compute S(j) = E[w{j}] (eq 15)
        S[j] = np.mean(wj)

        # Do the resampling step to obtain N samples from f_{j+1}(theta) and
        # then perform Metropolis-Hastings on each of these samples using as a
        # stationary PDF "fj1"
        # fj1 = @(t) fT(t).*log_fD_T(t).^pj1   % stationary PDF (eq 11) f_{j+1}(theta)
        log_fj1 = lambda t: np.log(fT(t)) + pj1*log_fD_T(t)

        # weighted mean
        mu = np.zeros(D)
        for l in range(N):
            mu = mu + wj_norm[l] * thetaj[:,l]  # 1 x N

        # scaled covariance matrix of fj1 (eq 17)
        cov_gauss = np.zeros([D,D])
        for k in range(N):
            # this formula is slightly different to eq 17 (the transpose)
            # because of the size of the vectors)m and because Ching and Chen
            # forgot to normalize the weight wj:
            tk_mu = thetaj[:,k] - mu
            cov_gauss = cov_gauss + wj_norm[k] * np.matrix(tk_mu).T * np.matrix(tk_mu)

        cov_gauss = beta**2 * cov_gauss
        if np.isinf(np.linalg.cond(cov_gauss)):
            print('Something is wrong with the likelihood.')
            return

        prop_rnd = lambda x: proprnd(x, cov_gauss, fT)
        prop_pdf = lambda x,y:  proppdf(x, y, cov_gauss, fT)

        # improve posterior sampling :
        if pj1 == 1:
            burnin = lastburnin

        # Start N different Markov chains
        weights_time_elapsed = time.time() - t2
        t3 = time.time()
        print('Computed weights and covariance matrix. Time elapsed {:.4f} (sec)'.format(weights_time_elapsed))
        print('Markov chains ...')

        idx = np.random.choice(N, N, replace=with_replacement,p=wj_norm)
        """
        Execution of the mhsample
        """
        if parallel:
                func = lambda t: mhparallelsample(log_fj1, thetaj[:,idx[t]], prop_rnd, prop_pdf,1, 1, burnin, thinning)
                #job_args = [(for i in ]
                #[(log_fj1, thetaj[:,idx[i]], cov_gauss, fT) for i in range(N)]
                results = p.map(func, range(N))
                results = np.array(results)
                thetaj1 = results.squeeze()

        else:                   #logpdf = log_fj1, proppdf = propdf, proprnd = prop_rnd
            acceptance_rate =[]
            for i in range(N):  #log_prop_pdf, log_pdf, prop_rnd, start, nsamples,
                thet, a1 = metropolis_hastings(log_fj1, thetaj[:,idx[i]], prop_rnd, prop_pdf,nsamples=1, nchain=1, burnin=burnin, thin=thinning)
                if i ==0:
                    thetaj1 = [thet[0]]
                else:
                    thetaj1 = np.concatenate((thetaj1,[thet[0]]),axis=0)
                acceptance_rate.append(a1)

        thetaj = thetaj1.T
        pj     = pj1
        mc_time_elapsed = time.time() - t3
        print('Completed evaluation of Markov Chains. Time elapsed {:.2f} (sec)'.format(mc_time_elapsed))
        print('N samples: {} - Burnin: {} - Thining: {} '.format(N,burnin,thinning))
        print('Completed evaluation of iteration {}. Time elapsed {:.4f} (min)\n'.format(j,(time.time()-t1)/60))

        if store:
            t4 = time.time()
            print('writing data to disk ...')
            save_iteration(filename,summary_data,j,pj,thetaj,mu,cov_gauss,log_fD_T_thetaj,t1)
            print('Completed data storage for iteration. Time elapsed {:.2f}\n'.format(time.time()-t4))

    # TMCMC provides N samples distributed according to f(T|D)
    samples_fT_D = thetaj
    #estimation of f(D) -- this is the normalization constant in Bayes
    log_fD = np.sum(np.log(S[range(j)]))
    print('Completed evaluating the likelihood in {} batches.'.format(n_batches))
    print('Computed TMCMC evaluation. Total time elapsed {:.4f} (min))\n\n'.format((time.time()-t)/60))
    return samples_fT_D, log_fD

"""
Auxillary parallel functions
"""

# def auxillary_parallel_execution(args):
#    Auxillary function to allow pool execution with multiple input arguments
    # return execute_function(*args)

def execute_function(n_per_batch,log_fD_T,thetaIn):
    log_fD_T_thetajIn = []
    for kk in range(n_per_batch):
        log_fD_T_thetajIn.append(log_fD_T(thetaIn[:,kk]))
    return log_fD_T_thetajIn

#def auxillary_metropolis(args):
#    return mhparallelsample(*args)

def mhparallelsample(log_fj1, thetaj, prop_rnd, prop_pdf,nsamples, nchain, burnin, thinning):
    A,ax = metropolis_hastings(log_fj1, thetaj, prop_rnd, prop_pdf,nsamples=nsamples, nchain=nchain, burnin=burnin, thin=thinning)
    return A

def xrange(x):
    return iter(range(x))
_func = None

"""
Auxillary save
"""
def save_iteration(filename,summary_data,j,pj,thetaj,mu,cov_gauss,log_fD_T_thetaj,t1):
    summary_data[j] = {'pj':pj,
                       'thetaj':thetaj,
                       'mu':mu,
                       'cov':cov_gauss,
                       'likelihood':log_fD_T_thetaj,
                       't_evaluation':time.time()-t1}
    with open(os.path.join(filename,'Iteration{}.pickle'.format(int(j))), 'wb') as handle:
        pickle.dump(summary_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#def worker_init(func):
#  global _func
#  _func = func

#def worker(x):
#  return _func(x)
"""
Cauchy Deviates
"""
"""def CD(self,M):
    # cauchydeviatesimulation(n,f,x):
    from random import random
    from math import pi,tan
    from scipy.optimize import fsolve
        # dim = len(x)
    dim = self.__dim
    x = self.__bunchofintervals
    f = self.__function

    xtilda = [(xi[0]+xi[1])/2 for xi in x]
    Delta = [(xi[0]-xi[1])/2 for xi in x]
    ytilda = f(xtilda)

    DELTA = []
    DELTA_append = DELTA.append

    X = []
    Y = []

    X_append = X.append
    Y_append = Y.append

    for i in range(M):
        r = [random() for _ in range(dim)]
        c = [tan(pi*(ri-0.5)) for ri in r]
        K = max(c)
        delta = [Delta_i * c_i / K for Delta_i,c_i in zip(Delta,c)]
        x_cd = [xtilda_i - delta_i for xtilda_i,delta_i in zip(xtilda,delta)]
        X_append(x_cd)
        y_cd = f(x_cd)
        Y_append(y_cd)
        DELTA_append( K * (ytilda - y_cd) )

        sold = fsolve(Z, random()*max(DELTA)/2, full_output=1)
    solution = float(sold[0])

    if solution>0:
        interval = [ytilda-solution, ytilda+solution]

    else:
        interval = [ytilda+solution, ytilda-solution]

    Ymin = interval[0]
    Ymax = interval[1]
    return X,Y,Ymin,Ymax

def Z(y):
    return M/2 - sum([1/(1+(Di/y)**2) for Di in DELTA])
"""

"""
Plot
"""

def scatterplot_matrix(data, names =[], **kwargs):
    import itertools
    import matplotlib.pyplot as plt
    numvars, numdata = np.shape(data)
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].scatter(data[x], data[y], **kwargs)

    # Label the diagonal subplots...
    for i in range(numvars):
        axes[i,i].hist(data[i],alpha=0.6)
        #annotate(label, (0.5, 0.5), xycoords='axes fraction',ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig

def test_confidence():
    alpha = .95
    Ms =80
    #data = np.zeros([2,Ms])
    data = np.random.randn(Ms)
    data = np.sort(data)
    #data[1] = data[0] + 0.01

    t = []
    for i in range(100):
        t= np.random.randn(Ms)+0.005
        t = np.sort(t)
        plt.step(t, np.linspace(0,1,Ms), alpha=0.2, color='green')

    L,R,xi = confidence_limits_distribution(data,alpha,interval=False,plot=True)


def uniform_prior_marginal(LB, UB):
    def func(x): return stats.uniform.rvs(LB, UB, size=x)
    return func


def joint_prior(bounds):
    F = []
    for i in range(len(bounds)):
        F.append(uniform_prior_marginal(bounds[i][0], bounds[i][1]))
    return F


def sample_prior(F, x):
    samp = []
    for f in F:
        samp.append(f(x))
    return np.array(samp)


"""
construct_prior
"""


def prior_marginal(bounds):
    def func(x): return stats.uniform.pdf(x, bounds[0], bounds[1])
    return func


def product_prior(l1, l2, i):
    def y(x): return l1(x[1]) * l2(x[0])
    return y


def prior_strictly_positive(w):

    if(w[0] <= 0 or w[1] <= 0):
        return 0
    else:
        return 1

"""
log file
"""


def timestamp_format(t0, t1):
    t_delta = t1-t0

    if t_delta/60 < 1:
        time = '{:.3f} (sec)'.format(t_delta)
    elif t_delta/60 > 1 and t_delta/60/60 < 1:
        time = '{:.3f} (min)'.format(t_delta/60)
    else:
        time = '{:.3f} (hrs)'.format(t_delta/60/60)

    return time

def copyR(logfile):
    """Print copyright information to file."""
    outputf=open(logfile, 'a')
    outputf.write('+'+'='*77+'+ \n')
    tl='Updating4Smardda'
    sp1=(77-len(tl))//2
    sp2=77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|'+ '\n')
    tl='Bayesian updating tools for Smardda'
    sp1=(77-len(tl))//2
    sp2=77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|'+ '\n')
    tl=' '
    sp1=(77-len(tl))//2
    sp2=77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|'+ '\n')
    tl=' Version: '+__version__
    sp1=(77-len(tl))//2
    sp2=77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|'+ '\n')
    outputf.write('|'+' '*77+'| \n')
    tl=__copyright__+' (c) '+ __author__
    sp1=(77-len(tl))//2
    sp2=77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|'+ '\n')
    outputf.write('+'+'='*77+'+'+ '\n')
    outputf.write('\n')
    outputf.close()
    return



"""
# Plot the sampled points
if (plot_graphics):
    fig= plt.subplots(1)
    fig.plot(thetaj[0], thetaj[1], 'b')

    ax=[np.min(thetaj[0]),np.max(thetaj[0]),np.min(thetaj[1]),np.max(thetaj[1])]
    xx, yy = np.meshgrid(np.linspace(ax[1],ax[2],100), np.linspace(ax[3], ax[4], 99))

    if j == 0:
        zz = np.reshape(fT([xx, yy]), 99, 100)
    else:
        zz = np.reshape(fj1([xx, yy]), 99, 100)

    fig.contour([xx, yy], zz, [50])
    fig.title('Samples of f_{} and contour levels of f_{} (red) and f_{} (black)'.format(j, j, j+1))

        if plot_graphics:
            # In the definition of fj1 we are including the normalization
            # constant prod(S(1:j))
            fj1 = lambda t: np.exp(np.log(fT(t)) + pj1*log_fD_T(t) - np.sum(np.log(S(range(0,j)))))
            zz = np.reshape(fj1([xx, yy]), 99, 100)
            plt.contour([xx, yy], zz, 50, 'k')

        # and using as proposal PDF a Gaussian centered at thetaj(idx,:) and
        # with covariance matrix equal to an scaled version of the covariance
        # matrix of fj1:

"""

if __name__ == '__main__':
    test_confidence()


    ###
    #   Tests samples. Give sampler a known density
    ###
    import sys
    #import smardda_updating as up
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt


    D = 5
    covmatReal = np.identity(8)
    muReal1 = [D,D,D,D,D,D,D,D]               #   Top octant
    muReal2 = [-D,-D,-D,-D,-D,-D,-D,-D]        #   Bottom octant

    # Denity of 2 gaussians at each corner
    target = lambda x: np.log(stats.multivariate_normal.pdf(x, muReal1, np.matrix(covmatReal)) + stats.multivariate_normal.pdf(x, muReal2, np.matrix(covmatReal)))


    def boxSampleFunc(N, limit):
        samps = stats.uniform.rvs(-limit, 2*limit, size=[8,N])
        return samps

    def boxFunc(x, limit):
        x = np.array(x)
        if (x > limit).any():
            return 0
        if (x < -limit).any():
            return 0
        return 1

    bounds = 10
    box = lambda x: boxFunc(x,bounds)
    boxSample = lambda N: boxSampleFunc(N,bounds)

    Nsamples = 200

    samples,logJ = tmcmc(target, box, boxSample, Nsamples,parallel=False, burnin = 20, lastburnin = 20,beta=2,plot=True)
    plt.figure()
    scatterplot_matrix(samples)
