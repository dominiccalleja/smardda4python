#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:50:29 2018
This function wraps the required smardda executable functions, there is no modification to the underlying smarrda modules.

Smardda4python
"""
# '\'
"""
Version history:

S4P  0.3: Inclusion of the new Parallel execution class, and complete set of
        logfile write
S4P  0.2: Minor updates to the original class.
        TO DO : Must modify the powcalexecute function to take inputs suitable
        for the bayesian model updating toolbox. (or other seqential simulation
        workflows)
S4P  0.1: Implimented the first version of Smardda for Python.
            Wrapping the essential smardda modules in simple toexecute Smardda
            modules in pyton functions.
            This implimentation modifies some standard ctl template files and
            exeuctes smardda as operating system commands.
"""

import pickle
__author__ = "Dominic Calleja"
__copyright__ = "Copyright 2018"
__credits__ = ["Wayne Arter"]
__license__ = "MIT"
__version__ = "0.3"
__date__ = '01/11/2018'
__maintainer__ = "Dominic Calleja"
__email__ = "d.calleja@liverpoo.ac.uk"
__status__ = "Draft"


import timeit
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
import shutil
import glob
InPath='/Users/dominiccalleja/smardda_workflow/exec'
sys.path.append(InPath)

import src.AnalysisS4P as AS
from src.FWmod import check_smardda_install
import src.FWmod as FW
import src.TStools as time_series
import src.smardda_updating as updating 
import src.smardda_updating as sensitivity 
import src.FEM_heat_equation as FEM
import src.heatequation as FDM
import src.PPF_Data as PPF
import src.AnalysisS4P as analysis


print('+====================================+')
print('|           smardda4python           |')
print('|Generic tools for executing smardda |')
print('|Modification of CTL files primarilly|')
print('|                                    |')
print('|           Version: '+__version__ + 12*' '+' |')
print('|                                    |')
print('| '+__copyright__+' (c) ' + __author__+' |')
print('+====================================+')
print(' ')

#smardda_path = '/Users/dominiccalleja/smiter/exec'
# install_path='/Users/dominiccalleja/demo_misalighnments/exec'


class smardda:
    def __init__(self, Target, Shadow, Equil, Equid, SimDir, geom_path =[], equil_path=[], install_path =InPath):
        self.smardda_path, flag, msg = check_smardda_install(skip=False)

        self.Target = Target
        self.Shadow = Shadow
        self.Equil = Equil  # "abreviated file name"
        self.Equid = Equid  # "full file name"
        self.SimDir = SimDir
        self.SimName = (self.Target).split('.')[0]+'_'+(self.Equil)
        self.ResultsVTK = self.SimName+'POW_powx.vtk'
        self.install_path = install_path

        self.geom_path  = geom_path
        self.equil_path = equil_path
        self._track_fields = False
        "------------------------------ Write Log Files ------------------------------"
        self.logfile = os.path.join(SimDir,self.SimName + '_log.txt')
        copyR(self.logfile)
        outputf=open(self.logfile, 'a')

        if flag:
            outputf.write('Smardda corectly installed : {}\n'.format(flag))
        else:
            outputf.write('Smardda corectly installed : {}\n'.format(flag))
            outputf.write(msg)
            return
        outputf.write('Smardda object initialised: parameters set' + '\n')

    def update_object(self):
        self.SimName = []
        self.ResultsVTK = []
        self.SimName = (self.Target).split('.')[0]+'_'+(self.Equil)
        self.ResultsVTK = self.SimName+'POW_powx.vtk'

    def plasma_parameters(self, Psol=1E6, Lambda_q=0.0012, Sigma=0.0001, equilibrium_displace=False ,displacement_r_z=[],profile='eich'):
        self.Psol = str(Psol)
        self.Lambda_q = str(Lambda_q)
        self.Sigma = str(Sigma)
        self.profile = profile

        self.equilibrium_displacement=equilibrium_displace
        if equilibrium_displace:
            if not len(displacement_r_z) == 2:
                print('ERROR: if equilibrium_displace set to true a list of displacement magnitudes of length 2 must be entered in displacement_r_z')
            else:
                self.eq_r_move = displacement_r_z[0]
                self.eq_z_move = displacement_r_z[1]

        "------------------------------ Write Log Files ------------------------------"
        outputf=open(self.logfile, 'a')
        outputf.write('Plasma parameters set' + '\n')
        outputf.close()

    def vessel_parameters(self, device='JET', nzetp=1, psiref=1):
        self.device = device
        self.nzetp = str(nzetp)

        self.cen_opt = str(4)
        self.bdry_opt_res = str(12)
        self.bdry_opt_shad = str(12)
        self.psi_min = str(0.9)
        self.psi_max = str(2)
        self.psiref = str(psiref)

        self.CTLpath = 'CTL/CTL_'+self.device
        self.resdem = 'res.ctl'
        self.shadem = 'shad.ctl'
        self.hdsctl = 'hds.ctl'
        self.powdem = 'pow.ctl'
        self.fieldtrackctl = 'fieldtrack.ctl'
        self.analysis = 'smanal.ctl'

        vars_string = []
        for keys,values in vars(self).items():
            vars_string.append(str(keys) +':   '+ str(values))

        "------------------------------ Write Log Files ------------------------------"
        outputf=open(self.logfile, 'a')
        outputf.write('Vessel parameters set' + '\n')
        outputf.write('##############-Simulation Input Parameters-##############' + '\n')
        outputf.write('\n'.join(vars_string) + '\n')
        outputf.close()

    def resultsCTL(self, RES_outputVTK='RES.vtk'):
        with open(os.path.join(self.install_path, self.CTLpath, self.resdem), "r+") as f:
            old = f.read()
            txt = old.replace('<Target>', self.Target)
            txt = txt.replace('<Equid>', self.Equid)
            txt = txt.replace('<OutputRes>', RES_outputVTK)
            txt = txt.replace('<Nzetp>', self.nzetp)
            txt = txt.replace('<cen_opt>', self.cen_opt)
            txt = txt.replace('<bdry_opt_r>', self.bdry_opt_res)
            txt = txt.replace('<psi_min>', self.psi_min)
            txt = txt.replace('<psi_max>', self.psi_max)
            txt = txt.replace('<psiref>', self.psiref)

            if self.equilibrium_displacement:
                txt = equil_displacement(self.eq_r_move,self.eq_z_move,txt=txt,write=False)

            Output = self.SimDir+'/'+self.SimName+'RES.ctl'
            fout = open(Output, "wt")
            fout.write(txt)
            fout.close()

        "------------------------------ Write Log Files ------------------------------"
        outputf=open(self.logfile, 'a')
        outputf.write('Target control file written for GEOQ' + '\n')
        outputf.write('Path:'+ Output + '\n')
        outputf.close()

    def shadowCTL(self, SHAD_outputVTK='SHAD.vtk'):
        self.SHAD_outputVTK = SHAD_outputVTK
        with open(os.path.join(self.install_path, self.CTLpath, self.shadem), "r+") as f:
            old = f.read()
            txt = old.replace('<Shadow>', self.Shadow)
            txt = txt.replace('<Equid>', self.Equid)
            txt = txt.replace('<OutputShad>', self.SHAD_outputVTK)
            txt = txt.replace('<cen_opt>', self.cen_opt)  #do i need any of these for SHAD? probably not!!!
            txt = txt.replace('<bdry_opt_s>', self.bdry_opt_shad)
            txt = txt.replace('<psi_min>', self.psi_min)
            txt = txt.replace('<psi_max>', self.psi_max)
            txt = txt.replace('<Nzetp>', self.nzetp)

            if self.equilibrium_displacement:
                txt = equil_displacement(self.eq_r_move,self.eq_z_move,txt=txt,write=False)

            Output = self.SimDir+'/'+self.SimName+'SHAD.ctl'
            fout = open(Output, "wt")
            fout.write(txt)
            fout.close()

        "------------------------------ Write Log Files ------------------------------"
        outputf=open(self.logfile, 'a')
        outputf.write('Shadow control file written for GEOQ' + '\n')
        outputf.write('Path:'+ Output + '\n')
        outputf.close()

    def HDSCTL(self):
        with open(os.path.join(self.install_path, self.CTLpath, self.hdsctl), "r+") as f:
            old = f.read()
            txt = old.replace('<Shadow>', self.SimName+'SHAD')

            Output = self.SimDir+'/'+self.SimName+'HDS.ctl'
            fout = open(Output, "wt")
            fout.write(txt)
            fout.close()

        "------------------------------ Write Log Files ------------------------------"
        outputf=open(self.logfile, 'a')
        outputf.write('Hybrid data structure control file written for HDSGEN' + '\n')
        outputf.write('Path:'+ Output + '\n')
        outputf.close()

    def powcalCTL(self):
        with open(os.path.join(self.install_path, self.CTLpath, self.powdem), "r+") as f:
            old = f.read()
            txt = old.replace('<SimName>', self.SimName)
            txt = txt.replace('<DECL>', self.Lambda_q)
            txt = txt.replace('<POW>', self.Psol)
            txt = txt.replace('<DIFL>', self.Sigma)
            txt = txt.replace('<PROF>', self.profile)

            Output = self.SimDir+'/'+self.SimName+'POW.ctl'
            fout = open(Output, "wt")
            fout.write(txt)
            fout.close()

        "------------------------------ Write Log Files ------------------------------"
        outputf=open(self.logfile, 'a')
        outputf.write('Power calculation control file written for POWCAL' + '\n')
        outputf.write('Path:'+ Output + '\n')
        outputf.close()
    
    def fieldTrackCTL(self,elementsID_of_interest,rename_ctl=[]):
        ctl_suffix = 'field_POW'
        if rename_ctl:
            ctl_suffix = ctl_suffix+'_{}'.format(rename_ctl)
        with open(os.path.join(self.install_path, self.CTLpath, self.fieldtrackctl), "r+") as f:
            old = f.read()
            txt = old.replace('<SimName>', self.SimName)
            txt = txt.replace('<DECL>', self.Lambda_q)
            txt = txt.replace('<POW>', self.Psol)
            txt = txt.replace('<DIFL>', self.Sigma)
            txt = txt.replace('<PROF>', self.profile)

            if isinstance(elementsID_of_interest,np.ndarray):
                Element_IDS = list(elementsID_of_interest.astype(str))
            elif not all(isinstance(s, str) for s in elementsID_of_interest):
                Element_IDS = [str(i) for i in elementsID_of_interest]
            elif isinstance(elementsID_of_interest,int):
                Element_IDS = [elementsID_of_interest]
            else:
                Element_IDS = elementsID_of_interest

            txt = txt.replace('<ElementIDs>', ','.join(Element_IDS))
            txt = txt.replace('<Nelements>', str(int(len(Element_IDS))))

            Output = self.SimDir+'/'+self.SimName+ ctl_suffix +'.ctl'
            fout = open(Output, "wt")
            fout.write(txt)
            fout.close()

        "------------------------------ Write Log Files ------------------------------"
        outputf = open(self.logfile, 'a')
        outputf.write(
            'Power calculation control file written for POWCAL' + '\n')
        outputf.write('Path:' + Output + '\n')
        outputf.close()

        self._track_fields = True

    def executefullsimulation(self):
        import shutil
        outputf=open(self.logfile, 'a')
        retDir = os.getcwd()
        try:
            if not os.path.exists(self.Equid):
                shutil.copy(os.path.join(self.equil_path,self.Equid),self.SimDir)
        except:
            print('Not Copied Equil')
        
        try:
            if not os.path.exists(self.Target):
                shutil.copy(os.path.join(self.geom_path,self.Target),self.SimDir)
        except:
            print('Not copied target GEOM')

        try: 
            if not os.path.exists(self.Shadow):
                shutil.copy(os.path.join(self.geom_path,self.Shadow),self.SimDir)
        except:
            print('Not copied target SHAD')
        os.chdir(self.SimDir)
        tic = timeit.default_timer()

        outputf.write('##############-Executing full simulation workflow-##############' + '\n')
        print('Initialising GEOQ run for {}'.format(self.SimName+'RES.ctl'))
        outputf.write('Initialising GEOQ run for {}'.format(self.SimName+'RES.ctl')+ '\n')

        os.system('{} {}'.format(self.smardda_path+'/geoq', self.SimName+'RES.ctl'))
        toc = timeit.default_timer()

        print('Completed processing {}'.format(self.SimName+'RES.ctl'))
        print('Time to complete: %.2f' % (toc-tic))
        print('Initialising GEOQ run for {}'.format(self.SimName+'SHAD.ctl'))

        outputf.write('Completed processing {}'.format(self.SimName+'RES.ctl')+'\n')
        outputf.write('Time to complete: %.2f' % (toc-tic)+'\n')
        outputf.write('Initialising GEOQ run for {}'.format(self.SimName+'SHAD.ctl')+'\n')

        os.system('{} {}'.format(self.smardda_path+'/geoq', self.SimName+'SHAD.ctl'))
        toc1 = timeit.default_timer()
        print('Completed processing {}'.format(self.SimName+'SHAD.ctl'))
        print('Time to complete: %.2f' % (toc1-toc))
        outputf.write('Completed processing {}'.format(self.SimName+'SHAD.ctl')+'\n')
        outputf.write('Time to complete: %.2f' % (toc1-toc)+'\n')

        print('Initialising HDSGEN run for {}'.format(self.SimName+'HDS.ctl'))
        os.system('{} {}'.format(self.smardda_path+'/hdsgen', self.SimName+'HDS.ctl'))
        toc2 = timeit.default_timer()
        print('Completed processing {}'.format(self.SimName+'HDS.ctl'))
        print('Time to complete: %.2f' % (toc2-toc1))
        outputf.write('Initialising HDSGEN run for {}'.format(self.SimName+'HDS.ctl')+'\n')
        outputf.write('Completed processing {}'.format(self.SimName+'HDS.ctl')+'\n')
        outputf.write('Time to complete: %.2f' % (toc2-toc1)+'\n')

        if not self._track_fields:
            print('Initialising POWCAL run for {}'.format(self.SimName+'POW.ctl'))
            os.system('{} {}'.format(self.smardda_path+'/powcal', self.SimName+'POW.ctl'))
            toc3 = timeit.default_timer()
            os.chdir(retDir)
            print('Completed processing {}'.format(self.SimName+'POW.ctl'))
        else:
            print('Initialising field tracking POWCAL run for {}'.format(self.SimName+'field_POW.ctl'))
            os.system('{} {}'.format(self.smardda_path+'/powcal', self.SimName+'field_POW.ctl'))
            toc3 = timeit.default_timer()
            os.chdir(retDir)
            print('Completed processing {}'.format(
                self.SimName+'field_POW.ctl'))

        print('Time to complete: %.2f' % (toc3-toc2))
        print('SMARDDA run complete. Total execution time: %.2f (sec)' % (toc3-tic))

        "------------------------------ Write Log Files ------------------------------"
        outputf.write('Initialising POWCAL run for {}'.format(self.SimName+'POW.ctl')+'\n')
        outputf.write('Completed processing {}'.format(self.SimName+'POW.ctl')+'\n')
        outputf.write('Time to complete: %.2f' % (toc3-toc2)+'\n')
        outputf.write('##############-Full simulation workflow complete-##############' + '\n')
        outputf.write('Total execution time: %.2f (sec)' % (toc3-tic)+'\n')
        outputf.close()

    def executepowcal(self):
        outputf=open(self.logfile, 'a')
        outputf.write('##############- Executing reduced workflow : Powcal only -##############' + '\n')
        vars_string = []
        for keys,values in vars(self).items():
            vars_string.append(str(keys +':   '+ values))

        outputf.write('\n'.join(vars_string) + '\n')

        retDir = os.getcwd()
        os.chdir(self.SimDir)
        tic = timeit.default_timer()
        os.system('{} {}'.format(self.smardda_path+'/powcal', self.SimName+'POW.ctl'))
        toc = timeit.default_timer()

        "------------------------------ Write Log Files ------------------------------"
        outputf.write('POWCAL execution time: %.2f (sec)' % (toc-tic)+'\n')
        outputf.write('Results in :' + retDir)
        outputf.close()
        os.chdir(retDir)

    def smanal(self):

        with open(os.path.join(self.install_path, self.CTLpath, self.analysis), "r+") as f:
            old = f.read()
            txt = old.replace('<TARGET>', self.Target)
            txt = txt.replace('<powcal_output>', self.ResultsVTK)
            "------------------------------ Write Log Files ------------------------------"
            Output = self.SimDir+'/'+self.SimName+'_smanal.ctl'
            fout = open(Output, "wt")
            fout.write(txt)
            fout.close()

        retDir = os.getcwd()
        os.chdir(self.SimDir)
        os.system('{} {}'.format(self.smardda_path+'/smanal', self.SimName+'_smanal.ctl'))
        os.chdir(retDir)

def equil_displacement(r_move,z_move,ctl_file=[],txt=[],write=False):
    """
    Write CTL file with a transformation of the Equilibria file leading to simulation_pd

    NB/ r_move and z_move is defined in (mm), the allowable accuracy is to 4 decimal places
    # TODO: fscale -- impliment scale of the equil

    """
    header = '&beqparameters'
    imp = header+'\n'+'      {:s} = {:.4f},\n      {:s} = {:.4f},\n'.format('beq_rmove',r_move,'beq_zmove',z_move)

    if write:
        with open(ctl_file,"r+") as f:
            old = f.read()
            tmp = old.split(header)
            out = tmp[0] + imp + tmp[1]

        fout = open(ctl_file, "wt")
        fout.write(out)
        fout.close()

    else:
        tmp = txt.split(header)
        out = tmp[0] + imp + tmp[1]
        return out
    #

class data_holder():
    pass


class Parallel_Smardda:
    def __init__(self):
        print('Specify run parameters or use model_init_')
        self.simulation_list = []
        self.simultion_directory =[]
        self.smardda_path, flag, msg = check_smardda_install(skip=False)
        self.install_path = []
        self.equil_path = []
        self.geom_path = []
        self.Target = []
        self.Shadow = []
        self.logfile = 'parallel_simulation_log.txt'
        "------------------------------ Write Log Files ------------------------------"
        outputf=open(self.logfile, 'a')
        outputf.write('Parallel Smardda object initialised: parameters set' + '\n')
        outputf.close()
    # Populate the initial atributes of the Parallel_Smardda object

    def model_init_(self, Model):
    #Target, Shadow, Equil, Equid, SimDir, smardda_path=smPath, install_path =InPath
        self.Model = Model
        self.smardda_path = Model.smardda_path
        self.install_path = Model.install_path
        self.Target = Model.Target
        self.Shadow = Model.Shadow
        self.Equil = Model.Equil  # "abreviated file name"
        self.Equid = Model.Equid  # "full file name"
        self.SimDir = Model.SimDir
        self.SimName = Model.SimName
        self.ResultsVTK = Model.ResultsVTK
        # TO DO: modify to create new logfile, write in old log file to say its been created
        self.logfile = Model.logfile
        self.CTLpath = Model.CTLpath
        self.powdem = Model.powdem
        self.equilibrium_displacement = Model.equilibrium_displacement

    def construct_simulation_list(self, Lambda_q, Sigma, PowerLoss, NSamples, eq_r_move=0, eq_z_move=0):
        """ Specify the bounds on Lambda_q, Sigma, and Psol"""

        self.simulation_data = data_holder()

        "Generate the LHS samples"
        data_labels = ['Lambda_q','Sigma','Psol']

        # simplify to automatically compile the list and compute the samples. Maybe write a class for it.
        setattr(self.simulation_data,data_labels[0],Lambda_q)
        setattr(self.simulation_data,data_labels[1],Sigma)
        setattr(self.simulation_data,data_labels[2],PowerLoss)
        setattr(self.simulation_data,'NSamples',NSamples)

        if self.equilibrium_displacement:
            "Samples if equilibrium displacements are included"
            data_labels.append('eq_r_move')
            data_labels.append('eq_z_move')
            setattr(self.simulation_data,data_labels[3],eq_r_move)
            setattr(self.simulation_data,data_labels[4],eq_z_move)
            samples = Latin_Hypercube_Sample(np.asarray([Lambda_q[0],Sigma[0],PowerLoss[0],eq_r_move[0],eq_z_move[0]]),np.asarray([Lambda_q[1],Sigma[1],PowerLoss[1],eq_r_move[1],eq_z_move[1]]),NSamples)

            Input_samp={data_labels[0]:samples[:,0],
                        data_labels[1]:samples[:,1],
                        data_labels[2]:samples[:,2],
                        data_labels[3]:samples[:,3],
                        data_labels[4]:samples[:,4]}
        else:
            "samples if equilibrium displacement is not included"
            samples = Latin_Hypercube_Sample(np.asarray([Lambda_q[0],Sigma[0],PowerLoss[0]]),np.asarray([Lambda_q[1],Sigma[1],PowerLoss[1]]),NSamples)
            Input_samp={data_labels[0]:samples[:,0],
                        data_labels[1]:samples[:,1],
                        data_labels[2]:samples[:,2]}

        "Data frame of samples"
        IS = pd.DataFrame(Input_samp, columns=data_labels)
        IS.to_csv(os.path.join(self.SimDir, self.SimName+'InputTable'+'.csv'))
        self.Input_Samples = IS
        self.input_dimension_labels = data_labels

        "------------------------------ Write Log Files ------------------------------"
        outputf=open(self.logfile, 'a')
        outputf.write('LHS sample of the input domain : this method is suitable initial surrogate training' + '\n')
        outputf.write('##########- LHS Input Bounds -##########\n')
        outputf.write(data_labels[0]+': ' +'Lower Bound -' +str(Lambda_q[0]) +' Upper Bound -'+str(Lambda_q[1]) + '\n')
        outputf.write(data_labels[1]+': ' +'Lower Bound -' +str(Sigma[0]) +' Upper Bound -'+str(Sigma[1]) + '\n')
        outputf.write(data_labels[2]+': ' +'Lower Bound -' +str(PowerLoss[0]) +' Upper Bound -'+str(PowerLoss[1]) + '\n')
        if self.equilibrium_displacement:
            outputf.write(data_labels[3]+': ' +'Lower Bound -' +str(eq_r_move[0]) +' Upper Bound -'+str(eq_r_move[1]) + '\n')
            outputf.write(data_labels[4]+': ' +'Lower Bound -' +str(eq_z_move[0]) +' Upper Bound -'+str(eq_z_move[1]) + '\n')
        outputf.write('Number of samples ='+str(NSamples) + '\n')

        "------------------------------ Write CSV of simulation object ------------------------------"
        list_RunID = list(range(self.simulation_data.NSamples))
        list_path = []
        output_pickle =[]
        output_vtk=[]
        for i in range(self.simulation_data.NSamples):
            list_path.append(self.SimDir+'/'+'Run'+str(i))
            output_pickle.append(self.SimDir+'/'+'Run'+str(i)+'.pickle')
            output_vtk.append(self.SimDir+'/'+'Run'+str(i)+'/' + self.ResultsVTK)

        SimStruct ={'Run_Indicies'  :list_RunID,
                    'SimDir_Path'   :list_path,
                    'output_pickle' :output_pickle,
                    'output_vtk'    :output_vtk}

        Simulation_Struture = pd.DataFrame(SimStruct,columns=['Run_Indicies','SimDir_Path','output_pickle','output_vtk'])
        Simulation_Struture = self.Input_Samples.join(Simulation_Struture)
        Simulation_Struture.to_csv(os.path.join(self.SimDir, self.SimName+'Simulation_Structure'+'.csv'))
        self.Simulation_Structure = Simulation_Struture

        "------------------------------ Write Log Files ------------------------------"
        outputf.write('Constructed simulation list.' + '\n')
        outputf.write('Simulation list saved in :'+ self.SimName+'Simulation_Structure'+'.csv' + '\n')
        outputf.write('Full path :' + os.path.join(self.SimDir, self.SimName+'InputTable'+'.csv') + '\n')
        outputf.close()
        # Generate a latin hypercube sample of the input space specified by the user. The aproach is modified to allow
    def execute_multi_processing_list(self,NBatches):
        p = multiprocessing.Pool(NBatches)
        model = {}

        for i, item in enumerate(self.simulation_list.index):
            model[i] = smardda(self.Target, self.Shadow, self.simulation_list['abrv_equil'][item], self.simulation_list['Equil'][item],
            self.simultion_directory,geom_path=self.geom_path, equil_path=self.equil_path, install_path=self.install_path)

        job_args = [(model[i], item, self.simultion_directory, self.simulation_list['Psol'][item],
                     self.simulation_list['Lambda_q'][item], self.simulation_list['Sigma'][item],
                     self.simulation_list['eq_r_move'][item], self.simulation_list['eq_z_move'][item])
        for i, item in enumerate(self.simulation_list.index)]

        p.map(auxillary_parallel_execution2, job_args)

    def execute_multi_processing(self, ModelObject, NBatches):
            # modify to allow you to pick a range of simulations to run as opposed to full batch
        if not self.equilibrium_displacement:
            p = multiprocessing.Pool(NBatches)
            job_args = [(ModelObject, item_RunID, self.SimDir, self.Simulation_Structure['Psol'][i],
            self.Simulation_Structure['Lambda_q'][i], self.Simulation_Structure['Sigma'][i])
            for i, item_RunID in enumerate(list(self.Simulation_Structure['Run_Indicies']))]

            p.map(auxillary_parallel_execution, job_args)

        else:
            p = multiprocessing.Pool(NBatches)
            job_args = [(ModelObject, item_RunID, self.SimDir, self.Simulation_Structure['Psol'][i],
            self.Simulation_Structure['Lambda_q'][i], self.Simulation_Structure['Sigma'][i],
            self.Simulation_Structure['eq_r_move'][i],self.Simulation_Structure['eq_z_move'][i])
            for i, item_RunID in enumerate(list(self.Simulation_Structure['Run_Indicies']))]

            p.map(auxillary_parallel_execution2, job_args)

    def plot_input_samples(self):
        spm = pd.plotting.scatter_matrix(self.Input_Samples, figsize=(15,15),diagonal='hist')
        # Plot the input samples


def auxillary_parallel_execution(args):
    # Auxillary function to allow pool execution with multiple input arguments
    return execute_function(*args)
#

def auxillary_parallel_execution2(args):
    # Auxillary function to allow pool execution with multiple input arguments
    return execute_function2(*args)
#

def execute_function(Object, RunID, TopPath, Psol, L_q, Sig):
    # Function to execute a single case of powcal on remote machine and extract the results
    # TO DO: Modify this for formal object execution

    outputf=open(Object.logfile, 'a')
    print('Process {} evaluation routine stated'.format(RunID))
    outputf.write('Process {} evaluation routine stated'.format(RunID))

    os.chdir(TopPath)
    tic = timeit.default_timer()
    print('Initialising execution %d' % RunID)

    # run powcal in new directory
    SimDir = os.path.join(TopPath,'Run'+str(RunID))
    SimName = 'Run'+str(RunID)
    os.mkdir(SimDir)
    print('Simulation directory: %s' % SimDir)
    outputf.write('Simulation directory: %s' % SimDir)

    parallel_execution(Object, SimDir, SimName, Psol, L_q, Sig, profile='eich')
    shutil.copy(Object.Target, SimDir)
    print('Process {} evaluation routine complete'.format(RunID))
    outputf.write('Process {} evaluation routine complete'.format(RunID))
#
    ##extract results and save in pickle
    Res=AS.Analysis(Object.Target, Object.ResultsVTK, SimDir,'logfile_'+str(RunID)+'.txt')
    Res.ExtractGeom()
    Res.ExtractStats()
    Fp_df21, F_Q21, RL21, C21 = Res.extract_profile(10,BodyNumber = '21')
    Fp_df22, F_Q22, RL22, C22 = Res.extract_profile(10,BodyNumber = '22')
    Fp_df23, F_Q23, RL23, C23 = Res.extract_profile(10,BodyNumber = '24')
    Fp_df24, F_Q24, RL24, C24 = Res.extract_profile(10,BodyNumber = '23')

    Description = ' RunID: {}  - Inp Param: Psol: {:.2E} MW  - L_q: {:f} m  - Sig: {:f} m'.format('Run0', Psol, L_q, Sig)
    Res.Save_pickle([21, 22, 23, 24], TopPath +'/'+ SimName, Description,F1=Fp_df21,F2=Fp_df22,F3=Fp_df23,F4=Fp_df24)

    Description2 = Description + ' \n MaxQ: {:.2E} Wm^-2  - IntQ:  {:.2E} W'.format(Res.summary_data['MaxQ'], Res.summary_data['Int_q'])
    # print and write log file
    toc = timeit.default_timer()

    outputf.write(Description2)
    print(Description2)
    print('SMARDDA run complete. Total execution time: %.2f (sec)' % (toc-tic))
    outputf.write('SMARDDA run complete. Total execution time: %.2f (sec)' % (toc-tic))
    outputf.close()
    return


def execute_function2(Object, run_id, top_dir, Psol, lambda_q, sigma, r_move, z_move):
    # Function to execute a single case of powcal on remote machine and extract the results
    # TO DO: Modify this for formal object execution

    outputf=open(Object.logfile, 'a')
    print('Process {} evaluation routine stated'.format(run_id))
    outputf.write('Process {} evaluation routine stated'.format(run_id))

    os.chdir(top_dir)
    tic = timeit.default_timer()
    print('Initialising execution %d' % run_id)

    # run powcal in new directory
    run_name = 'Run'+str(run_id)
    simulation_dir = os.path.join(top_dir,'Run'+str(run_id))

    os.mkdir(simulation_dir)
    print('Simulation directory: %s' % simulation_dir)
    outputf.write('Simulation directory: %s' % simulation_dir)

#####
    parallel_execution2(Object, simulation_dir, run_name, Psol, lambda_q, sigma, r_move, z_move)
#####
    print('Process {} evaluation routine complete'.format(run_id))
    outputf.write('Process {} evaluation routine complete'.format(run_id))
#
    ##extract results and save in pickle
    Res=AS.Analysis(Object.Target, Object.ResultsVTK, simulation_dir, 'logfile_'+str(run_id)+'.txt')
    Res.ExtractGeom()
    Res.ExtractStats()
    Fp_df21, F_Q21, RL21, C21 = Res.extract_profile(10,BodyNumber = '21')
    Fp_df22, F_Q22, RL22, C22 = Res.extract_profile(10,BodyNumber = '22')
    Fp_df23, F_Q23, RL23, C23 = Res.extract_profile(10,BodyNumber = '24')
    Fp_df24, F_Q24, RL24, C24 = Res.extract_profile(10,BodyNumber = '23')

    Description = ' RunID: {}  - Inp Param: Psol: {:.2E} MW  - L_q: {:f} m  - Sig: {:f} m'.format(run_id, Psol, lambda_q, sigma)
    Res.Save_pickle([21, 22, 23, 24], simulation_dir, Description,F1=Fp_df21,F2=Fp_df22,F3=Fp_df23,F4=Fp_df24)
    rm_generated_files(simulation_dir)
    print('generated files in  %s removed' % simulation_dir)
    outputf.write('generated files in  %s removed' % simulation_dir)
    Description2 = Description + ' \n MaxQ: {:.2E} Wm^-2  - IntQ:  {:.2E} W'.format(Res.summary_data['MaxQ'], Res.summary_data['Int_q'])
    # print and write log file

    toc = timeit.default_timer()
    outputf.write(Description2)

    print(Description2)
    print('SMARDDA run complete. Total execution time: %.2f (sec)' % (toc-tic))
    outputf.write('SMARDDA run complete. Total execution time: %.2f (sec)' % (toc-tic))
    outputf.close()
    return


def flatten(l): return [item for sublist in l for item in sublist]


def rm_generated_files(path_to_folder, additional_file_keep=[]):
    matching_files = [glob.glob(path_to_folder+"/*powx.vtk")]
    if additional_file_keep:
        for i in range(len(additional_file_keep)):
            matching_files.append(
                glob.glob(path_to_folder + '/*{}*'.format(additional_file_keep[i])))
    matching_files = flatten(matching_files)
    generated_files = glob.glob(path_to_folder+'/*')
    removal_list = np.setdiff1d(generated_files, matching_files)
    for j in range(len(removal_list)):
        os.remove(removal_list[j])



def parallel_execution(Object, NewDir, SimName, Psol, Lambda_q, Sigma, profile='eich'):
    retDir = os.getcwd()

    with open(os.path.join(Object.install_path, Object.CTLpath, Object.powdem), "r+") as f:
        old = f.read()
        txt = old.replace('<SimName>', '../' + Object.SimName)
        txt = txt.replace('<DECL>', str(Lambda_q))
        txt = txt.replace('<POW>', str(Psol))
        txt = txt.replace('<DIFL>', str(Sigma))
        txt = txt.replace('<PROF>', profile)

        Output = os.path.join(NewDir, Object.SimName+'POW.ctl')
        fout = open(Output, "wt")
        fout.write(txt)
        fout.close()
    os.chdir(NewDir)
    os.system('{} {}'.format(os.path.join(Object.smardda_path, 'powcal'), Object.SimName+'POW.ctl'))
    os.chdir(retDir)


def parallel_execution2(Object, simulation_dir, run_name, Psol, lambda_q, sigma, r_move, z_move, profile='eich'):
    retDir = os.getcwd()

    run_object = smardda(Object.Target, Object.Shadow, Object.Equil,
    Object.Equid, run_name, geom_path =Object.geom_path,
    equil_path=Object.equil_path ,install_path=Object.install_path)

    run_object.plasma_parameters(Psol, lambda_q, sigma, equilibrium_displace=True , displacement_r_z = [r_move,z_move])
    run_object.vessel_parameters(device='JET', nzetp=1, psiref=1.7760963)
    run_object.resultsCTL()
    run_object.shadowCTL()
    run_object.HDSCTL()
    run_object.powcalCTL()
    run_object.executefullsimulation()
    os.chdir(retDir)
#


def Latin_Hypercube_Sample(param_mins, param_maxes, num_samples):
    dim = param_mins.size
    latin_points = np.array([np.random.permutation(num_samples) for i in range(dim)]).T
    lengths = (param_maxes - param_mins)[None, :]

    return lengths*(latin_points + 0.5)/num_samples + param_mins[None, :]


def copyR(logfile):
    """Print copyright information to file."""
    outputf=open(logfile, 'a')
    outputf.write('+'+'='*77+'+ \n')
    tl='smardda4python'
    sp1=(77-len(tl))//2
    sp2=77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|'+ '\n')
    tl='Python wrappers for Smardda raytracing toolbox'
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


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " %
                  (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


def get_object_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_object_size(v, seen) for v in obj.values()])
        size += sum([get_object_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_object_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_object_size(i, seen) for i in obj])
    return size


def Penalty_factor_simulation(SHADDOW_ALL, TARGET, TRANSLATION_TYPE, translations, results_dir, sim_name, axis = [],rep_geom=False):
    logfile = results_dir + '{}_analysis_construction.txt'.format(sim_name)
    eqdsk_path = '/Users/dominiccalleja/demoaux/Data/Equilibrium/'
    EQDSK = 'ndunr.eqdsk'

    print('READING TARGET DATA ...')
    Target_VTK_extra = FW.ReadVTK(results_dir+'/'+TARGET, logfile)
    Target_vtk_data = {}
    Target_vtk_data['P'] = FW.VTK_P(Target_VTK_extra, logfile)
    Target_vtk_data['B'] = FW.VTK_B(Target_VTK_extra, logfile)
    Target_vtk_data['C'] = FW.VTK_C(Target_VTK_extra, logfile)
    print('TARGET DATA READ')

    print('COMPUTING TARGET DATA AXIS ...')
    x_c, y_c, z_c = FW.centroid(Target_vtk_data['P'])
    
    if not axis:
        normal = FW.normal_axis_unit_vec(Target_vtk_data['P'])
        poltil = FW.tile_poloidal_axis_unit_vec(Target_vtk_data['P'])
        tortil = FW.tile_toroidal_axis_unit_vec(Target_vtk_data['P'])
    else:
        print('Skipping Axis Compute!')
        if TRANSLATION_TYPE == 'NormTrans':
            normal = axis
        elif TRANSLATION_TYPE == 'NormTil':
            normal = axis
        elif TRANSLATION_TYPE == 'PolTil':
            poltil = axis
        elif TRANSLATION_TYPE == 'TorTil':
            tortil = axis
             
    print('TARGET DATA AXIS [complete]')

    evaluation_target_label = []

    for i, trans in enumerate(translations):
        evaluation_target_label.append(
            '{}_{}_delta:{}'.format(sim_name, TRANSLATION_TYPE, trans))
        if TRANSLATION_TYPE == 'NormTrans':
            new_points = FW.translation(Target_vtk_data['P'], normal, trans)

        elif TRANSLATION_TYPE == 'NormTil':
            new_points = FW.rotation(Target_vtk_data['P'], normal, trans)

        elif TRANSLATION_TYPE == 'PolTil':
            new_points = FW.rotation(Target_vtk_data['P'], [
                                     x_c, y_c, z_c], poltil, trans)
        elif TRANSLATION_TYPE == 'TorTil':
            new_points = FW.rotation(Target_vtk_data['P'], [
                                     x_c, y_c, z_c], tortil, trans)

        simulation_dir = results_dir+'/'+evaluation_target_label[i]
        try:
            os.mkdir(simulation_dir)
        except:
            print('----------------------reset work!!!!!!!!!!!!!!')
            continue 

        ret_dir = os.getcwd()
        FW.modFW(new_points, results_dir+'/'+TARGET,
                 results_dir+'/'+evaluation_target_label[i]+'/Target.vtk', logfile)

        os.chdir(simulation_dir)
        shutil.copy('../shaddow_all.vtk', 'tmp_shad.vtk')

        print(
            'COMBINING Shaddow :{} \n\t- and Results:{}'.format('tmp_shad.vtk', 'Target.vtk'))

        FW.combine_vtk_files(['tmp_shad.vtk', 'Target.vtk'], Geom_dir=simulation_dir, Body_No=[], VTK_output='Shaddow_for_run.vtk',
                             output_directory=simulation_dir, combine_ctl='/Users/dominiccalleja/smardda_workflow/DEMO_WORK/combine_res.ctl')

        if os.path.isfile(simulation_dir+'/'+'Shaddow_for_run.vtk'):
            flag = '.TRUE.'
        else:
            flag = '.FALSE.'
            print('Shaddow Check: \n\t{} \n\t ....{}'.format(
                simulation_dir+'/'+'Shaddow_for_run.vtk', flag))
            return
            
        Sim = smardda('Target.vtk', 'Shaddow_for_run.vtk', 'nundnr', EQDSK, simulation_dir,
                         geom_path=simulation_dir, equil_path=eqdsk_path)

        Sim.plasma_parameters(Psol=69E6, Lambda_q=0.08, Sigma=0.001,
                              equilibrium_displace=False, displacement_r_z=[], profile='exp')
        if not rep_geom:
            Sim.vessel_parameters(device='DEMO', nzetp=1)
        else:
            print('Executing with NZETP: {}'.format(rep_geom))
            Sim.vessel_parameters(device='DEMO', nzetp=rep_geom)

        Sim.resultsCTL()
        Sim.shadowCTL()
        Sim.HDSCTL()
        Sim.powcalCTL()
        Sim.executefullsimulation()
        os.chdir(ret_dir)
        rm_generated_files(simulation_dir, additional_file_keep=[
                           '*RES_geofldx.vtk'])
