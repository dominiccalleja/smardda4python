

from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import scipy.linalg as linalg
import time
import pathos.multiprocessing as mp
import pickle
import os
import pandas as pd
from math import sqrt, sin
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import cm

mpl.rcParams.update({'font.size':15})

from scipy.signal import savgol_filter
from scipy import integrate
from scipy.special import erfc
import scipy.stats as stats
import scipy.optimize as op

"Less standard things"
try :
    from jet.data import sal
except: 
    print('JET PPF repository did not respond : from jet.data import sal')

import sys
#sys.path.append('/Users/dominiccalleja/smardda_workflow/exec')
#import scoord as scoord
import scoord as scoord
import TStools as TS

class data_holder():
    pass


class PPF_Data():
    def __init__(self,pulseNo):
        """ default description
        Class : PPF_Data

        Description : This class is used to download and store PPF data from the JET repository

        Inputs : Pulse Number

        Returns : The init initialises a set of data_holder classes for the different PPF data types.
        """

        self.pn = pulseNo
        self.Times =[]
        self.power =    data_holder()
        self.footprint= data_holder()
        self.IRcam =    data_holder()
        self.eftf =     data_holder()
        self.efit =     data_holder()
        self.electrons= data_holder()
        self.elms =     data_holder()
        self.qfit =     data_holder()
        self.t6data =  data_holder()
        print('In PPF_Data :')
    def Power(self):
        """ default description
        Function : Power

        Description : Extract a variety of interesting data relating to energy during a pulse
        A full description of the data can be found in the 'description' attribute
        """

        print('\t Extracting JET Power data from jetppf scal')
        desc = 'Kl9_PPF data \n This data is the primary source for validastion '
        attr = ['description','PSol','PRad','PTot','NBLM','P_e','T_e','W_e']

        setattr(self.power,attr[1],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/scal/psol'))
        setattr(self.power,attr[2],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/scal/prad'))
        setattr(self.power,attr[3],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/scal/plth'))
        setattr(self.power,attr[4],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/nbi/nblm'))
        # setattr(self.power,attr[5],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/lidx/pe0'))    # Electron Pressure
        # setattr(self.power,attr[6],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/lidx/te0'))    # Electron Temp
        # setattr(self.power,attr[7],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/lidx/we'))     # Electron Energy

        D =[]
        for i,name in enumerate(vars(self.power)):
            tmp = getattr(self.power,name)
            s = getattr(tmp,'shape')
            S = [getattr(tmp,'shape')[0],0]
            if len(s)>1:
                S=[getattr(tmp,'shape')[0],getattr(tmp,'shape')[1]]
            D.append('Name :%-12s Descfiption : %-20s ' % (name,getattr(tmp,'description')))
            D.append('Shape: %-5i, %-4i  Units: %-4s, %-4s ' % (S[0],S[1],'sec',getattr(tmp,'units')))

        setattr(self.power,attr[0],D)
        setattr(self.power,'indi',1)
        return self.power

    def Footprint(self):
        """
        Function : Footprint Properties

        Description : Extract a variety of interesting data relating to the footprint of a pulse of outboard target
        A full description of the data can be found in the 'description' attribute
        """
        print('\t Extracting JET Footprint Data')
        desc = 'Kl9_PPF data \n This data is the primary source for validastion '
        attr = ['description','lamda_q','Sigma','SP_r','SP_z','SP_scoord','SP_Power']

        setattr(self.footprint,attr[1], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/ssilburn/feet/olt'))
        setattr(self.footprint,attr[2], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/ssilburn/feet/ost'))

        setattr(self.footprint,attr[3], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/ssilburn/qfit/rsol'))
        setattr(self.footprint,attr[4], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/ssilburn/qfit/zsol'))
        setattr(self.footprint,attr[5], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/ssilburn/qfit/ssol'))
        setattr(self.footprint,attr[6], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/ssilburn/feet/op'))

        D =[]
        for i,name in enumerate(vars(self.footprint)):
            tmp = getattr(self.footprint,name)
            s = getattr(tmp,'shape')
            S = [getattr(tmp,'shape')[0],0]
            if len(s)>1:
                S=[getattr(tmp,'shape')[0],getattr(tmp,'shape')[1]]
            D.append('Name :%-12s Descfiption : %-20s ' % (name,getattr(tmp,'description')))
            D.append('Shape: %-5i, %-4i  Units: %-4s, %-4s ' % (S[0],S[1],'sec',getattr(tmp,'units')))

        setattr(self.footprint,attr[0],D)
        setattr(self.footprint,'indi',1)
        return self.footprint

    def IRcameraQ(self):
        """
        Function : IRcameraQ

        Description : Extract the maximum and radial temporal Q onto tile 6 for a pulse
        A full description of the data can be found in the 'description' attribute
        """
        print('\t Extracting JET IR Camera Data kl9ppf ...')
        desc = 'Kl9_PPF data \n This data is the primary source for validastion '
        attr = ['description','T6_Qmax','T6_QProf']#'T6_Tmax','T6_Qmax',
        
        try:
            setattr(self.IRcam,attr[1], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/kl9ppf/9aqp/slt6'))  # smoothed

        except:
            print('No Max Q IR camera Data extracted')
        
        try:
            setattr(self.IRcam,attr[2], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/kl9ppf/9aqp/spt6'))
        except:         
            print('No smoothed temperature IR camera Data extracted')
            print('Trying RAW')
        
            try: 
                setattr(self.IRcam,attr[2], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/kl9ppf/9aqp/qpt6'))
            except:
                print('profile heat flux not availiable')
       # D =[]
       # for i,name in enumerate(vars(self.IRcam)):
       #     tmp = getattr(self.IRcam,name)
       #     s = getattr(tmp,'shape')
       #     S = [getattr(tmp,'shape')[0],0]
       #     if len(s)>1:
       #         S=[getattr(tmp,'shape')[0],getattr(tmp,'shape')[1]]
       #     D.append('Name :%-12s Descfiption : %-20s ' % (name,getattr(tmp,'description')))
       #     D.append('Shape: %-5i, %-4i  Units: %-4s, %-4s ' % (S[0],S[1],'sec',getattr(tmp,'units')))

       # setattr(self.IRcam,attr[0],D)
        setattr(self.IRcam,'indi',1)
        return self.IRcam

    def IRcameraTemp(self,verbose =True):
        """
        Function : IRcameraTemp

        Description : Extract the maximum and radial temporal Temp onto tile 6 for a pulse
        A full description of the data can be found in the 'description' attribute
        """
        print('\t Extracting JET IR Camera Data kl9ppf ...')
        desc = 'Kl9_PPF data \n This data is the primary source for validastion '
        attr = ['description','T6_Tmax','T6_TProf']

        setattr(self.IRcam,attr[1], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/kl9ppf/9atp/tmt6'))
        setattr(self.IRcam,attr[2], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/kl9ppf/9atp/tpt6'))

        #D =[]
        #if verbose:
        #    for i,name in enumerate(vars(self.IRcam)):
        #        tmp = getattr(self.IRcam,name)
        #        s = getattr(tmp,'shape')
        #        S = [getattr(tmp,'shape')[0],0]
        #        if len(S)>1:
        #            S=[getattr(tmp,'shape')[0],getattr(tmp,'shape')[1]]
        #        D.append('Name :%-12s Descfiption : %-20s ' % (name,getattr(tmp,'description')))
        #        D.append('Shape: %-5i, %-4i  Units: %-4s, %-4s ' % (S[0],S[1],'sec',getattr(tmp,'units')))

        #setattr(self.IRcam,attr[0],D)
        setattr(self.IRcam,'indi',1)
        return self.IRcam

    def EFIT(self):
        """
        Function : EFIT

        Description : Extract strikepoint position on outboard target
        A full description of the data can be found in the 'description' attribute
        """
        print('\t Extracting JET EFIT and EFTF data')
        desc = 'EFIT strike point data \n'
        attr = ['description','EFIT_SPr','EFIT_SPz']
        setattr(self.efit,attr[1],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/efit/rsom'))
        setattr(self.efit,attr[2],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/efit/zsom'))

        D =[]
        for i,name in enumerate(vars(self.EFIT)):
            tmp = getattr(self.EFIT,name)
            s = getattr(tmp,'shape')
            S = [getattr(tmp,'shape')[0],0]
            if len(s)>1:
                S=[getattr(tmp,'shape')[0],getattr(tmp,'shape')[1]]
            D.append('Name :%-12s Descfiption : %-20s ' % (name,getattr(tmp,'description')))
            D.append('Shape: %-5i, %-4i  Units: %-4s, %-4s ' % (S[0],S[1],'sec',getattr(tmp,'units')))

        setattr(self.efit,attr[0],D)
        setattr(self.efit,'indi',1)

    def EFTF(self):
        """
        Function : EFTF

        Description : Extract strikepoint position on outboard target
        A full description of the data can be found in the 'description' attribute
        """
        desc = 'EFIT strike point data \n'
        attr = ['description','EFTF_SPr','EFTF_SPz']

        setattr(self.eftf,'EFTF_SPr',sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/eftf/rsom'))
        setattr(self.eftf,'EFTF_SPz',sal.get('/pulse/'+str(self.pn)+'/ppf/signal/jetppf/eftf/zsom'))

        D =[]
        for i,name in enumerate(vars(self.eftf)):
            tmp = getattr(self.eftf,name)
            s = getattr(tmp,'shape')
            S = [getattr(tmp,'shape')[0],0]
            if len(s)>1:
                S=[getattr(tmp,'shape')[0],getattr(tmp,'shape')[1]]
            D.append('Name :%-12s Descfiption : %-20s ' % (name,getattr(tmp,'description')))
            D.append('Shape: %-5i, %-4i  Units: %-4s, %-4s ' % (S[0],S[1],'sec',getattr(tmp,'units')))

        setattr(self.eftf,attr[0],D)
        setattr(self.eftf,'indi',1)
        return self.efit, self.eftf

    def ELMS(self):
        """
        Function : ELMS Data

        Description : Extract ELM data
        A full description of the data can be found in the 'description' attribute
        """
        print('\t Extracting JET ELMS Data')
        desc = 'ELMS Data from the elms and elma under chain 1. \n used to extract elms from the time history \n'
        attr = ['description','ELM_duration','N_elm', 'feq_elm','free_elm_period','mu_amp_elm','max_amp_elm', 'base_amp_elm']


        setattr(self.elms,attr[1],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/elms/dura'))
        setattr(self.elms,attr[2],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/elms/ncnt'))
        setattr(self.elms,attr[3],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/elma/freq'))
        setattr(self.elms,attr[4],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/elma/free'))
        setattr(self.elms,attr[5],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/elma/mean'))
        setattr(self.elms,attr[6],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/elms/max'))
        setattr(self.elms,attr[7],sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/elma/base'))

        D =[]
        for i,name in enumerate(vars(self.elms)):
            tmp = getattr(self.elms,name)
            s = getattr(tmp,'shape')
            S = [getattr(tmp,'shape')[0],0]
            if len(s)>1:
                S=[getattr(tmp,'shape')[0],getattr(tmp,'shape')[1]]
            D.append('Name :%-12s Descfiption : %-20s ' % (name,getattr(tmp,'description')))
            D.append('Shape: %-5i, %-4i  Units: %-4s, %-4s ' % (S[0],S[1],'sec',getattr(tmp,'units')))

        setattr(self.elms,attr[0],D)
        setattr(self.elms,'indi',1)
        return self.elms

    def QFIT(self):

        attr = ['flux_expansion','targ_lambda_q','targ_s','Q_background','Q_sp','Q_background_error','Q_sp_error','OS_strike_point_s']

        setattr(self.qfit, attr[0], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/cburge/qfit/ofx'))        
        setattr(self.qfit, attr[1], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/cburge/qfit/olt'))
        setattr(self.qfit, attr[2], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/cburge/qfit/ost'))
        setattr(self.qfit, attr[3], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/cburge/qfit/oqb'))
        setattr(self.qfit, attr[4], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/cburge/qfit/oq0'))
        setattr(self.qfit, attr[5], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/cburge/qfit/oqbe'))
        setattr(self.qfit, attr[6], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/cburge/qfit/oq0e'))
        setattr(self.qfit, attr[7], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/cburge/qfit/ssol'))
        setattr(self.qfit, 'keys', attr)
        #

    def T6_data(self):
        setattr(self.t6,'Temperature',data_holder())
        print('\t Extracting T6 Temperature Data')
        attr = ['t6x8','t6y5','t6y8','t6z4','t6z5','Avg_tmp_M2306X','Avg_tmp_M2306Y']
        setattr(self.t6.Temperature, attr[0], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/kd1d/t6x8'))
        setattr(self.t6.Temperature, attr[1], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/kd1d/t6y5'))
        setattr(self.t6.Temperature, attr[2], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/kd1d/t6y8'))
        setattr(self.t6.Temperature, attr[3], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/kd1d/t6z4'))
        setattr(self.t6.Temperature, attr[4], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/kd1d/t6z5'))
        setattr(self.t6.Temperature, attr[5], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/kd1d/an6x'))
        setattr(self.t6.Temperature, attr[6], sal.get('/pulse/'+str(self.pn)+'/ppf/signal/chain1/kd1d/an6y'))
        setattr(self.t6.Temperature, 'keys', attr)

        setattr(self.t6, 'Energy', data_holder())
        attr = ['ea6x','ed6u','ediv_in','total_E_div','Ediv_scoord']
        setattr(self.t6.Energy, attr[0], sal.get('/pulse/90271/ppf/signal/chain1/dvtc/ea6x'))
        setattr(self.t6.Energy, attr[1], sal.get('/pulse/90271/ppf/signal/chain1/dvtc/ed6u'))
        setattr(self.t6.Energy, attr[2], sal.get('/pulse/90271/ppf/signal/chain1/dvtc/eind'))
        setattr(self.t6.Energy, attr[3], sal.get('/pulse/90271/ppf/signal/chain1/dvtc/etco'))
        setattr(self.t6.Energy, attr[4], sal.get('/pulse/90271/ppf/signal/chain1/dvtc/etcs'))
        setattr(self.t6.Energy, 'keys', attr)
        
class PPF_Analysis():
    def __init__(self,pulseNo,Times):
        """
        Class : PPF_Analysis

        Description : This class is used for extracting, cleaning, and preparing JET data for analysis.
        The tools utilise the PPF_data() class. Where any necessary PPH data is used the fucnitons will
        execute the PPF_data() class within this class.

        Inputs :
        pulseNo : JET pulse number
        Times   : User defined window for analysis

        Returns :
        """
        self.pn     = pulseNo
        self.times  = Times
        self.Data   =  PPF_Data(pulseNo)
        # Rmin, Rmax
        self.Tile6R = [1321, 1510]
        # add S-Coord option and if statements

        self.data_cleaning_options = data_holder()
        self.impute_method = 'Radial_Mean'
        self.imputation_param = [3,5]
        #self.__KN__ = KN # Smoothing Window for the missing data
        methods = ['Radial_Mean','krigging','Longitudinal_polynomial','KNN']

        if isinstance(self.impute_method, int):
            if self.impute_method>len(methods):
                print('ERROR: %s is not a valid option, please enter value between [0,%i] or a string from %s' % (str(self.impute_method),len(methods),methods))
                return

            self.__imputation_method_Prof__ = methods[self.impute_method]
        elif not np.isin(self.impute_method,methods):
            print('ERROR: %s is not a valid entry. Please enter value between [0,%i] or a string from %s' % (str(self.impute_method),len(methods),methods))
            return
        else:
            self.__imputation_method_Prof__ = self.impute_method


        self.plot_options = data_holder()
        setattr(self.plot_options,'PLOT',True)
        setattr(self.plot_options,'plot_max_long',True)
        setattr(self.plot_options,'plot_Q_prof',True)
        setattr(self.plot_options,'plot_Q_filt',True)
        setattr(self.plot_options,'temporal_cross_section',True)
        setattr(self.plot_options,'ELM_plot_window',[1,10])

    def Times_ELMS(self):
        """
        Function : Times_ELMS

        Description : This function extracts the ELM peak time and duration and builds
        a np array of the values of interest.
        The function appends the lead in and drop down user specified time interval.

        Inputs : NA
        Returns :
        Raw_ELM_Times = np.array : [t end of elm, t peak of next elm]
        """
        # needs slight change to work if the user specifies a sub time interval, at the moment only takes a bigger interval
        if not hasattr(self.Data.elms,'indi'):
            self.Data.ELMS()
        print('Constructing ELM Times Matrix')

        t_start_elm = self.Data.elms.N_elm.dimensions[0].data
        t_end_elm = t_start_elm + self.Data.elms.ELM_duration.data

        if self.times[0] > np.min(t_start_elm):
            t_start_elm = self.Data.elms.N_elm.dimensions[0].data[self.Data.elms.N_elm.dimensions[0].data>self.times[0]]
            t_end_elm = t_end_elm[self.Data.elms.N_elm.dimensions[0].data>self.times[0]]
        if self.times[1] < np.min(t_end_elm):
            t_start_elm = [t_end_elm < self.times[1]]
            t_end_elm = t_end_elm[t_end_elm < self.times[1]]

        tt = np.zeros([len(t_start_elm)-1,2])
        for i in range(len(t_start_elm)-1):
           tt[i,0] = t_end_elm[i]
           tt[i,1] = t_start_elm[i+1]

        T = tt[np.logical_and(tt[:,0]>self.times[0],tt[:,1]<self.times[1]),:]

        a = np.array([self.times[0],np.min(T)])
        b = np.array([np.max(T),self.times[1]])
        self.Raw_ELM_Times = np.concatenate(([a],T,[b]))
        return self.Raw_ELM_Times

    def Filter_ELMS(self,ELM_proportion=0.99):
        """
        Function : user_def_time_ELMS

        Description : This function allows the user to specify a perscentage of the ELM they wish to filter.
        The unfiltered ELM times typically include the 'rise' time, as the ELMS are usually identified by the
        peak of the ELM.
        The function

        Inputs :
        ELM_proportion = % of elm window of interest (Default 99%)
        plot           = a plot of the longitudinal cross section
        plot_window    = a smaller sub-interval to aid comprehension (Default [1,5] Elms)
        """

        #### Need to impliment some other data replacement methods when I have time!!!

        if not hasattr(self,'Raw_ELM_Times'):
            self.Times_ELMS()
        if not hasattr(self.Data.IRcam,'indi'):
            self.Data.IRcameraQ()

        print('Executing ELM Filtering \n')
        print('Inter ELM proportion {:.4f}\n'.format(ELM_proportion))
        E_times = self.Raw_ELM_Times

        QProf = self.Data.IRcam.T6_QProf
        ind = np.logical_and(QProf.dimensions[0].data>=self.times[0], QProf.dimensions[0].data<=self.times[1])
        q_t = QProf.dimensions[0].data[ind]
        q_x = QProf.dimensions[1].data
        q_dat = QProf.data[ind,:]
        S_scale = np.interp(q_x, (q_x.min(), q_x.max()), ([1363.3,1552.6]))
        R_scale = scoord.get_R_Z(S_scale)

        QQ = Profile_to_Pandas(QProf.data[ind,:],R_scale[0],q_t)
        q_max = np.max(q_dat,axis=1)

        ELM_Times = np.zeros(np.shape(E_times))
        logic = []
        delta_t =[]
        ind_int_elm =[]
        for i,t in enumerate(E_times[0:-1]):
            t[1]=(t[1]-t[0])*ELM_proportion +t[0]
            lo = np.logical_and(q_t>t[0],q_t<t[1])
            inelm = np.where(np.logical_and(q_t>t[0],q_t<t[1]))

            d_t = t[1]-t[0]

            ELM_Times[i] = [t[0],t[1]]
            logic.append(lo)
            delta_t.append(d_t)
            ind_int_elm.append(inelm)

        logic = np.logical_or.reduce(logic)

        # Make a copy of the Q pandas array
        Q_ELM = QQ.copy(deep=True)

        Q_ELM.iloc[np.logical_not(logic)]=np.nan

        if self.__imputation_method_Prof__ =='Radial_Mean':
            Q_ELM.fillna(Q_ELM.mean(axis=0),inplace=True)

        elif self.__imputation_method_Prof__ =='krigging':
            from sklearn.impute import SimpleImputer
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit_transform(Q_ELM)

        elif self.__imputation_method_Prof__ == 'Logitudinal_KN':
            D = Avg_kn_cross_secton(Q_ELM.values, q_x, q_t,logic,self.imputation_param[0])
            Q_ELM= Profile_to_Pandas(D,q_x,q_t,major_radius = True)

        elif self.__imputation_method_Prof__ == 'Longitudinal_polynomial':
            if len(self.imputation_param) == 2:
                D = Temporal_poly_impute(Q_ELM.values, q_x, q_t, logic, self.imputation_param[0],self.imputation_param[1])
                Q_ELM= Profile_to_Pandas(D,q_x,q_t,major_radius = True)
            else:
                KN = self.imputation_param
                polyorder = 5
                self.imputation_param = [KN,polyorder]
                D = Temporal_poly_impute(Q_ELM.values, q_x, q_t, logic, self.imputation_param[0],polyorder)
                Q_ELM= Profile_to_Pandas(D,q_x,q_t,major_radius = True)
            print('Averaging over %s radial locations - with polnomial order %s' % (str(self.imputation_param[0]),str(self.imputation_param[1])))

        delta_t = np.array(delta_t)
        Q_mu = np.mean(q_dat[logic,:],axis=0)
        #Q_rep_mu = np.mean(Q_ELM_REP,axis=0)

        Q_max_long = q_max[logic]

        self.ELM_Times = ELM_Times

        self.ELM_Inclusive = data_holder()
        self.Inter_ELM_Q = data_holder()

        Description = 'ELM Inclusive Q profile data- Time Int: %0.2d - %0.2d' % (self.times[0],self.times[1])
        setattr(self.ELM_Inclusive,'Q',QQ)
        setattr(self.ELM_Inclusive,'Time',q_t)
        setattr(self.ELM_Inclusive,'xq_camera',q_x)
        setattr(self.ELM_Inclusive,'RZ_location',R_scale)
        setattr(self.ELM_Inclusive,'Description',Description)

        Description = 'Inter ELM Q profile data- Time Int: %0.2d - %0.2d' % (self.times[0],self.times[1])
        # Standard outputs

        setattr(self.Inter_ELM_Q,'Q',Q_ELM)
        setattr(self.Inter_ELM_Q,'Time',q_t)
        setattr(self.Inter_ELM_Q,'logical',logic)
        setattr(self.Inter_ELM_Q,'xq_camera',q_x)
        setattr(self.Inter_ELM_Q,'RZ_location',R_scale)
        setattr(self.Inter_ELM_Q,'ELM_proportion',ELM_proportion)
        setattr(self.Inter_ELM_Q,'Description',Description)

        # some other helpful things
        setattr(self.Inter_ELM_Q,'Q_int_elm',q_dat[logic,:])
        setattr(self.Inter_ELM_Q,'Avg_Q_int_elm',Q_mu)
#        setattr(self.Inter_ELM_Q,'Avg_Q',Q_rep_mu)
        setattr(self.Inter_ELM_Q,'Q_max',Q_max_long)
        setattr(self.Inter_ELM_Q,'Ind_int_elm',ind_int_elm)
        setattr(self.Inter_ELM_Q,'times_Ex_elm',q_t[logic])
        setattr(self.Inter_ELM_Q,'times_mid_elm',ELM_Times[0:-1,0]+delta_t)
        setattr(self.Inter_ELM_Q,'ELM_Int_width',delta_t)

#
        if self.plot_options.PLOT:
            if self.plot_options.plot_max_long:
                plt.figure(figsize=(18,5))
                plt.plot(q_t,q_max,color='red',label='Raw Max Q')
                plt.plot(q_t,np.max(Q_ELM,axis=1),color='green',label='Replaced Max Method: %s, %s:%s' % (self.__imputation_method_Prof__,'Param',str(self.imputation_param)))
                for i in range(self.plot_options.ELM_plot_window[0],self.plot_options.ELM_plot_window[1]):
                    plt.plot(q_t[ind_int_elm[i]],q_max[ind_int_elm[i]],color='blue')
                plt.legend()
                plt.xlim(self.Raw_ELM_Times[self.plot_options.ELM_plot_window[0],0],self.Raw_ELM_Times[self.plot_options.ELM_plot_window[1],1])
                plt.xlabel('Time (sec)')
                plt.ylabel('Max Radial Q MWm-2')
                plt.title('Longitudinal MaxQ from ELM {:1d} to ELM {:1d} \n Max Q for specified {:.2f} interval'.format(self.plot_options.ELM_plot_window[0],self.plot_options.ELM_plot_window[1],ELM_proportion))

            if self.plot_options.plot_Q_prof:
                plot_heatmap(self.ELM_Inclusive.Q.values,self.ELM_Inclusive.Time,self.Inter_ELM_Q.RZ_location[0],'Q (Wm-2)','Tile 6 Heat Flux Profile Raw')
            if self.plot_options.plot_Q_filt:
                plot_heatmap(self.Inter_ELM_Q.Q.values,self.Inter_ELM_Q.Time,self.Inter_ELM_Q.RZ_location[0],'Q (Wm-2)','Tile 6 Heat Flux ELM Replaced') #self.Inter_ELM_Q.Avg_Q_int_elm,self.ELM_Times[:,0]+self.Inter_ELM_Q.ELM_Int_width,self.Inter_ELM_Q.RZ_location[0]

    def time_slice(self,t_vector,Int_ELM=True,user_def_interval=True,plot=True,scatter=False,ELM_proportion = 0.8):   # this needs fixing with the new plot options stuff
        """ default description
        Function : time_slice_inter_elm

        Description :

        Inputs :

        Returns :
        """
        if not hasattr(self,'Inter_ELM_Q'):
            self.Filter_ELMS(ELM_proportion=ELM_proportion,plot=False)
        if not hasattr(self.Data.efit,'indi'):
            self.Data.EFIT()
        if not hasattr(self.Data.footprint,'indi'):
            self.Data.Footprint()

        Q =[]
        Q_mu =[]
        SP = []
        SPeft =[]
        t_mid = []
        t_win = []
        self.user_specified_time_slice= data_holder()
        if user_def_interval:
            # if np.shape(t_vector)[2]==1:
                # print('WARNING: When using the user_def_interval Analysis expects shape of t_vector to be (n x 2)')
                # return

            __type__ = 'Inter ELM'
            __Number_of_slice__ = str(np.shape(t_vector))


            for i, t in enumerate(t_vector):
                ind = np.logical_and(self.ELM_Times[:,0]<=t,self.ELM_Times[:,1]>=t).argmax()
                Q.append(self.ELM_Inclusive.Q[self.Inter_ELM_Q.time_ind_list[ind][0],:])
                Q_mu.append(self.Inter_ELM_Q.Avg_Q_int_elm[ind])
                t_win.append(self.ELM_Times[ind,:])

                i_sp=abs(self.Data.efit.EFIT_SPr.dimensions[0].data-t).argmin()
                SP.append(self.Data.efit.EFIT_SPr.data[i_sp])
                i_speft=abs(self.Data.eftf.EFTF_SPr.dimensions[0].data-t).argmin()
                SPeft.append(self.Data.eftf.EFTF_SPr.data[i_speft])
            setattr(self.user_specified_time_slice,'Tmid',t_mid)
        else:
            __type__ = 'User specified time intervals'
            __Number_of_slice__ = str(np.shape(t_vector))

            for i, t in enumerate(t_vector):
                ind = np.logical_and(self.Inter_ELM_Q.times_int_elm >= t[0], self.Inter_ELM_Q.times_int_elm <= t[1])
                Q.append(self.Inter_ELM_Q.Q_int_elm[ind,:])
                print(Q[i])
                print(np.shape(Q[i]))
                Q_mu.append(self.Inter_ELM_Q.Avg_Q_int_elm[ind[0],:])
                t_mid.append(t[1]-t[0])

                i_sp_L=abs(self.Data.efit.EFIT_SPr.dimensions[0].data-t[0]).argmin()
                i_sp_R=abs(self.Data.efit.EFIT_SPr.dimensions[0].data-t[1]).argmin()
                SP.append(np.array(self.Data.efit.EFIT_SPr.data[i_sp_L],self.Data.efit.EFIT_SPr.data[i_sp_R]))
                i_speft_L=abs(self.Data.eftf.EFTF_SPr.dimensions[0].data-t[0]).argmin()
                i_speft_R=abs(self.Data.eftf.EFTF_SPr.dimensions[0].data-t[1]).argmin()
                SPeft.append(np.array(self.Data.eftf.EFTF_SPr.data[i_speft_L],self.Data.eftf.EFTF_SPr.data[i_speft_R]))
            setattr(self.user_specified_time_slice,'Tmid',t_mid)

        setattr(self.user_specified_time_slice,'Sliced_Q',Q)
        setattr(self.user_specified_time_slice,'Sliced_Q_mu',Q_mu)
        setattr(self.user_specified_time_slice,'Time_slices',t_vector)
        setattr(self.user_specified_time_slice,'EFIT_SP_R',SP)
        setattr(self.user_specified_time_slice,'EFTF_SP_R',SPeft)

        if self.plot_options.PLOT:
            if self.plot_options.temporal_cross_section:

                fig = plt.figure()
                fig.set_size_inches(15,4*len(t_vector))
                for i,t in enumerate(t_vector):
                    axs= fig.add_subplot(len(t_vector), 1, i+1)

                    if scatter:
                        X = self.Inter_ELM_Q.RZ_location[0]*np.ones(np.shape(self.user_specified_time_slice.Sliced_Q[i]))
                        Y = Q[i]
                        axs.scatter(X,Y)
                    if user_def_interval:
                        X = self.Inter_ELM_Q.RZ_location[0]
                        Y = Q_mu[i]
                    else:
                        X = self.Inter_ELM_Q.RZ_location[0]*np.ones(np.shape(Q_mu[i]))
                        Y = Q_mu[i]

                    axs.scatter(X,Y,label = 'Mean Q Footprint @ %0.3f (sec) Win:[%0.3f,%0.3f]' % (t,t_win[i][0],t_win[i][1]))

                    # Plot strike point
                    axs.plot(np.ones([2,1])*SP[i],[0,np.max(Q[i])],label = 'EFIT Xpoint %0.4f (m)' %(SP[i]))
                    axs.plot(np.ones([2,1])*SPeft[i],[0,np.max(Q[i])],label = 'EFTF Xpoint %0.4f (m)' %(SPeft[i]))
                    axs.set_ylim(-0.1E7,2E7)
                    axs.set_xlabel(' Radial Axis ')
                    axs.set_ylabel('Q MWm-2')
                    axs.legend()

        return self.user_specified_time_slice


def plot_heatmap(Dat,t,R,label,title):
    framestride = max(1,int(5e-4 / np.mean(np.gradient(t))))
    time = [min(t),max(t)]
    clim = [np.min(Dat),np.max(Dat)]#np.log(Dat.min())
    fig= plt.figure()
    fig.set_size_inches(12,10)
    #plt.gca().set_facecolor((0.2,0.2,0.2))
    im = plt.imshow(np.flip(Dat[::framestride,:].transpose(),0),extent=[time[0],time[1],np.min(R),np.max(R)],clim=clim,aspect = 'auto',cmap='hot')
    plt.xlabel('Time (sec)',size=18)
    plt.ylabel('Radial Location (m)',size=18)
    cbar = plt.colorbar(im)
    cbar.set_label(label,size=18)
    plt.title(title,size=20)
    plt.show()


def Temporal_inverse_weight(IN_Dat,Logical,KN):
    missing = np.where(np.logical_not(Logical))[0]
    present = np.where(Logical)[0]
    DATA =IN_Dat

    op = []
    for i,t in enumerate(missing):

        u = np.where(present<missing[i])[0][-KN:]
        l = present[np.where(present>missing[i])[0]][0:KN]
        ind = np.concatenate([u,l])
        distance =  np.concatenate([abs((missing[i]-u)),abs((l-missing[i]))])

        DATA[t,:] = np.sum(1/distance*IN_Dat[ind,:].T,axis=1)/np.sum(distance)

    return DATA

def Avg_kn_cross_secton(Data,X,T,logical,K):

    D = np.zeros([len(T),len(X)])
    Out = np.zeros([np.sum(np.logical_not(logical)),len(X)])

    Ex_col = np.where(np.logical_not(logical))[0]
    Inc_col = np.where(logical)[0]
    for i, col in  enumerate(Ex_col):

        d = np.abs(Inc_col-col)
        i_d = np.argsort(d)[1:K]

        Out[i,:] = np.mean(Data[Inc_col[i_d],:],axis=0)
        D = Data
    D[np.logical_not(logical),:] = Out
    return D

def Temporal_poly_impute(Data,X,T,logical,KN,poly_order):

    D = np.zeros([len(T),len(X)])
    Out = np.zeros([np.sum(np.logical_not(logical)),len(X)])

    for i in range(len(X)):
        L = i - KN
        U = i + KN

        if L <0:
            L = 0
        if U>len(X):
            U = len(X)

        y = Data[logical,L:U]
        x = T[logical]

        xx = x*np.ones([len(x),U-L]).T
        F = np.polyfit(x,y,poly_order)

        # can use different rules in here! impliment later
        F = np.median(F,axis=1)

        x_i = T[np.logical_not(logical)]
        Out[:,i]=np.polyval(F,x_i)

    D = Data
    D[np.logical_not(logical),:] = Out
    return D


def kalman_impute():
    """
    Kalman filters operate on state-space models of the form (there are several ways to write it; this is an easy one based on Durbin and Koopman (2012); all of the following is based on that book, which is excellent):

    𝑦𝑡  = 𝑍𝛼𝑡+𝜀𝑡              𝜀𝑡∼𝑁(0,𝐻)
    𝛼𝑡1 = 𝑇𝛼𝑡+𝜂𝑡              𝜂𝑡∼𝑁(0,𝑄)
    𝛼1∼𝑁(𝑎1,𝑃1)

    where 𝑦𝑡 is the observed series (possibly with missing values) but 𝛼𝑡 is fully unobserved.
    The first equation (the "measurement" equation) says that the observed data is related to the unobserved states in a particular way.
    The second equation (the "transition" equation) says that the unobserved states evolve over time in a particular way.

    The Kalman filter operates to find optimal estimates of 𝛼𝑡 (𝛼𝑡 is assumed to be Normal: 𝛼𝑡∼𝑁(𝑎𝑡,𝑃𝑡),
    so what the Kalman filter actually does is to compute the conditional mean and variance of the distribution for 𝛼𝑡 conditional on observations up to time 𝑡).

    In the typical case (when observations are available) the Kalman filter uses the estimate of the current state and the current observation 𝑦𝑡 to do the best it can to estimate the next state 𝛼𝑡+1, as follows:

    𝑎𝑡+1 = 𝑇𝑎𝑡+𝐾𝑡(𝑦𝑡−𝑍𝛼𝑡)
    𝑃𝑡+1 = 𝑇𝑃𝑡(𝑇−𝐾𝑡𝑍)′+𝑄
    where 𝐾𝑡 is the "Kalman gain".

    When there is not an observation, the Kalman filter still wants to compute 𝑎𝑡+1 and 𝑃𝑡+1 in the best possible way.
    Since 𝑦𝑡 is unavailable, it cannot make use of the measurement equation, but it can still use the transition equation.
    Thus, when 𝑦𝑡 is missing, the Kalman filter instead computes:

    𝑎𝑡+1 = 𝑇𝑎𝑡
    𝑃𝑡+1 = 𝑇𝑃𝑡𝑇′+𝑄
    Essentially, it says that given 𝛼𝑡, my best guess as to 𝛼𝑡+1 without data is just the evolution specified in the transition equation.
    This can be performed for any number of time periods with missing data.

    If there is data 𝑦𝑡, then the first set of filtering equations take the best guess without data, and add a "correction" in,
    based on how good the previous estimate was

    Imputing data :
    Once the Kalman filter has been applied to the entire time range, you have optimal estimates of the states 𝑎𝑡,𝑃𝑡 for 𝑡=1,2,…,𝑇.
    Imputing data is then simple via the measurement equation. In particular, you just calculate:

    𝑦̂ 𝑡 = 𝑍𝑎𝑡

    Durbin, J., & Koopman, S. J. (2012). Time series analysis by state space methods (No. 38). Oxford University Press.
    """

def Profile_to_Pandas(Data,X,T,major_radius = True):

    if major_radius:
        R = X
    else:
        S_scale = np.interp(X, (X.min(), X.max()), ([1363.3,1552.6]))
        R = scoord.get_R_Z(S_scale)[0]

    np.shape(T)
    np.shape(R)
    np.shape(Data)
    Index = T
    Cols = R
    df = pd.DataFrame(Data ,index=Index ,columns=Cols)

    return df


"""
functions for dealing with data PPF
"""


def get_JET_power_data(pulse_number):
    """ 
    Fetch data for dynamic power balance
    """
    power_data = {}
    data_names = []
    try :
        W_dia= sal.get('/pulse/'+str(pulse_number)+'/ppf/signal/chain1/ehpf/wdia')
        power_data['W_dia'] = W_dia
        data_names.append('W_dia')
    except:
        print('No Jet W_dia  ... skipping')
    # aproximation for storred energy
    try :
        W_mhd = sal.get('/pulse/'+str(pulse_number) +'/ppf/signal/chain1/ehpf/wp') #jetppf/efit
        power_data['W_mhd'] = W_mhd
        data_names.append('W_mhd')
    except:
        print('No Jet W_mhd  ... skipping')
    #P Ohm
    try :
        P_ohm = sal.get('/pulse/'+str(pulse_number)+'/ppf/signal/jetppf/efit/pohm')
        power_data['P_ohm'] = P_ohm
        data_names.append('P_ohm')
    except:
        print('No Jet P_ohm  ... skipping')
    # P irch tot
    try :
        P_ich = sal.get('/pulse/'+str(pulse_number)+'/ppf/signal/jetppf/icrh/ptot')
        power_data['P_ich'] = P_ich
        data_names.append('P_ich')
    except:
        print('No Jet P_ich  ... skipping')
    # P_NBI
    try :
        P_nbi = sal.get('/pulse/'+str(pulse_number)+'/ppf/signal/jetppf/nbi/ptot')
        power_data['P_nbi'] = P_nbi
        data_names.append('P_nbi')
    except:
        print('No Jet P_nbi  ... skipping')
    # P_RAD
    try:
        P_rad = sal.get('/pulse/'+str(pulse_number)+'/ppf/signal/jetppf/bolo/topi')
        power_data['P_rad'] = P_rad
        data_names.append('P_rad')
    except:
        print('No Jet P_rad  ... skipping')
    #P_tile5
    try : 
        P_Tile5 = sal.get('/pulse/'+str(pulse_number)+'/ppf/signal/kl9ppf/9aqp/plt5')
        power_data['P_tile5'] = P_Tile5
        data_names.append( 'P_tile5')
    except:
        print('No Jet P_tile5  ... skipping')
    
    power_data['variable_names'] = data_names

    return power_data


def slice_power_data(data, times, label=[]):
    """
    Slice formatted power data
    """
    t = data.dimensions[0].data
    P = data.data

    i_time = np.logical_and(t > times[0], t < times[1])
    if not label:
        label = data.description
    D = {}
    D['Time'] = t[i_time]
    D[label] = P[i_time]
    P_i = pd.DataFrame(columns=['Time', label], data=D)
    return P_i


def stored_energy(Wdia, Wmhd):

    W = ((2*Wmhd.values) + Wdia.values)/3
    return W

def JET_energy_deriv(stored_energy, scheme='grad'):
    """
    deriverative of plasma stored energy to find lost energy
    scheme = central  or diff
    """
    if scheme == 'diff':
        if isinstance(stored_energy,pd.DataFrame):
            diff = np.diff(stored_energy.values[::-1])
        else:
            diff = np.diff(stored_energy[::-1])
    elif scheme == 'grad':
        if isinstance(stored_energy,pd.DataFrame):
            diff = np.gradient(stored_energy.values[::-1])
        else:
            diff = np.gradient(stored_energy[::-1])
    else:
        print('must choose diff or central')
    return diff


def slice_Q_profile(QProf, sim_times=[]):
    q_t = QProf.dimensions[0].data
    q_x = QProf.dimensions[1].data
    if sim_times:
        ind = np.logical_and(q_t > sim_times[0], q_t < sim_times[1])
    else:
        ind = range(len(q_t))

    S_scale = np.interp(q_x, (q_x.min(), q_x.max()), ([1363.3, 1552.6]))
    R_scale = scoord.get_R_Z(S_scale)

    Q = Profile_to_Pandas(QProf.data[ind, :], R_scale[0], q_t[ind])
    Qs = Profile_to_Pandas(QProf.data[ind, :], S_scale, q_t[ind])
    return Q, Qs, R_scale, q_t[ind]


def format_JET_input_power_data(times, power_data=[], pulse_number=[], calc_lost_power=True, resample_rate=False):
    """specify pulse number or pass power data"""
    # output the raw signas, and mean imputation
    if pulse_number:
        power_data = get_JET_power_data(pulse_number)

    inputs = list(np.array(list(power_data.keys()))[
        [x in ['P_ohm', 'P_ich', 'P_nbi', 'W_dia', 'W_mhd'] for x in power_data.keys()]])

    d = []
    l = []
    for i, lab in enumerate(inputs):
        tmp = slice_power_data(power_data[lab], times, label=lab)
        d.append(tmp)
        l.append(len(tmp['Time']))

    j = np.argmax(l)
    time = d[j]['Time']
    time = pd.TimedeltaIndex(time, unit='s')

    d1 = d[j]
    d1.index = time
    d1 = d1.drop(columns='Time')
    if resample_rate:
        d1 = d1.resample(resample_rate).agg('nearest')

    input_power = d1.copy(deep=True)

    tmp_l = np.where(np.array(range(len(l))) != j)[0]

    for i, ind in enumerate(tmp_l):
        #if i == j:
        #    continue
        d[ind].index = pd.TimedeltaIndex(d[ind]['Time'], unit='s')
        if resample_rate:
            d[ind] = d[ind].resample(resample_rate).agg('nearest')
        input_power = pd.merge_asof(
            input_power, d[ind], right_index=True, left_index=True, direction='nearest')
    try:
        input_power = input_power.drop(columns=['Time_x', 'Time_y'])
    except:
        print('\n')
    input_power['P_IN'] = input_power[list(np.array(list(power_data.keys()))[[x in ['P_ohm', 'P_ich', 'P_nbi'] for x in power_data.keys()]])].sum(axis=1)
    input_power['Time'] = input_power.index.values.astype('float')/1E9
    #cols = list(input_power)
    #cols.insert(0, cols.pop(cols.index('Time')))
    #input_power.index = input_power['Time']  # = input_power.ix[:, cols]
    #input_power = input_power.drop(columns='Time')
    return input_power, power_data

def format_JET_radiation_data(sim_times, power_data, resample_rate='1ms'):

    inputs = 'P_rad'
    data_raw = slice_power_data(power_data[inputs], sim_times, inputs)

    resampled_data = data_raw.copy(deep=True)
    resampled_data.index = pd.TimedeltaIndex(resampled_data['Time'], unit='s')
    resampled_data = resampled_data.resample(resample_rate).agg('nearest')
    #resampled_data.index = resampled_data['Time']
    #resampled_data = resampled_data.drop(columns='Time')
    return resampled_data, data_raw


def format_JET_tile5_data(sim_times, power_data, resample_rate='1ms'):

    inputs = 'P_tile5'
    data_raw = slice_power_data(power_data[inputs], sim_times, inputs)

    resampled_data = data_raw.copy(deep=True)
    resampled_data.index = pd.TimedeltaIndex(resampled_data['Time'], unit='s')
    resampled_data = resampled_data.resample(resample_rate).agg('nearest')
    return resampled_data, data_raw


def format_JET_qfit_data(times, qfit_data, resample_rate='0.1ms'):
    """pass qfit data"""
    # output the raw signas, and mean imputation

    d = []
    l = []
    for i, lab in enumerate(qfit_data.keys):
        tmp = slice_power_data(getattr(qfit_data, lab), times, label=lab)
        d.append(tmp)
        l.append(len(tmp['Time']))

    j = np.argmax(l)
    time = d[j]['Time']
    time = pd.TimedeltaIndex(time, unit='s')

    d1 = d[j]
    d1.index = time
    #d1 = d1.drop(columns='Time')
    d1 = d1.resample(resample_rate).agg('nearest')
    #
    tmp_l = np.where(np.array(range(len(l))) != j)[0]
    for i, ind in enumerate(tmp_l):
        if i == j:
            continue
        d[ind].index = pd.TimedeltaIndex(d[ind]['Time'], unit='s')
        #d[ind] = d[ind].drop(columns='Time')
        d[ind] = d[ind].resample(resample_rate).agg('nearest')
    d[j] = d1

    qfit = pd.merge_asof(d[j], d[0], on='Time', direction='nearest')
    for i, ind in enumerate(tmp_l):
        if i == 0:
            continue
        qfit = pd.merge_asof(qfit, d[ind], on='Time', direction='nearest')
    R, Z = scoord.get_R_Z(qfit.OS_strike_point_s)
    qfit['R_strike_point'] = R
    qfit['Z_strike_point'] = Z
    return qfit


def Inter_ELM_times(PPF_ELMS, filter_window=[], inter_ELM_period=0.9):

    t_start = PPF_ELMS.N_elm.dimensions[0].data
    t_end = t_start + PPF_ELMS.ELM_duration.data
    elm_ind = np.array(range(len(t_start)))

    if filter_window:
        i_elm = np.logical_and(
            t_start > filter_window[0], t_end < filter_window[1])
        t_start = t_start[i_elm]
        t_end = t_end[i_elm]
        elm_duration = PPF_ELMS.ELM_duration.data[i_elm]
        elm_ind_global = elm_ind[i_elm]

    elm_ind_local = np.array(range(len(t_start)))

    t_elm_free = t_start[1:] - t_end[:-1]
    #t_mid = t_end[:-1] + (t_elm_free/2)

    t_elm_free_start = t_end[:-1]  # t_mid - (t_elm_free*inter_ELM_period)/2
    # t_mid + (t_elm_free*inter_ELM_period)/2
    t_elm_free_end = t_end[:-1] + (t_elm_free*inter_ELM_period)

    t_elm_free_start = np.append(t_elm_free_start, np.nan)
    t_elm_free_end = np.append(t_elm_free_end, np.nan)

    label = ['t_start', 't_end', 'elm_duration', 'pulse_elm_n',
             str(inter_ELM_period)+'int_elm_start', str(inter_ELM_period)+'int_elm_end']
    data = {label[0]: t_start, label[1]: t_end, label[2]: elm_duration, label[3]: elm_ind_global,
            label[4]: t_elm_free_start, label[5]: t_elm_free_end}

    ELMS = pd.DataFrame(data, index=elm_ind_local, columns=label)
    return ELMS


def time_split_radial_IR(QProf, time_interval):

    q_t = QProf.index.values
    q_t_ind = np.where(np.logical_and(
        q_t > time_interval[0], q_t < time_interval[1]))[0]
    Q = np.zeros([len(q_t_ind), len(QProf.columns.values)])

    for i, ind in enumerate(q_t_ind):
        Q[i, :] = QProf.iloc[ind].values
    return Q


def time_slice_qfit(Qfit, time_interval):
    q_t = Qfit.Time
    q_t_ind = np.where(np.logical_and(
        q_t > time_interval[0], q_t < time_interval[1]))[0]

    Q = Qfit.iloc[q_t_ind]
    Q = Q.reset_index(drop=True)
    return Q


def EFTF_strike_point(EFTF):
    eftf_sp_s = {'Time': EFTF.EFTF_SPr.dimensions[0].data,
                 'EFTF_R': EFTF.EFTF_SPr.data, 'EFTF_Z': EFTF.EFTF_SPz.data}
    EFTF_R = pd.DataFrame(eftf_sp_s, columns=['Time', 'EFTF_R', 'EFTF_Z'])
    EFTF_R['EFTF_S'] = scoord.get_s_coord(
        EFTF_R['EFTF_R'], EFTF_R['EFTF_Z'])
    return EFTF_R


def eich(x, lambda_q, S, q0, fx, q_bg):

    LFS = (S/(2*lambda_q*fx))
    def ADJ(x): return x/(lambda_q*fx)

    Q = np.zeros(np.shape(x))
    for i in range(len(x)):
        Q[i] = (q0/2)*np.exp((LFS**2) - ADJ(x[i]))*erfc(LFS - x[i]/S) + q_bg
    return Q


def compute_energy_balance(P_ohm, P_NBI, P_ICH, P_IR, P_RAD, dWdt, Psol):
    # https://doi.org/10.1088/1402-4896/aa8de7
    # Dynamic power balance analysis JET
    a = 1
    c = 1
    f = 1

    b = .96  # +/- .006
    d = 2.28  # +/- .014
    e = 1  # +/- .006
    # Equation2
    P_in = a*P_ohm + b*P_NBI + c*P_ICH
    P_out = d*P_IR + e*P_RAD + f*P_loss
    P_IR_pred = ((a*P_ohm + b*P_NBI + c*P_ICH)-(e*P_RAD + f*dWdt))/d

    P_balance = (P_RAD + Psol)-(P_ohm + P_NBI + P_ICH)

    return P_in, P_out, P_IR_pred, P_balance


def predict_IR(P_ohm, P_NBI, P_ICH, P_RAD, dWdt):
    # https://doi.org/10.1088/1402-4896/aa8de7
    # Dynamic power balance analysis JET
    a = 1
    c = 1
    f = 1

    b = .96  # +/- .006
    d = 2.28  # +/- .014
    e = 1  # +/- .006
    # Equation2
    P_IR = ((a*P_ohm + b*P_NBI + c*P_ICH)-(e*P_RAD + f*dWdt))/d
    return P_IR


def smooth_window(dt, T):
    time_smoothing = dt * 1E-3
    window = int(time_smoothing/(T[1]-T[0]))
    if window == 1:
        print('WARNING: smoothing interval small relative to signal sample frequency \n Smoothing interval: {}s \n Sample frequency: {:.2f}, int: {:.2f}'.format(
            time_smoothing, 1/(T[1]-T[0]), T[1]-T[0]))
    #if (window % 2) == 0:
    #    window = window-1
    return window


def smoothTriangle(data, degree, dropVals=False):
    triangle = np.array(list(range(degree)) +
                        [degree]+list(range(degree)[::-1]))+1
    smoothed = []
    for i in range(degree, len(data)-degree*2):
        point = data[i:i+len(triangle)]*triangle
        smoothed.append(sum(point)/sum(triangle))
    if dropVals:
        return smoothed
    smoothed = [smoothed[0]]*int(degree+degree/2)+smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


def IR_Q_to_Power_robust(QProf, torr_wet=1):
    """
    calculate divertor power
    """
    if isinstance(torr_wet, list):
        A = True
    else:
        A = False
    # intergrate, maximum, Lq and S at target
    t = QProf.index.values
    Int_Q = []
    for i in range(len(QProf.index)):
        Int_Q.append(integrate.trapz(2 * np.pi * QProf.columns.values *
                                     QProf.iloc[i].values * torr_wet, QProf.columns.values))

    Data = {'IntQ': Int_Q}
    df = pd.DataFrame(Data, index=t, columns=['IntQ'])
    return df

def IR_Q_to_Power(QProf,torr_wet=1,R = 2.79):
    """
    calculate divertor power
    """
    if isinstance(torr_wet,list):
        A = True
    else:
        A = False
    # intergrate, maximum, Lq and S at target
    t = QProf.index.values
    Int_Q = []
    for i in range(len(QProf.index)):
        if not A:
            Int_Q.append(2 * np.pi * R * torr_wet * integrate.trapz(
                QProf.iloc[i].values, QProf.columns.values))
        else:
            Int_Q.append(2 * np.pi * R * torr_wet[i] * integrate.trapz(
                QProf.iloc[i].values, QProf.columns.values))
        #integrate.trapz(
        #QProf.iloc[i].values, Q_pd.columns.values-min(Q_pd.columns.values)*0.94))

    Max_Q = QProf.max(axis=1).values

    Data = {'MaxQ': Max_Q, 'IntQ': Int_Q}
    df = pd.DataFrame(Data, index=t, columns=['MaxQ', 'IntQ'])
    return df


def IR_Q_to_Qr(QProf):
    """
    Intergrate radial heat flux profile to Power 
    """
    # intergrate, maximum, Lq and S at target
    t = QProf.index.values
    Int_Q = []
    for i in range(len(QProf.index)):
        Int_Q.append(integrate.trapz(
            QProf.iloc[i].values, QProf.columns.values))

        #integrate.trapz(
        #QProf.iloc[i].values, Q_pd.columns.values-min(Q_pd.columns.values)*0.94))

    Max_Q = QProf.max(axis=1).values

    Data = {'Qr': Int_Q}
    df = pd.DataFrame(Data, index=t, columns=['Qr'])
    return df


def get_BE_II(pulse_number):
    Outer = sal.get('/pulse/'+str(pulse_number)+'/ppf/signal/jetppf/edg8/tbeo')
    Inner = sal.get('/pulse/'+str(pulse_number)+'/ppf/signal/jetppf/edg8/tbei')
    return Inner,Outer

def calc_power_split(pulse_number,time):
    I, O = get_BE_II(pulse_number)
    Be_in = slice_power_data(I, time,'BeIn')
    Be_out = slice_power_data(O, time,'BeOut')
    Be_in['BeOut'] = Be_out['BeOut']
    split = integrate.trapz(Be_out['BeOut'], Be_out['Time']) / (integrate.trapz(Be_out['BeOut'], Be_out['Time']) + integrate.trapz(Be_in['BeIn'],Be_in['Time']))
    return Be_in, split

"""
def extract_ELM_profiles(Q_pd, ELMS, t_rise=0, P_threshold=2E7):
    p_threshold_ELM = 0.2E7
    Qprofile = [] # Panda array extracted ELM profile
    Q_intergrated_R = [] # Radial intergrated R
    stats_Q = [] # Max and integral of outside target
    Energy = [] # Energy of ELM 
    Deposition_R = []
    Deposition_Area = [] # ELM wetted radial area
    max_q = [] # Maximum Q

    Loading_parameter_S = []
    Loading_parameter_S_error = []
    Loading_time = []
    Loading_baseline_Q = []
    strike_point = []
    Loading_area = []
    Loading_q_max = []
    for i in range(len(ELMS.index)):

        t_ind = np.logical_and(
            Q_pd.index.values > ELMS.t_start[i]-t_rise, Q_pd.index.values < ELMS.t_end[i])
        Q = Q_pd.iloc[t_ind, :]
        
        while np.max(Q.values[0, :]) > P_threshold:
            t_rise = t_rise+1E-5
            t_ind = np.logical_and(
                Q_pd.index.values > ELMS.t_start[i]-t_rise, Q_pd.index.values < ELMS.t_end[i])
            Q = Q_pd.iloc[t_ind, :]

        Qprofile.append(Q)
        stats_Q.append(IR_Q_to_Power(Q))
        Q_intergrated_R.append(IR_Q_to_Qr(Q))

        Energy.append(integrate.trapz(
            stats_Q[i]['IntQ'].values, stats_Q[i].index.values))
        max_q.append(np.max(Q_pd.iloc[t_ind, :].values))
        Deposition_R.append(
            Q_pd.columns[(Q > P_threshold).any()].values)
        Deposition_Area.append(len(Q_pd.columns[(Q > P_threshold).any()].values)*(Q.columns.values[1] -Q.columns.values[0]))

        
        loading parameters
        
        #q0 = np.mean(Q.values[-3:,:])
        S, Error = fit_ELM_decay(
            Q, (Q > p_threshold_ELM).any()) #, q0)
        TMP,E = fit_ELM_profile(Q, P_threshold)
        strike_point.append(TMP[0])
        Loading_area.append(TMP[2])
        Loading_q_max.append(TMP[3])
        Loading_parameter_S.append(S)
        Loading_parameter_S_error.append(Error)
        Loading_time.append(
            Q.index.values-Q.max(axis=1).idxmax())
        Loading_baseline_Q.append(np.max(Q.values[-10:, :]))

    return Qprofile, Q_intergrated_R, stats_Q, max_q, Energy, Deposition_R, Deposition_Area, Loading_parameter_S, Loading_parameter_S_error, Loading_time, Loading_baseline_Q
"""

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def decay(x, q0, qmax, S):
    Q = np.ones(np.shape(x))*qmax
    Q[x < 0] = np.interp(x[x < 0], [np.min(x), 0], [q0, qmax])
    Q[x > 0] = ((qmax)*np.exp(-(S)*x[x > 0]*1E3))+q0
    return Q


def f(x):
    return s[0]*x**3 + s[1]*x**2 + s[2]*x + s[3]


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def fit_tau(Q, gr=2, gf=-10):
    """
    output
    T_max
    Tau_rise
    Tau_fall
    """
    q_prof = Q.max(axis=1)
    t_max = np.argmax(q_prof)

    q_rise = q_prof.index.values[:t_max]
    q_fall = q_prof.index.values[t_max:]

    dQ = q_prof/np.max(q_prof)

    grad_rise = np.gradient(dQ, q_prof.index.values)[:t_max]
    grad_fall = np.gradient(dQ, q_prof.index.values)[t_max:]

    ind_rise = np.where(grad_rise > gr)[0]
    ind_fall = np.where(grad_fall < gf)[0]

    # next(([i for i, j in enumerate(np.flip(ind_rise)) if not j]), None)
 
    try:
        i_rise = consecutive(ind_rise)[-1][0]
    except:
        try:
            i_rise = ind_rise[0]
        except:
            i_rise = len(ind_rise)

    tau_rise = q_prof.index.values[i_rise]
    tau_fall = np.max(q_prof.index.values[ind_fall])
    c = 0
    while tau_fall < q_prof.index[t_max]:
        print('itterating ELM fit : {}'.format(c))
        tau_fall = tau_fall+0.0002
        c += 1
    #if tau_fall < q_prof.index[t_max]:
    #    tau_fall = q_prof.index[t_max]+0.002
    if tau_rise >= q_prof.index[t_max]:
        tau_rise = q_prof.index[t_max] - (q_prof.index[-1]-q_prof.index[-2])
    return q_prof.index[t_max], np.min(tau_rise)-q_prof.index[t_max], tau_fall - q_prof.index[t_max]


"""
def fit_tau(Q, gr=1E6, gf=-1E3):
    q_prof = Q.max(axis=1)
    t_max = np.argmax(q_prof)
    q_rise = q_prof.iloc[:t_max]
    q_fall = q_prof.iloc[t_max:]

    ind_rise = np.diff(q_rise.values) > gr
    ind_fall = np.where(np.diff(q_fall.values) < gf)[0]

    ik = next((i for i, j in enumerate(np.flip(ind_rise)) if not j), None)
    tau_rise = q_prof.index.values[np.where(ind_rise)[0][-ik]]
    tau_fall = np.max(q_prof.index.values[ind_fall])
    if tau_fall < t_max:
        tau_fall = t_max+0.002
    if tau_rise >= t_max:
        tau_rise = t_max - (q_rise.iloc[-1]-q_rise.iloc[-2])
    return t_max, np.min(tau_rise)-t_max, tau_fall - t_max
"""

def fit_ELM_decay(Q, A):
    """
    S[0] = Q_baseline
    S[1] = Q_max
    S[2] = S - Decay parameter
    """
    # S[0] Optimal values for the parameters so that the sum of the squared residuals of
    # S[1] The estimated covariance of popt. The diagonals provide the variance of the parameter estimate. 
    # To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
    # How the sigma parameter affects the estimated covariance depends on absolute_sigma argument, as described above.
    #

    x = Q.index.values-Q.max(axis=1).idxmax()
    #q0 = np.mean(Q.values[-1, :])
    q_0 = np.max(Q.values[:3, :])
    q_m = np.max(Q.values)

    # np.mean(Q.values[:, A], axis=1)+((np.max(Q.values[:, A], axis=1) - np.mean(Q.values[:, A], axis=1))*.9)
    tm = np.max(Q.values[:, A], axis=1)
    #print('test max')
    Data = tm
    def q_hat(k, q0, qmax, S): return decay(k, q0, qmax, S)

    S = op.curve_fit(q_hat, x, Data, p0=[q_0, q_m, 2])
    s = S[0]
    Error = {}
    Error['rmse'] = np.sqrt(((q_hat(x, s[0],s[1],s[2])-Data)**2).mean())
    Error['Cov'] = S[1]
    return s, Error


def fit_ELM_profile(Q, P_threshold):
    """
    S[0] = Strike Point
    S[1] = Deposition width parameter 
    S[2] = Qmax
    """
    tmp = 1
    k = np.argmax(Q.max(axis=1).values)
    Dep_R = Q.columns[(Q > P_threshold).any()].values
    A = len(Dep_R)*(Q.columns.values[1] - Q.columns.values[0])
    x = Q.columns.values

    R = range((k-tmp), (k+tmp))
    #x = np.tile(x, [((k+tmp)-(k-tmp)),1])

    Data = np.max(Q.values, axis=0)  # Q.iloc[range((k-tmp),(k+tmp)), :].values
    Data[Data<P_threshold] = 0
    def q_hat(k, mu, sig, q): return q * gaussian(k, mu, sig)
    
    S = op.curve_fit(q_hat, x, Data, p0=[np.median(Dep_R), A/6, np.max(Q.values)], maxfev=int(1E5))

    s = S[0]
    Error = {}
    Error['rmse'] = np.sqrt(((q_hat(x, s[0], s[1], s[2])-Data)**2).mean())
    Error['Cov'] = S[1]
    return s, Error



def simulation_ELM_profile(Qmax, strike_point, width, x_r):
    """
    Qmax = intergrated radial Qmax profile
    strike_point = location of SP 
    width = width of gaussian profile
    x_r = Finite difference mesh points
    """
    Q_out = np.zeros(np.shape(x_r))
    Q_out = gaussian(x_r, strike_point, width)
    Q_out = Q_out * Qmax
    return Q_out


def simulation_ELM_integrate(times, x_r, q0, qmax, S, strike_point, Area, wetted_fraction = 0.94,power_split=0.5):
    q_decay = decay(times, q0, qmax, S)
    Q_ELM_MODEL = simulation_ELM_profile(qmax, strike_point, Area, x_r)

    Q_bdry = np.zeros([len(times), len(x_r)])
    QR = []
    Q_int = []
    for j in range(len(times)):
        Q_bdry[j, :] = (Q_ELM_MODEL/np.max(Q_ELM_MODEL))*q_decay[j]

    Q_bdry = Profile_to_Pandas(Q_bdry, x_r, times)
    QR = IR_Q_to_Qr(Q_bdry)
    Q_int = (IR_Q_to_Power_robust(Q_bdry, torr_wet=wetted_fraction))*(1/power_split)
    Energy = integrate.trapz(Q_int['IntQ'].values, Q_int.index.values)
    return Q_bdry, QR, Q_int, Energy

def compute_model_adjusted_ELM_Q_timeseries(times, x_r, ELM_pd, pulse_start_time):

    times = times + pulse_start_time
    Q = np.zeros([len(times),len(x_r)])

    for i in ELM_pd.index:
        t_start = ELM_pd.Time_ELM_Peak[i]-ELM_pd.Tau_rise[i]
        t_end = ELM_pd.Time_ELM_Peak[i]+ELM_pd.Tau_fall[i]

        t_ind = np.logical_and(times>t_start,times<t_end)
        ts = np.linspace(-ELM_pd.Tau_rise[i],ELM_pd.Tau_fall[i],sum(t_ind))
        q0 = ELM_pd.model_q_baseline[i]
        qmax = ELM_pd.model_Qmax[i]
        S = ELM_pd.model_S_decay[i]
        strike_point = ELM_pd.model_strike_point[i]
        Area = ELM_pd.model_area[i]
        Q_bdry, QR, Q_int, Energy = simulation_ELM_integrate(ts, x_r, q0, qmax, S, strike_point, Area, wetted_fraction = 0.94,power_split=0.5)

        Q[t_ind,:] = Q_bdry
        return Q

def extract_ELM_profiles(Q_pd, ELMS, x_r, strike_point, t_rise=0, t_fall=0.003,grad_rise=2,grad_fall=-10, P_threshold=2E7,wetted_fraction=0.9,power_split=1):
    p_threshold_ELM = 0.2E7
    #t_rise and t_fall are adjustment parameters for the detection window 
    print('Extracting ELM shape parameters ...')
    Qprofile = {}  # Panda array extracted ELM profile

    Energy = []
    max_q = []
    Deposition_Area = []
    Loading_strike_point = []
    Loading_area = []
    Loading_q_max_prof = []
    Loading_q_max_dec = []
    Loading_q_baseline = []
    Loading_S_decay = []
    q_max_fit = []
    model_e = []
    error_qmax = []
    error_rmse_decay = []
    error_rmse_prof = []
    time_elm_peak = []
    IR_elm_duration = []
    model_elm_duration= []
    Model_elm_duration = []
    Tau_rise = []
    Tau_fall = []

    drop_elms = []
    for l, i in enumerate(ELMS.index):
    
        t_ind = np.logical_and(
            Q_pd.index.values > ELMS.t_start[i]-t_rise, Q_pd.index.values < ELMS.t_end[i]+t_fall)#ELMS.t_end[i])
        Q = Q_pd.iloc[t_ind, :]

        t_elm, tau_rise, tau_fall = fit_tau(Q, gr=grad_rise, gf=grad_fall)
        t_ind = np.logical_and(
            Q_pd.index.values > t_elm+tau_rise, Q_pd.index.values < t_elm+tau_fall)  # ELMS.t_end[i])
        if np.sum(t_ind)<3:
            x = np.where(t_ind)[0]
            x = np.insert(x,0,x[0]-1)
            x = np.insert(x,len(x),x[-1]+1)
            t_ind = x 

            
        Q = Q_pd.iloc[t_ind, :]

        print('ind: {} - Time: {} Trise: {} Tfall: {}'.format(i,t_elm, tau_rise, tau_fall))
        #while np.max(Q.values[0, :]) > P_threshold:
        #    t_rise = t_rise+1E-5
        #    t_ind = np.logical_and(
        #        Q_pd.index.values > ELMS.t_start[i]-t_rise, Q_pd.index.values < ELMS.t_start[i]+t_fall)  # ELMS.t_end[i])
        #    Q = Q_pd.iloc[t_ind, :]
        time = Q.index.values-t_elm

        Qprofile[i] = {}
        Qprofile[i]['IR_Camera'] = {}
        Qprofile[i]['IR_Camera']['Qprofile'] = Q
        Qprofile[i]['IR_Camera']['IntQ_Divertor'] = IR_Q_to_Power_robust(Q,torr_wet=wetted_fraction)*(1/power_split)
        Qprofile[i]['IR_Camera']['IntQ'] = IR_Q_to_Qr(Q)
        Qprofile[i]['IR_Camera']['Deposition_coordinates'] = Q.columns[(
            Q > P_threshold).any()].values
        Qprofile[i]['IR_Camera']['R'] = Q.columns.values
        Qprofile[i]['ELM_time_loading'] = time

        Energy.append(integrate.trapz(
            Qprofile[i]['IR_Camera']['IntQ_Divertor']['IntQ'].values, Qprofile[i]['IR_Camera']['IntQ_Divertor']['IntQ'].index.values))
        max_q.append(np.max(Q.values))
        Deposition_Area.append(len(Q.columns[(Q > P_threshold).any(
        )].values)*(Q.columns.values[1] - Q.columns.values[0]))
        
        time_elm_peak.append(t_elm)
        IR_elm_duration.append(ELMS.t_end[i]-ELMS.t_start[i])
        """
        fitting parameters
        """
        S1, Error1 = fit_ELM_decay(Q, (Q > p_threshold_ELM).any()) 
        S2, Error2 = fit_ELM_profile(Q, P_threshold)

        QMax = S2[2]#((S1[1] - S2[2])/2) + S2[2]
        """
        Evaluate parameters
        """
        Qprofile[i]['Model'] = {}
        Q_bdry, model_QR, model_Q_int, model_Energy = simulation_ELM_integrate(
            time, x_r, S1[0], QMax, S1[2], S2[0] , S2[1],wetted_fraction=wetted_fraction,power_split=power_split)
        Qprofile[i]['Model']['x_r'] = x_r
        Qprofile[i]['Model']['Q_bdry'] = Q_bdry
        Qprofile[i]['Model']['IntQ'] = model_QR
        Qprofile[i]['Model']['IntQ_Divertor'] = model_Q_int

        Qprofile[i]['Model']['Error_decay'] = Error1
        Qprofile[i]['Model']['Error_prof'] = Error2

        Loading_strike_point.append(S2[0])
        Loading_area.append(S2[1])
        Loading_q_max_prof.append(S2[2])
        Loading_q_max_dec.append(S1[1])
        Loading_q_baseline.append(S1[0])
        Loading_S_decay.append(S1[2])
        q_max_fit.append(QMax)
        model_e.append(model_Energy)
        model_elm_duration.append(time[-1]-time[0])

        error_qmax.append((S1[1] - S2[2])/2)
        error_rmse_decay.append(Error1['rmse'])
        error_rmse_prof.append(Error2['rmse'])

        Model_elm_duration.append(time[-1]-time[0])
        Tau_rise.append(abs(time[0]))
        Tau_fall.append(abs(time[-1]))

        
            #drop_elms.append(i)
            #print('Failed to fit elm model to: \n \t ELM ind:{} - Time: {}-{} (sec)'.format(i, ELMS.t_start[i], ELMS.t_end[i]))

    ELMS_out = ELMS.copy(deep=True)
    #if len(drop_elms)>0:
    ELMS_out = ELMS_out.drop(index=drop_elms).reset_index(drop=True)

    ELMS_out['IR_Energy'] = np.array(Energy)
    ELMS_out['IR_max_q'] = np.array(max_q)
    ELMS_out['IR_Deposition_Area'] = np.array(Deposition_Area)
    ELMS_out['model_strike_point'] = np.array(Loading_strike_point)
    ELMS_out['model_area'] = np.array(Loading_area)
    ELMS_out['model_Energy'] = np.array(model_e)
    ELMS_out['model_Qmax'] = np.array(q_max_fit)
    ELMS_out['model_q_max_prof'] = np.array(Loading_q_max_prof)
    ELMS_out['model_q_max_dec'] = np.array(Loading_q_max_dec)
    ELMS_out['model_q_baseline'] = np.array(Loading_q_baseline)
    ELMS_out['model_S_decay'] = np.array(Loading_S_decay)
    ELMS_out['error_qmax'] = np.array(error_qmax)
    ELMS_out['Decay_rmse'] = np.array(error_rmse_decay)
    ELMS_out['Prof_rmse'] = np.array(error_rmse_prof)
    ELMS_out['Time_ELM_Peak'] = np.array(time_elm_peak)
    try :
        ELMS_out['EFIT_SP'] = np.interp(
            ELMS_out['Time_ELM_Peak'].values, strike_point['Times'].values, strike_point['SP_R'].values)
    except:
        print('No strike point data availiable')
    ELMS_out['ELM_duration'] = np.array(IR_elm_duration)
    ELMS_out['model_ELM_duration'] = np.array(model_elm_duration)
    ELMS_out['Tau_rise'] = np.array(Tau_rise)
    ELMS_out['Tau_fall'] = np.array(Tau_fall)
    ELMS_out = ELMS_out.dropna(axis=0).reset_index(drop=True)
    return ELMS_out, Qprofile


def derivative(f, a, method='central', h=0.01):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h            
    '''
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


def smooth_and_extract_ELMS(Q_pd, ELMS, R, strike_point, window_length=9, polyorder=3, wetted_fraction=0.9, power_split=1, t_rise=0.003, t_fall=0.004,  grad_rise=2, grad_fall=-10,  PN=90271, skip_elm=[]):
    print('smoothing_window : {}'.format(
        (Q_pd.columns.values[1]-Q_pd.columns.values[0])*window_length))
    
    if window_length>29:
        print('WARNING: Are you sure you want a smoothing windown this large!\n Defoming time series excessively!')
        print('recommend you investigat the smoothing against the maximum profile. ')

    if len(skip_elm):
        print('dropping ELMs: {} '.format(skip_elm))
        ELMS = ELMS.drop(index=skip_elm)
        ELMS = ELMS.reset_index(drop=True)

    T6_Q = np.zeros(np.shape(Q_pd.values))
    for i in range(len(Q_pd.columns.values)):
        T6_Q[:, i] = savgol_filter(
            Q_pd.values[:, i], window_length=window_length, polyorder=polyorder)

    T6_Q = pd.DataFrame(T6_Q, index=Q_pd.index.values,
                        columns=Q_pd.columns.values)

    ELMS_out_smooth, Qprofile_smooth = extract_ELM_profiles(
        T6_Q, ELMS, R, strike_point, P_threshold=2.5E7, t_rise=t_rise, t_fall=t_fall, grad_rise=grad_rise, grad_fall=grad_fall, wetted_fraction=wetted_fraction, power_split=power_split)


    return ELMS_out_smooth, T6_Q, Qprofile_smooth


def plot_ELM_3D(ELMS_pd, Q_dict, ELM_N):
    i = ELM_N
    X, Y = np.meshgrid(Q_dict[i]['ELM_time_loading'],
                       Q_dict[i]['Model']['x_r'])

    X_data, Y_data = np.meshgrid(
        Q_dict[i]['ELM_time_loading'], Q_dict[i]['IR_Camera']['R'])

    fig = plt.figure(figsize=[20, 20])
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.05, hspace=0)
    mpl.rcParams['legend.fontsize'] = 15
    ax = fig.add_subplot(gs[0, :2], projection='3d')
    #ax.view_init(elev=0., azim=0)
    ax.plot_surface(X_data, Y_data, Q_dict[i]['IR_Camera']
                    ['Qprofile'].values.T, alpha=0.4, label='KL9a IR profile')
    ax.plot_wireframe(
        X, Y, Q_dict[i]['Model']['Q_bdry'].T, color='red', label='ELM model fit')
    ax.set_ylabel('Major Radius (m)', fontsize=15, rotation=35)
    ax.set_xlabel('Time (s)', fontsize=15, rotation=-5)
    ax.set_zlabel('Heat Flux (W)', fontsize=15, rotation=85)
    #fig = plt.figure(figsize=[20,20])
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(Q_dict[i]['IR_Camera']['R'], Q_dict[i]['IR_Camera']
             ['Qprofile'].values.T, c='C0')
    ax1.plot(Q_dict[i]['Model']['x_r'], Q_dict[i]['Model']['Q_bdry'].max(
        axis=0), linewidth=3, color='red')
    ax1.set_xlabel('Major Radius (m)', fontsize=15)
    ax1.set_ylabel('Heat Flux (W)', fontsize=15)
    ax2 = fig.add_subplot(gs[1, 1])
    #ax2.view_init(elev=0., azim=-90)
    #ax2.plot_surface(X_data, Y_data, Qprofile[i].values.T, alpha=0.4)
    #ax2.plot_wireframe(X, Y, Q_bdry.T, color='red')
    ax2.plot(Q_dict[i]['ELM_time_loading'], Q_dict[i]['IR_Camera']
             ['Qprofile'], color='#1f77b4', alpha=0.4)
    ax2.plot(Q_dict[i]['ELM_time_loading'], Q_dict[i]['IR_Camera']
             ['Qprofile'].iloc[:, -1], color='#1f77b4', alpha=0.4, label='KL9a IR profile')
    ax2.plot(Q_dict[i]['ELM_time_loading'], np.max(Q_dict[i]['IR_Camera']
                                                   ['Qprofile'], axis=1), linewidth=2, color='black')
    ax2.plot(Q_dict[i]['ELM_time_loading'], Q_dict[i]['Model']['Q_bdry'].max(axis=1),
             linewidth=3, color='red', label='ELM model fit')
    ax2.set_xlabel('Time (s)', fontsize=15)
    ax2.legend()
    plt.show()


def plot_ELM_MODEL(ELMS_out,colour='red'):
    from matplotlib.ticker import FormatStrFormatter

    plot_elm_array = ELMS_out[['model_area', 'model_Qmax', 'model_q_baseline', 'model_S_decay',
                               'model_ELM_duration', 'Tau_rise', 'Tau_fall', 'model_strike_point']].copy(deep=True)
    plot_elm_array = plot_elm_array.drop(
        index=np.where(plot_elm_array['model_area'] < 0.01)[0])
    #plot_elm_array['model_area'] = plot_elm_array['model_area'] /1E3
    plot_elm_array = plot_elm_array.rename(columns={'model_area': 'Width (m)', 'model_Qmax': 'Q_max (W)', 'model_q_baseline':'Q0 (W)',
                                                    'model_S_decay': 'S', 'model_ELM_duration': 'tau (s)', 'model_strike_point': 'Major Rad (m)', 'Tau_rise':'tau_rise (s)', 'Tau_fall':'tau_fall (s)'})

    cols = list(plot_elm_array.columns.values)
    cols.pop(cols.index('Q_max (W)'))
    plot_elm_array = plot_elm_array[cols + ['Q_max (W)']]
    fig = pd.plotting.scatter_matrix(
        plot_elm_array, marker='+', color=colour, figsize=[20, 20], diagonal='kde')
    for ax in fig.flatten():
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        if ymax > 1E3 or ymax < 1E-2:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        if xmax > 1E3 or xmax <1E-2:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))

        deltay = (ymax - ymin)*0.05
        deltax = (xmax - xmin)*0.05

        ax.set_yticks(np.round(np.linspace(ymin+deltay, ymax-deltay, 5), 4))
        ax.set_xticks(np.round(np.linspace(xmin+deltax, xmax-deltax, 5), 4))



def plot_elm_in_context(ELMS_out, Qprofile, Q_pd, i,ir_q_max = []):
    plot_times = [ELMS_out.t_start[i]-0.05, ELMS_out.t_end[i]+0.05]

    ind_q_prof = np.logical_and(
        Q_pd.index.values > plot_times[0], Q_pd.index.values < plot_times[1])
    ind_elms = np.logical_and(
        ELMS_out.t_start > plot_times[0], ELMS_out.t_end < plot_times[1])

    plt.figure(figsize=[15, 10])
    plt.plot(Q_pd.index.values[ind_q_prof],
             Q_pd.values[ind_q_prof, :], color='#1f77b4')

    for j in range(np.sum(ind_elms)):
        plt.plot(Qprofile[np.where(ind_elms)[0][j]]['ELM_time_loading']+ELMS_out.Time_ELM_Peak[np.where(ind_elms)[0][j]],
                 Qprofile[np.where(ind_elms)[0][j]]['Model']['Q_bdry'].max(axis=1), color='red')

    if ir_q_max:
        plt.plot(ir_q_max.dimensions[0].data,ir_q_max.data,color='black')
        plt.xlim([Q_pd.index.values[ind_q_prof].min(),Q_pd.index.values[ind_q_prof].max()])
    plt.plot(Qprofile[i]['ELM_time_loading']+ELMS_out.Time_ELM_Peak[i],
             Qprofile[i]['Model']['Q_bdry'].max(axis=1), color='green')

    plt.xlabel('Time (s)')
    plt.ylabel('Heat Flux (W)')

    plt.figure()
    plt.bar('IR Energy', ELMS_out['IR_Energy'][i])
    plt.bar('Model Energy', ELMS_out['model_Energy'][i])


def plot_mixed_gaussian_model(X_test, X_train, sensitivity_indicies = [] , identify_point = None, additional_plot=[],save_figure=[]):
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter
    mpl.rcParams.update({'font.size': 18})
    
    n_vars = np.shape(X_train)[1]
    labels = X_train.columns.values
    delta = []
    per = 0.1
    for i in range(n_vars):
        delta.append((X_train[labels[i]].max()-X_train[labels[i]].min())*per)

    fig = plt.figure(figsize=[30, 30])
    gs = gridspec.GridSpec(n_vars, n_vars)
    gs.update(wspace=0.05, hspace=0)
    axs = []
    k = 0
    for j in range(n_vars-1,0,-1):
        for i in range(j):
            axs.append(fig.add_subplot(gs[j, i]))
            axs[-1].set_ylim([X_test[labels[j]].min() - delta[j],
                              X_test[labels[j]].max()+delta[j]])
            axs[-1].set_xlim([X_test[labels[i]].min() - delta[i],
                              X_test[labels[i]].max()+delta[i]])
            if i == 0:
                axs[-1].set_ylabel(labels[j])
                axs[-1].set_ylim([X_test[labels[j]].min() -delta[j], X_test[labels[j]].max()+delta[j]])
            else:
                axs[-1].axes.yaxis.set_visible(False)
            if j == n_vars-1:
                axs[-1].set_xlabel(labels[i])
                axs[-1].set_xlim([X_test[labels[i]].min() -delta[i], X_test[labels[i]].max()+delta[i]])
            else:
                axs[-1].axes.xaxis.set_visible(False)
            axs[-1].scatter(X_test[labels[i]], X_test[labels[j]], s=40, c='C0')
            axs[-1].scatter(X_train[labels[i]],X_train[labels[j]], s=40, c='red')

            if identify_point is not None:
                axs[-1].plot(identify_point[labels[i]]*np.ones(2),[X_test[labels[j]].min(), X_test[labels[j]].max()], c='green')
                axs[-1].plot([X_test[labels[i]].min(), X_test[labels[i]].max()],identify_point[labels[j]]*np.ones(2), c='green')
        k = len(axs)-1

    for i in range(n_vars):
        ax = fig.add_subplot(gs[i, i])
        ax.axes.yaxis.set_visible(False)
        ax.axes.xaxis.set_visible(False)
        try:
            sns.kdeplot(X_test[labels[i]].values, ax=ax, color='blue')
            sns.kdeplot(X_train[labels[i]].values, ax=ax, color='red')
        except:
            ax.hist(X_test[labels[i]].values, 300, color='blue')
            ax.hist(X_train[labels[i]].values, 300, color='red')
        if i == n_vars-1:
            ax.set_xlabel(labels[i])
            ax.set_xticks(np.round(np.linspace(X_test[labels[i]].min(
            ) - delta[i], X_test[labels[i]].max() + delta[i], 3), 2))
            ax.axes.xaxis.set_visible(True)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if X_test[labels[i]].max() > 1E3 or X_test[labels[i]].max() < 1E-2:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    if save_figure:
        fig.savefig(save_figure)
    plt.show()
    if sensitivity_indicies:
        adds = fig.add_subplot(gs[0:2, 2:6])
        adds.bar(np.array(range(0, np.shape(sensitivity_indicies[0])[0]*2, 2))+0.3, sensitivity_indicies[0], yerr=sensitivity_indicies[1], alpha=0.8, label='Total effect')
        adds.bar(np.array(range(0,np.shape(sensitivity_indicies[2])[0]*2,2))-0.3, sensitivity_indicies[2], yerr=sensitivity_indicies[3], alpha=0.5,label='First order')
        adds.set_xticks(np.array(range(0, np.shape(sensitivity_indicies[1])[0]*2, 2)))
        adds.set_xticklabels(labels)
        adds.tick_params(axis="y",direction="in", pad=-30)
        adds.tick_params(axis="x",direction="in", pad=-20)
        adds.set_yticks(np.round(np.linspace(-0.25,0.6, 4), 1))
        adds.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        adds.legend()
    if additional_plot:
        addit = fig.add_subplot(gs[2:4, 4:6])
        fig.axes.append(additional_plot)
        #fig.add_axes(additional_plot)
        additional_plot.set_position(addit.get_position())
    


def add_subplot_axes(ax, rect):
    """
    USAGE
    rect = [0.48, 0.001, 0.28, 0.4]
    subax1 = add_subplot_axes(axs[2], rect)
    subax1.plot(x, y, label='P_T5', c='green')
    """

    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

"""
Inter ELM update 
"""


def clean_Q_prof(QProf, time):
    ind = np.logical_and(QProf.dimensions[0].data >= time[0],  # E_90271.pulse_times[0],
                         QProf.dimensions[0].data <= time[1])  # E_90271.pulse_times[1])
    T_t = QProf.dimensions[0].data[ind]
    T_x = QProf.dimensions[1].data
    T_dat = QProf.data[ind, :]
    S_scale = np.interp(T_x, (T_x.min(), T_x.max()), ([1363.3, 1552.6]))
    R_scale = scoord.get_R_Z(S_scale)
    Q_prof = Profile_to_Pandas(QProf.data[ind, :], R_scale[0], T_t)
    return Q_prof


def extract_inter_ELM(QProf, window):

    ind = np.logical_and(QProf.index.values >
                         window[0], QProf.index.values < window[1])

    Q = QProf.values[ind, :]
    return Q


def clean_inter_ELM_profile(QProf, interps=[.05, .5, .95]):

    q_mu = []
    q_max = []
    q_min = []

    for i in range(np.shape(QProf)[1]):
        delt = (np.max(QProf[:, i])-np.min(QProf[:, i]))*.1
        x_d = np.linspace(np.min(QProf[:, i])-delt,
                          np.max(QProf[:, i])+delt, int(1E4))
        kd = gaussian_kde(QProf[:, i], bw_method=.1).evaluate(x_d)

        limit = np.interp(interps, (1/sum(kd))*np.cumsum(kd), x_d)
        limit[limit < 0] = 0
    return limit


def construct_training_data(QProf, ELM):

    training_data = {}

    for i in range(len(ELM.index)):
        training_data[i] = {}
        Q = extract_inter_ELM(
            QProf, [ELM['0.9int_elm_start'][i], ELM['0.9int_elm_end'][i]])
        limit = clean_inter_ELM_profile(Q, interps=[.25, .5, .95])
        training_data[i]['limit'] = limit
        training_data[i]['strike_point_ind'] = np.argmax(limit[-1])
        training_data[i]['strike_point'] = QProf.columns.values[np.argmax(
            limit[-1])]
        training_data[i]['maxQ'] = Q.max()
        training_data[i]['min_IntQ'] = integrate.trapz(
            limit[0], QProf.columns.values)
        training_data[i]['max_IntQ'] = integrate.trapz(
            limit[-1], QProf.columns.values)

    return training_data


def make_training_data_table(training_data):
    labels = ['maxQ', 'minmaxQ', 'mumaxQ', 'maxmaxQ',
              'minint', 'muint', 'maxint', 'strike_point']
    Data = np.zeros([len(training_data.keys()), len(labels)])
    for i in training_data.keys():
        A = training_data[i]['maxQ']
        QQ = np.concatenate([training_data[i]['limit']])
        B, C, D = np.max(QQ, axis=1)
        E = training_data[i]['min_IntQ']
        F = (training_data[i]['min_IntQ'] + training_data[i]['max_IntQ'])/2
        G = training_data[i]['max_IntQ']
        H = training_data[i]['strike_point']

        data = [A, B, C, D, E, F, G, H]
        Data[i, :] = data
    Data = pd.DataFrame(Data, columns=labels)
    return Data


def evaluate_gp(GP, gp_scale, x_r, lambda_q, s, pbaseline, psol):
    pred = np.concatenate([[np.ones(len(x_r))*lambda_q],[np.ones(len(x_r))*s], [x_r]], axis=0)
    Q_pred, Q_std = GP.predict(pred.T)
    Q_pred = np.reshape(Q_pred, [len(Q_pred)])
    Q = Q_pred*gp_scale/pbaseline
    return Q*psol

def split_training_data_tables(Data, splits):
    train_data_tables = {}
    for i in range(len(splits)):
        tmp = Data.copy(deep=True)
        ind = np.logical_and(tmp['strike_point'].values >splits[i][0], tmp['strike_point'].values < splits[i][1])
        train_data_tables[i] = tmp.iloc[ind]
    return train_data_tables

def format_footprint_training_data(training_data,x_r):

    minimum_profile = np.zeros([len(training_data.keys()),len(training_data[0]['limit'][0])])
    maximum_profile = np.zeros([len(training_data.keys()),len(training_data[0]['limit'][2])])
    mean_profile =    np.zeros([len(training_data.keys()),len(training_data[0]['limit'][1])])

    for i in training_data.keys():
        minimum_profile[i,:] = sig.savgol_filter(training_data[i]['limit'][0],9,2)
        maximum_profile[i, :] = sig.savgol_filter(training_data[i]['limit'][2],9,2)
        mean_profile[i, :] = sig.savgol_filter(training_data[i]['limit'][1],9,2)

    minP = pd.DataFrame(minimum_profile, columns=x_r)
    maxP = pd.DataFrame(maximum_profile, columns=x_r)
    muP  = pd.DataFrame(mean_profile, columns=x_r)

    return minP, maxP, muP


"""
SENSITVITY 
"""

def compute_sobol(model, P_model, Nsamples, labels=[], conf_level=0.9, bootstrap=True, boot_n=100, verbose=True, plot=True):
    p = mp.Pool(8)
    print('Evaluating Sobol Sensitivity Indicies.')
    print('Smapling model ...')
    A = P_model(Nsamples)
    B = P_model(Nsamples)
    n_dim = np.shape(A)[1]

    if not labels:
        labels = []
        for i in range(n_dim):
            labels.append('Input_{}'.format(i))

    print('Building shuffle matrix ...')
    d = B
    D = np.repeat(B[None, ...], n_dim, axis=0)
    for i in range(n_dim):
        D[i][:, i] = A[:, i]

    t_0 = time.time()
    print('Evaluating model {} times...'.format(Nsamples))

    def func(i):
        o_a = model(A[i, :])
        o_b = model(B[i, :])
        O = np.zeros(n_dim)
        for j in range(n_dim):
            O[j] = model(D[j][i, :])
        Y = np.concatenate([np.array([o_a, o_b]), O])
        return Y
    results = p.map(func, range(Nsamples))

    Res = np.concatenate([results], axis=1)
    Y = Res
    
    print('Completed model evaluation : Time Elapsed : {}(sec)'.format(time.time()-t_0))
    # normalize Y
    Y = (Y-np.mean(Y)) / Y.std()

    # conf interval
    Z = stats.norm.ppf(0.5 + conf_level / 2)

    Var_X = first_order(Y[:, 0], Y[:, 1], Y[:, 2:])
    E_X = total_order(Y[:, 0], Y[:, 1], Y[:, 2:])

    print('Bootstrapping : {}'.format(bootstrap))
    V_err = []
    E_err = []
    #
    d_ind = np.random.randint(Nsamples, size=[Nsamples, boot_n])
    for i in range(boot_n):
        # V_err.append(first_order(Y[d_ind, 0], Y[d_ind, 1], Y[d_ind, 2:]))
        V_err.append(first_order(Y[d_ind[i], 0],
                                 Y[d_ind[i], 1], Y[d_ind[i], 2:]))
        # E_err.append(total_order(Y[d_ind, 0], Y[d_ind, 1], Y[d_ind, 2:]))
        E_err.append(total_order(Y[d_ind[i], 0],
                                 Y[d_ind[i], 1], Y[d_ind[i], 2:]))
    V_err = Z * np.array(V_err).std(ddof=1, axis=0)
    E_err = Z * np.array(E_err).std(ddof=1, axis=0)

    #if verbose:
    #    for i in range(n_dim):
    #    print('Sensitivity Incicies for {} dimensions'.format(n_dim))
    #                                                    Var_X[i], labels, E_X[i]))
    #        print('S_{} : {:2f} - S_T{} : {:2f}'.format(labels[i],
    #                                                    V_err[i], labels, E_err[i]))
    #        print('Var_s{} : {} - Var_sT{} : {}'.format(labels[i],

    return Var_X, E_X, V_err, E_err, Y, Res, A, B, D


def first_order(A_p, B_p, D_p):
    N, n_dim = np.shape(D_p)

    Var_X = np.zeros(n_dim)
    for i in range(n_dim):
        Var_X[i] = np.mean(B_p * (D_p[:, i]-A_p), axis=0) / \
            np.var(np.r_[A_p, B_p], axis=0)
    return Var_X


def total_order(A_p, B_p, D_p):
    N, n_dim = np.shape(D_p)

    E_X = np.zeros(n_dim)
    for i in range(n_dim):
        E_X[i] = 1/2 * (np.mean((A_p - D_p[:, i])**2, axis=0)
                        ) / np.var(np.r_[A_p, B_p], axis=0)
    return E_X


def rank_correlation(X, Y):
    xk = X - np.mean(X)
    yk = Y - np.mean(Y)
    RCC = np.sum((xk)*(yk))/(np.sum(xk**2) * np.sum(yk**2))**0.5
    return RCC


def partial_corr(C):

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def PRCC(X_pd, Y_pd):

    r = partial_corr(X_pd.values)  # input-input partial correlations
    rcc_y_i = np.zeros(len(X_pd.columns.values))
    rcc_i_y = np.zeros(len(X_pd.columns.values))
    for i, label in enumerate(X_pd.columns.values):
        rcc_y_i[i] = rank_correlation(
            X_pd[label].values, np.squeeze(Y_pd.values))
        rcc_i_y[i] = rank_correlation(
            np.squeeze(Y_pd.values), X_pd[label].values)

    C = np.ones(np.array(np.shape(r))+1)
    C[:np.shape(r)[0], :np.shape(r)[0]] = r
    C[:-1, -1] = rcc_i_y
    C[-1, :-1] = rcc_y_i

    prcc = np.zeros(np.shape(r)[0])
    for j in range(np.shape(r)[0]):
        prcc[j] = -(C.T[j, j+1])/np.sqrt(C.T[j, j]*C.T[j+1, j+1])
    return prcc


"""
def plot_mixed_gaussian_model(X_test, X_train):
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter

    Q_delta = (X_train.model_Qmax.max() - X_train.model_Qmax.min())*0.1
    A_delta = (X_train.model_area.max() - X_train.model_area.min())*0.1
    S_delta = (X_train.model_S_decay.max() - X_train.model_S_decay.min())*0.1
    t_delta = (X_train.model_ELM_duration.max() -
               X_train.model_ELM_duration.min())*0.1
    SP_delta = (X_train.model_strike_point.max() -
                X_train.model_strike_point.min())*0.1

    fig = plt.figure(figsize=[30, 30])
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0)

    "bottom row"
    axs0 = fig.add_subplot(gs[4, 0])
    axs0.scatter(X_test[:, 0], X_test[:, 1],
                 s=40, cmap='red')  # c=L_test,
    axs0.scatter(X_train.model_area.values, X_train.model_Qmax.values,
                 color='red')
    axs0.set_xlabel('Width (m)')
    axs0.set_ylabel('Qmax (W)')
    # axs0.set_xlim([X_train.model_area.min()-A_delta, X_train.model_area.max()+A_delta])
    axs0.set_xlim([X_test[:, 0].min()-A_delta, X_test[:, 0].max()+A_delta])
    # axs0.set_ylim([X_train.model_Qmax.min()-Q_delta, X_train.model_Qmax.max()+Q_delta])
    axs0.set_ylim([X_test[:, 1].min()-Q_delta, X_test[:, 1].max()+Q_delta])
    axs0.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs0.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    axs1 = fig.add_subplot(gs[4, 1], sharey=axs0)
    axs1.scatter(X_test[:, 2], X_test[:, 1],
                 s=40, cmap='red')
    axs1.scatter(X_train.model_S_decay.values,
                 X_train.model_Qmax.values, color='red')
    #c=L_test,
    axs1.set_xlabel('S')
    axs1.set_xlim([X_test[:, 2].min() -
                   S_delta, X_test[:, 2].max()+S_delta])
    axs1.axes.yaxis.set_visible(False)
    axs1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axs2 = fig.add_subplot(gs[4, 2], sharey=axs0)
    axs2.scatter(X_test[:, 3], X_test[:, 1],
                 s=40, cmap='red')  # c=L_test,
    axs2.scatter(X_train.model_ELM_duration.values,
                 X_train.model_Qmax.values, color='red')
    axs2.set_xlim([X_test[:, 3].min()-t_delta,
                   X_test[:, 3].max()+t_delta])
    axs2.set_xticks(np.round(np.linspace(
        X_test[:, 3].min(), X_test[:, 3].max(), 3), 3))
    axs2.set_xlabel('tau (s)')
    axs2.axes.yaxis.set_visible(False)
    axs2.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    axs3 = fig.add_subplot(gs[4, 3], sharey=axs0)
    axs3.scatter(X_test[:, 4], X_test[:, 1],
                 s=40, cmap='red')  # c=L_test,
    axs3.scatter(X_train.model_strike_point.values,
                 X_train.model_Qmax.values, color='red')
    axs3.set_xlabel('Major Radius (m)')
    axs3.axes.yaxis.set_visible(False)
    axs3.set_xlim([X_test[:, 4].min() -
                   SP_delta, X_test[:, 4].max()+SP_delta])
    axs3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    "second row"
    axs4 = fig.add_subplot(gs[3, 0], sharex=axs0)
    axs4.scatter(X_test[:, 0], X_test[:, 4],
                 s=40, cmap='red')
    axs4.scatter(X_train.model_area.values, X_train.model_strike_point.values,
                 color='red')
    axs4.set_ylim(
        [X_test[:, 4].min()-SP_delta, X_test[:, 4].max()+SP_delta])
    axs4.set_ylabel('Major Radius (m)')
    axs4.axes.xaxis.set_visible(False)
    axs4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axs5 = fig.add_subplot(gs[3, 1], sharey=axs4, sharex=axs1)
    axs5.scatter(X_test[:, 2], X_test[:, 4],
                 s=40, cmap='red')
    axs5.scatter(X_train.model_S_decay.values,
                 X_train.model_strike_point.values, color='red')
    axs5.axes.yaxis.set_visible(False)
    axs5.axes.xaxis.set_visible(False)

    axs6 = fig.add_subplot(gs[3, 2], sharey=axs4, sharex=axs2)
    axs6.scatter(X_test[:, 3], X_test[:, 4],
                 s=40, cmap='red')
    axs6.scatter(X_train.model_ELM_duration.values,
                 X_train.model_strike_point.values, color='red')
    axs6.axes.yaxis.set_visible(False)
    axs6.axes.xaxis.set_visible(False)
    "third row"
    axs7 = fig.add_subplot(gs[2, 0], sharex=axs0)
    axs7.scatter(X_test[:, 0], X_test[:, 3],
                 s=40, cmap='red')
    axs7.scatter(X_train.model_area.values, X_train.model_ELM_duration.values,
                 color='red')
    axs7.set_ylim(
        [X_test[:, 3].min()-t_delta, X_test[:, 3].max()+t_delta])
    axs7.set_ylabel('tau (s)')
    axs7.axes.xaxis.set_visible(False)
    axs7.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axs8 = fig.add_subplot(gs[2, 1], sharey=axs7, sharex=axs1)
    axs8.scatter(X_test[:, 2], X_test[:, 3],
                 s=40, cmap='red')
    axs8.scatter(X_train.model_S_decay.values,
                 X_train.model_ELM_duration.values, color='red')
    axs8.axes.yaxis.set_visible(False)
    axs8.axes.xaxis.set_visible(False)
    "Top row"
    axs9 = fig.add_subplot(gs[1, 0], sharex=axs0)
    axs9.scatter(X_test[:, 0], X_test[:, 2],
                 s=40, cmap='red')
    axs9.scatter(X_train.model_area.values, X_train.model_S_decay.values,
                 color='red')
    axs9.set_ylim(
        [X_test[:, 2].min()-S_delta, X_test[:, 2].max()+S_delta])
    axs9.set_ylabel('S')
    axs9.axes.xaxis.set_visible(False)
    axs9.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    "ECDF ['model_area', 'model_Qmax', 'model_S_decay','ELM_duration', 'model_strike_point']"

    axs_W = fig.add_subplot(gs[0, 0], sharex=axs0)
    axs_W.axes.xaxis.set_visible(False)
    sns.kdeplot(X_test[:, 0], ax=axs_W, color='blue')
    sns.kdeplot(X_train.model_area.values, ax=axs_W,  color='red')
    axs_W.axes.yaxis.set_visible(False)
    #axs_W.text(0.1, 0.9, 'A: {:2e}'.format(
    #    up.area_metric_robust(X_test[:, 0], X_train.model_area.values)[0]),size=12)

    axs_S = fig.add_subplot(gs[1, 1], sharex=axs1)
    axs_S.axes.xaxis.set_visible(False)
    axs_S.axes.yaxis.set_visible(False)
    sns.kdeplot(X_test[:, 2], ax=axs_S, color='blue')
    sns.kdeplot(X_train.model_S_decay.values, ax=axs_S,  color='red')
    #axs_S.text(0.1, 0.9, 'A: {:2e}'.format(up.area_metric_robust(
    #    X_test[:, 2], X_train.model_S_decay.values)[0]),size=12)

    axs_T = fig.add_subplot(gs[2, 2], sharex=axs2)
    axs_T.axes.xaxis.set_visible(False)
    axs_T.axes.yaxis.set_visible(False)
    sns.kdeplot(X_test[:, 3], ax=axs_T, color='blue')
    sns.kdeplot(X_train.model_ELM_duration.values, ax=axs_T,  color='red')
    #axs_T.text(0.1, 0.9, 'A: {:2e}'.format(up.area_metric_robust(
    #    X_test[:, 3], X_train.ELM_duration.values)[0]),size=12)

    axs_SP = fig.add_subplot(gs[3, 3], sharex=axs3)
    axs_SP.axes.xaxis.set_visible(False)
    axs_SP.axes.yaxis.set_visible(False)
    sns.kdeplot(X_test[:, 4], ax=axs_SP, color='blue')
    sns.kdeplot(X_train.model_strike_point.values, ax=axs_SP,  color='red')
    #axs_SP.text(0.1, 0.9, 'A: {:2e}'.format(up.area_metric_robust(
    #    X_test[:, 4], X_train.model_strike_point.values)[0]),size=12)

    axs_Q = fig.add_subplot(gs[4, 4])
    axs_Q.axes.yaxis.set_visible(False)
    sns.kdeplot(X_test[:, 1], ax=axs_Q, color='blue')
    sns.kdeplot(X_train.model_Qmax.values, ax=axs_Q,  color='red')
    axs_Q.set_xlabel('Qmax (W)')
    axs_Q.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    axs_Q.set_xticks(np.round(np.linspace(
        X_test[:, 1].min()-Q_delta, X_test[:, 1].max()+Q_delta, 3), 2))
    #axs_Q.text(0.1, 0.9, 'A: {:2e}'.format(up.area_metric_robust(X_test[:, 1], X_train.model_Qmax.values)[0]),size=12)
"""



"""
compicated plot of the reduced model 
"""
"""
mpl.rcParams.update({'font.size': 18})
X_train = X_pd
sensitivity_indicies = ln
n_vars = np.shape(X_train)[1]
labels = X_train.columns.values
delta = []
per = 0.1
for i in range(n_vars):
    delta.append((X_train[labels[i]].max()-X_train[labels[i]].min())*per)

fig = plt.figure(figsize=[30, 30])
gs = gridspec.GridSpec(n_vars, n_vars)
gs.update(wspace=0.05, hspace=0)
axs = []
k = 0
for j in range(n_vars-1, 0, -1):
    for i in range(j):
        axs.append(fig.add_subplot(gs[j, i]))
        axs[-1].set_ylim([X_test[labels[j]].min() - delta[j],
                          X_test[labels[j]].max()+delta[j]])
        axs[-1].set_xlim([X_test[labels[i]].min() - delta[i],
                          X_test[labels[i]].max()+delta[i]])
        if i == 0:
            axs[-1].set_ylabel(labels[j])
            axs[-1].set_ylim([X_test[labels[j]].min() - delta[j],
                              X_test[labels[j]].max()+delta[j]])
        else:
            axs[-1].axes.yaxis.set_visible(False)
        if j == n_vars-1:
            axs[-1].set_xlabel(labels[i])
            axs[-1].set_xlim([X_test[labels[i]].min() - delta[i],
                              X_test[labels[i]].max()+delta[i]])
        else:
            axs[-1].axes.xaxis.set_visible(False)
        axs[-1].scatter(X_test[labels[i]], X_test[labels[j]], s=40, c='C0')
        axs[-1].scatter(X_train[labels[i]], X_train[labels[j]], s=40, c='red')

        if identify_point is not None:
            axs[-1].plot(identify_point[labels[i]]*np.ones(2),
                         [X_test[labels[j]].min(), X_test[labels[j]].max()], c='green')
            axs[-1].plot([X_test[labels[i]].min(), X_test[labels[i]].max()],
                         identify_point[labels[j]]*np.ones(2), c='green')
    k = len(axs)-1

for i in range(n_vars):
    ax = fig.add_subplot(gs[i, i])
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    sns.kdeplot(X_test[labels[i]].values, ax=ax, color='blue')
    sns.kdeplot(X_train[labels[i]].values, ax=ax,  color='red')
    if i == n_vars-1:
        ax.set_xlabel(labels[i])
        ax.set_xticks(np.round(np.linspace(X_test[labels[i]].min(
        ) - delta[i], X_test[labels[i]].max() + delta[i], 3), 2))
        ax.axes.xaxis.set_visible(True)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if X_test[labels[i]].max() > 1E3 or X_test[labels[i]].max() < 1E-2:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
if sensitivity_indicies:
    adds = fig.add_subplot(gs[0:2, 2:6])
    adds.bar(np.array(range(0, np.shape(sensitivity_indicies[0])[
             0]*2, 2))+0.3, sensitivity_indicies[0], yerr=sensitivity_indicies[1], alpha=0.8, label='Total effect')
    adds.bar(np.array(range(0, np.shape(sensitivity_indicies[2])[
             0]*2, 2))-0.3, sensitivity_indicies[2], yerr=sensitivity_indicies[3], alpha=0.5, label='First order')
    adds.set_xticks(
        np.array(range(0, np.shape(sensitivity_indicies[1])[0]*2, 2)))
    adds.set_xticklabels(labels)
    adds.tick_params(axis="y", direction="in", pad=-30)
    adds.tick_params(axis="x", direction="in", pad=-20)
    adds.set_yticks(np.round(np.linspace(-0.25, 0.6, 4), 1))
    adds.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    adds.legend()
ax = fig.add_subplot(gs[2:4, 4:6])
#fig.add_subplot(111)
ax.fill_betweenx(P_E_M_LB, X_E_M_LB, X_E_M_UB, alpha=0.2, color='red')
ax.step(LB, P_E, c='C0', alpha=0.6)
ax.step(UB, P_E, c='C0', alpha=0.6)
ax.fill_betweenx(P_E, LB, UB, alpha=0.4, color='C0')
ax.step(X_E, P_E, label='Reduced Stochstic Model', c='C0', linewidth=3)

ax.step(X_E_M_LB, P_E_M_LB, c='red', alpha=0.6)
ax.step(X_E_M_UB, P_E_M_UB, c='red', alpha=0.6)
ax.step(X_E_M, P_E_M, label='Regression Model', c='red', linewidth=3)
ax.set_xlabel('Energy (J)')
ax.set_ylabel('P(x)')
ax.legend()
ax.tick_params(axis="y", direction="in", pad=-30)
ax.tick_params(axis="x", direction="in", pad=-20)
ax.xaxis.set_label_coords(0.75, -0.01)
ax.yaxis.set_label_coords(-0.01, 0.85)
"""
