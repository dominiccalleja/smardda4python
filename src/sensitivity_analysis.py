
import numpy as np
import pandas as pd
from math import floor, ceil
from scipy import fft
from scipy import stats
import matplotlib as mpl

import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 19})

class Sensitivity():
    """ Sensitivity analysis given data

    Includes methods for the main effect analysis of inputs, and analysis of nth order groups. 
    """

    bootstrap = True
    Nbootstrap = 100
    M = 6  #Harmonics 
    conf_interval = .95

    def __init__(self,x,y,input_names=[]):
        self.inputs = x
        self.outputs = y 

        if not input_names:
            self.input_names = ['inp_{}'.format(i) for i in range(np.shape(x)[1])]
        else:
            self.input_names = input_names
    
        self.results = {}
        for name in self.input_names:
            self.results[name] = {}

    def compute_main_order(self):
        Si = self._main(self.inputs,self.outputs)

        if self.bootstrap:
            distSi = self._main_bootstrap(self.Nbootstrap)
            CI = self._compute_confidence(distSi)
            SCi = []
            for i in range(len(CI)):
                SCi.append([Si[i], CI[i]])
            Si = SCi 

        for i, name in enumerate(self.input_names):
            self.results[name]['main'] = Si[i]

    def compute_group_effect(self, groups, group_names):
        self.group_names = group_names
        for i, g in enumerate(group_names):
            self.results[g] = {}
            self.results[g]['inputs'] = groups[i]
            gSi, STi, Sone, info = self._group(self.inputs,self.outputs, groups[i])
            
            if self.bootstrap:
                A, B, C, D = self._group_bootstrap(groups[i], self.Nbootstrap)
                gSi = [gSi, self._compute_confidence(A)]

            self.results[g]['Si'] = gSi
            self.results[g]['STi'] = STi
            self.results[g]['Sone'] = Sone
            self.results[g]['info'] = info

    def _compute_confidence(self,X):
        mu = np.mean(X, axis=0)
        sem = stats.sem(X, axis=0)
        mi = np.min(X, axis=0)
        ma = np.max(X, axis=0)
        #std = np.std(X, axis=0)

        try:
            d = np.shape(X)[1]
        except:
            d = 1
            mu = [mu]
            #std = [std]
            sem = [sem]
            mi = [mi]
            ma = [ma]
            
        Ci = []
        for i in range(d):
            C_5, C_95 = mi[i],ma[i]
            #stats.t.interval(alpha=self.conf_interval,df=np.shape(X)[0]-1, loc=mu[i], scale=sem[i])
            #stats.norm.interval(self.conf_interval, loc=mu[i], scale=std[i])
            Ci.append((C_5, C_95))
        return Ci 

    def _main(self,inputs,outputs):
        Si = easi(inputs, outputs, M=self.M)
        return Si 

    def _group(self, inputs, outputs, group):
        gSi, STi, Sone, info = xeasi(inputs, outputs, group, M=self.M)
        gSi = 1 - gSi
        return gSi, STi, Sone, info

    def _main_bootstrap(self, Nbootstrap):
        Si = []
        for i in range(Nbootstrap):
            ind = np.arange(np.shape(self.inputs)[0]) != np.random.randint(
                0, np.shape(self.inputs)[0], 1)[0]
            Si.append(self._main(self.inputs[ind, :], self.outputs[ind]))
        Si = np.concatenate([Si],axis=1)
        return Si
    
    def _group_bootstrap(self, group, Nbootstrap):
        gSi, STi, Sone, info =[],[],[],[]
        
        for i in range(Nbootstrap):
            ind = np.arange(np.shape(self.inputs)[0]) != np.random.randint(
                0, np.shape(self.inputs)[0], 1)[0]
            g, S, O, I = self._group(self.inputs[ind, :], self.outputs[ind], group)
            gSi.append(g)
            STi.append(S)
            Sone.append(O)
            info.append(I)

        gSi = np.array(gSi)
        STi = np.concatenate([STi],axis=1)
        Sone= np.concatenate([Sone],axis=1)
        info= np.concatenate([info],axis=1)
        return gSi,STi,Sone,info

    def plot_main_effect(self,save_address=[],color='C0'):
        x_pos = np.arange(len(self.get_input_names()))
        
        fig, ax = plt.subplots(figsize=(len(x_pos)*1,8))

        ax.bar(x_pos, self.get_main(), yerr=abs(np.array(self.get_conf_main()).T-np.array(self.get_main())), align='center',
               alpha=0.5, ecolor='black', capsize=10, color=color)
        ax.set_ylabel('Main Effect Sensitivity Index')
        ax.set_xticks(x_pos)
        
        ax.set_xticklabels(self.get_input_names(), rotation=45)

        if save_address:
            fig.savefig(save_address)
        plt.show()
    
    def plot_group_effect(self,save_address=[],color='red'):
        
        x_pos = np.arange(len(self.get_group_names()))
        fig, ax = plt.subplots(figsize=(len(x_pos)*1,8))
        
        ax.bar(x_pos, self.get_group_main(), yerr=abs(np.concatenate(self.get_conf_group_main()).T-np.array(self.get_group_main())), align='center',
               alpha=0.5, ecolor='black', capsize=10, color=color)
        ax.set_ylabel('Combined Effect Sensitivity Index')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.get_group_names(), rotation=45)

        if save_address:
            fig.savefig(save_address)
        plt.show()

    def get_main(self):
        return [self.results[i]['main'][0] for i in self.input_names]
    
    def get_conf_main(self):
        return [self.results[i]['main'][1] for i in self.input_names]

    def get_group_main(self):
        return [self.results[i]['Si'][0] for i in self.group_names]
    
    def get_conf_group_main(self):
        return [self.results[i]['Si'][1] for i in self.group_names]

    def get_input_names(self):
        return self.input_names
    
    def get_group_names(self):
        return self.group_names



def easi(x,y, M=6):
    """Calculation of sensitivity indices from given data.
    
    Inputs :
        x - [n x k] array of inputs 
        y - [n x d] array of outputs
    
    Outputs :
        Si - Sensitivity indicies for the input dimensions
    """
    # Adapted for python by Dominic Calleja. Original matlab code by Elmar Plischke 
    #References:
    #      E. Plischke, "An Effective Algorithm for Computing
    #       Global Sensitivity Indices(EASI)",
    #       Reliability Engineering & Systems Safety, 95(4), 354-360, 2010
    #      E. Plischke, "How to compute variance-based sensitivity
    #       indicators with your spreadsheet software",
    #       Environmental Modelling & Software, In Press, 2012

    n = np.shape(x)[0]
    k = np.shape(x)[1]
    if n != np.shape(y)[0]:
        print('ERROR X and Y must have same number of samples')
        return
            
    index = np.argsort(x,axis=0)

    if np.mod(n, 2) == 0:
        # even no. of samples
        shuffle = np.concatenate([np.array(range(0,n,2)),np.array(range(n-1,0,-2))])
    else:
        # odd no. of samples
        shuffle = np.concatenate([np.array(range(0,n+1,2)),np.array(range(n-1,0,-2))])
        
    indexqper = index[shuffle, :]

    if len(np.shape(y)) == 1:
        yr = y[indexqper]
    else:
        yr = np.zeros([n, k * np.shape(y)[1]])

        for i in range(np.shape(y)[1]):
            z = y[:,i]
            yr[:, i*k+1:i*k+k] = z[indexqper]
    
    F = abs(fft.fft(yr.T).T)
    spectrum = F**2/(n-1)

    Vi=2*sum(spectrum[1:M+1,:])
    V=sum(spectrum[1:,:])
    Si=Vi/V
    return Si



def xeasi(x,y,group,M=6):
    """Calculate main and total effect indicies
    
    Inputs :
        x - [n x k] array of inputs 
        y - [n x d] array of outputs
        group - list of input indicies contained in the group 
    

    Outputs :
        Si - sensitivity index for the interaction of the index set GROUP
        STi - cumulative effects
        Sone - first order effects for individual factors of GROUP
        info - quality of signal
    """


    # Adapted for python by Dominic Calleja. Original matlab code by Elmar Plischke
    # References:
    #     E. Plischke, "An Effective Algorithm for Computing
    #     Global Sensitivity Indices(EASI)",
    #     Reliability Engineering & Systems Safety, 95(4), 354-360, 2010
        
    n = np.shape(x)[0]
    k = np.shape(x)[1]

    index = np.argsort(x[:,group], axis=0)
    if np.mod(n, 2) == 0:
        # even no. of samples
        shuffle = np.concatenate(
            [np.array(range(0, n, 2)), np.array(range(n-1, 0, -2))])
    else:
        # odd no. of samples
        shuffle = np.concatenate(
            [np.array(range(0, n, 2)), np.array(range(n-1, 0, -2))-1])


    l = len(group)

    if l ==1:
        P=n
    elif l ==2:
        P = min(2*M+1, floor(n/(2*M))-1)
    else:
        P = min(2*M+1,floor((n+1)**(1/l)))

    hyperrank = to_curve_address((index2rank(index)-0.5)/n, P)
    hyperrank = hyperrank[shuffle]
    F = abs(fft.fft(y[hyperrank].T)).T
    spectrum = F**2/n

    s = np.ones([l,2**(l-1)])
    for i in range(0,2**(l-1)):
        B = binbits(i, l)
        s[:,i] = 1-2*np.array([i for i in B[::-1]]).astype(bool)
            
    indexx = np.reshape(np.arange(0,M),[M,1])
    for i in range(1,l):
        indexx = np.concatenate([[np.tile(indexx[:,0], M)],P*((np.kron(indexx.T, np.ones([M])))+1)]).T

    indexx = np.dot(indexx,s)

    Vi = np.sum(spectrum[1: n])
    Sone = np.zeros([l])

    Si = 2*sum(spectrum[list(indexx.flatten().astype(int))])/Vi
    # total indices
    STi = np.zeros([l]);
    for i in range(1,l+1):
        STi[i-1]=2*sum(spectrum[np.arange(1,(M*(P**i-1)/(P-1))).astype(int).tolist()])/Vi

    # first order effects
    for i in range(1,l+1):
        Sone[i-1] = 2*sum(spectrum[P**(l-i)*np.arange(1, M+1)])/Vi

    info = np.zeros([l])
    xspectrum = (abs(fft.fft(x[hyperrank][:, group].T)).T)**2/n
    Vx = np.sum(xspectrum[1:,:],axis=0)
    for i in range(1,l+1):
        info[i-1] = 2*xspectrum[P **(l-i), i-1]/Vx[i-1]
    return Si, STi, Sone, info


def binbits(x, n):
    """Return binary representation of x with at least n bits"""
    bits = bin(x).split('b')[1]

    if len(bits) < n:
        return '0' * (n - len(bits)) + bits


def to_curve_address(x,P):

    n = np.shape(x)[0]
    k = np.shape(x)[1]

    if k ==1:
        g = x
    else:
        xi = np.floor(x*P)
        xx = np.zeros(x.shape)
        xx[:, 0] = x[:, 0]*(P-1)
        xx[:, 1:] = xi[:, 1:k]
        signs = (-1)**xi
        cumsigns = np.ones([n, k])
        s = signs[:, k-1]
        for i in range((k-2),-1,-1):
            cumsigns[: , i] = s
            s = s * signs[:, i]
        
        cs = (cumsigns+1)/2
        gx = xx * cs+(P-1-xx)*(1-cs)
        g = np.sum(gx*(P ** np.arange(0,k).T),axis=1)

    index = np.argsort(g)
    return index


def index2rank(index):
    "transform index to rank matrix"
    n = np.shape(index)[0]
    k = np.shape(index)[1]

    rank = np.zeros(index.shape)
    for i in range(k):
        rank[index[:,i],i] = np.arange(0,n)
    return rank


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #### Functional usage 
    x = np.random.normal(0, 1, size=[1001, 9])
    y = x[:, 1]**3 + (x[:, 0]/5)  # np.concatenate([[x[:, 1]**2], [x[:, 0]*2]]).T

    Si = easi(x, y)
    group = list(range(5))

    plt.figure()
    plt.bar(list(range(len(Si))),Si)
    plt.show()

    Si, STi, Sone, info = xeasi(x, y, group, M=6)

    plt.figure()
    plt.bar(list(range(len(STi))), Sone)
    plt.show()

    plt.figure()
    plt.bar(list(range(len(STi))), STi)
    plt.show()
