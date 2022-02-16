#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:55:37 2019
@author: dominiccalleja

Analysis Smardda 4 python
AnalysisS4P

"""

__author__ = "Dominic Calleja"
__copyright__ = "Copyright 2019"
__credits__ = ["Wayne Arter"]
__license__ = "MIT"
__version__ = "0.1"
__date__ = '01/04/2019'
__maintainer__ = "Dominic Calleja"
__email__ = "d.calleja@liverpool.ac.uk"
__status__ = "Draft"

"Standard things"
import pickle
import os
import pandas as pd
from math import sqrt, sin
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

"Less standard things"
#from jet.data import sal
#import scoord
#import smardda4python as sm
import FWmod as FW

print('+====================================+')
print('|             AnalysisS4P            |')
print('|  Analysis Tools For SMARDDA in Py  |')
print('|                                    |')
print('|           Version: '+__version__ + 12*' '+' |')
print('|                                    |')
print('| '+__copyright__+' (c) ' + __author__+' |')
print('+====================================+')
print(' ')


class Analysis:
    def __init__(self, Target=[], Results=[], WorkingDir=[], logfile=[], smardda_object=[]):
        """ default description
        Class :

        Description :

        Inputs :

        Returns :
        """
        if smardda_object:
            self.Target = smardda_object.Target
            self.Results = smardda_object.ResultsVTK
            self.SimDir = smardda_object.SimDir
            self.logfile_a = os.path.join(self.SimDir, smardda_object.SimName + '_analysis.txt')
            copyR(self.logfile_a)
        else:
            self.Target = Target
            self.Results = Results
            self.SimDir = WorkingDir
            self.logfile_a = os.path.join(self.SimDir, logfile)
            copyR(self.logfile_a)
        

    def ExtractGeom(self,feild_line_suffix=[]):
        """ default description
        Function :

        Description :

        Inputs :

        Returns :
        """

        VTK = FW.ReadVTK(os.path.join(self.SimDir, self.Target),
                         os.path.join(self.SimDir, self.logfile_a))
        Res = FW.ReadVTK(os.path.join(self.SimDir, self.Results),
                         os.path.join(self.SimDir, self.logfile_a))

        files = os.listdir(self.SimDir)
        if not feild_line_suffix:
            GeoFldx = [i for i in files if 'RES_geofldx' in i]
        else: 
            GeoFldx = [i for i in files if 'RES_geofldx' in i]
            
        try :
            self.Normals, self.Feild_line, Bodies, Points = Extract_Normals(GeoFldx[0],self.SimDir,logfile = self.logfile_a)
        except:
            print('missing normals')
            self.Normals    = []
            self.Feild_line = []

        self.Points = FW.VTK_P(VTK, os.path.join(self.SimDir, self.logfile_a))
        self.Cells  = FW.VTK_C(VTK, os.path.join(self.SimDir, self.logfile_a))
        self.Bodies = FW.VTK_B(VTK, os.path.join(self.SimDir, self.logfile_a))
        self.Q      = FW.ExtractResultData(Res, os.path.join(self.SimDir, self.logfile_a))

        Area = []
        Centroid = []
        for i in range(len(self.Cells)):
            Cord = []
            Cord.append(self.Points[int(self.Cells[i][0])])
            Cord.append(self.Points[int(self.Cells[i][1])])
            Cord.append(self.Points[int(self.Cells[i][2])])
            "numpyfy these"
            Element = np.array(Cord)

            ons = np.array([1.0, 1.0, 1.0])
            A = np.concatenate([Element[:, 0], Element[:, 1], ons])
            B = np.concatenate([Element[:, 1], Element[:, 2], ons])
            C = np.concatenate([Element[:, 2], Element[:, 0], ons])

            ASq = 0.5*sqrt(np.linalg.det(A.reshape([3, 3]))**2 + np.linalg.det(
                B.reshape([3, 3]))**2 + np.linalg.det(C.reshape([3, 3]))**2)

            c_x = (max(Element[:, 0])-min(Element[:, 0]))/2+min(Element[:, 0])
            c_y = (max(Element[:, 1])-min(Element[:, 1]))/2+min(Element[:, 1])
            c_z = (max(Element[:, 2])-min(Element[:, 2]))/2+min(Element[:, 2])

            Cen = [c_x, c_y, c_z]

            Area.append(ASq/1000)
            Centroid.append(Cen)

        self.Area = Area
        self.cell_centre = Centroid
        
        return self.Points, self.Cells, self.Normals, self.Feild_line, self.Bodies, self.Area, self.cell_centre
        

    def ExtractStats(self, idfun=None):
        """ default description
        Function :

        Description :

        Inputs :

        Returns :
        """

        Entries = ('Body_Num', 'Max_q', 'Intergrated_q',
                   'HotSpot', 'RadialCoordHS', 'Area_Cells',
                   'Cell_Centres', 'Cell_Normals', 'Cell_Feild',
                   'Len_Cells', 'VTK_Cell_Indicies','HeatFlux',
                   'Body_Centre')

        BOD = np.array(self.Bodies)
        Q  = np.array(self.Q)
        VTKInd = np.array(range(len(self.Q)))
        A  = np.array(self.Area)
        CC = np.array(self.cell_centre)
        try:
            N  = np.array(self.Normals)
        except:
            N = []
        F  = np.array(self.Feild_line)

        Data = {}
        Body_Numbers = []
        Output_string = []
        [Body_Numbers.append(i) for i in self.Bodies if not Body_Numbers.count(i)]
        for i in Body_Numbers:
            Q_bod = Q[BOD == i]
            A_bod = A[BOD == i]
            CC_bod = CC[BOD == i]
            try:
                N_bod = N[BOD == i]
            except:
                N_bod = []
            try:
                F_bod = F[BOD == i]
            except:
                F_bod =[]
            VTK_bod = VTKInd[BOD == i]
            maxQ = max(Q_bod)
            intQ = sum(A_bod*Q_bod)
            HS = []  # " self.cell_centre.index(maxQ)"
            radHS = []  # sqrt(HS[0]**2+HS[1]**2)
            Body_Centre = ((np.asarray([max(CC_bod[:,0]),max(CC_bod[:,1]),max(CC_bod[:,2])])- np.asarray([min(CC_bod[:,0]),min(CC_bod[:,1]),min(CC_bod[:,2])]))/2)+np.asarray([min(CC_bod[:,0]),min(CC_bod[:,1]),min(CC_bod[:,2])])
            Data['BodyNumber_'+str(i)] = {Entries[1]: maxQ,
                                          Entries[2]: intQ,
                                          Entries[3]: HS,
                                          Entries[4]: radHS,
                                          Entries[5]: A_bod,
                                          Entries[6]: CC_bod,
                                          Entries[7]: N_bod,
                                          Entries[8]: F_bod,
                                          Entries[9]: len(Q_bod),
                                          Entries[10]: VTK_bod,
                                          Entries[11]: Q_bod,
                                          Entries[12]: Body_Centre}

            Output_string.append(' \n ### Body Number :'+str(int(i)) + '###')
            Output_string.append('{}          :{:.1E} Wm^-2'.format(
                Entries[1], maxQ))
            Output_string.append('{}  :{:.1E} W'.format(
                Entries[2], intQ))
#            Output_string.append(Entries[3] +'         :'+ str(HS)  + 'mm')
#            Output_string.append(Entries[4] +'  :'+ str(radHS)+ 'mm')
            Output_string.append('Body_Centre' +'    :'+ str(Body_Centre)+ 'mm')
        outputf=open(self.logfile_a, 'a')
        outputf.write('##############-Summary Individual Body Parameters Parameters-##############' + '\n')
        outputf.write('\n'.join(Output_string) + '\n')

        self.Body_Numbers = Body_Numbers
        self.Total_IntQ = sum(Q*A)
        self.Total_MaxQ = max(Q)
        self.Individual_Body_Data = Data

        outputf.write('##############-Extracted Statistics-##############' + '\n')
        outputf.write('Total Cells Read : ' + str(int(len(self.Cells))) +'\n')
        outputf.write('Maximum Q        : %.2E' % (self.Total_MaxQ)+ '(Wm^{-2})' + '\n')
        outputf.write('Intergrated Q    : %.2E' % (self.Total_IntQ) + '(W)' + '\n')
        outputf.close()

    def extract_profile(self,WT,BodyNumber = '23',plot= True):
        """ default description
        Function :

        Description :

        Inputs :

        Returns :
        """
        from math import tan
        # WT = Window Thickness : the strip of results to consider on the tiles

        X = self.Individual_Body_Data['BodyNumber_'+str(BodyNumber)+'.0']['Cell_Centres'][:,0]
        Y = self.Individual_Body_Data['BodyNumber_'+str(BodyNumber)+'.0']['Cell_Centres'][:,1]
        R = np.sqrt(X**2 + Y**2)

        # find midpoint of tile
        C_x,C_y = np.median(X),np.median(Y)
        qX,qY = rotate([0,0],[X,Y],-tan(C_y/C_x))
        # Four courners of the observation window
        A = np.array([qX[np.argmin(R)],0+WT])#+10 little fudge factor to remove inboard facing plate
        B = np.array([qX[np.argmin(R)],0-WT])
        C = np.array([qX[np.argmax(R)],0+WT])
        D = np.array([qX[np.argmax(R)],0-WT])

        ind = []
        A_sq = triangle_area(A[0],A[1],B[0],B[1],C[0],C[1]) + triangle_area(D[0],D[1],B[0],B[1],C[0],C[1])
        for i in range(len(X)):
            a1 = triangle_area(qX[i],qY[i],A[0],A[1],B[0],B[1])
            a2 = triangle_area(qX[i],qY[i],B[0],B[1],C[0],C[1])
            a3 = triangle_area(qX[i],qY[i],C[0],C[1],D[0],D[1])
            a4 = triangle_area(qX[i],qY[i],D[0],D[1],A[0],A[1])
            pA = a1+a2+a3+a4
            if not pA > A_sq:
                ind.append(i)

        self.Footprint_Q  = self.Individual_Body_Data['BodyNumber_'+str(BodyNumber)+'.0']['HeatFlux'][ind]
        self.Coordinates  = self.Individual_Body_Data['BodyNumber_'+str(BodyNumber)+'.0']['Cell_Centres'][ind,:]
        self.Foot_Normals = self.Individual_Body_Data['BodyNumber_'+str(BodyNumber)+'.0']['Cell_Normals'][ind,:]
        self.Foot_Feilds  = self.Individual_Body_Data['BodyNumber_'+str(BodyNumber)+'.0']['Cell_Feild'][ind,:]
        self.Radial_location = R[ind]

        Dict = {    'Ind': ind,
            'Footprint_Q':self.Footprint_Q,
        'Radial_location':self.Radial_location,
                'X_Coord':self.Coordinates[:,0],
                'Y_Coord':self.Coordinates[:,1],
                'Z_Coord':self.Coordinates[:,2],
                 'X_Norm':self.Foot_Normals[:,0],
                 'Y_Norm':self.Foot_Normals[:,1],
                 'Z_Norm':self.Foot_Normals[:,2],
                'X_Feild':self.Foot_Feilds[:,0],
                'Y_Feild':self.Foot_Feilds[:,1],
                'Z_Feild':self.Foot_Feilds[:,2]}

        Output = pd.DataFrame(Dict, columns=["Ind","Footprint_Q","Radial_location","X_Coord","Y_Coord","Z_Coord","X_Norm","Y_Norm","Z_Norm","X_Feild","Y_Feild","Z_Feild"])
        self.Footprint_data_frame = Output
        outputf=open(self.logfile_a, 'a')
        outputf.write('##############-Extracted Footprint Table-##############' + '\n')
        outputf.write('Body Number Observed :' + BodyNumber)
        outputf.write('\n-------- Observation Window --------\n')
        outputf.write(str(A)+'\n')
        outputf.write(str(B)+'\n')
        outputf.write(str(C)+'\n')
        outputf.write(str(D)+'\n')
        outputf.write('\n-------- Extracted Footprint --------\n')
        outputf.write(Output.to_string())
        outputf.close()

        if plot:
            plt.scatter(R[ind], self.Individual_Body_Data['BodyNumber_'+str(BodyNumber)+'.0']['HeatFlux'][ind])
            #plt.xlabel('Radial distance')
            #plt.ylabel('Heat Flux')
            plt.title('Radial SOL footprint on T6 Body Number:' + str(BodyNumber))

        return self.Footprint_data_frame, self.Footprint_Q, self.Radial_location, self.Coordinates

    def extract_tile6(self):
        Dict2= {'Tile6_a': self.Individual_Body_Data['BodyNumber_21.0'],
                'Tile6_b': self.Individual_Body_Data['BodyNumber_22.0'],
                'Tile6_c': self.Individual_Body_Data['BodyNumber_23.0'],
                'Tile6_d': self.Individual_Body_Data['BodyNumber_24.0']}
        self.Tile6_Results = Dict2
        return Dict2
        
    def Save_pickle(self, BodyNumbers, filename, Description = None, **kwargs):
        """ default description
        Function :
        Description :
        Inputs :
        Returns :
        """
        summary_data = {}

        if not hasattr(self,'Footprint_Q'):
            FP = []
            RL = []
            C  = []
            for i in range(len(BodyNumbers)):
                FP,RL,C = self.extract_profile(BodyNumber=str(BodyNumbers[i]),plot=False)

        # Prepare inputs Tile6 : BodyNumbers = [21,22,23,24]
        for i in range(len(BodyNumbers)):
            if i == 0:
                CC = self.Individual_Body_Data['BodyNumber_' +
                                               str(BodyNumbers[i])+'.0']['Cell_Centres']
                Q = self.Individual_Body_Data['BodyNumber_'+str(BodyNumbers[i])+'.0']['HeatFlux']
                Int_q = self.Individual_Body_Data['BodyNumber_' +
                                                  str(BodyNumbers[i])+'.0']['Intergrated_q']
            else:
                Q = np.append(
                    Q, self.Individual_Body_Data['BodyNumber_'+str(BodyNumbers[i])+'.0']['HeatFlux'], axis=0)
                CC = np.append(
                    CC, self.Individual_Body_Data['BodyNumber_'+str(BodyNumbers[i])+'.0']['Cell_Centres'], axis=0)
                Int_q = Int_q + \
                    self.Individual_Body_Data['BodyNumber_' +
                                              str(BodyNumbers[i])+'.0']['Intergrated_q']

        self.summary_data = {'MaxQ': max(Q),
                             'Int_q': Int_q,
                             "Cell_Centres": CC,
                             "Heat_Flux": Q,
                             "BodyNumbers": BodyNumbers}
        if Description is not None:
            self.summary_data['Description']=Description

        if hasattr(self,'Footprint_Q'):
            self.summary_data['Footprint_Q'] = self.Footprint_Q
            self.summary_data['Foot_radial'] = self.Radial_location

        if kwargs:
            self.summary_data['additional_data'] = add_to_pickle(**kwargs)

        # Store data (serialize)
        if "pickle" not in filename:
            filename = filename+'.pickle'
        else:
            filename = filename

        with open(filename, 'wb') as handle:
            pickle.dump(self.summary_data, handle, protocol=min(4,pickle.HIGHEST_PROTOCOL))

    def plot_tile6(self):
        """ default description
        Function :

        Description :

        Inputs :

        Returns :
        """

        Tile6_a = self.Individual_Body_Data['BodyNumber_21.0']
        Tile6_b = self.Individual_Body_Data['BodyNumber_22.0']
        Tile6_c = self.Individual_Body_Data['BodyNumber_23.0']
        Tile6_d = self.Individual_Body_Data['BodyNumber_24.0']

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Col = (Tile6_a['HeatFlux']-min(Tile6_a['Heat_Flux']))/(max(Tile6_a['Heat_Flux'])-min(Tile6_a['Heat_Flux']))
        im = ax.scatter(Tile6_a['Cell_Centres'][:, 0], Tile6_a['Cell_Centres'][:, 1],
                        Tile6_a['Cell_Centres'][:, 2], c=Tile6_a['HeatFlux'], cmap=cm.hot, marker='x')
        ax.scatter(Tile6_b['Cell_Centres'][:, 0], Tile6_b['Cell_Centres'][:, 1],
                   Tile6_b['Cell_Centres'][:, 2], c=Tile6_b['HeatFlux'], cmap=cm.hot, marker='x')
        ax.scatter(Tile6_c['Cell_Centres'][:, 0], Tile6_c['Cell_Centres'][:, 1],
                   Tile6_c['Cell_Centres'][:, 2], c=Tile6_c['HeatFlux'], cmap=cm.hot, marker='x')
        ax.scatter(Tile6_d['Cell_Centres'][:, 0], Tile6_d['Cell_Centres'][:, 1],
                   Tile6_d['Cell_Centres'][:, 2], c=Tile6_d['HeatFlux'], cmap=cm.hot, marker='x')

        ax.set_xlabel('X')
        ax.set_zlabel('Y')
        ax.set_ylabel('Z')

        plt.colorbar(im, ax=ax)
        plt.show()


def extract_profile_shoulder_plane_projection(analysis_object, WT, BodyNumber='23', shoulder_R=[2870, 2940], plot=True):
    """ default description
        Function : extract_profile_shoulder_plane_projection

        Description : This extracts the footprint profile, projected onto the plane of the shoulder. In the typical JET pulse 
        the majority of the power deposition tends to be focused on the shoulder of tile 6. Hence, there is a small error induced
        when calculating the 2 dimensional heat equation as the projection onto the base plane major radius causes a slight 
        narrowing of the deposition profile. 

        Inputs : 
        anaysis_object - This is the object created by evaluating the smardda analysis class (object handle)
        WT             - Window thickness (n cells either side of plane of extraction)
        shoulder_R     - 2 points near the min and max of the shoulder radial distance.

        Returns : 
        shoulder_projection_x_distance  - major radius shoulder plane projection
        shoulder_projection_footprint_Q - Q profile shoulder plane projection
        """
    from math import tan
    from math import atan
    from math import degrees
    from math import radians
    # WT = Window Thickness : the strip of results to consider on the tiles

    X = analysis_object.Individual_Body_Data['BodyNumber_' +
                                             str(BodyNumber)+'.0']['Cell_Centres'][:, 0]
    Y = analysis_object.Individual_Body_Data['BodyNumber_' +
                                             str(BodyNumber)+'.0']['Cell_Centres'][:, 1]
    Z = analysis_object.Individual_Body_Data['BodyNumber_' +
                                             str(BodyNumber)+'.0']['Cell_Centres'][:, 2]
    R = np.sqrt(X**2 + Y**2)

    # find midpoint of tile
    C_x, C_y = np.median(X), np.median(Y)
    qX, qY = rotate([0, 0], [X, Y], -tan(C_y/C_x))
    # Four courners of the observation window
    # +10 little fudge factor to remove inboard facing plate
    A = np.array([qX[np.argmin(R)], 0+WT])
    B = np.array([qX[np.argmin(R)], 0-WT])
    C = np.array([qX[np.argmax(R)], 0+WT])
    D = np.array([qX[np.argmax(R)], 0-WT])

    ind = []
    A_sq = triangle_area(A[0], A[1], B[0], B[1], C[0], C[1]) + \
        triangle_area(D[0], D[1], B[0], B[1], C[0], C[1])
    for i in range(len(X)):
        a1 = triangle_area(qX[i], qY[i], A[0], A[1], B[0], B[1])
        a2 = triangle_area(qX[i], qY[i], B[0], B[1], C[0], C[1])
        a3 = triangle_area(qX[i], qY[i], C[0], C[1], D[0], D[1])
        a4 = triangle_area(qX[i], qY[i], D[0], D[1], A[0], A[1])
        pA = a1+a2+a3+a4
        if not pA > A_sq:
            ind.append(i)

    #find shoulder plane
    r_surface = R[ind]
    z_surface = Z[ind]

    shoulder_z = z_surface[np.logical_and(
        r_surface > shoulder_R[0], r_surface < shoulder_R[1])]
    shoulder_r = r_surface[np.logical_and(
        r_surface > shoulder_R[0], r_surface < shoulder_R[1])]
    shoulder_m = np.polyfit(shoulder_r, shoulder_z, 1)
    alpha_x_s = atan(shoulder_m[0])

    xi = np.linspace(R.min(), R.max(), 50)
    yi = np.polyval(shoulder_m, xi)

    mid_r_s = shoulder_r.min() + (shoulder_r.max() - shoulder_r.min())/2
    mid_z_s = shoulder_z.min() + (shoulder_z.max() - shoulder_z.min())/2

    rotated_s_r, rotated_s_z = rotate(
        [mid_r_s, mid_z_s], [r_surface, z_surface], -alpha_x_s)

    # results projected onto tile shoulder plane
    q_projection = analysis_object.Individual_Body_Data['BodyNumber_' +
                                                        str(BodyNumber)+'.0']['HeatFlux'][ind]
    r_projection = rotated_s_r
    shoulder_projection_footprint_Q = q_projection
    shoulder_projection_x_distance = r_projection

    if plot:
        plt.figure()
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        axs[0].scatter(R/1000, Z/1000, alpha=0.3, color='grey')
        axs[0].plot(xi/1000, yi/1000, label='shoulder plane: angle {:.2f}'.format(
            degrees(alpha_x_s)))
        axs[0].scatter(
            R[ind]/1000, Z[ind]/1000, label='extracted profile surface elements')
        axs[0].scatter(shoulder_r/1000, shoulder_z /
                       1000, label='shoulder elements')
        axs[0].scatter(mid_r_s/1000, mid_z_s/1000, marker='+', s=100, color='red',
                       label='origin {:.2f}-{:.2f}'.format(mid_r_s, mid_z_s))
        axs[0].set_xlabel('Major Radius (m)')
        axs[0].set_ylabel('Z (m)')
        axs[0].legend()

        axs[1].scatter(rotated_s_r/1000, rotated_s_z/1000)
        axs[1].plot([xi[0]/1000, xi[-1]/1000], [mid_z_s/1000, mid_z_s/1000], label='shoulder plane: angle {}'.format(
            degrees(alpha_x_s)))
        #axs[1].plot([0, rotated_s_r.max()], [0,0])
        axs[1].scatter(rotated_s_r[np.logical_and(r_surface > shoulder_R[0], r_surface < shoulder_R[1])]/1000, rotated_s_z[np.logical_and(
            r_surface > shoulder_R[0], r_surface < shoulder_R[1])]/1000)
        axs[1].scatter(mid_r_s/1000, mid_z_s/1000, marker='+', s=1000, color='red',
                       label='origin {:.2f}-{:.2f}'.format(mid_r_s, mid_z_s))
        axs[1].set_xlabel('r tile6 (m)')
        axs[1].set_ylabel('z tile6 (m)')

        axs[2].scatter(R[ind]/1000, analysis_object.Individual_Body_Data['BodyNumber_' +
                                                                         str(BodyNumber)+'.0']['HeatFlux'][ind], label='baseplane projection', alpha=0.5)
        axs[2].scatter(r_projection/1000, q_projection,
                       label='shoulder plane projection')
        axs[2].set_xlabel('r tile6 (m)')
        axs[2].set_ylabel('Heat Flux Wm^-2')
        axs[2].legend()
        axs[2].ticklabel_format(style='sci', scilimits=(1, 5), axis='y')
    return shoulder_projection_x_distance, shoulder_projection_footprint_Q


def mesh_element_area(Element):
    ons = np.array([1, 1, 1])
    A = np.concatenate([Element[:, 0], Element[:, 1], ons])
    B = np.concatenate([Element[:, 1], Element[:, 2], ons])
    C = np.concatenate([Element[:, 2], Element[:, 0], ons])

    ASq = 0.5*sqrt(np.linalg.det(A.reshape([3, 3]))**2 + np.linalg.det(
        B.reshape([3, 3]))**2 + np.linalg.det(C.reshape([3, 3]))**2)
    return ASq

def Load_pickle(filename,pickle_protocol=5):
    """ default description
    Function :

    Description :

    Inputs :

    Returns :
    """

    if 'pickle' not in filename:
        filename = filename+'.pickle'
    else:
        filename = filename

    with open(filename, 'rb') as handle:
        Data = pickle.load(handle)
    return Data

def add_to_pickle(**kwargs):
    dict = {}
    for key,value in kwargs.items():
        dict[key] = value
    return dict

def plot_pickle(DATA, **kwargs):
    """ default description
    Function :

    Description :

    Inputs :

    Returns :
    DATA should be a single entry of the pickle: i.e a single output of the Aanalysis pickle function
    ** Kwargs can contain any other key value pairs that wish to be entered into the title of the figure produced.
    """

    x = DATA['Cell_Centres'][:,0]
    y = DATA['Cell_Centres'][:,1]
    z = DATA['Cell_Centres'][:,2]
    Q = DATA['Heat_Flux']

    maxQ = DATA['MaxQ']
    intQ = DATA['Int_q']
    # Generate label
    Label =[]
    Label.append(('Output Arguments: MaxQ = {:.3e}Wm-2 -- IntQ = {:.3e}W \n'.format(maxQ,intQ)))

    for key,value in kwargs.items():
        if kwargs.__len__() == 0:
            break
        else:
            Label.append('Input Arguments: ')
        lab = ("%s = %.2E" %(key,value))
        Label.append(lab)
    # 3D subplot of full output
    fig = plt.figure()
    fig.set_size_inches(15,20)
    fig.suptitle(' '.join(Label))
    axs = fig.add_subplot(2, 1, 1,projection='3d')
    im =axs.scatter(x, y, z, c=Q, cmap=cm.hot, marker='x')
    plt.colorbar(im, ax=axs)
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.set_zlabel('Z')
    # Footprint profile
    axs = fig.add_subplot(2, 1, 2)
    axs.scatter(DATA['Foot_radial'],DATA['Footprint_Q'])
    axs.set_xlabel('Radial Location mm')
    axs.set_ylabel('Heat Flux W')

def copyR(logfile):
    """Print copyright information to file."""
    outputf=open(logfile, 'a')
    outputf.write('+'+'='*77+'+ \n')
    tl='AnalysisS4P'
    sp1=(77-len(tl))//2
    sp2=77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|'+ '\n')
    tl='Analysis package for Smardda'
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
# Copy the log file


def AnsysMeshmapping(Heat_Flux, Cell_centres, Ansys_mesh_file):
    """
    Description : map smardda output to ansys mesh
    """
    if '.csv' in Ansys_mesh_file:
        Data = pd.read_csv(Ansys_mesh_file)
    else:
        raise ValueError("Invalid file type enter CSV")
    Header = Data.columns.values
    X = [i for i, s in enumerate(Header) if 'X' in s]
    Y = [i for i, s in enumerate(Header) if 'Y' in s]
    Z = [i for i, s in enumerate(Header) if 'Z' in s]

    Cell_centres = np.array(Cell_centres)
    if abs(np.mean(Cell_centres)) < 900:
        print("Order of magnitude of Cell Centres not the same as Ansys mesh")
        Cell_centres = Cell_centres/1000

    QId = []
    for i in range(len(Data.ix[:, X])):
        QId.append(abs(sqrt((Data.ix[i, X] + Cell_centres[:, 0])**2 + (Data.ix[i, Y] +
                                                                       Cell_centres[:, 1])**2 + (Data.ix[i, Z] + Cell_centres[:, 2])**2)).idxmin())
    return QId
# Mapping between Smardda and Ansys meshes

def triangle_area(x1, y1, x2, y2, x3, y3):
    """
    Description : Find smardda cell area
    """
    return abs(0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)))
# calculate area of cells in smardda mesh

def rotate(origin, point, angle):
    """
    Description : Rotate a point counterclockwise by a given angle around a given origin.
    """
    import math
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
#rotate points around given axis

def print_table(table):
    """
    Description : Print pandas array
    """
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3)
        for i in range(len(table[0]))
    ]
    row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
    for row in table:
        str(row_format.format(*row))
# print table area

def Extract_Normals(GeoFldx,directory,logfile = 'Extract_Normals.txt'):
    """
    Description : Function to extract surface normals from smardda file
    """
    VTK = FW.ReadVTK(os.path.join(directory, GeoFldx),logfile)

    Points  = FW.VTK_P(VTK,logfile)
    #Cells   = FW.VTK_C(VTK,logfile)  No cells in the gefldx files
    Bodies  = FW.VTK_B(VTK,logfile)
    Normals = FW.VTK_N(VTK,logfile)
    Feild_line = FW.VTK_F(VTK,logfile,type_name = 'Bcell')
    return Normals, Feild_line, Bodies, Points
# extract cell normals from VTK Geoflds files

def Calc_Q_Norm(Q,ang):
    """
    Description : Calculate Perpendicular Q
    """
    Q_norm = []
    for i in range(len(ang)):
        Q_norm.append(sin(ang[i]) * Q[i])

    return Q_norm
# Calculate Normal Q at the tile surface

def Calc_Q_Par(Q,ang):
    """
    Description : Calculate Parallel Q
    (TODO: Check)
    """
    from math import cos
    Q_par =[]
    for i in range(len(ang)):
        Q_par.append(cos(ang[i]) * Q[i])
    return Q_par

class data_holder():
    pass
# Class to make a temporary data holder

def extract_sub_values(Dict,Key):
    """
    Description : Simple function to pull key value pairs from sub dicts of a dictionary
    """
    val =[]
    for i,k in enumerate(Dict):
        val.append(Dict[k][Key])
    return val
# Extract the sub values corresponding to a key in the sub directory of a hyeracical dict structure

def format_footprint(x,r_f,q_f):
    """
    Description : Simple function to interpolate unsorted smardda extracted footprint Q values
    x = user specified linspace interpolation points
    r_f = unsorted radial footprint locations
    q_f = unsorted footprint q
    """
    #x = np.linspace(np.min(r_f),np.max(r_f),np.size(x))
    #rs =  2878 #shoulder

    i_sorted = np.argsort(r_f)
    r_sorted = r_f[i_sorted]
    q_sorted = q_f[i_sorted]
    Qf = np.interp(x,r_sorted,q_sorted)
    return Qf

def format_pickle_list(pickle_path):
    """
    Description : Simple function to find and sort avaliable pickles in a directory
    """
    import re
    files = os.listdir(pickle_path)
    pickle_list = []
    pickle_ind = []
    for i in range(len(files)):
        if '.pickle' in files[i]:
            pickle_list.append(files[i])
            pickle_ind.append(int(re.findall('\d+', files[i])[0]))

    arg_ind = np.argsort(np.array(pickle_ind)).tolist()
    p_list = [pickle_list[x] for x in arg_ind]
    return p_list, pickle_ind
# returns formatted pickle list, and the indicies of avaliable pickles

def format_training_data(simulation_directory, simulation_structure, pickle_list, interp_points=50, user_def_interp=[], nk=[], Lambda_q = 'Lambda_q'):
    # TODO needs to be modified for the mov_r and mov_z cases
    """
    Function : format_training_data
    Description : Function formats 2 data arrays
    1  :  data_frame   (standard square array [Inputs - Max & Int - Q radial])
    2  :  gp_data      (GP style block wise output with a single output vector for radial location)
    3 (out) x[ind]     (returns the radial locations wetted for GPs)
    INPUTS :
    simulation directory        = path to results directory
    simulation structure        = pandas array of the stadard simulation structure produced by analysis class
    pickle list                 = list of pickles avaliable in the directory
    interp points (default=100) = interpolated points along radial direction
    nk (default=[])             = number of points either side of the wetted cells radially selected
                                if [] the whole array is taken not just the wetted cells
    Lambda_q (TODO : deprecated)= used to fix a previous spelling mistake in code. Will be removed once all results with that mistake are no longer useful
    """
    run_data = {}
    for i,pick in enumerate(pickle_list):
        run_data[i]=Load_pickle(os.path.join(simulation_directory,pick))

    maximum_q = extract_sub_values(run_data,'MaxQ')
    intergrated_q = extract_sub_values(run_data,'Int_q')
    footprint_q = extract_sub_values(run_data,'Footprint_Q')
    radial_footprint = run_data[0]['Foot_radial']
    if not sum(user_def_interp):
        x = np.linspace(np.min(radial_footprint),np.max(radial_footprint),interp_points)
    else:
        x = user_def_interp
    f_q =[]
    for i , qs in enumerate(footprint_q):
        f_q.append(format_footprint(x,radial_footprint,  qs))

    f_q = np.array(f_q)
    if not nk:
        f_q_s = f_q
        ind = np.array(range(np.shape(f_q)[1]))
    else:
        ind = np.where(np.any(f_q >0, axis = 0))[0]
        ind = np.concatenate([np.array(range(np.min(ind)-nk,np.min(ind))),ind,np.array(range(np.max(ind)+1,np.max(ind)+nk+1))])
        f_q_s = f_q[:,ind]

    Data = {'Psol':simulation_structure.Psol,
    'Lambda_q':simulation_structure[Lambda_q] ,'Sigma':simulation_structure.Sigma,
    'MaxQ':maximum_q,'IntQ':intergrated_q}
    cols = ['Psol','Lambda_q','Sigma','MaxQ','IntQ']
    if 'eq_r_move' in simulation_structure:
        Data['r_move']= simulation_structure['eq_r_move']
        cols.append('r_move')
    if 'eq_z_move' in simulation_structure:
        Data['z_move']= simulation_structure['eq_z_move']
        cols.append('z_move')

    df2 = pd.DataFrame(Data, columns=cols)
    df = pd.DataFrame(f_q_s, index = range(len(f_q_s[:,0])),columns=ind)

    data_frame = pd.merge(df2,df,left_index=True,right_index=True)

    shp = np.ones(len(f_q_s[0]))
    gp_data =pd.DataFrame([])

    if 'eq_r_move' in simulation_structure:
        for i in range(len(footprint_q)):
            Data = np.array([int(i)*shp,df2['Lambda_q'][i]*shp,df2['Sigma'][i]*shp,
            df2['r_move'][i]*shp, df2['z_move'][i]*shp, x[ind], f_q_s[i,:]])
            Data = pd.DataFrame(Data.T, columns = ['run_ind','Lambda_q','Sigma','r_move','z_move','R','Q'])
            gp_data = gp_data.append(Data,ignore_index=True)
    else:
        for i in range(len(footprint_q)):
            Data = np.array([int(i)*shp,df2['Lambda_q'][i]*shp,df2['Sigma'][i]*shp,x[ind],f_q_s[i,:]])
            Data = pd.DataFrame(Data.T, columns = ['run_ind','L_q','Sig','R','Q'])
            gp_data = gp_data.append(Data,ignore_index=True)

    return data_frame, gp_data, x[ind]

# returns the simulation data frame, and the radial location of interp footprint used after removing zeros


def irregular_space(x_r, wetted, N, order=3):
    b = (x_r[wetted][0]-x_r[0])
    a = (x_r[-1]-x_r[wetted][-1])

    xi = np.linspace(0, b, 10)
    xu = np.linspace(-a, 0, 10)

    si = xi**order
    su = xu**order

    xsi = np.linspace(0, max(si), int(N*(b/(a+b))))
    xso = np.linspace(0, min(su), int(N*(a/(a+b))))

    xi1 = xsi**(1/order)
    xu1 = abs(xso)**(1/order)
    #xu1 = xu1-max(xu1)

    return np.unique(np.concatenate([xi1+x_r[0], x_r[wetted], x_r[-1]-xu1], axis=0))

def plot_heatmap(Dat,t,R,label,title, xlim = [],xlabel = 'Time (sec)',ylabel = 'Radial Location (m)',return_fig=False,save_address=[],cmap='hot',clim=[]):
    framestride = max(1,int(5e-4 / np.mean(np.gradient(t))))
    time = [min(t),max(t)]

    if not clim:
        clim = [np.min(Dat),np.max(Dat)]#np.log(Dat.min())
    fig= plt.figure()
    fig.set_size_inches(13.5,4.5)
    #plt.gca().set_facecolor((0.2,0.2,0.2))
    im = plt.imshow(np.flip(Dat[::framestride, :].transpose(), 0), extent=[
                    time[0], time[1], np.min(R), np.max(R)], clim=clim, aspect='auto', cmap=cmap)
    plt.xlabel(xlabel,size=12)
    plt.ylabel(ylabel,size=12)
    cbar = plt.colorbar(im)
    cbar.set_label(label,size=12)
    plt.title(title,size=15)
    if return_fig:
        return fig
    if save_address:
        plt.savefig(save_address)
    else:
        plt.show()
    
# Plot a heatmap from PPF data

def rm_outlier(data):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    drop = s<5
    return drop


def ansys_xls_to_csv(directory, return_dict=True):
    files = [_ for _ in os.listdir(directory) if _.endswith('.xls')]
    new_dir = directory + '/CSV_coversion'
    os.mkdir(new_dir)

    DICT = {}
    for i, f in enumerate(files):
        csv = pd.read_table(directory + '/' + f, encoding='unicode_escape')
        output = pd.DataFrame(csv.values, columns=['node', 'X', 'Y', 'Z', 'T'])
        fnd = f.split('.')[0]
        output.to_csv('{}/{}.csv'.format(new_dir, fnd))
        if return_dict:
            DICT[fnd.split('t')[1]] = output
    if return_dict:
        return DICT


"""
Midplane heat map!
"""


#
#
#
