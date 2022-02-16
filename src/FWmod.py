#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 

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
#Created on Tue Oct 16 14:10:16 2018


import shutil
__author__ = "Dominic Calleja"
__copyright__ = "Copyright 2018"
__credits__ = ["Zsolt Vizvary", "Lorena Richiusa","Dominic Calleja"]
__license__ = "MIT"
__version__ = "0.4"
__date__='10/12/2018'
__maintainer__ = "Dominic Calleja"
__status__ = "Draft"

"""
Version history:
FWmod  0.4: Included new functions to extract the cell and body numbers from
            VTK files and a function to combine VTK files. This currently uses
            vtktfm module of Wayne Arters SMARDDA library. With the new atribute
            extraction functions this should be modified to execute entierly in
            this module. (Development intended to be independent of device
            - currently compatible with JET and DEMO)
FWmod  0.3: Further modifications to the transformation funtions. There is now
            a generic function for cartesian translation and rotation with any
            specified axis of rotation or vector of translation]
FWmod  0.2: Modified transformation so that the gap between FW outboard and
            divertor is constant.
FWmod  0.1: First attempt.
"""

"""
Function list:
- copyR(logfile)
- Printer()
- ReadVTK(vtkfile,logfile)
- vtkSearch(VTKlist,term,logfile)
- translation
- rotation
- VTK_P(SourceFile,unit,Limits,logfile)
- VTK_C
- VTK_B
- mod_OB(Points,R,Z,pb,Zlim,Zmin,Ymin,Ymax,Sector_rot)
- mod_IB(Points,R,Z,pb,Zlim,Zmax,Sector_rot)
- WriteVTK(VTK_List,vtkfile,logfile)
- modFW(input_VTK,output_VTK,OBpushback,IBpushback,Sector_pos,logfile)
- CombineVTK
"""

import sys, math, csv
import os
from math import cos ,sin ,pi
from time import time
import numpy as np

#from vtk import *

print('+====================================+')
print('|               FWmod                |')
print('| VTK mesh modifier for the DEMO FW  |')
print('| Currently also compatible with JET |')
print('|                                    |')
print('|           Version: '+__version__ +12*' '+' |')
print('|                                    |')
print('|  '+__copyright__+' (c) '+ __author__+'  |')
print('+====================================+')
print(' \n\n TO DO: \n\t - Need to change explict references to ctl files. ')


def check_smardda_install(skip):
    #TODO: modify this function for FREIA install. Currently the assumption is smiter has been installed from the git lab repo
    if skip:
        msg = 'overwright self.smardda_path in attributes'

    else:

        msg = []
        with open(os.path.join(os.path.expanduser('~'), '.bashrc'), 'r') as f:
            look_up = 'SMITER_DIR'
            txt = f.read()
            if look_up in txt:
                msg.append('It seems smiter is correctly installed\n')
                for line in txt.split('\n'):
                    if look_up in line:
                        smardda_path = line.split("=")[1] + '/exec'
                msg.append('smardda install path : {}\n'.format(smardda_path))
                flag = True
            else:
                msg.append(
                    'Whoops!!!!! looks like smiter is not installed correctly. It can be cloned from the smardda gitlab repository\n')
                msg.append(
                    'If using smardda on Freia this check install function can be skipped with skipped=True\n')
                flag = False
                return
    return smardda_path, flag, msg

   
smardda_path, flag, msg = check_smardda_install(skip=False)



def copyR(logfile):
    """Print copyright information to file."""
    outputf=open(logfile, 'w')
    outputf.write('+'+'='*77+'+ \n')
    tl='FWmod'
    sp1=(77-len(tl))//2
    sp2=77-sp1-len(tl)
    outputf.write('|'+' '*sp1+tl+' '*sp2+'|'+ '\n')
    tl='VTK mesh modifier for the DEMO FW'
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

class Printer():
    """Print things to stdout on one line dynamically"""
    def __init__(self,data):
        #sys.stdout.write('\033[K\r'+data.__str__())
        sys.stdout.write('\r'+data.__str__())
        sys.stdout.flush()

def ReadVTK(vtkfile,logfile):
    """
    Read in a vtk file and search for the term in it. Collect the relevant
    data into a list.
    It is expected the different sets of data is seperated by an empty line
    in the vtk file.
    """
    DataIn=[]
    outputf=open(logfile, 'a')
    try:
        open(vtkfile)
    except (OSError, IOError) as e:
        print(e)
        print(' ')
        print('Aborted! '+vtkfile+' is missing!')
        outputf=open(logfile, 'a')
        outputf.write('Aborted! '+vtkfile+' is missing! \n')
        outputf.close()
        sys.exit

    #Use space as a limiter (or perhapd tab \t ?)
    with open(vtkfile, newline='') as csvfile:
        originalcontent = csv.reader(csvfile, delimiter=' ')
        for row in originalcontent:
            DataIn.append(row)
    print('VTK file "'+vtkfile+'" has been read!')
    outputf=open(logfile, 'a')
    outputf.write('VTK file "'+vtkfile+'" has been read! \n')
    outputf.close()
    VTK=[]
    VTKHeader=[]
    VTKData=[]
    VTKHeaders=[]
    VTKDats=[]
    for row in range(len(DataIn)):
        while True:
            try:
                DataIn[row].remove(' ')
            except ValueError:
                break
    for row in range(len(DataIn)):
        while True:
            try:
                DataIn[row].remove('')
            except ValueError:
                break
    for row in range(len(DataIn)):
        if len(DataIn[row])!=0:
            #if type(DataIn[row][0])==str:
            try:
                test=float(DataIn[row][0])
                VTKData.append(DataIn[row])
                lastrow='data'
            except ValueError:
                VTKHeader.append(DataIn[row])
                lastrow='header'
            #Check what's comming!
            if row!=len(DataIn)-1:
                if len(DataIn[row+1])!=0:
                    try:
                        test=float(DataIn[row+1][0])
                        nextrow='data'
                    except ValueError:
                        nextrow='header'
                else:
                    nextrow='header'
            else:
                nextrow='header'

            if nextrow=='header' and lastrow=='data':
                VTKHeaders.append(VTKHeader)
                VTKDats.append(VTKData)
                VTKHeader=[]
                VTKData=[]

    VTK.append(VTKHeaders)
    VTK.append(VTKDats)

    print('VTK file "'+vtkfile+'" has been processed! \n')
    outputf=open(logfile, 'a')
    outputf.write('VTK file "'+vtkfile+'" has been processed! \n')
    outputf.write(' \n')
    outputf.close()
    return VTK

def vtkSearch(VTKlist,term,logfile):
    """
    Searches the list that has been created from the VTK file
    """
    HeaderCont=[]
    DataCont=[]
    for i in range(len(VTKlist[0])):
        for j in range(len(VTKlist[0][i])):
            if term in VTKlist[0][i][j]:
                DataCont=VTKlist[1][i]
                HeaderCont=VTKlist[0][i]
    print('Search for "'+term+'" has been completed!')
    outputf=open(logfile, 'a')
    outputf.write('Search for "'+term+'" has been completed! \n')
    outputf.close()
    return HeaderCont,DataCont

def translation(Points,UnitVector,Mag):   #old mod_model
    """
    Modify point coordinates by rigidly moving them along any specified vector.
    """

    #List of new points
    New_P=[]
    for i in range(len(Points)):

        Px=Points[i][0]+ UnitVector[0]*Mag
        Py=Points[i][1]+ UnitVector[1]*Mag
        Pz=Points[i][2]+ UnitVector[2]*Mag

        New_P.append([Px,Py,Pz])
    return New_P

def Rot_Vec(u, theta):
    R = [[cos(theta) + u[0]**2 * (1-cos(theta)),
          u[0] * u[1] * (1-cos(theta)) - u[2] * sin(theta),
                 u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta)],
         [u[0] * u[1] * (1-cos(theta)) + u[2] * sin(theta),
                 cos(theta) + u[1]**2 * (1-cos(theta)),
                 u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta)],
         [u[0] * u[2] * (1-cos(theta)) - u[1] * sin(theta),
                 u[1] * u[2] * (1-cos(theta)) + u[0] * sin(theta),
                 cos(theta) + u[2]**2 * (1-cos(theta))]]
    return R

def rotation(Points,Origin,UnitVector,angle):
    """
    Modify point coordinates by rigid rotation about a unity vector, giving as input the origin of the axis and the angle of rotation
    Points: list of points read from the vtk file
    Origin: vector containing the coordinates of the origin of the axis of rotation
    rot_vect: unitary vector of rotation
    theta: angle of rotation
    """
    P = np.array(Points) - np.array(Origin)
    theta = np.deg2rad(angle)
    u = UnitVector
    New_P = []
    R = Rot_Vec(u, theta)
    for i in range(len(Points)):
        P2= np.dot(P[i],R)

        New_P.append(P2)
    New_P = New_P + np.array(Origin)
    return New_P


def rotate_to_alighn_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    A, B = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 /
                                                      np.linalg.norm(vec2)).reshape(3)
    v = np.cross(A, B)
    c = np.dot(A, B)
    s = np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + k.dot(k) * ((1 - c) / (s ** 2))
    return R

def centroid(Points):
    x_coords = [p[0] for p in Points]
    y_coords = [p[1] for p in Points]
    z_coords = [p[2] for p in Points]
    _len = len(Points)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    centroid_z = sum(z_coords)/_len
    return [centroid_x, centroid_y, centroid_z]


def cell_centres(points, cells):
    Cent = np.zeros(cells.shape)
    for i in range(len(cells)):

        a = points[int(cells[i][0])]
        b = points[int(cells[i][1])]
        c = points[int(cells[i][2])]
        P = np.concatenate([[a], [b], [c]], axis=0)
        Cent[i] = centroid(P)
    return Cent


def radial_axis_unitVec(Points):
    [Xc,Yc,Zc] = centroid(Points)
    Norm = math.sqrt(Xc**2+Yc**2+Zc**2)
    Unit_Vector = [Xc/Norm,Yc/Norm,Zc/Norm]
    return Unit_Vector

def normal_axis_unit_vec(Points,memory_reduction_limit=True):
    [Xc, Yc, Zc] = centroid(Points)

    if memory_reduction_limit:
        element_max_size = 2E4
        interp = 1
        if max(np.shape(Points)) > element_max_size:
            interp = math.ceil(max(np.shape(Points))/element_max_size)
            print('WARNING: \n Points size very large for SVD \n\tPoints matrix size : {} \n\tAccessing every {} elements'.format(
                np.shape(Points), interp))

    result = np.linalg.svd(np.concatenate([Points], axis=0)[::interp] - np.array([Xc, Yc, Zc]))
    normal = np.cross(result[2][0], result[2][1])
    return normal


def tile_poloidal_axis_unit_vec(Points):
    norm = normal_axis_unit_vec(Points)
    print('WARNING: Only correct for Modules. Need to change Y_axis for bodies')
    poloid = np.zeros(3)
    poloid[0] = -norm[2]
    poloid[1] = norm[1] #0
    poloid[2] = norm[0]
    return poloid


def tile_toroidal_axis_unit_vec(Points):
    print('WARNING: Only correct for Modules. Need to change X_axis for bodies')
    norm = normal_axis_unit_vec(Points)

    toroid = np.zeros(3)
    toroid[0] = norm[2] #0
    toroid[1] = -norm[0]
    toroid[2] = norm[1]
    return toroid


def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [
                 0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(R, vector)


def Ang_Vec(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def Vec(p1, p2):
    distance = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
    norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2 + distance[2] ** 2)
    direction = [distance[0] / norm, distance[1] / norm, distance[2] / norm]
    return direction


def tile_poloidal_axis(centroid, origin=np.zeros(3)):
    cent = np.zeros(3)
    cent[:2] = centroid[:2]
    v0 = Vec(np.concatenate([cent[:2], [0]]), [0, 0, 0])
    theta = Ang_Vec([1, 0], v0[:2])*[1 if cent[1] > 0 else -1][0]
    poloidal = z_rotation(np.array([0, 1, 0]), theta)
    return poloidal


def tile_toroidal_axis(centroid, origin=np.array([8500, 0, 0])):
    # TODO: change geometry references to implicit

    print('Warning : This function is incorrect')
    """
    poltil = tile_poloidal_axis(centroid, origin=np.zeros(3))

    cent = np.zeros(3)
    cent[0] = centroid[0]
    cent[2] = centroid[2]

    T = np.dot(norm[-1], FW.Rot_Vec(poltil[-1], np.deg2rad(90)))
    v0 = Vec(T, np.concatenate([cent[:2], [-6000]]))
    n_a = np.rad2deg(Ang_Vec(v0, N))

    if n_a < 90:
        T = T*-1

    v0 = np.array([1, 0, 0])
    v1 = Vec(origin, cent)
    theta = Ang_Vec(v0[::2], v1[::2])*[1 if cent[2] > 0 else -1][0]
    toroidal = y_rotation(np.array([0, 0, 1]), theta)
    """
    return toroidal


def tile_coord_system(points):
    x_c, y_c, z_c = centroid(points)
    cent = np.concatenate([[x_c], [y_c], [z_c]])

    #### --------Fit Normals--------- ####
    normal = normal_axis_unit_vec(points)
    v0 = Vec(cent, [8500, 0, 0])

    n_a = np.rad2deg(Ang_Vec(v0, normal))
    if n_a > 90:
        normal = normal*-1

    #### --------Fit poloidal--------- ####
    poloidal = tile_poloidal_axis(cent)

    #### --------Fit toroidal-------- ####
    toroidal = np.dot(normal, Rot_Vec(poloidal, np.deg2rad(90)))
    v0 = Vec(toroidal, np.concatenate([cent[:2], [-6000]]))
    n_a = np.rad2deg(Ang_Vec(v0, normal))
    if n_a < 90:
        toroidal = toroidal*-1
    return cent, normal, poloidal, toroidal

#from math import sqrt
#
#def euqli_dist(Points, centroid_x, centroid_y, centroid_z, squared=False):
#    # Calculates the euclidean distance, the "ordinary" distance between two
#    # points
#    #
#    # The standard Euclidean distance can be squared in order to place
#    # progressively greater weight on objects that are farther apart. This
#    # frequently used in optimization problems in which distances only have
#    # to be compared.
#    if squared:
#        return ((Points[0] - centroid_x) ** 2) + ((Points[1] - centroid_y) ** 2) + ((Points[2] - centroid_z) ** 2)
#    else:
#        return sqrt(((Points[0] - centroid_x) ** 2) + ((Points[1] - centroid_y) ** 2)) + ((Points[2] - centroid_z) ** 2)
#
#def closest(cur_pos, Points):
#    low_dist = float('inf')
#    closest_pos = None
#    for p in range(len(Points)):
#        dist = euqli_dist(Points,centroid_x,centroid_y,centroid_z)
#        if dist < low_dist:
#            low_dist = dist
#            closest_x = Points[0]
#            closest_y = Points[1]
#            closest_z = Points[2]
#    return [closest_x,closest_y,closest_z]           # center of the local coordinate system

#############################################################################

def VTK_P(In_list,logfile):
    """
    Collect point coordinates from the VTK list file.
    """
    #List of Points
    Points=[]
    #List of coordinates
    Clist=[]
    #Collect point coordinates
    print('Collecting point coordinates from VTK file.')
    outputf=open(logfile, 'a')
    outputf.write('Collecting point coordinates from VTK file. \n')
    outputf.close()
    Head,Data=vtkSearch(In_list,'POINTS',logfile)
    for i in range(len(Data)):
        for j in range(len(Data[i])):
            if len(Data[i][j])!=0:
                Clist.append(float(Data[i][j]))
    if len(Clist)%3==0:
        for i in range(len(Clist)//3):
            j=3*i
            Points.append([Clist[j],Clist[j+1],Clist[j+2]])
    else:
        print('Number of coordinates is not a multiple of 3!')
        outputf=open(logfile, 'a')
        outputf.write('Number of coordinates is not a multiple of 3! \n')
        outputf.close()

    print('Point coordinates have been collected. \n')
    outputf=open(logfile, 'a')
    outputf.write('Point coordinates have been collected. \n')
    outputf.write(' \n')
    outputf.close()
    return Points

def VTK_C(In_list,logfile):
    """
    Collect Cells from the VTK list file.
    """
    #List of Polygons
    Polygons=[]
    #List of Polygons Connectivity
    Plist=[]
    #Collect Polygons
    print('Collecting Polygons from VTK file.')
    outputf=open(logfile, 'a')
    outputf.write('Collecting Polygons from VTK file. \n')
    outputf.close()
    Head,Data=vtkSearch(In_list,'POLYGONS',logfile)
    if not Head:
        Head, Data = vtkSearch(In_list, 'CELLS', logfile)
    
    for i in range(len(Data)):
        for j in range(len(Data[i])):
            if len(Data[i][j])!=0:
                Plist.append(float(Data[i][j]))
    if len(Plist)%4==0:
        for i in range(len(Plist)//4):
            j=4*i
            Polygons.append([Plist[j+1],Plist[j+2],Plist[j+3]])
    else:
        print('Number of Polygons is not a multiple of 3! Check mesh is tetrahedral')
        outputf=open(logfile, 'a')
        outputf.write('Number of Polygons is not a multiple of 3! \n Check mesh is tetrahedral')
        outputf.close()

    print('Polygons have been collected. Each corresponds to one Cell\n')
    outputf=open(logfile, 'a')
    outputf.write('Polygons have been collected. \n')
    outputf.write(' \n')
    outputf.close()

    CellNo = range(len(Polygons))
    return Polygons

def VTK_B(In_list,logfile):
    """
    Collect Bodies from the VTK list file.
    """
    #List of Bodies
    Bodies=[]
    #List of Bodies Connectivity
    Blist=[]

    #Collect Bodies
    print('Collecting Bodies from VTK file.')
    outputf=open(logfile, 'a')
    outputf.write('Collecting Bodies from VTK file. \n')
    outputf.close()
    Head,Data=vtkSearch(In_list,'Body',logfile)
    for i in range(len(Data)):
        for j in range(len(Data[i])):
            if len(Data[i][j])!=0:
                Blist.append(float(Data[i][j]))
    for i in range(len(Blist)):
            Bodies.append(Blist[i])
    print('Bodies have been collected. Each corresponds to one Cell\n')
    outputf=open(logfile, 'a')
    outputf.write('Bodies have been collected. \n')
    outputf.write(' \n')
    outputf.close()
    return Bodies

def VTK_N(In_list,logfile):
    """
    Collect Normals from the VTK list file.
    """
    #List of Normals
    Normals=[]
    #List of Normals Connectivity
    Nlist=[]
    #Collect Normals
    print('Collecting Normals from VTK file.')
    outputf=open(logfile, 'a')
    outputf.write('Collecting Normals from VTK file. \n')
    outputf.close()

    Head,Data=vtkSearch(In_list,'Normal',logfile)
    for i in range(len(Data)):
        for j in range(len(Data[i])):
            if len(Data[i][j])!=0:
                Nlist.append(float(Data[i][j]))
    if len(Nlist)%3==0:
        for i in range(len(Nlist)//3):
                j=3*i
                Normals.append([Nlist[j],Nlist[j+1],Nlist[j+2]])
    else:
        print('Number of Normals is not a multiple of 3! Check Normals in cartesian coords')
        outputf=open(logfile, 'a')
        outputf.write('Number of Normals is not a multiple of 3! \n Check Normals in cartesian coords')
        outputf.close()
    print('Normals have been collected. Each corresponds to one Cell\n')
    outputf=open(logfile, 'a')
    outputf.write('Normals have been collected. \n')
    outputf.write(' \n')
    outputf.close()
    return Normals

def VTK_F(In_list,logfile,type_name = 'Bcart'):
    """
    Collect Feild incidence from the VTK list file.
    """
    #List of Feilds
    Feild=[]
    #List of Feilds Connectivity
    Flist=[]
    #Collect Feilds
    print('Collecting {} from VTK file.'.format(type_name))
    outputf=open(logfile, 'a')
    outputf.write('Collecting {} from VTK file. \n'.format(type_name))
    outputf.close()

    Head,Data=vtkSearch(In_list,'{}'.format(type_name),logfile)
    #return Head, Data
    for i in range(len(Data)):
        for j in range(len(Data[i])):
            if len(Data[i][j])!=0:
                Flist.append(float(Data[i][j]))
    if len(Flist)%3==0:
        for i in range(len(Flist)//3):
                j=3*i
                Feild.append([Flist[j],Flist[j+1],Flist[j+2]])
    else:
        print('Number of {} is not a multiple of 3! Check {} in cartesian coords'.format(type_name,type_name))
        outputf=open(logfile, 'a')
        outputf.write('Number of {} is not a multiple of 3! \n Check {} in cartesian coords'.format(type_name,type_name))
        outputf.close()
    print('{} have been collected. Each corresponds to one Cell\n'.format(type_name))
    outputf=open(logfile, 'a')
    outputf.write('{} have been collected. \n'.format(type_name))
    outputf.write(' \n')
    outputf.close()
    return Feild

def WriteVTK(VTK_List,vtkfile,logfile):
    print('Writing new VTK file.')
    outputf=open(logfile, 'a')
    outputf.write('Writing new VTK file. \n')
    VTKHeader=[]
    VTKData=[]
    outputVTKf=open(vtkfile, 'w')
    for i in range(len(VTK_List[0])):
        VTKHeader=VTK_List[0][i]
        VTKData=VTK_List[1][i]
        for data in range(len(VTKHeader)):
            for data_item in range(len(VTKHeader[data])):
                outputVTKf.write(VTKHeader[data][data_item])
                outputVTKf.write(' ')
            outputVTKf.write('\n')
        for data in range(len(VTKData)):
            for data_item in range(len(VTKData[data])):
                outputVTKf.write(str(VTKData[data][data_item]))
                outputVTKf.write(' ')
            outputVTKf.write('\n')
        outputVTKf.write('\n')
    outputVTKf.close()
    print('Writing VTK file completed.')
    outputf.write('Writing VTK file completed. \n')
    print('New VTK file "'+vtkfile+'" has been created.')
    outputf.write('New VTK file "'+vtkfile+'" has been created. \n')
    outputf.write(' \n')
    outputf.close()
    return

""" needs some work to correct the log file generation in the new elaborated framework

"""
def modFW(New_points,input_VTK,output_VTK,logfile):

    #New_points=[]
    copyR(logfile)

#    #Read the source file
    VTK_L=ReadVTK(input_VTK,logfile)
#    #Find the nodes that need to be modified
#    AllPoints=VTK_P(VTK_L,logfile)
#    #Apply modification
#

    print('Modifying point coordinates.')
    outputf=open(logfile, 'a')
    outputf.write('Modifying point coordinates. \n')
#    print('Checking outboard segments...')
#    outputf.write('Checking outboard segments... \n')
#    mod_points=mod_OB(AllPoints)#,ROB,ZOB,OBpushback,OB_Zlim,OB_Zmin,OBC_Ymin,OBC_Ymax,Sector_pos)
    print('Checking model...')
    outputf.write('Checking model... \n')
   # New_points=     #mod_model(AllPoints,d)
    print('Modification done. \n')
    outputf.write('Modification done. \n')
    outputf.write(' \n')
    outputf.close()
    #Replace point coordinates in VTK list
    for i in range(len(VTK_L[0])):
        for j in range(len(VTK_L[0][i])):
            if 'POINTS' in VTK_L[0][i][j]:
                VTK_L[1][i]=New_points
    #Write new VTK file
    WriteVTK(VTK_L,output_VTK,logfile)
    return

def CombineVTK(listVTKs,OutputName,vtktfm):
    " correct to relative path"

    BodNum = range(len(listVTKs))
    BodNum = [(x+1)*100+1 for x in BodNum]

    fpath = '/Users/dominiccalleja/demo_misalighnments/exec/geom/CTL/CombineVTK.ctl'

    fm =str()
    for i in range(len(listVTKs)):
        fm=fm+''''{}','''

    with open(fpath, "r+") as f:
     old = f.read() #
     txt = old.replace('<VTKFiles>',fm.format(*listVTKs))
     txt = txt.replace('<BodyNumbers>',",".join(map(str,BodNum)))
     txt = txt.replace('<LogicOutput>',str(len(listVTKs))+'*1')

     Output = OutputName.split('.')
     fout = open(Output[0]+'.ctl',"wt")
     fout.write(txt)
     fout.close()
    rtn = os.system('{} {}'.format(vtktfm,fout.name))
    return rtn





def combine_vtk_files(VTK_list, Geom_dir=[], Body_No=[], VTK_output=[], output_directory=[], combine_ctl='/Users/dominiccalleja/smardda_workflow/DEMO_WORK/combine_geometry.ctl'):
    """
    Use this for compining VTK files. Uses VTKTFM correctly! 
    """
    if not Body_No:
        Body_No = (np.array(range(len(VTK_list)))+1)*100+1

    bodies = ''
    vtk_list_string = ''
    for i in range(len(VTK_list)):
        try:
            shutil.copy(Geom_dir+'/'+VTK_list[i], output_directory)
        except:
            print('Attempted to copy :{} \n File already exists in :{} \n\t Skipping..'.format(
                VTK_list[i], output_directory))
        vtk_list_string += '\'{}\', '.format(VTK_list[i])
        bodies += '{}, '.format(Body_No[i])
    N_files = len(VTK_list)

    with open(combine_ctl, "r+") as f:
        old = f.read()
        txt = old.replace('<n_files>', str(N_files))
        txt = txt.replace('<list_files>', vtk_list_string)
        txt = txt.replace('<panel_body_numbers>', bodies)

        Output = output_directory+'/'+VTK_output.split('.vtk')[0]+'.ctl'
        fout = open(Output, "wt")
        fout.write(txt)
        fout.close()
    ret_dir = os.getcwd()
    os.chdir(output_directory)
    os.system('{} {}'.format(smardda_path+'/vtktfm',
                             VTK_output.split('.vtk')[0]+'.ctl'))
    os.chdir(ret_dir)




def ExtractResultData(VTK,logfile='LogAnalysis.txt'):

    #List of flux values
    Q=[]
    qlist=[]

    #Collect heat flux
    print('Collecting Q from VTK file.')
    outputf=open(logfile, 'a')
    outputf.write('Collecting Q from VTK file. \n')
    outputf.close()
    Head,Data=vtkSearch(VTK,'Q',logfile)
    for i in range(len(Data)):
        for j in range(len(Data[i])):
            if len(Data[i][j])!=0:
                qlist.append(float(Data[i][j]))
    for i in range(len(qlist)):
            Q.append(qlist[i])
    print('HeatFlux has been collected. Each value corresponds to one Cell\n')
    outputf=open(logfile, 'a')
    outputf.write('Flux values have been collected. \n')
    outputf.write(' \n')
    outputf.close()
    return Q

def plot_module_axis(points,norm,pol,tor,projection=[30,110], save_address=[]):
    import matplotlib.pyplot as plt
    
    x_c, y_c, z_c =centroid(points)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d',)
    ax.view_init(elev=projection[0], azim=projection[1])
    ax.scatter(points[:, 0],
            points[:, 1], points[:, 2], alpha=0.02)
    ax.scatter(x_c,y_c,z_c,color='red',marker='+',s=1000)
    ax.quiver(x_c, y_c, z_c, -3000*norm[0], -3000*norm[1], -3000*norm[2], color='red')
    ax.quiver(x_c, y_c, z_c, -3000*pol[0], -3000*0, -3000*pol[2], color='green')
    ax.quiver(x_c, y_c, z_c, -3000*0, -3000*tor[1], -3000*tor[2], color='purple')
    ax.scatter(0,0,0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    if save_address:
        fig.savefig(save_address)
    else:
        return fig
