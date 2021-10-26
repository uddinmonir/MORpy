# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 01:37:15 2020

@author:
Dr. Mohammad Monir Uddin
Md. Tanzim Hossain
"""
"""import necessary libraries"""
import math
import scipy.io as sio
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from tf_plot import tf_plot
#from tf_plot2 import tf_plot2

if __name__ == "__main__": 
    
    file = sio.loadmat('E:/STUDY MATERIAL/MONIR SIR/2020/freq_lim_IRKA/generalized_ex/tchain_35.mat')
    #file = sio.loadmat('E:/STUDY MATERIAL/MONIR SIR/2020/index1/bps_1142_p.mat')
    #file = sio.loadmat('E:/STUDY MATERIAL/MONIR SIR/2020/index1/bps_1450_p.mat')
    #file = sio.loadmat('E:/STUDY MATERIAL/MONIR SIR/2020/index1/bps_1693_p.mat')
    
    A = file['A']
    B = file['B']
    C = file['C']
    E = file['E']
    Ar = file['Ar']
    Br = file['Br']
    Cr = file['Cr']
    Er = file['Er']
    Ar_fl = file['Ar_fl']
    Br_fl = file['Br_fl']
    Cr_fl = file['Cr_fl']
    Er_fl = file['Er_fl']
    
    
    A = csc_matrix(A)
    B = csc_matrix(B)
    C = csc_matrix(C)
    E = csc_matrix(E)
    Ar = csc_matrix(Ar)
    Br = csc_matrix(Br)
    Cr = csc_matrix(Cr)
    Er = csc_matrix(Er)
    Ar_fl = csc_matrix(Ar_fl)
    Br_fl = csc_matrix(Br_fl)
    Cr_fl = csc_matrix(Cr_fl)
    Er_fl = csc_matrix(Er_fl)
    
    
    
    low_point = 0  # Lower point of the plotting domain
    up_point = 4  # Higher point of the plotting domain
    tot_point = 200
    space,tf = tf_plot(A,B,C,E,Ar,Br,Cr,Er,Ar_fl,Br_fl,Cr_fl,Er_fl,low_point,up_point,tot_point)
        
    # print("\n\n***** Ok at this point *****\n")
    
    
