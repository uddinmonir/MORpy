# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:10:15 2020

@author: Md. Tanzim Hossain
"""
"""
This script test frequency-limited 
H2 optimal model reduction generalized system
"""
import math
import scipy.io as sio
from scipy import sparse
from scipy.sparse.linalg import spsolve
from IRKA import IRKA
from gen_TSIA import gen_TSIA
from gen_Fl_TSIA import gen_Fl_TSIA
from tf_plot import tf_plot
def call():
        
        file = sio.loadmat('E:/STUDY MATERIAL/MONIR SIR/ALL/data/1st_order/dae/index1/bps_606.mat')
    
        B1 = file['B1']
        B2 = file['B2']
        C1 = file['C1']
        C2 = file['C2']
        E1 = file['E1']
        J1 = file['J1']
        J2 = file['J2']
        J3 = file['J3']
        J4 = file['J4']
        
        B1 = sparse.csc_matrix(B1)
        B2 = sparse.csc_matrix(B2)
        C1 = sparse.csc_matrix(C1)
        C2 = sparse.csc_matrix(C2)
        E1 = sparse.csc_matrix(E1)
        J1 = sparse.csc_matrix(J1)
        J2 = sparse.csc_matrix(J2)
        J3 = sparse.csc_matrix(J3)
        J4 = sparse.csc_matrix(J4)
        
        E = E1
        A = J1 - J2 * spsolve(J4, J3)
        B = B1 - J2 * spsolve(J4, B2)
        C = C1 - C2 * spsolve(J4, J3)
        
        A = sparse.csc_matrix(A)
        E = sparse.csc_matrix(E)
        B = sparse.csc_matrix(B)
        C = sparse.csc_matrix(C)
    
        """
        #########################
        ###### COMPUTE ROM ######
        #########################
        """
        r = 20    
        maxiter = 30  
        tol = math.pow(10,-8)
        
        """IRKA for initial ROM"""
        print("*** IRKA for initial ROM ***\n")
        Ar,Br,Cr = IRKA(E,A,B,C,r,maxiter,tol)
        
        """TSIA for frequency unrestricted"""
        print("\n*** TSIA for frequency unrestricted ***\n")
        Ar,Br,Cr = gen_TSIA(E,A,B,C,Ar,Br,Cr,maxiter,tol);
        
        """Frequency range [w1,w2]"""
        w1 = 6
        w2 = 10
        
        """TSIA for frequency restricted"""
        print("\n*** TSIA for frequency restricted ***\n")
        Ar_fl, Br_fl, Cr_fl = gen_Fl_TSIA(E,A,B,C,Ar,Br,Cr,maxiter,tol,w1,w2)
        
        
    
call()