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
from IRKA import IRKA
from gen_TSIA import gen_TSIA
from gen_Fl_TSIA import gen_Fl_TSIA
from tf_plot import tf_plot
def call():
        
        file = sio.loadmat('E:/STUDY MATERIAL/MONIR SIR/ALL/data/1st_order/generalized/tchain.mat')
   
   
        A = file['A']
        B = file['B']
        C = file['C']
       
        A = sparse.csc_matrix(A)
        E = sparse.eye(A.shape[0])
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
        w1 = 1
        w2 = 2
        
        """TSIA for frequency restricted"""
        print("\n*** TSIA for frequency restricted ***\n")
        Ar_fl, Br_fl, Cr_fl = gen_Fl_TSIA(E,A,B,C,Ar,Br,Cr,maxiter,tol,w1,w2)
        
        
        low_point = 0;      """Lower point of the plotting domain"""
        up_point = 4;     """Higher point of the plotting domain"""
        tot_point = 200;   
        Er = sparse.eye(Ar.shape[0])
        Er_fl = Er
        
        """Ploat transfer function"""
        space,tf = tf_plot(A,B,C,E,Ar,Br,Cr,Er,Ar_fl,Br_fl,Cr_fl,Er_fl,low_point,up_point,tot_point)

    
call()