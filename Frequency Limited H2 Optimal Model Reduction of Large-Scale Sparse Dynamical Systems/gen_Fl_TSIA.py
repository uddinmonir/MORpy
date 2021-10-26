# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 18:16:50 2020

@author: 
Dr. Mohammad Monir Uddin
Md. Tanzim Hossain
"""

import numpy as np
import time
from numpy import diag, pi
from scipy import sparse
from scipy.sparse.linalg import eigs, spsolve
from scipy.linalg import logm, eig, qr
from scipy.sparse import csc_matrix

from sylv_ls_sg import sylv_ls_sg 

def gen_Fl_TSIA(E,A,B,C,Ar,Br,Cr,maxiter,tol,w1,w2):
    
    """Initialization"""
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    # fullAr = sparse.csr_matrix(Ar).toarray()
    # fullEr = sparse.csr_matrix(Er).toarray()
    # sol = np.linalg.solve(fullEr, fullAr)
    # sol1 = np.linalg.eigvals(sol)
    # sol1 = sparse.csr_matrix(sol1).toarray()
    # S = diag(sol1)
    S, _ = eig(Ar)
    I = np.eye(A.shape[0])
    Ir = np.eye(Ar.shape[0])
    tmp1 = sparse.csr_matrix(A+1j*w1*E).toarray()
    tmp2 = sparse.csr_matrix(A+1j*w2*E).toarray()
    S_omega = np.linalg.solve(tmp1,tmp2)
    
    """Patrick"""
    S_omega = np.real((1j/pi) * logm(S_omega))
    
    # tmp1 = sparse.csr_matrix(A+1j*w1*E).toarray()
    # tmp2 = sparse.csr_matrix(A+1j*w2*E).toarray()
    # A_omg1 = np.real((1j/pi) * logm(np.linalg.solve(tmp1.T, tmp2.T)))  
    # temp = np.linalg.solve(tmp1,tmp2) 
    # A_omg2 = np.real((1j/pi) * logm(temp))
    exe_time = 0;
    """Start iteration"""
    for i in range(maxiter):
        S_old = S
        """Patrick"""
        tmp3 = sparse.csr_matrix(Ar+1j*w1*Ir).toarray()
        tmp4 = sparse.csr_matrix(Ar+1j*w2*Ir).toarray()
        tmp5 = np.linalg.solve(tmp3, tmp4)
        Sr_omega = np.real((1j/pi) * logm(tmp5))
        
        E = sparse.csr_matrix(E).toarray()
        B = sparse.csr_matrix(B).toarray()
        
        E_B = np.linalg.solve(E, B)
        
        
        B = csc_matrix(B)
        Br = csc_matrix(Br)
        E_B = csc_matrix(E_B)
        S_omega = csc_matrix(S_omega)
        Sr_omega = csc_matrix(Sr_omega)
        
        P = (E * S_omega * E_B * Br.H) + (B * Br.H * Sr_omega.H)
        Q = -((S_omega.H * C.H * Cr) + (C.H * Cr * Sr_omega))
        
        E = csc_matrix(E)
        A = csc_matrix(A)
        Ar = csc_matrix(Ar)
        
        start = time.time()
        X, _ = sylv_ls_sg(A, E, Ar.H, P)
        end = time.time()
        exe_time = exe_time + (end - start)
        
        Y, _ = sylv_ls_sg(A.H,E.H,Ar, Q)
        
        #V, _ = np.linalg.qr(X)
        #W, _ = np.linalg.qr(Y)
        X = sparse.csr_matrix(X).toarray()
        Y = sparse.csr_matrix(Y).toarray()
        V, _ = qr(X, mode='economic')
        W, _ = qr(Y, mode='economic')
        
        tmp5 = np.dot(V.T, W)
        W = np.linalg.solve(tmp5.T, W.T) 
        W = csc_matrix(W)
        W = W.H
        V = csc_matrix(V)
    
        """Compute ROM"""
        Er = W.H * E * V
        
        t1 = sparse.csr_matrix(Er).toarray()
        t2 = sparse.csr_matrix(W.H*A*V).toarray()
        Ar = np.linalg.solve(t1,t2)
        
        t3 = sparse.csr_matrix(W.H*B).toarray()
        Br = np.linalg.solve(t1,t3)
        
        Cr = C*V 
        
        """Update interpolation points/tangential directions"""
        S, _ = eig(Ar)
        
        # fullAn = sparse.csr_matrix(Ar).toarray()
        # fullEn = sparse.csr_matrix(Er).toarray()   
        # solve = np.linalg.solve(fullEn,fullAn)
        # soleig = np.linalg.eigvals(solve)
        # soleig = sparse.csr_matrix(soleig).toarray()
        # S = diag(soleig)
        
        """Check for convergence"""
        err = np.linalg.norm(np.sort(S)-np.sort(S_old)) / np.linalg.norm(S_old)
        print("IRKA step ",i+1, ", conv. crit. = ",err , "\n")
        
        if err < tol:
           break
       
    print("Iteration :: ", i+1)
    exe_time = exe_time / (i+1)
    print ('--- %0.3fms. --- ' % (exe_time*1000.))
    print("--- %s seconds ---" % (exe_time))
    print("--- %s minutes ---" % (exe_time / 60)) 
    if iter == maxiter and err > tol:
        print("IRKA: No convergence in ", maxiter, " iterations.\n")
    return Ar,Br,Cr
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
