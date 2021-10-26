# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:15:22 2020

@author: rabbi
"""
import numpy as np
from numpy import diag, pi
from scipy import sparse
from scipy.linalg import logm, eig, qr
from scipy.sparse import csc_matrix
from sylv_ls_sg import sylv_ls_sg 

def gen_TSIA(E,A,B,C,Ar,Br,Cr,maxiter,tol):

    """Initialization"""
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    S,_ = eig(Ar)
    I = np.eye(A.shape[0]);
    Ir = np.eye(Ar.shape[0]);
    
    
    """Start iteration"""
    for i in range (maxiter):
        Ar = csc_matrix(Ar)
        Br = csc_matrix(Br)
        Cr = csc_matrix(Cr)
        S_old = S
        P = B * Br.H
        Q = -C.H * Cr
        
        X, _ = sylv_ls_sg(A,E,Ar.H,P)
        Y, _ = sylv_ls_sg(A.H,E.H,Ar,Q)
        #V,_ = np.linalg.qr(X
        #W,_ = np.linalg.qr(Y)
        X = sparse.csr_matrix(X).toarray()
        Y = sparse.csr_matrix(Y).toarray()
        V, _ = qr(X, mode='economic')
        W, _ = qr(Y, mode='economic')
        
        
        
        tmp5 = np.dot(V.T, W)
        W = np.linalg.solve(tmp5.T, W.T) 
        W = csc_matrix(W)
        W = W.H
        V = csc_matrix(V)
        
        Er = W.T * E * V
        
        t1 = sparse.csr_matrix(Er).toarray()
        t2 = sparse.csr_matrix(W.H*A*V).toarray()
        Ar = np.linalg.solve(t1,t2)
        
        t3 = sparse.csr_matrix(W.H*B).toarray()
        Br = np.linalg.solve(t1,t3)
        
        Cr = C*V   
        S,_ = eig(Ar)
        
    
        err = np.linalg.norm(np.sort(S)-np.sort(S_old)) / np.linalg.norm(S_old)
        print("IRKA step ",i+1, ", conv. crit. = ",err , "\n")
        
        if err < tol:
           break
    if iter == maxiter and err > tol:
        print("IRKA: No convergence in ", maxiter, " iterations.\n")
    
    return Ar,Br,Cr
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    








