# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:27:07 2020

@author: Md. Tanzim Hossain
"""
import numpy as np
from scipy import sparse
from scipy.linalg import eig, qr
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

def IRKA(E,A,B,C,r,maxiter,tol):
    """Initialization"""   
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    S = 100*np.random.rand(r,1)
    b = np.random.randn(m,r)
    c = np.random.randn(p,r)
   
    
    """Start iteration"""
    for i in range(maxiter):
        S_old = S
        
        """Compute projection subspaces"""
        V = np.zeros((n,r))
        W = np.zeros((n,r))
        j = 0
        
        while j < r:
            btemp = csc_matrix(b[:,j])
            ctemp = csc_matrix(c[:,j])
            
            pc=S[j]
            if pc.imag == 0:
                pc=float(pc.real)
            else:
                pc=complex(pc)    
                
            Atil_x = pc*E - A
            # print("Atil_x = ", Atil_x.shape)
            # print("B = ", B.shape)
            # print("btemp = ", btemp.T.shape)
            X = spsolve(csc_matrix(Atil_x),csc_matrix(B * btemp.T))      
            Atil_y = pc*E.H - A.H   
            # print("Atil_y = ", Atil_y.shape)
            # print("C.H = ", C.H.shape)
            # print("ctemp = ", ctemp.shape)
            Y = spsolve(csc_matrix(Atil_y),csc_matrix(C.H * ctemp.T))
            # print("X.real = ", X.shape)
            # print("type(Y) = ", type(Y))
            
            con = np.isreal(pc)
            # print("type(con) = ", type(con))
            if con == False:
                V[:,j] = X.real
                W[:,j] = Y.real   
                V[:,j+1] = X.imag
                W[:,j+1] = Y.imag 
                j = j+2
            else:
                # print("V[:,j] = ", V[:,j].shape)
                # print("X.real = ", X.real.shape)
                V[:,j] = X.real
                W[:,j] = Y.real
                j = j+1
                
                               
        V, _ = qr(V, mode='economic')
        W, _ = qr(W, mode='economic')
        
        W = csc_matrix(W)
        V = csc_matrix(V)
        
        """Compute ROM"""
        Er = W.H * E * V
        
        t1 = sparse.csr_matrix(Er).toarray()
        t2 = sparse.csr_matrix(W.H*A*V).toarray()
        Ar = np.linalg.solve(t1,t2)
        
        t3 = sparse.csr_matrix(W.H*B).toarray()
        Br = np.linalg.solve(t1,t3)
        
        Cr = C*V
        # Er = W.T * E * V
        # Ar = W.T * A * V
        # Br = W.T * B
        # Cr = C * V
        
        """Update interpolation points/tangential directions"""
        S, T = eig(Ar)
        S = -(S)
        Bn = sparse.csr_matrix(Br).toarray()
        En = sparse.csr_matrix(Er).toarray() 
        solve = np.linalg.solve(En,Bn)  
        solve2 = np.linalg.solve(T, solve)  
        b = sparse.csr_matrix(solve2.T).toarray()
        c = Cr*T    
        
        """Check for convergence""" 
        err = np.linalg.norm(np.sort(S)-np.sort(S_old)) / np.linalg.norm(S_old)
        """comment out for non-verbose mode"""
        print("IRKA step ",i+1, ", conv. crit. = ",err , "\n")
        
        if err < tol:
           break
    if iter == maxiter and err > tol:
        print("IRKA: No convergence in ", maxiter, " iterations.\n")
    return Ar,Br,Cr
        