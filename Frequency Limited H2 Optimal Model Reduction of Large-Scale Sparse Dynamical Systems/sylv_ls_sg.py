# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 22:50:07 2020

@author: 
Dr. Mohammad Monir Uddin
Md. Tanzim Hossain
"""
import sys
import scipy as sp
import numpy as np
from numpy import diag
from scipy import sparse
from scipy.sparse import csc_matrix

def sylv_ls_sg(A,E,H,M):
    """
    function [ X, eigH ] = sylv_ls_sg( A,E,H,M, Trans )

    Solve the semi-generalized Sylvester Equation 
    
         A*X + E*X*H + M = 0                                              (1)
    
    with A and E large and sparse and H small and dense matrices. 
    The algorithm is described in [1].
    
    Inputs: 
     A,E      large and sparse input matrix of (1), dimension n0 x n0
     H        small and dense input matrix of (1), dimension n1 x n1
     M        right hand side of (1), dimension n0 x n1
     Trans    Optional flag. If Trans=='T' 
                  A'*X + E'*X*H' + M = 0                                  (2)
              is solved instead of (1). 
         
    
    Outputs: 
     X        Solution of (1) or (2), truncated to a real matrix.
     eigH     Eigenvalues of H
    
    
    [1] Sparse-Dense Sylvester Equations in H2-Model Order Reduction;
        Benner, Peter; KÃ¶hler, Martin; Saak, Jens;
        MPI Magdeburg Preprints 2011. 
    
    Matlab code Copyright 2011-2012, Martin KÃ¶hler
    MPI Magdeburg

    """
    n0 = A.shape[0]
    n1 = H.shape[0]
    
    if M.shape[0] != n0 and M.shape[1] != n1: 
        print("The dimension of the matrix M does not fit.")
        sys.exit()
    
    if A.shape != E.shape: 
        print("Dimension of A and E does not fit.")
        sys.exit()
    
    q = n1
    H = sparse.csr_matrix(H).toarray()
    U, S = sp.linalg.schur(H, output='complex')
    
    
    M = csc_matrix(M)
    U = csc_matrix(U)
    Mtilde = M * U
    Xtilde = np.zeros((A.shape[0], q))
    Xtilde = csc_matrix(Xtilde)
    for j in range(q):
        rhs = np.zeros((Mtilde.shape[0], 1))
        rhs = csc_matrix(rhs)
        
        for i in range(j-1):
            #tmp3 = sparse.csr_matrix(np.dot(S[i,j], Xtilde[:,i])).toarray()
            rhs = rhs + (S[i,j] * Xtilde[:,i])
            rhs = csc_matrix(rhs)
       
        Mtil = csc_matrix(Mtilde[:,j])
        rhs = -(Mtil) - (E * rhs)
        
        rhs = sparse.csr_matrix(rhs).toarray()
        tmp1 = sparse.csr_matrix(A + (S[j,j] * E)).toarray()
        tmp2 = np.linalg.solve(tmp1,rhs)
        tmp2 = csc_matrix(tmp2)
        # tmp2 = np.array(tmp2, dtype = "complex_")
        # Xtilde[:,j] = np.column_stack(tmp2)
        Xtilde[:,j] = tmp2
    
    
    Xtilde = csc_matrix(Xtilde)
    X = Xtilde * U.H
    X = X.real
    eigH = diag(S)
    
    return X, eigH
