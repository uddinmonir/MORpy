# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:26:20 2020

@author: 
Dr. Mohammad Monir Uddin
Md. Tanzim Hossain
"""
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

def tf_plot2(A,B,C,E,Ar,Br,Cr,Er,Ar_fl,Br_fl,Cr_fl,Er_fl,low_point,up_point,tot_point):
    space = np.linspace(low_point,up_point,tot_point)
    tf = np.zeros(1000)
    tf_rom = np.zeros(1000)
    tf_rom_fl = np.zeros(1000)
    abs_err = np.zeros(1000)
    abs_err_fl = np.zeros(1000)

    for k in range (tot_point):
        G1 = C*(spsolve((1j*space[k]*E-A), B))
        G2 = Cr*(spsolve(csc_matrix((1j)*space[k]*Er-Ar),csc_matrix(Br)))
        G3 = Cr_fl*(spsolve(csc_matrix((1j)*space[k]*Er_fl-Ar_fl),csc_matrix(Br_fl)))
        
        G1 = csc_matrix(G1)
        G2 = csc_matrix(G2)
        G3 = csc_matrix(G3)
        print("G1.shape = ", G1.shape)
        print("G2.shape = ", G2.shape)
        print("G3.shape = ", G3.shape)
        
        U1,sigma1,Ub1 = np.linalg.svd(csc_matrix(G1).toarray(),full_matrices=True)
        U2,sigma2,Ub2 = np.linalg.svd(csc_matrix(G2).toarray(),full_matrices=True)
        U3,sigma3,Ub3 = np.linalg.svd(csc_matrix(G3).toarray(),full_matrices=True)       
        U4,sigma4,Ub4 = np.linalg.svd(csc_matrix(G1-G2).toarray(),full_matrices=True)
        U5,sigma5,Ub5 = np.linalg.svd(csc_matrix(G1-G3).toarray(),full_matrices=True)

        tf[k] = max(sigma1)
        tf_rom[k] = max(sigma2)
        tf_rom_fl[k] = max(sigma3)
        abs_err[k] = max(sigma4)
        abs_err_fl[k] = max(sigma5)
    
    rel_err = abs_err / tf
    rel_err_fl = abs_err_fl / tf

    # Transfer Function Plot for full,frequency limited and frequency unrestricted reduced model       
    plt.figure()
    plt.title('Transfer function')
    plt.xlabel('omega')
    plt.ylabel('sigma')
    plt.semilogy(space,tf,'green', label='Full model')
    plt.semilogy(space,tf_rom,'--b', label='Frequency unrestricted')
    plt.semilogy(space,tf_rom_fl,'--r', label='Frequency restricted')
    plt.legend()
    plt.savefig('Transfer function.tiff', format='tiff', dpi=1200)
    plt.show()
      

    # Absolute Error
    plt.title('Absolute model reduction error')
    plt.xlabel('omega')
    plt.ylabel('sigma')
    plt.semilogy(space,abs_err,'--b', label='Frequency unrestricted') 
    plt.semilogy(space,abs_err_fl,'r', label='Frequency restricted')
    plt.legend()
    plt.savefig('Absolute error.tiff', format='tiff', dpi=1200)
    plt.show()

    # Relative error;
    plt.title('Relative model reduction error')
    plt.xlabel('omega')
    plt.ylabel('sigma')
    plt.semilogy(space,rel_err,'b', label='Frequency unrestricted') 
    plt.semilogy(space,rel_err_fl,'r', label='Frequency restricted')
    plt.legend()
    plt.savefig('Relative model reduction error.tiff', format='tiff', dpi=1200)
    plt.show()

   
    return space, tf
