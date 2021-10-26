# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:26:20 2020

@author: 
Dr. Mohammad Monir Uddin
Md. Tanzim Hossain
"""
import numpy as np
from numpy.linalg import solve
from scipy import sparse
from scipy.sparse.linalg import spsolve, svds
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import tikzplotlib

def tf_plot(A,B,C,E,Ar,Br,Cr,Er,Ar_fl,Br_fl,Cr_fl,Er_fl,low_point,up_point,tot_point):
    
    space = np.linspace(low_point,up_point,tot_point)
    tf = np.zeros(200)
    tf_rom = np.zeros(200)
    tf_rom_fl = np.zeros(200)
    abs_err = np.zeros(200)
    abs_err_fl = np.zeros(200)

    for k in range (tot_point):
        G1 = C*(spsolve((1j*space[k]*E-A), B))
        G2 = Cr*(spsolve(csc_matrix((1j)*space[k]*Er-Ar),csc_matrix(Br)))
        G3 = Cr_fl*(spsolve(csc_matrix((1j)*space[k]*Er_fl-Ar_fl),csc_matrix(Br_fl)))
        
        G1 = csc_matrix(G1)
        G2 = csc_matrix(G2)
        G3 = csc_matrix(G3)
        # print("G1.shape = ", G1.shape)
        # print("G2.shape = ", G2.shape)
        # print("G3.shape = ", G3.shape)
        
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
    #plt.title('Transfer function')
    plt.xlabel('$\\alpha$')
    plt.ylabel('$\\sigma$')
    plt.semilogy(space,tf,'green',marker='*', markersize=3, label='Full model')
    plt.semilogy(space,tf_rom,'--b', marker='^', markersize=3, label='Frequency unrestricted')
    plt.semilogy(space,tf_rom_fl,'--r', label='Frequency restricted')
    plt.legend()
    #plt.savefig('Transfer function3078.tiff', format='tiff', dpi=1200)
    tikzplotlib.save('tchain_tf.tex')
    # tikzplotlib.save("Transfer function.tex")
    plt.show()
      

    # Absolute Error
    #plt.title('Absolute model reduction error')
    plt.xlabel('$\\alpha$')
    plt.ylabel('$\\sigma$')
    plt.semilogy(space,abs_err,'--b',marker='*', markersize=3, label='Frequency unrestricted') 
    plt.semilogy(space,abs_err_fl,'r', marker='^', markersize=3, label='Frequency restricted')
    plt.legend()
    #plt.savefig('Absolute error3078.tiff', format='tiff', dpi=1200)
    tikzplotlib.save('tchain_abs_err.tex')
    # tikzplotlib.save("Absolute error.tex")
    plt.show()

    # Relative error;
    #plt.title('Relative model reduction error')
    plt.xlabel('$\\alpha$')
    plt.ylabel('$\\sigma$')
    plt.semilogy(space,rel_err,'b', marker='*', markersize=5, label='Frequency unrestricted') 
    plt.semilogy(space,rel_err_fl,'r', marker='^', markersize=5, label='Frequency restricted')
    plt.legend()
    #plt.savefig('Relative model reduction error3078.tiff', format='tiff', dpi=1200)
    tikzplotlib.save('tchain_rel_err.tex')
    # tikzplotlib.save("Relative model reduction error.tex")
    plt.show()

   
    return space, tf
