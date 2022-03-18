import numpy as np
import numba as nb
import math

def CosmicWebClassification( density_field, L_box, Nmesh, smoothing_scale, lambda_th):
    '''
    This function is used to classify the cosmic web based on the Hessian matrix of the gravitational potential
    
    Input:
      'density_field': 3-dimenional array-like, the density field at 3-dimensional regular mesh points;
      'L_box': float, the box size of the density field;
      'Nmesh': integer, the number of mesh points on each side of the simulation box;
      'smoothing_scale': float, the smoothing scale used to smooth the density field by Gaussian filter;
      'lambda_th': float, the threshold used to classify the cosmic web
    
    Output:
      'class_field': 3-dimensional array-like, the classification field'''
    
    # Fourier transform of the smoothed density field
    k =  (2.0*np.pi/L_box)*np.linspace(-float(Nmesh/2), float(Nmesh/2)-1.0, Nmesh, endpoint=True, dtype=np.float64)
    kkx, kky, kkz = np.meshgrid(k,k,k, indexing="ij")
    kk_mod=kkx**2+kky**2+kkz**2
    dens_k=((L_box/float(Nmesh))**3)*np.fft.fftn(density_field, axes=(-3,-2,-1), norm="backward")
    dens_k = np.fft.fftshift(dens_k)
    smooth_dens_k = dens_k*Gaussian_filter_k(smoothing_scale, kkx,kky,kkz)
    #-------------
    print("Ok-1:density field has been smoothed.")
    #-------------
    # Compute the gravitational potential
    phi_k = np.zeros((Nmesh, Nmesh, Nmesh), np.complex64)
    phi_k = - np.divide(smooth_dens_k, kk_mod, out=np.zeros_like(smooth_dens_k), where=kk_mod!=0)
    del smooth_dens_k
    #---------------
    print("Ok-2:the graviational potential in k space has been computed.")
    #---------------

    # Compute the Hessian in k space
    Hess_k00 = - (kkx**2)*phi_k
    Hess_k01 = - (kkx*kky)*phi_k
    Hess_k02 = - (kkx*kkz)*phi_k
    Hess_k10 = - (kky*kkx)*phi_k
    Hess_k11 = - (kky**2)*phi_k
    Hess_k12 = - (kky*kkz)*phi_k
    Hess_k20 = - (kkz*kkx)*phi_k
    Hess_k21 = - (kkz*kky)*phi_k
    Hess_k22 = - (kkz**2)*phi_k
    del phi_k
    #---------------------
    print ("Ok-3: the Hessian matrix in k space has been computed.")
    #---------------------

    # Inverse Fourier transform of the Hessian
    Hessian = np.zeros((Nmesh,Nmesh,Nmesh,3,3), dtype=np.float64)
    Hessian[:,:,:,0,0] = np.real( np.fft.ifftn( np.fft.ifftshift(Hess_k00, axes=(-3,-2,-1)), 
                                     axes=(-3,-2,-1), norm="backward" ) / ((L_box/float(Nmesh))**3))
    del Hess_k00
    print("Ok-4a: the 00 component of the Hessian is Ok.")
    Hessian[:,:,:,0,1] = np.real( np.fft.ifftn( np.fft.ifftshift(Hess_k01, axes=(-3,-2,-1)), 
                                     axes=(-3,-2,-1), norm="backward" ) / ((L_box/float(Nmesh))**3))
    del Hess_k01
    print("Ok-4b: the 01 component of the Hessian is Ok.")
    Hessian[:,:,:,0,2] = np.real( np.fft.ifftn( np.fft.ifftshift(Hess_k02, axes=(-3,-2,-1)), 
                                     axes=(-3,-2,-1), norm="backward" ) / ((L_box/float(Nmesh))**3))
    del Hess_k02
    print("Ok-4c: the 02 component of the Hessian is Ok.")
    Hessian[:,:,:,1,0] = np.real( np.fft.ifftn( np.fft.ifftshift(Hess_k10, axes=(-3,-2,-1)), 
                                     axes=(-3,-2,-1), norm="backward" ) / ((L_box/float(Nmesh))**3))
    del Hess_k10
    print("Ok-4d: the 10 component of the Hessian is Ok.")
    Hessian[:,:,:,1,1] = np.real( np.fft.ifftn( np.fft.ifftshift(Hess_k11, axes=(-3,-2,-1)), 
                                     axes=(-3,-2,-1), norm="backward" ) / ((L_box/float(Nmesh))**3))
    del Hess_k11
    print("Ok-4e: the 11 component of the Hessian is Ok.")
    Hessian[:,:,:,1,2] = np.real( np.fft.ifftn( np.fft.ifftshift(Hess_k12, axes=(-3,-2,-1)), 
                                     axes=(-3,-2,-1), norm="backward" ) / ((L_box/float(Nmesh))**3))
    del Hess_k12
    print("Ok-4f: the 12 component of the Hessian is Ok.")
    Hessian[:,:,:,2,0] = np.real( np.fft.ifftn( np.fft.ifftshift(Hess_k20, axes=(-3,-2,-1)), 
                                     axes=(-3,-2,-1), norm="backward" ) / ((L_box/float(Nmesh))**3))
    del Hess_k20
    print("Ok-4g: the 20 component of the Hessian is Ok.")
    Hessian[:,:,:,2,1] = np.real( np.fft.ifftn( np.fft.ifftshift(Hess_k21, axes=(-3,-2,-1)), 
                                     axes=(-3,-2,-1), norm="backward" ) / ((L_box/float(Nmesh))**3))
    del Hess_k21
    print("Ok-4h: the 21 component of the Hessian is Ok.")
    Hessian[:,:,:,2,2] = np.real( np.fft.ifftn( np.fft.ifftshift(Hess_k22, axes=(-3,-2,-1)), 
                                     axes=(-3,-2,-1), norm="backward" ) / ((L_box/float(Nmesh))**3))
    del Hess_k22
    print("Ok-4i: the 22 component of the Hessian is Ok.")

    # Compute the eigenvalues of the Hessian
    eigenvalues = np.zeros((Nmesh,Nmesh,Nmesh, 3), dtype=np.float64)
    eigenvalues = EigenvalsofHessian( Hessian, Nmesh )
    del Hessian
    print("Ok-5: eigenvalues of Hessian are OK.")

    # Classify the density field as knots, filaments, sheets, voids according to eigenvalues
    temp_arr = np.zeros((Nmesh,Nmesh,Nmesh, 3), dtype=np.int64)
    temp_arr[eigenvalues >= lambda_th] = 1
    class_field = np.sum(temp_arr, axis=3)
    return class_field


@nb.jit(nopython=True)
def EigenvalsofHessian( Hessian, Nmesh ):
    eigenvals = np.zeros((Nmesh,Nmesh,Nmesh, 3), dtype=np.float64)
    for i in range(Nmesh):
        for j in range(Nmesh):
            for m in  range(Nmesh):
                eigenvals[i,j,m,:] = np.real( np.linalg.eigvals(Hessian[i,j,m,:,:]) )
    return eigenvals

def Gaussian_filter_k(scale, kx, ky, kz):
    p1 = -0.5*(scale**2)*(kx**2 + ky**2 + kz**2)
    return np.exp(p1)