# created by Pedro Herrero-Vidal

import numpy as np
import random

def FA_EM(X_cov, xDim, zDim, eps, T, penalty=3, TL_flag=0):
    A = abs(np.random.uniform(0,1, (xDim,zDim))/np.sqrt(zDim))  # initiate A
    R = np.diag(np.diag(X_cov))                                 # initiate R           

    # eps = 1e-7                                                # define stopping value
    LL_prev = 0                                                 # initiate LL reference
    LL_step = eps+1                                             # non-stop value for while loop
    LL_cache = []                                               # array with LL values
    counter = 0 
    while LL_step > eps:                                        # EM FA
        # E-step
        delta = np.linalg.pinv(A @ A.T + R)
        beta = A.T @ delta

        # M-step
        A = (X_cov @ beta.T @ 
             np.linalg.pinv(np.identity(zDim) - beta @ A + beta @ X_cov @ beta.T))
        R = np.diag(np.diag(X_cov - A @ beta @ X_cov))

        # avg. LL
        if np.linalg.slogdet(delta)[0] > 0:
            LL = -T/2*np.trace(delta @ X_cov) + T/2*np.linalg.slogdet(delta)[1] - T*xDim/2*np.log(2*np.pi) 
                                            # N*sum(log(diag(chol(MM))))
        elif np.linalg.slogdet(delta)[0] < 0:
#             print(str(zDim)+'Negative determinant')
            LL = -T/2*np.trace(delta @ X_cov) + T/2*np.linalg.slogdet(delta)[1] - T*xDim/2*np.log(2*np.pi) 
                                                                
        LL_step = abs((LL-LL_prev)/abs(LL))
        LL_prev = LL
        LL_cache.append(LL)
        counter += 1
        if counter > 1e4:
            break

    LL_corrected = LL #- zDim ** penalty;

    return A, R, LL_cache, LL_corrected

def FA_project(A, R, X, X_mu):
    return A.T @ np.linalg.pinv(A @ A.T + R) @ (X.T - np.tile(X_mu, (X.shape[0], 1)).T)



