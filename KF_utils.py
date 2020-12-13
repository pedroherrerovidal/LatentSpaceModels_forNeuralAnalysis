# created by Pedro Herrero-Vidal

import numpy as np

def KL_filter(A, Q, C, R, mu0, E0, X, n_dim_state):
    """
    Method that performs Kalman filtering
    @param X: a numpy 2D array whose dimension is [n_example, self.n_dim_obs]
    @output: filtered_state_means: a numpy 2D array whose dimension is [n_example, self.n_dim_state]
    @output: filtered_state_covariances: a numpy 3D array whose dimension is [n_example, self.n_dim_state, self.n_dim_state]
    """
    # validate inputs
    n_example, observed_dim = X.shape
#     assert observed_dim==self.n_dim_obs

    # create holders for outputs
    filtered_state_means = np.zeros( [n_example, n_dim_state] )
    filtered_state_covariances = np.zeros( [n_example, n_dim_state, n_dim_state] )
    pred_state_covariances = np.zeros( (n_example, n_dim_state, n_dim_state) )

    # My KF filter

    temp_state_mean_t = mu0.copy()
    temp_state_cov_t = E0.copy()

    filtered_state_means[0, :] = mu0.copy()
    filtered_state_covariances[0, :,:] = E0.copy()

    for t in range(1, n_example):
        # estimate next state
        pred_state_mean_t = np.dot(A, temp_state_mean_t)
        pred_state_cov_t = np.dot(np.dot(A, temp_state_cov_t), A.T) + Q

        pred_state_covariances[t, :] = pred_state_cov_t
        # predict next obs 
#             temp_obs_mean_t = C * temp_state_mean_t
#             temp_obs_cov_t = C @ temp_state_cov_t @ C.T + R

        # correct latent prediction with data
        K = np.dot(np.dot(pred_state_cov_t, C.T), np.linalg.pinv(np.dot(np.dot(C,pred_state_cov_t),C.T) + R))

        temp_state_mean_t = pred_state_mean_t + np.dot(K,(X[t, :]- np.dot(C,pred_state_mean_t)))
        temp_state_cov_t = np.dot((np.eye(pred_state_cov_t.shape[0]) - np.dot(K, C) ), pred_state_cov_t)

        filtered_state_means[t, :] = temp_state_mean_t.copy()
        filtered_state_covariances[t, :] = temp_state_cov_t.copy()
    return filtered_state_means, filtered_state_covariances, pred_state_covariances

def KF_smooth(A, Q, C, R, mu0, E0, X, n_dim_state):
    # depends on KL_filter
    """
    Method that performs the Kalman Smoothing
    @param X: a numpy 2D array whose dimension is [n_example, self.n_dim_obs]
    @output: smoothed_state_means: a numpy 2D array whose dimension is [n_example, self.n_dim_state]
    @output: smoothed_state_covariances: a numpy 3D array whose dimension is 
             [n_example, self.n_dim_state, self.n_dim_state]
    """

    # validate inputs
    n_example, observed_dim = X.shape
#     assert observed_dim==self.n_dim_obs

    # run the forward path
    mu_list, v_list, v_pred = KL_filter(A, Q, C, R, mu0, E0, X, n_dim_state)

    # create holders for outputs
    smoothed_state_means = np.zeros( (n_example, n_dim_state) )
    smoothed_state_covariances = np.zeros( (n_example, n_dim_state, n_dim_state) )

    ## My KF smoothing
    smoothed_state_means[n_example-1,:] = np.copy(mu_list[n_example-1,:])
    smoothed_state_covariances[n_example-1,:,:] = np.copy(v_list[n_example-1,:,:])
    Js = []

    for t in np.flip(np.arange(n_example-1)): 
        pt = np.copy(v_pred[t+1, :,:])
        J = np.dot(v_list[t,:,:], np.dot(A.T, np.linalg.pinv(pt)))
        Js.append(J)

        smoothed_state_means[t,:] = mu_list[t,:]+ np.dot(J,(smoothed_state_means[t+1,:]- np.dot(A, mu_list[t,:])))

        smoothed_state_covariances[t,:,:] = v_list[t,:,:]+np.dot(J,np.dot((smoothed_state_covariances[t+1,:,:] - pt),J.T))

    # append last J
    v_predT =  np.copy(np.dot(np.dot(A, v_list[-1, :, :]), A.T) + Q)
    Js = list(reversed(Js))
    Js.append( np.dot(v_list[t,:,:], np.dot(A.T, np.linalg.pinv(v_predT)) )  )
    return smoothed_state_means, smoothed_state_covariances, Js

def KF_em_MT(A, Q, C, R, mu0, cov0, X, n_dim_state, max_iter=10, LL_flag= True):
    """
    Method that perform the EM algorithm to update the model parameters
    Note that in this exercise we ignore offsets
    @param X: a numpy 2D array whose dimension is [n_example, self.n_dim_obs]
    @param max_iter: an integer indicating how many iterations to run
    """
    # validate inputs have right dimensions
    trials, n_example, observed_dim = X.shape

    # keep track of log posterior (use function calculate_posterior below)
    avg_em_log_posterior = np.zeros(max_iter,)*np.nan
    
    mu0_holder = np.zeros((trials, n_dim_state))*np.nan
    cov0_holder = np.zeros((trials, n_dim_state, n_dim_state))*np.nan
    A_holder = np.zeros((trials, n_dim_state, n_dim_state))*np.nan
    Q_holder = np.zeros((trials, n_dim_state, n_dim_state))*np.nan
    C_holder = np.zeros((trials, observed_dim, n_dim_state))*np.nan
    R_holder = np.zeros((trials, observed_dim, observed_dim))*np.nan

    for step in range(max_iter):
        for t in range(trials):
            x = X[t, :, :]
            # E-step
            filtered_state_covariances, smoothed_state_means, smoothed_state_covariances, Js, Ezn, Eznznminus, Eznzn = KF_em_E_step(A, Q, C, R, mu0, cov0, x, n_dim_state, n_example)
            # M-step
            mu0_new, cov0_new, A_new, Q_new, C_new, R_new = KF_em_M_step(x, smoothed_state_means, smoothed_state_covariances, Ezn, Eznznminus, Eznzn, observed_dim, n_example)
            # update all variables

            mu0_holder[t, :] = mu0_new
            cov0_holder[t, :, :] = cov0_new
            A_holder[t, :, :] = A_new
            Q_holder[t, :, :] = Q_new
            C_holder[t, :, :] = C_new
            R_holder[t, :, :] = R_new
            
        mu0 = np.mean(mu0_holder, 0)
        cov0 = np.mean(cov0_holder, 0)
        A = np.mean(A_holder, 0)
        Q = np.mean(Q_holder, 0)
        C = np.mean(C_holder, 0)
        R = np.mean(R_holder, 0)
        # log likelihood
        
        # plot C - C.T 
        # better way to estimate likelihood (another loop?)
        if LL_flag == True:
            foo_ll = 0
            for t in range(trials):
                x = X[t, :, :]
                smoothed_state_means, _, _ = KF_smooth(A, Q, C, R, mu0, cov0, x, n_dim_state)  
                foo_ll += np.mean(KF_calculate_posterior(A,Q,C,R,mu0,cov0,
                                                         x,n_dim_state, smoothed_state_means))
            avg_em_log_posterior[step] = foo_ll/trials
    return mu0, cov0, A, Q, C, R, avg_em_log_posterior
    
    
def KF_em_E_step(A, Q, C, R, mu0, cov0, X, n_dim_state, n_example):
    # E-step
    _, filtered_state_covariances, _ = KL_filter(A, Q, C, R, mu0, cov0, X, n_dim_state)
    smoothed_state_means, smoothed_state_covariances, Js = KF_smooth(A, Q, C, R, mu0, cov0, X, n_dim_state)  

    Ezn = []
    Eznznminus = []
    Eznzn = []
    for n in range(n_example):
        Ezn.append(smoothed_state_means[n,:])
        Eznzn.append(smoothed_state_covariances[n, :, :] + 
                     np.outer(smoothed_state_means[n,:].T,smoothed_state_means[n,:]))

        # Eznznminus is n-1 dimensional
        if n != 0:
            Eznznminus.append( np.dot(Js[n-1],
#                                        (filtered_state_covariances[n-1, :, :]@ 
#                                        self.transition_matrices.T@
#                                        np.linalg.pinv(v_pred[n, :, :]))@ 
                               smoothed_state_covariances[n, :, :]) +
                               np.outer(smoothed_state_means[n,:],smoothed_state_means[n-1,:].T) )

    Ezn = np.array(Ezn)
    Eznznminus = np.array(Eznznminus)
    Eznzn = np.array(Eznzn)
    
    return filtered_state_covariances, smoothed_state_means, smoothed_state_covariances, Js, Ezn, Eznznminus, Eznzn 
    
def KF_em_M_step(X, smoothed_state_means, smoothed_state_covariances, Ezn, Eznznminus, Eznzn, observed_dim, n_example):
    # updata initial parameters
    mu0_new = np.copy(smoothed_state_means[0, :])
    cov0_new = np.copy(smoothed_state_covariances[0, :, :]) #Eznzn[0,:,:] - np.outer(Ezn[0, :] , Ezn[0, :].T)

    # update latent parameters
    Eznznminus_F = np.sum(Eznznminus, axis=0)
    Eznzn_F = np.sum(Eznzn, axis=0)
    Eznzn_1 = np.sum(Eznzn[1:,:,:], axis=0)
    Eznzn_n = np.sum(Eznzn[:-1,:,:], axis=0)
    A_new = np.dot(Eznznminus_F, np.linalg.pinv(Eznzn_n))

    A_Eznznminus = np.dot(Eznznminus_F , A_new.T)
    Q_new = (Eznzn_1- 
             A_Eznznminus -
             A_Eznznminus.T+
             np.dot(np.dot(A_new, Eznzn_n), 
             A_new.T) ) / (n_example-1)

    # update observation parameters            
    C_new = np.dot(np.dot(X.T,Ezn), np.linalg.pinv(Eznzn_F.T)) # np.dot(np.dot(X.T,Ezn), np.linalg.pinv(Eznzn_F.T))

    R_new = np.zeros((observed_dim, observed_dim))
    for nn in range(n_example):
        err = (X[nn] - np.dot(C_new, smoothed_state_means[nn]))
        R_new += (np.outer(err, err) + np.dot(C_new, np.dot(
                                        smoothed_state_covariances[nn],
                                        C_new.T) ))

    R_new /= n_example
    
    return mu0_new, cov0_new, A_new, Q_new, C_new, R_new
    
    
def KF_calculate_posterior(A, Q, C, R, mu0, E0, X, n_dim_state, state_mean, v_n=None, nPSD = True):
    """
    Method that calculates the log posterior
    @param X: a numpy 2D array whose dimension is [n_example, self.n_dim_obs]
    @param state_mean: a numpy 2D array whose dimension is [n_example, self.n_dim_state]
    @output: a numpy 1D array whose dimension is [n_example]
    """

    if v_n is None:
        _, v_n, _ = KL_filter(A, Q, C, R, mu0, E0, X, n_dim_state)
    llh = []
    for i in range(1,len(state_mean)):
        normal_mean = np.dot(C, np.dot(A, state_mean[i-1]))
        p_n = A.dot(v_n[i].dot(A))+Q
        #normal_cov = np.matmul(self.observation_matrices, np.matmul(self.p_n_list[i], self.observation_matrices.T)) + self.observation_covariance
        normal_cov = np.matmul(C, np.matmul(p_n, C.T)) + R
        if nPSD == True:
            try:
                np.linalg.cholesky(normal_cov)
            except:
                normal_cov = getNearPSD(normal_cov)
        try:
            pdf_val = multivariate_normal.pdf(X[i], normal_mean, normal_cov)
        except:
            pdf_val = 1e-10
        # replace 0 to prevent numerical underflow
        if pdf_val < 1e-10:
            pdf_val = 1e-10
        llh.append(np.log(pdf_val))  
    return np.array(llh)


