## Purpose: Bilinear slerp interpolation ##
## Updates from 'slerp1.py': Slerp after realizing that the data input convention is scalar last (permute the indices prior to upsampling), and keep scalar first convention for output. #

import numpy as np
import torch
import math
from mat_sci_torch_quats.symmetries import fcc_syms
from mat_sci_torch_quats.quats import matrix_hamilton_prod, outer_prod, inverse, scalar_first2last, scalar_last2first, slerp_calc, slerp_calc2



# This was actually used for rolling (moving objects in the same dimension). What I need is to permute dimensions
def permute_X(X):
    restore_shape = X.shape
    X_flat = X.flatten()
    X_length = X_flat.shape[0]
    for index in range(int(X_length / 4)):

        k = 4 * index
        temp = X_flat[k + 1]
        X_flat[k + 1] = X_flat[k]
        X_flat[k] = X_flat[k + 3]
        X_flat[k + 3] = X_flat[k + 2]
        X_flat[k + 2] = temp

    X = X_flat.reshape(restore_shape)
    return(X)

# This time, we slerp in the frame of reference of unit cell 2, and then return the slerped result to the sample frame of reference, for subsequent display.
def slerp_ref_symm(q1, q2, t, syms):

    # import pdb; pdb.set_trace()

    syms = syms.to(torch.device('cuda:0'))
    R = matrix_hamilton_prod(q1, inverse(q2)) # transformation from crystal frame 1, to crystal frame 2
    R_syms = outer_prod(R, syms) # symmetry equivalent transformations
    R_syms *= torch.sign(R_syms[...,:1]) # ensure we are on 

    R_syms = R_syms.view(-1, 24, 4)

    theta = torch.arccos(R_syms[...,0])
    min_ind = theta.min(-1)[1]

    R_min = R_syms[torch.arange(len(R_syms)), min_ind]
    R_min = R_min.reshape(q1.shape)

    qs = inverse(matrix_hamilton_prod(inverse(q1), R_min))
    q3 = slerp_calc(q1, qs, t)

    return q3


# spherical interpolation, used for quaternions
def slerp(q1, q2, t):

    # check if Cos(Omega) is negative, and if so, negate one end.
    q = q2*q1.inverse()
    if (q.q[0] < 0):
        # import pdb; pdb.set_trace()
        q2_neg = Quat(-q2.q)
        return(((q2_neg*q1.inverse())**t)*q1)
    else:
        return(((q2*q1.inverse())**t)*q1)
    
    return 

def slerp2(q1, q2, t, syms):

    syms = syms.to(torch.device('cuda:0'))

    A = matrix_hamilton_prod(q1, inverse(q2)) # transformation from crystal frame 1, to crystal frame 2
    A_syms = outer_prod(A, syms) # symmetry equivalent transformations
    A_syms *= torch.sign(A_syms[...,:1]) # ensure we are on 
    a_min_indices = torch.max(A_syms[...,0],-1)[1]
    A_min = A_syms[torch.arange(len(A_syms)), a_min_indices]

    qs = inverse(matrix_hamilton_prod(inverse(q1), A_min))
    q3 = slerp_calc2(q1, qs, t)

    return q3

def quat_upsampling_symm3(X, scale=4):

    X = X.permute(1,2,0)
    device = torch.device('cuda:0')

    interp_rows = (scale-1)*(X.shape[0]-1)
    interp_cols = (scale-1)*(X.shape[1]-1)

    X_scaled = torch.zeros((scale*X.shape[0], scale*X.shape[1], 4), dtype=X.dtype, device=device)
    X_scaled[::scale, ::scale, :] = X

    delta_t = 1 / scale
    t = torch.tensor([delta_t * k for k in range(1, scale)], device=device) # try to make 't' outer product friendly
    q1 = torch.tensor([], device=device); q2 = torch.tensor([], device=device)

    # perform slerp column-wise
    # for j in range(X.shape[1]-1):
    # import pdb; pdb.set_trace() 
    for j in range(X.shape[1]-1):
        q1 = torch.cat([q1, X[:,j,:]])
        q2 = torch.cat([q2, X[:,j+1,:]])

    q_interp_cols = slerp2(q1, q2, t, fcc_syms) # seems like there are no NaN values here.
    q_interp_cols = q_interp_cols.reshape(-1,X.shape[0],3,4)
    import pdb; pdb.set_trace() 
    # dimensions should be the amount of interpolation pairs, rows, interpolations per pair, 4

    for i in range(q_interp_cols.shape[0]): # looping through all interpolations pairs
            indices2 = range(scale*i + 1, scale*i + scale, 1)
            X_scaled[::scale, indices2, :] = q_interp_cols[i,:,:,:]

    ## Check if individual columns of X_scaled are completely filled --> looks like Yes.
    import pdb; pdb.set_trace()
    if (X.shape[1] % 2 == 0):
        X_scaled[::scale, range(X_scaled.shape[1] - scale + 1, X_scaled.shape[1], 1), :] = X_scaled[::scale, X_scaled.shape[1] - scale,:].unsqueeze(1)

    # Now, perform row-based interplation
    q1 = torch.tensor([], device=device); q2 = torch.tensor([], device=device)
    for i in range(X.shape[0]-1):
            q1 = torch.cat([q1, X_scaled[i*scale,:,:]])
            q2 = torch.cat([q2, X_scaled[(i+1)*scale,:,:]])

    q_interp_rows = slerp2(q1, q2, t, fcc_syms)
    q_interp_rows = q_interp_rows.reshape(-1,X_scaled.shape[1],3,4)
    q_interp_rows = torch.movedim(q_interp_rows, 2, 1)

    for i in range(q_interp_rows.shape[0]): # looping through all interpolations pairs
            indices2 = range(scale*i + 1, scale*i+scale, 1)
            X_scaled[indices2, :, :] = q_interp_rows[i,:,:,:]
 
    # Re-permute dimensions order to [channel, width, height], as per PyTorch convention
    import pdb; pdb.set_trace()
    X_scaled = X_scaled.permute(2, 0, 1)

    return X_scaled

# Bi-linear quaternion upsampling #
def quat_upsampling(X, scale=4): # X is low-resolution numpy-array
        
        # import pdb; pdb.set_trace()
        # X = permute_X(X) # convert from quaternion scalar_last convention to scalar_first.

        X = X.permute(1,2,0)
        # X = scalar_last2first(X)

        X_rows = X.shape[0]; X_cols = X.shape[1]
        interp_factor = scale
        X_SR = torch.zeros([scale * X_rows, scale * X_cols, 4], dtype=X.dtype) # Super-resolved numpy-array

        for i in range(scale * X_rows):

            for j in range(scale * X_cols):

                x1 = math.floor(j/scale)
                x2 = math.floor(j/scale) + 1
                y1 = math.floor(i/scale)
                y2 = math.floor(i/scale) + 1

                # Check algorithm didn't exit bounds of original quat matrix
                if (y2 == X.shape[0]):
                    y2 = X.shape[0] - 1
                if (x2 == X.shape[1]):
                    x2 = X.shape[1] - 1

                q1 = X[y2][x1] # quats from LR cell
                q2 = X[y2][x2]
                q3 = X[y1][x1]
                q4 = X[y1][x2]

                t_x = (j % scale) / scale # find mod, and normalize to obtain interpolation parameter, 't'.

                # import pdb; pdb.set_trace()

                q_interp_x1 = slerp_ref_symm(q1, q2, t_x, fcc_syms)
                q_interp_x2 = slerp_ref_symm(q3, q4, t_x, fcc_syms)
                # y-interpolation:
                t_y = (i % scale) / scale

                # import pdb; pdb.set_trace()
                X_SR[i][j] = slerp_ref_symm(q_interp_x1, q_interp_x2, t_y, fcc_syms) # slerp returns Quat object, so get its numpy array.

        # Re-permute dimensions order to [channel, width, height], as per PyTorch convention
        X_SR = X_SR.permute(2, 0, 1)

        return X_SR
