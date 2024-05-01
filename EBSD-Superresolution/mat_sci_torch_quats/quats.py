from math import pi

import torch
import numpy as np

# Defines mapping from quat vector to matrix. Though there are many
# possible matrix representations, this one is selected since the
# first row, X[...,0], is the vector form.
# https://en.wikipedia.org/wiki/Quaternion#Matrix_representations
q1 = np.diag([1,1,1,1])
qj = np.roll(np.diag([-1,1,1,-1]),-2,axis=1)
qk = np.diag([-1,-1,1,1])[:,::-1]
qi = np.matmul(qj,qk)
Q_arr = torch.Tensor([q1,qi,qj,qk])
Q_arr_flat = Q_arr.reshape((4,16))


# Checks if 2 arrays can be broadcast together
def _broadcastable(s1,s2):
        if len(s1) != len(s2): return False
        else: return all((i==j) or (i==1) or (j==1) for i,j in zip(s1,s2))

# Converts an array of quats as vectors to matrices. Generally
# used to facilitate quat multiplication.
def vec2mat(X):
        assert X.shape[-1] == 4, 'Last dimension must be of size 4'
        new_shape = X.shape[:-1] + (4,4)
        dtype = X.dtype
        Q = Q_arr_flat.type(X.dtype).to(X.device)
        #print('Q', Q.dtype)
        return torch.matmul(X,Q).reshape(new_shape)


def normalize(x):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
            # make ||q|| = 1
    y_norm = torch.div(x, x_norm) 

    return y_norm


# Hamilton product
def hadamard_prod(q1,q2):
        assert _broadcastable(q1.shape,q2.shape), 'Inputs of shapes ' \
                        f'{q1.shape}, {q2.shape} could not be broadcast together'
        # import pdb; pdb.set_trace()
        # q1.to('cuda:0')
        q2 = q2.to('cuda:0')
        X1 = vec2mat(q1)
        X_out = (X1 * q2[...,None,:]).sum(-1)
        return X_out



# Performs outer product on ndarrays of quats
# Ex if X1.shape = (s1,s2,4) and X2.shape = (s3,s4,s5,4),
# output will be of size (s1,s2,s3,s4,s5,4)
def outer_prod(q1,q2):
        # q1 = q1.cuda(); q2 = q2.cuda()
        X1 = vec2mat(q1)
        X2 = torch.movedim(q2,-1,0)
        X1_flat = X1.reshape((-1,4))
        X2_flat = X2.reshape((4,-1))
        X_out = torch.matmul(X1_flat,X2_flat)
        X_out = X_out.reshape(q1.shape + q2.shape[:-1])
        X_out = torch.movedim(X_out,len(q1.shape)-1,-1)
        return X_out


# Utilities to create random vectors on the L2 sphere. First produces
# random samples from a rotationally invariantt distibution (i.e. Gaussian)
# and then normalizes onto the unit sphere

# Produces random array of the same size as shape.
def rand_arr(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        X = torch.randn(shape).type(dtype)
        X /= torch.norm(X,dim=-1,keepdim=True)
        return X

# Produces array of 3D points on the unit sphere.
def rand_points(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        return rand_arr(shape + (3,), dtype)

# Produces random unit quaternions.
def rand_quats(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        return rand_arr(shape+(4,), dtype)


# arccos, expanded from range [-1,1] to all real numbers
# values outside of [-1,1] and replaced with a line of slope pi/2, such that
# the function is continuous
def safe_arccos(x):
    mask = (torch.abs(x) < 1).float()
    x_clip = torch.clamp(x,min=-1,max=1)
    output_arccos = torch.arccos(x_clip)
    output_linear = (1 - x)*pi/2
    output = mask*output_arccos + (1-mask)*output_linear
    return output

# for loss transformation, q1 is the nn output, and q2 is the respective hr pixel
def transformation_matrix_tensor(q1, q2, syms):
        
        # with torch.no_grad():

        # import pdb; pdb.set_trace()

        ## removed torch no grad for the conversion, see if misorientation starts to go down now, w/o errors.

        # with torch.no_grad(): --> using this incorrectly stops the gradients from flowing; even returning R_min fails.

        # q1 = normalize(q1)

        syms = syms.to(torch.device('cuda:0'))

        inv = inverse_matrix_generate(q1) # only uses q1 to obtain tensor shape.

        q1_inv = q1 * inv
        q2_inv = q2 * inv
        R = hadamard_prod(q1, q2_inv)
        R_syms = outer_prod(R, syms)
        R_syms = R_syms.view(-1, 24, 4)

        theta = torch.arccos(R_syms[...,0])
        min_ind = theta.min(-1)[1] # still differentiable --> gradient flows through only for min.
        min_ind_flat = min_ind.view(-1)

        R_min = R_syms[torch.arange(len(R_syms)), min_ind_flat]
        R_min = R_min.reshape(q1.shape)

        # import pdb; pdb.set_trace()
        q_loss_inv = hadamard_prod(q1_inv, R_min) ## Perhaps the error is here, can't backpropagate current inverse function applied to q_nn.
        q_loss = q_loss_inv * inv

        # R_min works, q_loss_inv works
        # return q_loss
        return R_min

# generate an inverse matrix for the size of input matrix
def inverse_matrix_generate(q):

        # q = q.to(torch.device('cuda:0'))

        # used for element-wise product with [1,-1,-1,-1]

        # import pdb; pdb.set_trace()

        data_shape = q.shape

        magnitudes = torch.norm(q,2,-1)

        inverse_matrix = torch.ones(data_shape).to(torch.device('cuda:0'))
        inverse_matrix[...,1:4] = -1 * inverse_matrix[...,1:4]

        inverse_matrix = 1/magnitudes.unsqueeze(-1) * inverse_matrix

        return inverse_matrix

## ! issue was likely here, make sure this is performed as differentiable matrix operation
def inverse(q):
        # import pdb; pdb.set_trace()
        magnitudes = torch.norm(q,2,-1)
        q_inv = q
        q_inv[...,1:4] = -1 * q[...,1:4]
        q_inv = 1/magnitudes.unsqueeze(-1) * q_inv

        return q_inv

def quat_dist(q1,q2=None):
        """
        Computes distance between two quats. If q1 and q2 are on the unit sphere,
        this will return the arc length along the sphere. For points within the
        sphere, it reduces to a function of MSE.
        """
        #import pdb; pdb.set_trace()
        if q2 is None: mse = (q1[...,0]-1)**2 + (q1[...,1:]**2).sum(-1)
        else: mse = ((q1-q2)**2).sum(-1)
        
        corr = 1 - (1/2)*mse
        corr_clamp = torch.clamp(corr,-1,1)
        return safe_arccos(corr)

# just calculate the theta of a quaternion
def misorientation(q1, q2=None):

        # import pdb; pdb.set_trace()

        if (q2 == None):
                q2 = torch.Tensor([1,0,0,0])
        q_dot = torch.tensordot(q1, q2, dims=[[-1], [-1]]).squeeze() # dot product across last dimension for multi-dimensional matrices
        # q_dot = q1 @ q2
        theta = 2*torch.arccos(torch.clamp(q_dot,-1,1))
        return theta

def rot_dist(q1,q2=None):
        """ Get dist between two rotations, with q <-> -q symmetry """
        #import pdb; pdb.set_trace()
        q1_w_neg = torch.stack((q1,-q1),dim=-2)
        if q2 is not None: q2 = q2[...,None,:]
        dists = quat_dist(q1_w_neg,q2)
        dist_min = dists.min(-1)[0]
        return dist_min

# Calculates minimum symmetry misorientation of all pixels
def rot_dist_relative(q1, q2, syms):

        # import pdb; pdb.set_trace()
        
        # q1 = normalize(q1) # normalize q1

        # this is the validation stage, so q1 = q_nn, generated by current weights established during training.

        R = hadamard_prod(q1, inverse(q2))
        R_syms = outer_prod(R, syms)
        R_syms = R_syms.view(-1, 24, 4)

        theta = torch.arccos(R_syms[...,0])
        min_ind = theta.min(-1)[1]

        # theta_min = theta[torch.arange(len(theta)), min_ind]
        # import pdb; pdb.set_trace()
        R_min = R_syms[torch.arange(len(R_syms)), min_ind]
        R_min = R_min.reshape(q1.shape)

        zero_broadcast_tensor = torch.Tensor([1,0,0,0])
        zero_broadcast_tensor = zero_broadcast_tensor.reshape(1,1,1,4).to(torch.device('cuda:0'))


        q_loss = inverse(hadamard_prod(inverse(q1), R_min))
        
        # added fz reduction, with the intention of checking if this reduces the euclidean distance calc.
        q_loss_fz = fz_reduce(q_loss, syms)

        # COULD THE ERROR BE SOMEWHERE HERE, BELOW?:
        # euclid_dist = torch.linalg.norm(q_loss - q2,2,dim=-1)
        # euclid_dist = torch.linalg.norm(q_loss_fz - q2, 2, dim=-1)

        euclid_dist = torch.linalg.norm(R_min - zero_broadcast_tensor, 2, dim=-1)

        # import pdb; pdb.set_trace() ## WHY DID I PLACE A 0 INDEX?
        dist = 4*torch.arcsin(euclid_dist / 2)
   
        return dist
        
def fz_reduce(q,syms):
        shape = q.shape
        q = q.reshape((-1,4))
        syms = syms.cuda()
        q_w_syms = outer_prod(q,syms)
        dists = rot_dist(q_w_syms)
        inds = dists.min(-1)[1]
        q_fz = q_w_syms[torch.arange(len(q_w_syms)),inds]
        q_fz *= torch.sign(q_fz[...,:1])
        q_fz = q_fz.reshape(shape)
        return q_fz


def scalar_first2last(X):
        return torch.roll(X,-1,-1)

def scalar_last2first(X):
        return torch.roll(X,1,-1)

def conj(q):
        q_out = q.clone()
        q_out[...,1:] *= -1
        return q_out

def rotate(q,points,element_wise=False):
        points = torch.as_tensor(points)
        P = torch.zeros(points.shape[:-1] + (4,),dtype=q.dtype,device=q.device)
        assert points.shape[-1] == 3, 'Last dimension must be of size 3'
        P[...,1:] = points
        if element_wise:
                X_int = hadamard_prod(q,P)
                X_out = hadamard_prod(X_int,conj(q))
        else:
                X_int = outer_prod(q,P)
                inds = (slice(None),)*(len(q.shape)-1) + \
                                (None,)*(len(P.shape)) + (slice(None),)
                X_out = (vec2mat(X_int) * conj(q)[inds]).sum(-1)
        return X_out[...,1:]



# A simple script to test the quats class for numpy and torch
if __name__ == '__main__':

        np.random.seed(1)
        N = 700
        M = 1000
        K = 13

        def test(dtype,device):

                q1 = rand_quats(M,dtype).to(device)
                q2 = rand_quats(N,dtype).to(device)
                q3 = rand_quats(M,dtype).to(device)
                p1 = rand_points(K,dtype).to(device)

                p2 = rotate(q2,rotate(q1,p1))
                p3 = rotate(outer_prod(q2,q1),p1)
                p4 = rotate(conj(q1[:,None]),rotate(q1,p1),element_wise=True)

                print('Composition of rotation error:')
                err = abs(p2-p3).sum()/len(p2.reshape(-1))
                print('\t',err)

                print('Rotate then apply inverse rotation error:')
                err = abs(p4-p1).sum()/len(p1.reshape(-1))
                print('\t',err,'\n')

        
        print('CPU Float 32')
        test(torch.cuda.FloatTensor,'cpu')

        print('CPU Float64')
        test(torch.cuda.DoubleTensor,'cpu')     

        if torch.cuda.is_available():

                print('CUDA Float 32')
                test(torch.cuda.FloatTensor,'cuda')

                print('CUDA Float64')
                test(torch.cuda.DoubleTensor,'cuda') 

        else:
                print('No CUDA')

