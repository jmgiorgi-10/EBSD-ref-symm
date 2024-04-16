import torch
from mat_sci_torch_quats.quats import hadamard_prod, outer_prod, misorientation, inverse

def normalize(x):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
            # make ||q|| = 1
    y_norm = torch.div(x, x_norm) 

    return y_norm

class RotDistLoss(torch.nn.Module):
    def forward(self,q_pred,q_gt):
        #import pdb; pdb.set_trace()
        q_pred_neg = torch.stack((q_pred,-q_pred),dim=-2)
        if q_gt is not None: q_gt = q_gt[...,None,:]

        euclid_dist = torch.linalg.norm(q_pred_neg-q_gt,dim=-1)
        theta = EuclidToRotApprox()(euclid_dist) # this was keeping the system differentiable
        theta_min = theta.min(-1)[0]
        return theta_min
    
class RotDistRelative2(torch.nn.Module):
    # q_pred: ML output, q_gt: ground-truth (hr)
    def __init__(self, syms):
        super(RotDistRelative, self).__init__() # call the superclass's '__init__',so that nn.Module class's initialization is executed before that of this subclass's init.
        self.syms = syms

class RotDistRelative(torch.nn.Module):

    # q_pred: ML output, q_gt: ground-truth (hr)
    def __init__(self, syms):
        super(RotDistRelative, self).__init__() # call the superclass's '__init__',so that nn.Module class's initialization is executed before that of this subclass's init.
        self.syms = syms

    def forward(self, q_pred, q_gt):

        # normalize the tensor on GPU, which is part of the network backpropagation
        q_pred = normalize(q_pred)
        
        # Place a COPY of q_pred and q_gt in the cpu for intermediate computations
        q_pred_cpu = q_pred.cpu(); q_gt_cpu = q_gt.cpu()

        # All of the below is now happening on the cpu, and is not connected to backpropagation
        A = hadamard_prod(q_pred_cpu, inverse(q_gt_cpu))
        data_shape = A.shape
        A = A.view(len(A), -1, 4)    # flatten for pixel-wise symmetry minimization
        A_sym = outer_prod(A, self.syms)
        # import pdb; pdb.set_trace()       
        zero_tensor = torch.Tensor([1,0,0,0])
        zero_tensor = zero_tensor.reshape(1,1,1,4)
        # flatten for computing thetas, and then reshape to matrix dimensions, before computing the indices.
        # misorientation is producing incorrect dimensions
        thetas = misorientation(A_sym, zero_tensor) # broadcasting
        min_ind = thetas.min(-1)[1]
        min_ind_flat = min_ind.view(-1)

        A_sym_flat = A_sym.view(-1, 24, 4)
        A_min = A_sym_flat[torch.arange(len(A_sym_flat)), min_ind_flat]

        A_min = A_min.reshape(data_shape) # reshape to original data dimensions
        # bring back A_min to the reference frame of the sample (to avoid previous 'noise' issues):
        q_pred_sample_frame = inverse(hadamard_prod(inverse(q_pred), A_min))

        if (q_pred_sample_frame.shape != q_gt.shape):
            import pdb; pdb.set_trace()
            print("mismatching dimensions, what's going on here?")

        # right now q_pred_sample_frame is not linked to q_pred, for network backpropagation (chain rule)
        euclid_dist = torch.linalg.norm(q_pred_sample_frame - q_gt, dim=-1)
        # del q_pred_sample_frame
        theta = EuclidToRotApprox()(euclid_dist) # applies the same rotational distance function, but using the reference symmetry-reduced q_pred_sample_frame
        # theta_min = theta.min(-1)[0]
        # import pdb; pdb.set_trace()
        # do we need to fz reduce wrt sample?
        theta = theta.to('cuda:0')
        theta.requires_grad_() # re-enable gradients, since we had pushed q_pred and q_gt to the cpu
        return theta

def euclid2rot(x):
    return torch.arccos(1 - 0.5*x**2)

class EuclidToRotApprox:
    def __init__(self,beta=0.1,eps=0.01):
        self.t = 2 - beta
        self.eps = eps
        t = torch.Tensor([self.t])
        t.requires_grad = True
        y = euclid2rot(t)
        import pdb; pdb.set_trace()
        y.backward()
        self.m = float(t.grad)
        self.b = float(y) - self.m*self.t

    # __call__ : objects that behave like functions but also have additional state or behavior associated with them
    # input is the euclidean distance between the nn output, and the hr ground truth.
    def __call__(self, x):
        x_clip = torch.clamp(x, self.eps, self.t)
        #mask = (x < self.beta).float()
        y_abs = torch.abs(x)
        y_lin = self.m*x + self.b # linear approximation computed for d_euclid > 1.9 (since otherwise slope approaches infinity)
        y_rot = euclid2rot(x_clip)
        # piece-wise function for loss, to avoid infinite gradients (see paper for details).
        y_out = y_abs * (x <= self.eps).float() + \
                y_rot * torch.logical_and(x > self.eps,x < self.t).float() + \
                y_lin * (x >= self.t).float()

        return y_out



        

