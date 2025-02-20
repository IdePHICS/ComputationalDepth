
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def He_2(x):
    return (x**2 - 1)/2
def He_3(x):
    return (x**3 - 3 * x)/6
def He_4(x):
    return (x**4 - 6*x**2 + 3)/24
def relu_deriv(x):
    return (x > 0).to(dtype)
def relushift(x):
    return torch.clamp(x - 0.5, min=0).to(dtype)
def quadratic_kernel(x1, x2):
    return (torch.mm(x1, x2.T) ** 2) + torch.mm(x1, x2.T) + 1
def relu(x):
    return torch.max(x, torch.tensor(0., dtype=dtype))

class ThreeLayerNN:
    def __init__(self, p1, p2, d, lr, reg1, reg2, lambda_ridge, eps_init, FlagReinitialize=False, fraction_batch=0.1):
        # Layer dimensions
        self.d = d
        self.p1 = p1
        self.p2 = p2
        self.lr = lr
        self.reg1 = reg1
        self.reg2 = reg2
        self.lambda_ridge = lambda_ridge
        self.eps_init = eps_init
        # Weights initialization
        self.W1 = torch.randn(p1, d, dtype=dtype) / np.sqrt(d)  
        self.W2 = eps_init*torch.randn(p2, p1, dtype=dtype) / np.sqrt(p1)
        self.W3 = eps_init*torch.randn(1, p2, dtype=dtype) / np.sqrt(p2)
        # Fixed random biases initialization for first two layers with matching norms
        self.b1 = (2*torch.rand(p1, 1, dtype=dtype)-1)
        self.b2 = eps_init*(2*torch.rand(p2, 1, dtype=dtype)-1)
        # Move to device
        self.W1 = self.W1.to(device)
        self.W2 = self.W2.to(device)
        self.W3 = self.W3.to(device)
        self.b1 = self.b1.to(device)
        self.b2 = self.b2.to(device)
        # Initialize lists to store weight trajectories
        self.W1_trajectory = [self.W1.clone()]
        self.W2_trajectory = [self.W2.clone()]
        # reinitialization flag to reinitialize second layer before training the third
        self.FlagReinitialize = FlagReinitialize
        # Validation/training loss trajectory when training layer 1
        self.validation_loss_trajectory_layer1 = []
        self.training_loss_trajectory_layer1 = []
        # Validation loss trajectory when training layer 2
        self.validation_loss_trajectory_layer2 = []
        self.training_loss_trajectory_layer2 = []
        # fraction batch 
        self.fraction_batch = fraction_batch
    def forward(self, x):
        # First layer (input to hidden layer 1)
        h1 = relu(torch.mm(self.W1, x.t()) + self.b1).t()

        # Second layer (hidden layer 1 to hidden layer 2)
        h2 = relu(torch.mm(self.W2, h1.t()) + self.b2).t()

        # Third layer (hidden layer 2 to output)
        output = torch.mm(self.W3, h2.t()).t()

        return output, h1, h2
    
    def train_layer1(self, x_batch, y_batch,  x_valid, y_valid, max_iter):
        snapshot_times = [int(max_iter - 1), int(max_iter/2), int(max_iter/5), int(max_iter/10)]
        snapshot_errors = []; 
        fraction_batch = self.fraction_batch
        mini_batch_size = int(x_batch.shape[0] * fraction_batch)
        num_batches = int(max_iter / fraction_batch)
        for i in range(num_batches):
            indices = torch.randperm(x_batch.shape[0])[:mini_batch_size]
            x_mini_batch = x_batch[indices]
            y_mini_batch = y_batch[indices]
            output, h1, h2 = self.forward(x_mini_batch)
            error = output - y_mini_batch
            grad_output = error.to(dtype)
            grad_h2 = torch.mm(grad_output, self.W3) * relu_deriv(h2); grad_h2 = grad_h2.to(dtype)
            grad_h1 = torch.mm(grad_h2, self.W2) * relu_deriv(h1); grad_h1 = grad_h1.to(dtype)
            grad_W1 = torch.mm(grad_h1.t(), x_mini_batch) / x_mini_batch.shape[0]; grad_W1 = grad_W1.to(dtype)
            # Update W1
            self.W1 -= self.lr*(np.sqrt(self.p1)*grad_W1*1/(self.eps_init**2) + self.reg1*self.W1).to(dtype)
            # Store updated W1
            self.W1_trajectory.append(self.W1.clone())
            if i in snapshot_times:
                snapshot_errors.append(torch.mean(error**2).item())
            # validation loss
            if i % 10 == 0:
                output_valid, _, _ = self.forward(x_valid)
                validation_loss = torch.mean((output_valid - y_valid) ** 2).item()
                self.validation_loss_trajectory_layer1.append(validation_loss)
                self.training_loss_trajectory_layer1.append(torch.mean(error**2).item())
            # Explicitly delete tensors and cache
            del output, h1, h2, error, grad_W1, grad_h2, grad_h1, grad_output
            torch.cuda.empty_cache()
            

    def train_layer2(self, x_batch, y_batch, x_valid, y_valid, max_iter):
        snapshot_times = [int(max_iter - 1), int(max_iter/2), int(max_iter/5), int(max_iter/10)]
        snapshot_errors = []
        FlagReinitialize = self.FlagReinitialize
        if FlagReinitialize:
            p1 = self.p1 ; p2 = self.p2; eps_init = self.eps_init
            self.W2 = eps_init*torch.randn(p2, p1, dtype=dtype) / np.sqrt(p1)
            self.W2 = self.W2.to(device)
            self.W2_trajectory = [self.W2.clone()]
        fraction_batch = self.fraction_batch
        mini_batch_size = int(x_batch.shape[0] * fraction_batch)
        num_batches = int(max_iter / fraction_batch)
        for i in range(num_batches):
            indices = torch.randperm(x_batch.shape[0])[:mini_batch_size]
            x_mini_batch = x_batch[indices]
            y_mini_batch = y_batch[indices]
            output, h1, h2 = self.forward(x_mini_batch)
            error = output - y_mini_batch
            # Compute the gradient for W2 manually
            grad_output = error; grad_output = grad_output.to(dtype)
            grad_h2 = torch.mm(grad_output, self.W3) * relu_deriv(h2); grad_h2 = grad_h2.to(dtype)
            grad_W2 = torch.mm(grad_h2.t(), h1) / x_batch.shape[0]; grad_W2 = grad_W2.to(dtype)
            self.W2 -= 4*self.lr*(np.sqrt(self.p2)*grad_W2/(self.eps_init*self.p1) + self.reg2*1/self.d**1.5*self.W2).to(dtype)
            # Store updated W2
            self.W2_trajectory.append(self.W2.clone())
            if i in snapshot_times:
                snapshot_errors.append(torch.mean(error**2).item())
            # validation loss
            if i % 50 == 0:
                output_valid, _, _ = self.forward(x_valid)
                validation_loss = torch.mean((output_valid - y_valid) ** 2).item()
                self.validation_loss_trajectory_layer2.append(validation_loss)
                self.training_loss_trajectory_layer2.append(torch.mean(error**2).item())
            # Explicitly delete tensors and cache
            del output, h1, h2, error, grad_W2, grad_output, grad_h2
            torch.cuda.empty_cache()
    def ridge_regression_update_W3(self, x_batch, y_batch):
        nbatch = x_batch.shape[0]
        _, _, H2 = self.forward(x_batch)  # Compute H2
        H2_t_H2_size = H2.shape[1] * H2.shape[1] * 4  # Estimate memory for H2_t_H2
        FlagReinitialize = self.FlagReinitialize
        if FlagReinitialize:
            p1 = self.p1 ; p2 = self.p2; eps_init = self.eps_init
            self.W3 = eps_init*torch.randn(1, p2, dtype=dtype) / np.sqrt(p2)
            self.W3 = self.W3.to(device)
        # Select device based on estimated memory
        regularization_matrix = torch.eye(H2.shape[1], dtype=torch.float32, device=device)
        H2_t_H2 = torch.zeros((H2.shape[1], H2.shape[1]), dtype=torch.float32, device=device)
        batch_size = min(1024, nbatch)  # Define a manageable batch size based on actual nbatch
        for i in range(0, nbatch, batch_size):
            H2_batch = H2[i : i + batch_size]
            H2_t_H2 += torch.mm(H2_batch.t(), H2_batch).to(device)
        H2_t_H2 += self.lambda_ridge * nbatch * regularization_matrix

        y_batch_sol = y_batch.to(device).to(torch.float32)
        H2_sol = H2.to(device).to(torch.float32)
        try:
            solution = torch.linalg.solve(H2_t_H2, torch.mm(H2_sol.t(), y_batch_sol))
        except RuntimeError as e:
            if 'singular' in str(e):
                print("Matrix is singular, returning zero estimator.", flush=True)
                solution = torch.zeros_like(self.W3, device=device).t()
            else:
                raise e
        self.W3 = solution.t().to(device).to(dtype)
        del H2_t_H2, solution, y_batch_sol, H2_sol, regularization_matrix, H2 
        torch.cuda.empty_cache()
    
    def joint_training(self, x_batch, y_batch, x_valid, y_valid, max_iter):
        snapshot_times = [int(max_iter - 1), int(max_iter/2), int(max_iter/5), int(max_iter/10)]
        snapshot_errors = []
        fraction_batch = self.fraction_batch
        mini_batch_size = int(x_batch.shape[0] * fraction_batch)
        num_batches = int(max_iter / fraction_batch)

        for i in range(num_batches):
            indices = torch.randperm(x_batch.shape[0])[:mini_batch_size]
            x_mini_batch = x_batch[indices]
            y_mini_batch = y_batch[indices]
            output, h1, h2 = self.forward(x_mini_batch)
            error = output - y_mini_batch
            # Compute the gradient for W2 manually
            grad_output = error.to(dtype)
            grad_h2 = torch.mm(grad_output, self.W3) * relu_deriv(h2); grad_h2 = grad_h2.to(dtype)
            grad_W2 = torch.mm(grad_h2.t(), h1) / x_batch.shape[0]; grad_W2 = grad_W2.to(dtype)
            grad_W3 = torch.mm(grad_output.t(), h2) / x_batch.shape[0]; grad_W3 = grad_W3.to(dtype)
            grad_h1 = torch.mm(grad_h2, self.W2) * relu_deriv(h1); grad_h1 = grad_h1.to(dtype)
            grad_W1 = torch.mm(grad_h1.t(), x_mini_batch) / x_mini_batch.shape[0]; grad_W1 = grad_W1.to(dtype)

            self.W1 -= 0.15*self.lr*(np.sqrt(self.p1)*grad_W1*1/(self.eps_init**2) + self.reg1*self.W1).to(dtype)
            self.W2 -= 0.15*self.lr*(np.sqrt(self.p2)*grad_W2/(self.eps_init*self.p1) + self.reg2*1/self.d**1.5*self.W2).to(dtype)
            self.W3 -= 0.15*self.lr*(np.sqrt(1)*grad_W3/(self.eps_init*self.p2) + self.lambda_ridge*self.W3).to(dtype) # here lambda_ridge plays the role of reg3 

            # Store updated W2
            self.W1_trajectory.append(self.W1.clone())
            self.W2_trajectory.append(self.W2.clone())
            if i in snapshot_times:
                snapshot_errors.append(torch.mean(error**2).item())
            # validation loss
            if i % 50 == 0:
                output_valid, _, _ = self.forward(x_valid)
                validation_loss = torch.mean((output_valid - y_valid) ** 2).item()
                self.validation_loss_trajectory_layer2.append(validation_loss)
                self.training_loss_trajectory_layer2.append(torch.mean(error**2).item())
            # Explicitly delete tensors and cache
            del output, h1, h2, error, grad_W2, grad_output, grad_h2, grad_W3, grad_h1, grad_W1
            torch.cuda.empty_cache()
            

class TwoLayerNN:
    def __init__(self, p, d, lr, reg, lambda_ridge, eps_init, fraction_batch):
        # Layer dimensions
        self.d = d
        self.p = p
        self.lr = lr
        self.reg = reg
        self.lambda_ridge = lambda_ridge
        self.eps_init = eps_init
        # Weights initialization
        self.W1 = torch.randn(p, d, dtype=dtype) / np.sqrt(d)
        self.W2 = eps_init*torch.randn(1, p, dtype=dtype) / np.sqrt(p)
        # Fixed random biases initialization for first two layers with matching norms
        self.b1 = (2*torch.rand(p, 1, dtype=dtype)-1)
        self.b2 = eps_init*(2*torch.rand(1, 1, dtype=dtype)-1)
        # Move to device
        self.W1 = self.W1.to(device)
        self.W2 = self.W2.to(device)
        self.b1 = self.b1.to(device)
        self.b2 = self.b2.to(device)
        # Initialize lists to store weight trajectories
        self.W1_trajectory = [self.W1.clone()]
        self.W2_trajectory = [self.W2.clone()]
        # Validation/training loss trajectory when training layer 1
        self.validation_loss_trajectory_layer1 = []
        self.training_loss_trajectory_layer1 = []
        self.fraction_batch = fraction_batch
    def forward(self, x):
        # First layer (input to hidden layer 1)
        h1 = relu(torch.mm(self.W1, x.t()) + self.b1).t()

        # Second layer (hidden layer 1 to output)
        output = torch.mm(self.W2, h1.t()).t()

        return output, h1
    def train_layer1(self, x_batch, y_batch, x_valid, y_valid, max_iter):
        snapshot_times = [int(max_iter - 1), int(max_iter/2), int(max_iter/5), int(max_iter/10)]
        snapshot_errors = []
        fraction_batch = self.fraction_batch
        mini_batch_size = int(x_batch.shape[0] * fraction_batch)
        num_batches = int(max_iter / fraction_batch)
        for i in range(num_batches):
            indices = torch.randperm(x_batch.shape[0])[:mini_batch_size]
            x_mini_batch = x_batch[indices]
            y_mini_batch = y_batch[indices]
            output, h1 = self.forward(x_mini_batch)
            error = output - y_mini_batch
            grad_output = error.to(dtype)
            grad_h1 = torch.mm(grad_output, self.W2) * relu_deriv(h1); grad_h1 = grad_h1.to(dtype)
            grad_W1 = torch.mm(grad_h1.t(), x_mini_batch) / x_mini_batch.shape[0]; grad_W1 = grad_W1.to(dtype)
            # Update W1
            self.W1 -= self.lr*(np.sqrt(self.p)*grad_W1*1/(self.eps_init) + self.reg*self.W1).to(dtype)
            # Store updated W1
            self.W1_trajectory.append(self.W1.clone())
            if i in snapshot_times:
                snapshot_errors.append(torch.mean(error**2).item())
            # validation loss
            if i % 50 == 0:
                output_valid, _ = self.forward(x_valid)
                validation_loss = torch.mean((output_valid - y_valid) ** 2).item()
                self.validation_loss_trajectory_layer1.append(validation_loss)
                self.training_loss_trajectory_layer1.append(torch.mean(error**2).item())
            # Explicitly delete tensors and cache
            del output, h1, error, grad_W1, grad_output, grad_h1
            torch.cuda.empty_cache()
            
    def ridge_regression_update_W2(self, x_batch, y_batch):
        # Ridge regression hyperparameters
        lambda_ridge = float(self.lambda_ridge)
        nbatch = x_batch.shape[0]
        _, H1 = self.forward(x_batch)  # Compute H1
        H1_t_H1_size = H1.shape[1] * H1.shape[1] * 4  # Estimate memory for H1_t_H1 (float32 = 4 bytes per value)

        # Select device based on estimated memory
        regularization_matrix = torch.eye(H1.shape[1], dtype=torch.float32, device=device)
        H1_t_H1 = torch.zeros((H1.shape[1], H1.shape[1]), dtype=torch.float32, device=device)
        batch_size = min(1024, nbatch)  # Define a manageable batch size based on actual nbatch
        for i in range(0, nbatch, batch_size):
            H1_batch = H1[i : i + batch_size]
            H1_t_H1 += torch.mm(H1_batch.t(), H1_batch).to(device)

        H1_t_H1 += lambda_ridge * nbatch * regularization_matrix

        y_batch_sol = y_batch.to(device).to(torch.float32)
        H1_sol = H1.to(device).to(torch.float32)
        try:
            solution = torch.linalg.solve(H1_t_H1, torch.mm(H1_sol.t(), y_batch_sol))
        except RuntimeError as e:
            if 'singular' in str(e):
                print("Matrix is singular, returning zero estimator.", flush=True)
                solution = torch.zeros_like(self.W2, device=device).t()
            else:
                raise e
        self.W2 = solution.t().to(device).to(dtype)
        del H1_t_H1, solution, y_batch_sol, H1_sol, regularization_matrix, H1
        torch.cuda.empty_cache()

def cosine_similarity_linear(W1_stars, W1s, FlagOverlapLin=False):
    "Function that computes the linear overlap M_W"
    W1_star_tensor = torch.stack(W1_stars, dim=2).to(device).to(dtype)  # shape: (d, deff, k)
    if FlagOverlapLin:
        # W1s is a list of matrices for each time step, each W1 has shape (p1, d)
        T = len(W1s)  # Number of time step
        # Stack all W1s into a single tensor
        W1_tensor = torch.stack(W1s).to(device).to(dtype)  # shape: (T, p1, d)
        # Compute numerator for all time steps at once
        numerator = torch.einsum('tpd,dek->tpek', W1_tensor, W1_star_tensor)  # shape: (T, p1, deff, k)
        n2 = torch.norm(numerator,dim=2) # shape: (T, p1, k)
        W1_norms = torch.norm(W1_tensor, dim=2)  # shape: (T, p1)
        denominator =  torch.unsqueeze(W1_norms,-1) # shape: (T, p1, 1)
        CosSim = n2 / denominator # shape: (T, p1, k)
        CosSimMetrics = torch.sqrt(torch.mean(CosSim**2, dim=(1)))  # shape: (T,k)
    else:
        # skip the computation of overlaps and allocate zero matrices of the correct dimensions
        CosSimMetrics = torch.zeros((len(W1s), W1_star_tensor.shape[2])).to(device).to(dtype)
    return CosSimMetrics

def cosine_similarity_nonlinear_joint_evolution(W1s, W2s, b1, b2, W1_stars, ws, FlagOverlapNonlin=False, sigma_star1=He_2):
    "Function that computes the non-linear overlap M_h"
    if FlagOverlapNonlin: 
        T = len(W2s)  # Number of time steps
        assert len(W1s) == len(W2s), "Length of W1s and W2s must be the same"
        d = W1s[0].shape[1]
        k = len(W1_stars)
        W1_star_tensor = torch.stack(W1_stars, dim=2).to(device).to(dtype)
        n_overlap = 1000
        X = torch.randn(n_overlap, d).to(device).to(dtype)
        deff = W1_star_tensor.shape[1]
        
        # Initialize overlap tensor of shape (T, p2, k)
        overlap_tensor = torch.zeros(T, W2s[0].shape[0], len(W1_stars)).to(device).to(dtype)
        first_term = torch.einsum('tpd,nd->tnp', torch.stack(W1s), X)
        bias_term = b1.squeeze(-1).unsqueeze(0).unsqueeze(1).expand(T, X.shape[0], b1.size(0))
        h1s = relu(first_term + bias_term)
        
        
        # Compute U_star once since it's independent of W2
        W1_star_permuted = W1_star_tensor.permute(1, 0, 2)  # shape: (deff, d, k)
        preactivations = sigma_star1(torch.einsum('dik,ni->dnk', W1_star_permuted, X))  # shape: (deff, n_overlap, k)
        ws = torch.stack(ws).to(device).to(dtype)
        ws = ws.view(k, deff) # shape: (k, deff)
        U_star = (1/deff**.5) * torch.einsum('kd,dnk->nk', ws, preactivations)  # shape: (n_overlap, k)
        U_star_norm = torch.norm(U_star, dim=0)
        
        # Stack all W2s into a single tensor
        W2_tensor = torch.stack(W2s).to(device).to(dtype)  # shape: (T, p2, p1)
        
        # Compute U for all time steps at once
        U = torch.bmm(W2_tensor, h1s.permute(0, 2, 1)) + b2.unsqueeze(0)  # shape: (T, p2, n_overlap)
        U = U.permute(0, 2, 1)  # shape: (T, n_overlap, p2)
        
        # Compute overlap matrix for all time steps at once
        # contract U and U_star over n_overlap
        numerator = torch.einsum('tnp,nk->tpk', U, U_star)  # shape: (T, p2, k)
        U_norm = torch.norm(U, dim=1)  # shape: (T, p2)
        denominator = torch.einsum('tp,k->tpk', U_norm, U_star_norm)  # shape: (T, p2, k)
        overlap_tensor = numerator / denominator
    else:
        # skip the computation of overlaps and allocate zero matrices of the correct dimensions 
        overlap_tensor = torch.zeros(len(W2s), W2s[0].shape[0], len(W1_stars)).to(device).to(dtype)
    return overlap_tensor

def generate_weights(d, k, epsilon):
    deff = int(d**epsilon)
    W1_stars = [0]*k
    ws = [torch.sign(torch.randn(deff, 1, dtype=dtype)) for _ in range(k)] 
    random_matrix,_ = np.linalg.qr(np.random.randn(d, k*deff))
    for i in range(k):
        W1_stars[i] = random_matrix[:, i*deff:(i+1)*deff]
        W1_stars[i]= torch.tensor(W1_stars[i], dtype=dtype).to(device)
    return W1_stars, ws

def generate_data(W1_stars, ws, n, d, epsilon, k, sigma_star1, sigma_star2):
    deff = int(d**epsilon)
    x = torch.randn(n, d).to(device).to(dtype)
    hs = torch.zeros(n, k).to(device).to(dtype)
    for i in range(k):
        "Generate non-linear features for every branch"
        w = ws[i].to(device).to(dtype)
        h2 = torch.mm(x, W1_stars[i])
        hs[:,i] = (1 / np.sqrt(deff)) * torch.squeeze(torch.mm(sigma_star1(h2),w)) 
    " Commitee machine with sigma_star2 activation"
    y = 1/np.sqrt(k)*torch.sum(sigma_star2(hs), dim=1).unsqueeze(1).to(device).to(dtype)
    return x, y, hs

# Generalization error function
def generalization_error(nn, x_test, y_test):
    with torch.no_grad():
        y_pred = nn.forward(x_test)[0]
        return torch.mean((y_pred - y_test) ** 2).item()
