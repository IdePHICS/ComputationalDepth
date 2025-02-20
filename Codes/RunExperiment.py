import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import yaml
import os
import sys
from contextlib import redirect_stdout
import copy
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device, flush=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
work_folder = f'ParamsFolder'

if 'SLURM_JOB_ID' in os.environ:
    output_name = os.getenv('SLURM_JOB_ID', 'default_output_name')
else:
    output_name = input("Enter the output name: ")
# Set the default tensor type to float16
dtype = torch.float32


# Main function to compute three metrics: generalization error, cosine similarity, and loss trajectories
def ComputeMetrics(d, n_values, lr, trials, k, epsilon, regs, eps_init=0.2, delta=0.1, p2=100, FlagOverlapLin=False, FlagOverlapNonlin=False, FlagTrain2Layer=False,  seed = 1846, coef_iter=2, test_size=1000, fraction_batch=0.1):
    "Input: d: dimension, n_values: number of samples probed, lr: learning rate, trials: number of trials, k: number of branches in the tree hierarchical model, epsilon: effective dimension parameter, regs: list of regularizations for different training methods, eps_init: initialization scale, delta: hidden layer size parameter, p2: second hidden layer size, FlagOverlapLin: flag to compute cosine similarity for linear features, FlagOverlapNonlin: flag to compute cosine similarity for nonlinear features, FlagTrain2Layer: flag to train 2layer network, seed: random seed, coef_iter: coefficient for the number of iterations, test_size: size of the test set, fraction_batch: fraction of the batch size used for training in mini-batch GD iterations"
    
    "Output: errors_results: list of generalization errors for different training methods, alignment_results: list of cosine similarity metrics, trajectory_results: list of validation and training loss trajectories"

    errors_setup1 = [] ; errors_setup4 = []; errors_setup2layer = []; errors_setup_joint = []
    CosSimLinear = [] ; CosSimNonlinear = []; CosSimLinear_joint = []; CosSimLinear_2layer = []; CosSimNonLinearJoint = []
    # Validation and training loss trajectories 
    val_loss_setup1, train_loss_setup1 = [], []
    val_loss_joint, train_loss_joint = [], []
    val_loss_setup2layer, train_loss_setup2layer = [], []
    val_loss_setup1_layer1, train_loss_setup1_layer1 = [], []

    # First hidden layer size is fixed with delta. $p1 = d^{kappa_max*(1-delta)}$
    nmax = int(n_values[-1]) ; p1 = int(nmax**(1-delta))
    # First hidden layer size of 2LNN 
    p = int(p1/15) 
    ### SET  MAXITERS: iteration time for graident descent on the first layer (max_iter1) and second layer (max_iter2) ###
    # layerwise training
    max_iter2 = int(coef_iter*d**(1.5)) 
    max_iter1 = int(15*np.log(d))
    # joint training iterated for max_iter2
    max_iter_joint = max_iter2
    # Regularizations for both gradient descent and ridge regression
    reg1_setup1, reg2_setup1, lambda_ridge_setup1 = regs[0]
    lambda_ridge_setup_kernel = regs[1]
    reg1_setup_2layer, lambda_ridge_setup_2layer = regs[2]
    reg1_setup_joint, reg2_setup_joint, reg3_setup_joint = regs[3]

    set_seed(seed)
    for iter_n, n in enumerate(n_values):
        # Allocate auxiliary variables
        errors1 = []; errors4 = []; errors_2layer = []; errors_joint = []
        CosSimLinear_trial = [] ; CosSimNonlinear_trial = []; CosSimLinear_joint_trial = []
        CosSimLinear_2layer_trial = [] ; CosSimNonLinearJoint_trial = []
        val_loss_setup1_fixedn = [] ; train_loss_setup1_fixedn = [] ; val_loss_joint_fixedn = [] ; train_loss_joint_fixedn = [] ; val_loss_setup2layer_fixedn = [] ; train_loss_setup2layer_fixedn = []; train_loss_setup1_layer1_fixedn = []; val_loss_setup1_layer1_fixedn = []
        
        # total computation  budget
        ntot = n
        for stat in range(trials):
            ### Generate target weights and data ###
            W1_stars, a_star = generate_weights(d, k, epsilon)
            x_test, y_test, _ = generate_data(W1_stars, a_star, test_size, d, epsilon, k, sigma_star1, sigma_star2)
            x_valid, y_valid, _ = generate_data(W1_stars, a_star, test_size, d, epsilon, k, sigma_star1, sigma_star2)
            xtrain, ytrain, _ = generate_data(W1_stars, a_star, ntot, d, epsilon, k, sigma_star1, sigma_star2)

            ### Train the networks ###

            # Layerwise
            
            nn_setup1 = ThreeLayerNN(p1, p2, d, lr=lr, reg1=reg1_setup1, reg2=reg2_setup1, lambda_ridge=lambda_ridge_setup1, FlagReinitialize=False, fraction_batch=fraction_batch, eps_init=eps_init)
            nn_setup1.train_layer1(xtrain, ytrain, x_valid, y_valid, max_iter1)
            nn_setup1.train_layer2(xtrain, ytrain, x_valid, y_valid, max_iter2)
            nn_setup1.ridge_regression_update_W3(xtrain, ytrain)
            error = generalization_error(nn_setup1, x_test, y_test)
            errors1.append(error)

            # 2layer network
            nn_setup_2layer = TwoLayerNN(p, d, lr=lr, reg=reg1_setup_2layer, lambda_ridge=lambda_ridge_setup_2layer, fraction_batch=fraction_batch, eps_init=eps_init)
            if FlagTrain2Layer:
                nn_setup_2layer.train_layer1(xtrain, ytrain, x_valid, y_valid, max_iter2)
                nn_setup_2layer.ridge_regression_update_W2(xtrain, ytrain)
                error = generalization_error(nn_setup_2layer, x_test, y_test)
                traj2layer = nn_setup_2layer.W1_trajectory 
            else: 
                error = 0
                traj2layer = [nn_setup_2layer.W1.clone()] * len(nn_setup1.W1_trajectory)
            errors_2layer.append(error)

            # Joint training 
            nn_setup_joint = ThreeLayerNN(p1, p2, d, lr=lr, reg1=reg1_setup_joint, reg2=reg2_setup_joint, lambda_ridge=reg3_setup_joint, fraction_batch=fraction_batch, eps_init=eps_init)
            nn_setup_joint.joint_training(xtrain, ytrain, x_valid, y_valid, max_iter_joint)
            error = generalization_error(nn_setup_joint, x_test, y_test)
            errors_joint.append(error)


            # Kernel Ridge Regression
            K_train = quadratic_kernel(xtrain, xtrain)
            alpha = torch.linalg.solve(K_train + lambda_ridge_setup_kernel * ntot * torch.eye(ntot).to(device).to(dtype), ytrain)
            K_test = quadratic_kernel(xtrain, x_test)
            y_pred = torch.mm(K_test.t(), alpha)
            error = torch.mean((y_pred - y_test) ** 2).item()
            errors4.append(error)    

            ### compute overlap and cosine similarity metrics ###

            # Layerwise training
            W1_trajectory_setup1 = copy.deepcopy(nn_setup1.W1_trajectory)
            W1_trajectory_setup1.extend([nn_setup1.W1.clone()] * len(nn_setup1.W2_trajectory))
            training_traj_W2 = nn_setup1.W2_trajectory
            W2_init = training_traj_W2[0]
            W2_fixed_trajectory = [W2_init] * len(nn_setup1.W1_trajectory)
            W2_trajectory_setup1 = W2_fixed_trajectory
            W2_trajectory_setup1.extend(training_traj_W2)
            CosSimLinear_trial.append(cosine_similarity_linear(W1_stars, W1_trajectory_setup1, FlagOverlapLin))
            CosSimNonlinear_trial.append(cosine_similarity_nonlinear_joint_evolution(W1_trajectory_setup1, W2_trajectory_setup1, nn_setup1.b1, nn_setup1.b2, W1_stars, a_star, FlagOverlapNonlin, sigma_star1))

            # 2Layer 
            CosSimLinear_2layer_trial.append(cosine_similarity_linear(W1_stars, traj2layer))
            # Joint training
            CosSimLinear_joint_trial.append(cosine_similarity_linear(W1_stars, nn_setup_joint.W1_trajectory, FlagOverlapLin))
            CosSimNonLinearJoint_trial.append(cosine_similarity_nonlinear_joint_evolution(nn_setup_joint.W1_trajectory, nn_setup_joint.W2_trajectory, nn_setup_joint.b1, nn_setup_joint.b2, W1_stars, a_star, FlagOverlapNonlin, sigma_star1))

            ### store loss trajectories ###
            val_loss_setup1_fixedn.append(nn_setup1.validation_loss_trajectory_layer2); val_loss_setup1_fixedn[-1].append(errors1[-1])
            train_loss_setup1_fixedn.append(nn_setup1.training_loss_trajectory_layer2)
            train_loss_setup1_layer1_fixedn.append(nn_setup1.training_loss_trajectory_layer1)
            val_loss_setup1_layer1_fixedn.append(nn_setup1.validation_loss_trajectory_layer1)
            val_loss_joint_fixedn.append(nn_setup_joint.validation_loss_trajectory_layer2)
            train_loss_joint_fixedn.append(nn_setup_joint.training_loss_trajectory_layer2)
            val_loss_setup2layer_fixedn.append(nn_setup_2layer.validation_loss_trajectory_layer1); val_loss_setup2layer_fixedn[-1].append(errors_2layer[-1])
            train_loss_setup2layer_fixedn.append(nn_setup_2layer.training_loss_trajectory_layer1)

        ### Aggregate results for different trials ###

        # compute the mean of the cosine similarity metrics
        CosSimLinear.append(torch.mean(torch.abs(torch.stack(CosSimLinear_trial)), dim=0))
        CosSimNonlinear.append(torch.mean(torch.abs(torch.stack(CosSimNonlinear_trial)), dim=0))
        CosSimLinear_2layer.append(torch.mean(torch.abs(torch.stack(CosSimLinear_2layer_trial)), dim=0))
        CosSimLinear_joint.append(torch.mean(torch.abs(torch.stack(CosSimLinear_joint_trial)), dim=0))
        CosSimNonLinearJoint.append(torch.mean(torch.abs(torch.stack(CosSimNonLinearJoint_trial)), dim=0))


        # use median to aggregate different errors
        errors_setup1.append(np.median(errors1)); errors_setup4.append(np.median(errors4));  errors_setup2layer.append(np.median(errors_2layer)); errors_setup_joint.append(np.median(errors_joint))

        # store all the loss trajectories
        val_loss_setup1.append(val_loss_setup1_fixedn)        
        train_loss_setup1.append(train_loss_setup1_fixedn)
        train_loss_setup1_layer1.append(train_loss_setup1_layer1_fixedn)
        val_loss_setup1_layer1.append(val_loss_setup1_layer1_fixedn)
        val_loss_joint.append(val_loss_joint_fixedn)
        train_loss_joint.append(train_loss_joint_fixedn)
        val_loss_setup2layer.append(val_loss_setup2layer_fixedn)
        train_loss_setup2layer.append(train_loss_setup2layer_fixedn)

        print(f"FOR n={n} and d={d}  (log(n)/log(d) ={np.log(n)/np.log(d):.3f})-- AFTER AVERAGING: Layerwise = {errors_setup1[-1]:.3f}, 2layer = {errors_setup2layer[-1]:.3f}, Joint = {errors_setup_joint[-1]:.3f}, Random Features = {errors_setup4[-1]:.3f}", flush = True)
    ### return the results ###
    CosSim_results = [CosSimLinear,  CosSimNonlinear, CosSimLinear_2layer, CosSimLinear_joint, CosSimNonLinearJoint]
    errors_results = [errors_setup1, errors_setup4, errors_setup2layer,  errors_setup_joint]
    val_train_trajectories = [val_loss_setup1, train_loss_setup1, val_loss_joint, train_loss_joint, val_loss_setup1_layer1, train_loss_setup1_layer1, val_loss_setup2layer, train_loss_setup2layer]
    return  errors_results, CosSim_results, val_train_trajectories

# read all the files from hyperparams
for hyperparameters_file in os.listdir(work_folder):
    with open(f'{work_folder}/{hyperparameters_file}') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    ### Defining the ranges of dimensions and number of samples. n=np.logspace(exp_min,exp_max,num=num,base=d) and for each n we run "trials" experiments"
    ds = parameters['ds'] ; exp_min = parameters['exp_min'] ; exp_max = parameters['exp_max'] ; num = parameters['num']; trials = parameters['trials']
    ### Activation functions: sigma1 and sigma2, k is the number of branches, coef is the coefficient modulating the tanh ###
    sigma1_code = parameters['sigma1_code'] ; sigma2_code = parameters['sigma2_code'];  k = parameters['k']; coef = parameters['coef']
    ### Effective dimension $d_{eff} = d^{\epsilon}$ ###
    epsilon = parameters['epsilon']
    ### Hyperparameters for the neural networks ###
    # Scale of the initialization of the weights, learning rate strength, minibatch size (fraction of the total batch), and number of GD iterations
    lr = parameters['lr'] ; fraction_batch = parameters['fraction_batch']; coef_iter = parameters['coef_iter'];
    # Hidden layer sizes for the first and second layer; $p1 = d^{kappa_max*(1-delta)}$ and p2 is chosen from the user
    delta = parameters['delta']; p2 = parameters['p2']
    # Regularization for different training methods/architectures
    lambda_ridge = parameters['lambda_ridge']; reg1 = parameters['reg1'] ; reg2 = parameters['reg2']; lambda_ridge = parameters['lambda_ridge']
    reg1_setup_2layer = parameters['reg1_setup_2layer']; lambda_ridge_setup_2layer = parameters['lambda_ridge_setup_2layer']
    lambda_ridge_setup_kernel = parameters['lambda_ridge_setup_kernel']
    reg1_setup_joint = parameters['reg1_setup_joint']; reg2_setup_joint = parameters['reg2_setup_joint']; reg3_setup_joint = parameters['reg3_setup_joint']

    ### Setup simulation for efficiency: flags to skip calculations if not wanted from the user ### 
    # FlagOverlapLin and FlagOverlapNonlin are used to activate the computation of the cosine similarity metrics
    # FlagTrain2Layer is used to train the 2layer network
    FlagOverlapLin = parameters['FlagOverlapLin'] ; FlagOverlapNonlin = parameters['FlagOverlapNonLin'] ; FlagTrain2Layer = parameters['FlagTrain2Layer'] 
    # activation dictionary to select with codewords
    activation_dict = {'tanh': lambda x: torch.tanh(x).to(device), 'relu': lambda x: torch.relu(x).to(device), 'tanh_coef' : lambda x: torch.tanh(coef*x).to(device), 'h2+1h3': lambda x: (He_2(x) + He_3(x)).to(device), 'h2+2h3': lambda x: (He_2(x) + 2*He_3(x)).to(device)}
    regs = [(reg1, reg2, lambda_ridge), (lambda_ridge_setup_kernel), (reg1_setup_2layer, lambda_ridge_setup_2layer), (reg1_setup_joint, reg2_setup_joint, reg3_setup_joint)]
    sigma_star1 = activation_dict[sigma1_code] ; sigma_star2 = activation_dict[sigma2_code]
    # remove yaml from hyperparameters_file 
    hyperparameters_file = hyperparameters_file.replace('.yaml', '')
    # create directory /data/hyperparameters_file if it does not exist
    os.makedirs(f'data/{hyperparameters_file}', exist_ok=True)
    filename = f'data/{hyperparameters_file}/3layer_results.pt'
    output_file = f'data/{hyperparameters_file}/{output_name}.txt'
    with open(output_file, 'w') as f:
        with redirect_stdout(f):
            alignment_resultss = [] ; errors_resultss = [] ; trajectory_resultss = []
            if os.path.exists(filename):
                print(f"Results already computed in {filename}")
            else:
                print(f"Results for {hyperparameters_file} not computed yet. Starting the experiment")
                sys.stdout.flush()
                for d in ds: 
                    my_seed = 1926 ^ d
                    print(f"Start experiment for d = {d}", flush = True)
                    start_time = time.time()
                    n_values = np.logspace(exp_min,exp_max, num = num, base=d, dtype=int)
                    errors_results, alignment_results, trajectory_results = ComputeMetrics(d, n_values, lr, trials=trials, k=k, epsilon=epsilon, regs=regs, delta=delta, p2=p2, FlagOverlapLin=FlagOverlapLin, FlagOverlapNonlin=FlagOverlapNonlin, FlagTrain2Layer=FlagTrain2Layer, seed = my_seed, coef_iter=coef_iter, fraction_batch=fraction_batch)
                    alignment_resultss.append(alignment_results)
                    errors_resultss.append(errors_results)
                    trajectory_resultss.append(trajectory_results)
                    end_time = time.time()
                    print(f"It took {end_time-start_time} seconds to finish the experiment for d = {d}", flush = True)

                torch.save({
                    'alignment_resultss': alignment_resultss,
                    'errors_resultss': errors_resultss, 
                    'trajectory_resultss': trajectory_resultss
                }, f'{filename}')
                # move the hyperparameters file yaml to the data folder close to the .pt file associated
                os.rename(f'{work_folder}/{hyperparameters_file}.yaml', f'data/{hyperparameters_file}/{hyperparameters_file}.yaml')
                