import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import torch

device = 'cpu'

idxs = [0, 1]

### Labels ###
FlagYLabel = True
FlagXLabel = True

FlagLegend = True
# FlagLegend = False
FlagLogScale = True

d_to_start = 0
markersize = 6  # Add markersize variable
fontsize = 10  # Add fontsize variable
fontsize_legend = 8  # Adjust legend font size for subplot clarity

# Prepare markers for different dimensions
markers = ['o', 's', 'x', 'v', '^', '<', '>']


## Baselines if the target function is built as the experiments in the paper: sigma1_code: h2+1h3; sigma_2code: tanh with coef = 3
baselines = {3: 0.14}    #2LNN
random_performances = {3: 0.65} # Random performance
baselines_rf = {3: 0.25} # Kernel methods

for data_folder in os.listdir('Codes/Data'):
    hyperparameters_file = [f for f in os.listdir(f'Codes/Data/{data_folder}') if f.endswith('.yaml')][0]
    hyperparameters_file = hyperparameters_file.replace('.yaml', '')
    
    # Load parameters
    with open(f'Codes/Data/{data_folder}/{hyperparameters_file}.yaml') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    ds = parameters['ds']
    lr = parameters['lr']
    exp_min = parameters['exp_min']
    exp_max = parameters['exp_max']
    num = parameters['num']
    trials = parameters['trials']
    coef = parameters['coef']
    # Load data
    filename = f'Codes/Data/{hyperparameters_file}/3layer_results.pt'
    print(f"Plotting {hyperparameters_file}")
    os.makedirs(f'Codes/Data/{hyperparameters_file}', exist_ok=True)
    data = torch.load(filename, map_location=device)
    errors_resultss = data['errors_resultss']
    baseline = baselines[coef]
    random_performance = random_performances[coef]

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 4.3), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1})
    plt.subplots_adjust(hspace=0.1, bottom=0.1, top=0.95)
    textsymbols = ["(1)", "(2)"]
    for subplot_idx, idx in enumerate(idxs):

        ax = axes[subplot_idx]  
        
        for i, d in enumerate(ds):
            if i >= d_to_start:
                xaxis_n = np.linspace(exp_min, exp_max, num)                
                errors_setup1s, errors_setup4s, errors_setup2layers, errors_joint, = [torch.tensor(errors_resultss[i][j]).cpu().numpy() for j in range(4)]
                    
                if idx == 0:  
                    ax.errorbar(xaxis_n, errors_setup2layers,
                                 color='green', marker=markers[i], markersize=markersize)
                    ax.errorbar(xaxis_n, errors_setup4s, 
                                 color='orange', marker=markers[i], markersize=markersize)
                    ax.errorbar(xaxis_n, errors_setup1s,
                                label=f"d = {d}", color='blue', marker=markers[i], markersize=markersize)
                if idx == 1:  
                    ax.errorbar(xaxis_n, errors_setup1s,
                                label=f"d = {d}", color='blue', marker=markers[i], markersize=markersize)
                    ax.errorbar(xaxis_n, errors_joint, 
                                 marker=markers[i], color='red', markersize=markersize)

        vertical_line_kernel = np.log(int(d*(d-1)*.5 + d + 1)) / np.log(d)
        ax.axhline(y=baseline, alpha=0.5, color='green', linestyle='--')
        x_axis_aux = np.linspace(exp_min, exp_max+0.2, 1000)
        y_2lnn = np.where(x_axis_aux < 1.5, random_performance, baseline)
        ax.plot(x_axis_aux, y_2lnn, color='green')
        ax.axhline(y=random_performance, alpha=1, color='purple')
        y_rf = np.where(x_axis_aux < vertical_line_kernel, random_performance, baselines_rf[coef])
        ax.plot(x_axis_aux, y_rf, color='orange')
        ax.axhline(y=baselines_rf[coef], alpha=0.5, color='orange', linestyle='--')
        ax.axvline(x=1.5, color='black', linestyle='--')
        
        ax.set_ylim([.5*baseline, 1])
        ax.set_xlim([exp_min, exp_max+0.05])
        # Titles and labels
    axes[1].plot(x_axis_aux, 1*np.ones_like(x_axis_aux)*random_performance, color='purple', label='Random')
    axes[0].plot(x_axis_aux, 1*np.ones_like(x_axis_aux)*random_performance, color='purple', label='Random')
    axes[0].axvline(x=vertical_line_kernel, color='orange', linestyle='--')
    axes[0].plot(x_axis_aux, y_2lnn, color='green', label='2LNN')
    axes[0].plot(x_axis_aux, y_rf, color='orange', label='Kernel')
    axes[1].plot(x_axis_aux, y_rf, color='orange', label='Kernel')
    axes[0].plot(x_axis_aux, 0*np.ones_like(x_axis_aux)*random_performance, color='blue', label='Layerwise')
    axes[1].plot(x_axis_aux, 0*np.ones_like(x_axis_aux)*random_performance, color='blue', label='Layerwise')
    axes[1].plot(x_axis_aux, 0*np.ones_like(x_axis_aux)*random_performance, color='red', label='Joint')
    
    if FlagXLabel and subplot_idx == 1:
        ax.set_xlabel(r'$\kappa = $log(n) / log(d)', fontsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    if FlagLogScale:
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')    
    axes[1].legend(fontsize=fontsize_legend, ncol=1)
    axes[0].legend(fontsize=fontsize_legend, ncol=1)
    # plt.tight_layout()
    plt.savefig(f'Codes/Data/{hyperparameters_file}/ErrorPlot{FlagLogScale}.pdf')
    plt.show()
