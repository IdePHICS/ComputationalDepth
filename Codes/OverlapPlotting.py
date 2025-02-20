import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import torch

# data struct is: 
# CosSim_results = [CosSimLinear,  CosSimNonlinear, CosSimLinear_2layer, CosSimLinear_joint, CosSimNonLinearJoint]


dict_names = {0: 'Layerwise', 1: 'Layerwise', 3: "Joint", 4: "Joint"}
device = 'cpu'
markersize = 6  # Add markersize variable
fontsize = 10  # Add fontsize variable
fontsize_legend = 8  # Adjust legend font size for subplot clarity

for data_folder in os.listdir('Codes/Data'):
    hyperparameters_file = [f for f in os.listdir(f'Codes/Data/{data_folder}') if f.endswith('.yaml')][0]
    hyperparameters_file = hyperparameters_file.replace('.yaml', '')
    with open(f'Codes/Data/{data_folder}/{hyperparameters_file}.yaml') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    ds = parameters['ds'] ; exp_min = parameters['exp_min'] ; exp_max = parameters['exp_max'] ; num = parameters['num']; k = parameters['k'] 
    filename = f'Codes/Data/{hyperparameters_file}/3layer_results.pt'
    print(f"Plotting {hyperparameters_file}")
    os.makedirs(f'Codes/Data/{hyperparameters_file}', exist_ok=True)
    data = torch.load(filename, map_location=device)
    alignment_resultss = data['alignment_resultss']
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
    markers = ['o', 's', 'x', 'v', '^', '<', '>', '1', '2', '3', '4', '8']
    
    # nplot 
    jslin = [0, 3]
    jsnonlin = [1, 4]
    # time plot
    jlin_values = [0, 3]
    jnonlin_values = [1, 4]
    textsymbols = [r"||$M_W$||$_2$", r"||$M_h$||$_2$"]
    fig, axs = plt.subplots(2, 1, figsize=(3.3, 4.3), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1})
    plt.subplots_adjust(hspace=0.1, bottom=0.1, top=0.95)
    # Plot dependence with n for time = Tmax for linear case for all the different training strategies in one plot (indexed by jslin) and only for the highest size
    for j_idx, j in enumerate(jslin):
        for i, d in enumerate(ds):
            xaxis_n = np.linspace(exp_min, exp_max, num)
            if i == len(ds) - 1:
                cos_linear = alignment_resultss[i][j]
                cos_linear = torch.stack(cos_linear, dim=0).cpu().numpy() # shape: (num, T, k)
                T = cos_linear.shape[1]
                for idx in range(k):
                    axs[0].plot(xaxis_n, cos_linear[:, -1, idx], color=colors[j_idx], marker=markers[i+ 2*idx], markersize=markersize)
    axs[0].axvline(x = 1.5, color = 'black', linestyle = '--')
    axs[0].set_ylim([0.35, 0.9])
    for j_idx, j in enumerate(jsnonlin):
        for i, d in enumerate(ds):
            xaxis_n = np.linspace(exp_min, exp_max, num)
            if i == len(ds) - 1:
                cos_non_linear = alignment_resultss[i][j]
                cos_non_linear = torch.stack(cos_non_linear, dim=0).cpu().numpy()  # shape: (num, T, p2, k)
                l2_norm_non_linear = np.sqrt(np.mean(cos_non_linear**2, axis=2))  # shape: (num, T, k)
                T = l2_norm_non_linear.shape[1]
                for idx in range(k):
                    axs[1].plot(xaxis_n, l2_norm_non_linear[:, -1, idx], label=f'{dict_names[j]}', color=colors[j_idx], marker=markers[i+ 2*idx], markersize=markersize)
    axs[1].set_xlabel(r'$\kappa = $ log(n) / log(d)', fontsize=fontsize)
    # axs[1].set_ylabel(r'||$M_h$||$_2$', fontsize=fontsize)
    axs[1].axvline(x = 1.5, color = 'black', linestyle = '--')
    axs[1].legend(fontsize=fontsize_legend)
    axs[1].set_ylim([0,0.35])
    for count, symbol in enumerate(textsymbols):
        ax = axs[count]
        ax.text(0.03, 0.7, symbol, transform=ax.transAxes, fontsize=fontsize+2, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(f"Codes/Data/{hyperparameters_file}/CosSimAllTogether.pdf")
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    for j_idx, jlin in enumerate(jlin_values):
        for i, d in enumerate(ds):
            if i == len(ds) - 1:
                cos_linear = alignment_resultss[i][jlin]
                cos_linear = torch.stack(cos_linear, dim=0).cpu().numpy()
                T = cos_linear.shape[1]
                xaxis_T = np.arange(T)
                for idx in range(k):
                    axs[0].plot(xaxis_T, cos_linear[-1, :, idx], label=f'{dict_names[jlin]} - Dir {idx+1}', color=colors[j_idx], marker=markers[j_idx + 2*idx], markersize=markersize)
    axs[0].set_xlabel('Time', fontsize=fontsize)
    axs[0].set_xscale('log')
    axs[0].set_ylabel(r'||$M_W$||$_2$', fontsize=fontsize)
    axs[0].legend(fontsize=fontsize_legend)
    axs[0].set_ylim([0.35, 0.9])
    for j_idx, jnonlin in enumerate(jnonlin_values):
        for i, d in enumerate(ds):
            if i == len(ds) - 1:
                cos_non_linear = alignment_resultss[i][jnonlin]
                cos_non_linear = torch.stack(cos_non_linear, dim=0).cpu().numpy()
                l2_norm_non_linear = np.sqrt(np.mean(cos_non_linear**2, axis=2))
                T = l2_norm_non_linear.shape[1]
                xaxis_T = np.arange(T)
                for idx in range(k):
                    axs[1].plot(xaxis_T, l2_norm_non_linear[-1, :, idx], label=f'{dict_names[jnonlin]} - Dir {idx+1}', color=colors[j_idx], marker=markers[j_idx + 2*idx], markersize=markersize)
    axs[1].set_xlabel('Time', fontsize=fontsize)
    axs[1].set_xscale('log')
    axs[1].set_ylabel(r'||$M_h$||$_2$', fontsize=fontsize)
    axs[1].legend(fontsize=fontsize_legend)
    axs[1].set_ylim([0,0.35])
    plt.tight_layout()
    plt.savefig(f"Codes/Data/{hyperparameters_file}/CosSimAllTogetherTime.pdf")
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # repeat the same subplot for the case n = nmax/2
    for j_idx, jlin in enumerate(jlin_values):
        for i, d in enumerate(ds):
            if i == len(ds) - 1:
                cos_linear = alignment_resultss[i][jlin]
                cos_linear = torch.stack(cos_linear, dim=0).cpu().numpy()
                T = cos_linear.shape[1]
                xaxis_T = np.arange(T)
                for idx in range(k):
                    axs[0].plot(xaxis_T, cos_linear[num // 2 - 1, :, idx], label=f'{dict_names[jlin]} - Dir {idx+1}', color=colors[j_idx], marker=markers[j_idx + 2*idx], markersize=markersize)
    axs[0].set_xlabel('Time', fontsize=fontsize)
    axs[0].set_xscale('log')
    axs[0].set_ylabel(r'||$M_W$||$_2$', fontsize=fontsize)
    axs[0].legend(fontsize=fontsize_legend)
    axs[0].set_ylim([0.35, 0.9])
    for j_idx, jnonlin in enumerate(jnonlin_values):
        for i, d in enumerate(ds):
            if i == len(ds) - 1:
                cos_non_linear = alignment_resultss[i][jnonlin]
                cos_non_linear = torch.stack(cos_non_linear, dim=0).cpu().numpy()
                l2_norm_non_linear = np.sqrt(np.mean(cos_non_linear**2, axis=2))
                T = l2_norm_non_linear.shape[1]
                xaxis_T = np.arange(T)
                for idx in range(k):
                    axs[1].plot(xaxis_T, l2_norm_non_linear[num // 2 - 1 , :, idx], label=f'{dict_names[jnonlin]} - Dir {idx+1}', color=colors[j_idx], marker=markers[j_idx + 2*idx])
    axs[1].set_xlabel('Time')
    axs[1].set_xscale('log')
    axs[1].set_ylabel(r'||$M_h$||$_2$', fontsize=12)
    axs[1].set_ylim([0,0.35])
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(f"Codes/Data/{hyperparameters_file}/CosSimAllTogetherTimeNmax2.pdf")
