import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from medis.twilight_colormaps import sunlight
from matplotlib.colors import LogNorm
import medis.medis_main as mm

from master2 import params

params['sp'].save_to_disk = True
params['tp'].prescription = 'general_telescope'
params['sp'].numframes = 1
params['ap'].n_wvl_init = 3
params['ap'].n_wvl_final = 3
params['sp'].memory_limit = 9.e9 /15
params['sp'].save_list = np.array(['atmosphere', 'add_aber', 'deformable mirror', 'add_aber', 'pre_coron', 'detector'])
save_labels = np.array(['Entrance Pupil', 'After CPA', 'After DM', 'After NCPA', 'Before Occult.', 'Detector'])
params['tp'].use_atmos = True

if __name__ == '__main__':  # required for multiprocessing - make sure globals are set before though

    sim = mm.RunMedis(params=params, name='figure1_2', product='fields')
    observation = sim()
    cpx_sequence = observation['fields']

    # fp_sampling = sampling[-1][:]
    # print(fp_sampling, cpx_sequence.shape)
    spectral_train_phase = np.angle(cpx_sequence[0,:-2,:,0])
    spectral_train_amp = np.abs(cpx_sequence[0,-2:,:,0]**2)
    spectral_train_grid = np.concatenate((spectral_train_phase,spectral_train_amp), axis=0)
    crop = spectral_train_grid.shape[2] // 4
    spectral_train_grid = spectral_train_grid[:,:,crop:-crop,crop:-crop]
    fig, axes = plt.subplots(6,3, figsize=(4,7))

    for j in range(3):
        for i in range(2):
            im1 = axes[i, j].imshow(spectral_train_grid[i, j], cmap=sunlight, vmin=-np.pi, vmax=np.pi)
        for i in range(2,4):
            im2 = axes[i, j].imshow(spectral_train_grid[i, j], cmap=sunlight, vmin=-1, vmax=1)
        for i in range(4,6):
            im3 = axes[i,j].imshow(spectral_train_grid[i,j], cmap='inferno', norm=LogNorm(), vmin=1e-8, vmax=1e-3)

    [[axes[i, j].axis('off') for i in range(6)] for j in range(3)]

    axes[0,0].set_title('800 nm')
    axes[0,1].set_title('1150 nm')
    axes[0,2].set_title('1500 nm')

    props = dict(boxstyle='square', facecolor='k', alpha=0.5)
    for j, text in enumerate(save_labels):
        axes[j, 0].text(0.05, 0.075, text, transform=axes[j, 0].transAxes, fontweight='bold',
                         color='w', fontsize=7.5, bbox=props)

    cax1 = fig.add_axes([0.86, 0.645, 0.02, 0.29])
    cax2 = fig.add_axes([0.86, 0.325, 0.02, 0.29])
    cax3 = fig.add_axes([0.86, 0.0145, 0.02, 0.29])
    cbar1 = plt.colorbar(im1, cax1)
    cbar2 = plt.colorbar(im2, cax2)
    cbar3 = plt.colorbar(im3, cax3)

    cbar1.ax.tick_params(labelsize=8)
    cbar2.ax.tick_params(labelsize=9)
    cbar3.ax.tick_params(labelsize=10)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.0, bottom=0.01, right=0.84, top=0.935, wspace=0.1, hspace=0.15)

    plt.savefig('/Users/dodkins/MEDIS2train.pdf', dpi=1000)
    plt.show(block=True)
