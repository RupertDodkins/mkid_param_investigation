'''Example Code for conducting SDI with MKIDs'''

import os
import matplotlib as mpl
import numpy as np
mpl.use("Qt5Agg")
import matplotlib.pylab as plt
import copy as copy
import pickle as pickle
from medis.params import mp, ap, iop
from medis.plot_tools import quick2D, view_spectra
from medis.utils import dprint
import medis.Detector.mkid_artefacts as MKIDs
#import master

metric_name = __file__.split('/')[-1].split('.')[0]
metric_vals = [1, 50, 1000]

master.set_field_params()
master.set_mkid_params()

iop.set_testdir(f'FirstPrincipleSim/{metric_name}')
iop.set_atmosdata('190823')
iop.set_aberdata('Palomar512')

print(sp.numframes)

comps = False
mp.hot_pix = True

def adapt_dp_master():
    if not os.path.exists(iop.testdir):
        os.mkdir(iop.testdir)
    with open(master.dp, 'rb') as handle:
        dp = pickle.load(handle)
    iop.device = iop.device[:-4] + '_'+metric_name
    new_dp = copy.copy(dp)
    for metric_val in metric_vals:
        new_dp.hot_pix = metric_val
        new_dp.hot_locs = MKIDs.create_false_pix(mp, amount=new_dp.hot_pix)
        new_dp.hot_per_step = int(np.round(ap.sample_time * mp.hot_bright *new_dp.hot_pix))
        dprint((new_dp.hot_locs, new_dp.hot_per_step))
        iop.device = iop.device.split('_'+metric_name)[0] + f'_{metric_name}={metric_val}.pkl'
        dprint((iop.device, metric_val))
        with open(iop.device, 'wb') as handle:
            pickle.dump(new_dp, handle, protocol=pickle.HIGHEST_PROTOCOL)

def form():
    if not os.path.exists(f'{iop.device[:-4]}_{metric_name}={metric_vals[0]}.pkl'):
        adapt_dp_master()
    # stackcubes, dps = get_stackcubes(metric_vals, metric_name, comps=comps, plot=True)
    # master.eval_performance(stackcubes, dps, metric_vals, comps=comps)

    comps_ = [True, False]
    pca_products = []
    for comps in comps_:
        stackcubes, dps = master.get_stackcubes(metric_vals, metric_name, comps=comps, plot=False)
        pca_products.append(master.pca_stackcubes(stackcubes, dps, comps))

    maps = pca_products[0]
    rad_samps = pca_products[1][1]
    conts = pca_products[1][4]

    master.combo_performance(maps, rad_samps, conts, metric_vals)

if __name__ == '__main__':
    form()
    # if not os.path.exists(f'{iop.device[:-4]}_{metric_name}={metric_vals[0]}.pkl'):
    #     adapt_dp_master()
    # stackcubes, dps = master.get_stackcubes(metric_vals, metric_name, comps=comps)
    # # plt.show(block=True)
    # master.eval_performance(stackcubes, dps, metric_vals, comps=comps)