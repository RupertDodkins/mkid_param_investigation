'''Example Code for conducting SDI with MKIDs'''

import os
import matplotlib as mpl
import numpy as np
mpl.use("Qt5Agg")
import matplotlib.pylab as plt
import copy as copy
import pickle as pickle
from medis.params import ap, iop

from medis.plot_tools import quick2D, view_spectra
from medis.utils import dprint
import medis.Detector.readout as read
#import master

metric_name = __file__.split('/')[-1].split('.')[0]

master.set_field_params()
master.set_mkid_params()

median_val = ap.sample_time
metric_multiplier = np.logspace(np.log10(1), np.log10(sp.numframes), 3)
metric_vals = np.around(median_val * metric_multiplier, decimals=1)

dprint(metric_vals)
iop.set_testdir(f'{os.path.dirname(iop.testdir[:-1])}/{metric_name}/')

def adapt_dp_master():
    if not os.path.exists(iop.testdir):
        os.mkdir(iop.testdir)
    with open(master.master_dp, 'rb') as handle:
        dp = pickle.load(handle)
    iop.device = iop.device[:-4] + '_'+metric_name
    iop.device = iop.device.split('_' + metric_name)[0] + f'_{metric_name}={metric_vals}.pkl'
    new_dp = copy.copy(dp)
    with open(iop.device, 'wb') as handle:
        pickle.dump(new_dp, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_stackcubes(metric_vals, metric_name, master_cache, comps=True, plot=False):
    _, master_fields = master_cache

    dprint(iop.device)
    dprint(iop.form_photons)
    dprint(iop.testdir)
    dprint(master_fields)

    iop.fields = master.master_fields

    dprint(iop.device)
    dprint(iop.form_photons)
    dprint(iop.testdir)
    dprint(master_fields)

    iop.device = iop.device[:-4] + '_'+metric_name
    iop.form_photons = iop.form_photons[:-4] +'_'+metric_name

    dprint(iop.device)
    dprint(iop.form_photons)
    dprint(iop.testdir)
    dprint(master_fields)

    fields = mm.RunMedis('field).telescope()[0]

    stackcubes, dps =  [], []
    iop.device = iop.device.split('_'+metric_name)[0] + f'_{metric_name}={metric_vals}.pkl'
    for metric_val in metric_vals:
        dprint(metric_val)
        ap.exposure_time = metric_val
        iop.form_photons = iop.form_photons.split('_'+metric_name)[0] + f'_{metric_name}={metric_val}_comps={comps}.pkl'
        reduced_fields = read.take_fields_exposure(fields)
        dprint(reduced_fields.shape)

        if os.path.exists(iop.form_photons):
            dprint(f'Formatted photon data already exists at {iop.form_photons}')
            with open(iop.form_photons, 'rb') as handle:
                stackcube, dp = pickle.load(handle)

        else:
            stackcube, dp = master.get_form_photons(reduced_fields, comps=comps)

        if plot:
            plt.figure()
            plt.hist(stackcube[stackcube!=0].flatten(), bins=np.linspace(0,1e4, 50))
            plt.yscale('log')
            dprint(stackcube.shape)
            view_spectra(stackcube[0], logAmp=True, show=False)
            view_spectra(stackcube[:, 0], logAmp=True, show=False)

        stackcube /= np.sum(stackcube)  # /sp.numframes
        stackcube = stackcube
        stackcube = np.transpose(stackcube, (1, 0, 2, 3))
        stackcubes.append(stackcube)
        dps.append(dp)

    return stackcubes, dps

def detect_obj_photons(metric_vals, metric_name, plot=False):
    iop.device = iop.device[:-4] + '_'+metric_name
    iop.form_photons = iop.form_photons[:-4] +'_'+metric_name

    iop.fields = master.master_fields
    fields = gpd.run_medis()

    objcubes, dps =  [], []
    # view_spectra(fields[0,:,1], logAmp=True)
    iop.device = iop.device.split('_' + metric_name)[0] + f'_{metric_name}={metric_vals}.pkl'
    for metric_val in metric_vals:
        iop.form_photons = iop.form_photons.split('_'+metric_name)[0] + f'_{metric_name}={metric_val}_obj.pkl'
        reduced_fields = fields[:metric_val]

        # fcy = np.array([256, 196, 256, 336, 256, 157, 256, 376])
        # fcx = np.array([207, 256, 327, 256, 167, 256, 367, 256])
        # from vip_hci.metrics.contrcurve import noise_per_annulus, aperture_flux
        # injected_flux = aperture_flux(np.mean(reduced_fields[:,:,1], axis=(0, 1)), fcy, fcx, 4, ap_factor=1)
        # plt.plot(injected_flux)
        # plt.show(block=True)

        if os.path.exists(iop.form_photons):
            dprint(f'Formatted photon data already exists at {iop.form_photons}')
            with open(iop.form_photons, 'rb') as handle:
                objcube, dp = pickle.load(handle)
        else:
            objcube, dp = master.get_obj_photons(reduced_fields)

        # fcy = np.array([75,  44,  75, 116,  75,  23,  75, 137])
        # fcx = np.array([49,  75, 111,  75,  28,  75, 132,  75])
        # injected_flux = aperture_flux(np.mean(objcube[:,:,1], axis=(0, 1)), fcy, fcx, 4, ap_factor=1)
        # plt.plot(injected_flux)
        # plt.show(block=True)

        if plot:
            dprint(objcube.shape)
            for o in range(3):
                plt.figure()
                plt.hist(objcube[o, objcube[o]!=0].flatten(), bins=np.linspace(0,1e4, 50))
                plt.yscale('log')
                view_spectra(objcube[o, 0], logAmp=True, show=False)
                view_spectra(objcube[o, :, 0], logAmp=True, show=False)

        objcube /= np.sum(objcube)  # /sp.numframes
        objcube = objcube
        objcube = np.transpose(objcube, (0, 2, 1, 3, 4))
        objcubes.append(objcube)
        dps.append(dp)

    return objcubes, dps

# def form2():
#     if not os.path.exists(f'{iop.device[:-4]}_{metric_name}={metric_vals[0]}.pkl'):
#         adapt_dp_master()
#     objcubes, dps = detect_obj_photons(metric_vals, metric_name, plot=False)
#
#     maps, rad_samps, conts = [], [], []
#     # fwhm = np.linspace(2,6,ap.n_wvl_final)
#     for objcube, dp in zip(objcubes, dps):
#         cont_data = contrcurve(objcube, dp=dp)
#         maps.append(cont_data[0])
#         rad_samps.append(cont_data[1])
#         conts.append(cont_data[2])
#
#     master.combo_performance(maps, rad_samps, conts, metric_vals)

# def plot_sum_perf():
#     comps = False
#     stackcubes, dps = get_stackcubes(metric_vals, metric_name, comps=comps)
#     # master.eval_performance(stackcubes, dps, metric_vals, comps=comps)
#     master.eval_performance_sum(stackcubes, dps, metric_vals, comps=comps)

if __name__ == '__main__':
    master.check_contrast_contriubtions(metric_vals, metric_name)
    # plot_sum_perf()