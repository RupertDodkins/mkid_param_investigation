'''Example Code for conducting SDI with MKIDs'''

import os
import matplotlib as mpl
import numpy as np
mpl.use("Qt5Agg")
import matplotlib.pylab as plt
import copy as copy
import pickle as pickle

import medis.medis_main as mm
# from medis.params import ap, iop
from medis.plot_tools import quick2D, view_spectra
from medis.utils import dprint

from master2 import get_form_photons

# metric_name = __file__.split('/')[-1].split('.')[0]
#

#
#
# median_val = 10
# metric_multiplier = np.logspace(np.log10(0.2), np.log10(5), 7)
# metric_vals = np.int_(np.round(median_val * metric_multiplier))
#
# # iop.set_testdir(f'{os.path.dirname(iop.testdir[:-1])}/{metric_name}/')

class MetricConfigur():
    def __init__(self, master_cam, testdir):
        self.name = __file__.split('/')[-1].split('.')[0]

        median_val = 10
        metric_multiplier = np.logspace(np.log10(0.2), np.log10(5), 7)
        self.vals = np.int_(np.round(median_val * metric_multiplier))
        self.master_cam = master_cam
        self.testdir = testdir
        self.cams = []

        if not os.path.exists(self.testdir):
            os.mkdir(self.testdir)

    def create_adapted_cams(self):
        # with open(self.master_dp, 'rb') as handle:
        #     dp = pickle.load(handle)
        # self.device = os.path.join(self.testdir, f'device_{self.name}={self.vals}.pkl')

        # cam_filename = os.path.join(self.testdir, f'device_{self.name}={self.vals}.pkl')
        # iop.device = iop.device[:-4] + '_'+self.name
        # iop.device = iop.device.split('_' + self.name)[0] + f'_{self.name}={self.vals}.pkl'
        # with open(cam_filename, 'wb') as handle:
        #     pickle.dump(self.new_cam, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for i, val in enumerate(self.vals):
            # cam_filename = os.path.join(self.testdir, f'device_{self.name}={val}.pkl')
            new_cam = copy.copy(self.master_cam)
            new_cam.name = os.path.join(self.testdir, f'camera_{self.name}={val}.pkl')
            self.cams.append(new_cam)

    def get_stackcubes(self, master_fields, comps=True, plot=False):

        for i, cam, metric_val in zip(range(len(self.cams)), self.cams, self.vals):
            reduced_fields = master_fields[:metric_val]
            if not hasattr(cam, 'stackcube'):
                cam = get_form_photons(reduced_fields, cam, comps=comps)

            if plot:
                plt.figure()
                plt.hist(cam.stackcube[cam.stackcube!=0].flatten(), bins=np.linspace(0,1e4, 50))
                plt.yscale('log')
                view_spectra(cam.stackcube[0], logZ=True, show=False)
                view_spectra(cam.stackcube[:, 0], logZ=True, show=False)

            cam.stackcube /= np.sum(cam.stackcube)  # /sp.numframes
            cam.stackcube = np.transpose(cam.stackcube, (1, 0, 2, 3))
            self.cams[i] = cam

        # return self.cams

# def detect_obj_photons(vals, name, plot=False):
#     iop.device = iop.device[:-4] + '_'+name
#     iop.form_photons = iop.form_photons[:-4] +'_'+name
#
#     iop.fields = master.master_fields
#     fields = gpd.run_medis()
#
#     objcubes, dps =  [], []
#     # view_spectra(fields[0,:,1], logAmp=True)
#     iop.device = iop.device.split('_' + name)[0] + f'_{name}={vals}.pkl'
#     for metric_val in vals:
#         iop.form_photons = iop.form_photons.split('_'+name)[0] + f'_{name}={metric_val}_obj.pkl'
#         reduced_fields = fields[:metric_val]
#
#         # fcy = np.array([256, 196, 256, 336, 256, 157, 256, 376])
#         # fcx = np.array([207, 256, 327, 256, 167, 256, 367, 256])
#         # from vip_hci.metrics.contrcurve import noise_per_annulus, aperture_flux
#         # injected_flux = aperture_flux(np.mean(reduced_fields[:,:,1], axis=(0, 1)), fcy, fcx, 4, ap_factor=1)
#         # plt.plot(injected_flux)
#         # plt.show(block=True)
#
#         if os.path.exists(iop.form_photons):
#             dprint(f'Formatted photon data already exists at {iop.form_photons}')
#             with open(iop.form_photons, 'rb') as handle:
#                 objcube, dp = pickle.load(handle)
#         else:
#             objcube, dp = master.get_obj_photons(reduced_fields)
#
#         # fcy = np.array([75,  44,  75, 116,  75,  23,  75, 137])
#         # fcx = np.array([49,  75, 111,  75,  28,  75, 132,  75])
#         # injected_flux = aperture_flux(np.mean(objcube[:,:,1], axis=(0, 1)), fcy, fcx, 4, ap_factor=1)
#         # plt.plot(injected_flux)
#         # plt.show(block=True)
#
#         if plot:
#             dprint(objcube.shape)
#             for o in range(3):
#                 plt.figure()
#                 plt.hist(objcube[o, objcube[o]!=0].flatten(), bins=np.linspace(0,1e4, 50))
#                 plt.yscale('log')
#                 view_spectra(objcube[o, 0], logAmp=True, show=False)
#                 view_spectra(objcube[o, :, 0], logAmp=True, show=False)
#
#         objcube /= np.sum(objcube)  # /sp.numframes
#         objcube = objcube
#         objcube = np.transpose(objcube, (0, 2, 1, 3, 4))
#         objcubes.append(objcube)
#         dps.append(dp)
#
#     return objcubes, dps

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