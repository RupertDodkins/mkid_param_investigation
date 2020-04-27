import os, sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

from vip_hci import phot, metrics, pca
from vip_hci.metrics.contrcurve import noise_per_annulus, contrast_curve

from medis.telescope import Telescope
from medis.MKIDs import Camera
from medis.utils import dprint
from medis.plot_tools import quick2D, view_spectra, body_spectra

from master2 import params
import metrics
from diagrams import contrcurve_plot, combo_performance
from substitution import get_form_photons

mode = 'develop'
# mode = 'test'

if mode == 'develop':
    params['ap'].n_wvl_init = 2
    params['ap'].n_wvl_final = 2
    params['sp'].numframes = 1
else:
    params['ap'].n_wvl_init = 8
    params['ap'].n_wvl_final = 16
    params['sp'].numframes = 10

investigation = f'figure3_{mode}'

class ObservatoryMaster():
    """ Each repeat has new fields and a median device params to seed from, as well as noise data for scaling """
    def __init__(self, params, iteration=0):
        self.params = params
        self.name = str(iteration)
        self.masterdir = os.path.join(params['iop'].datadir, investigation, self.name, 'master')
        self.save_extension = 'camera.pkl'
        self.fields_extension = 'fields.pkl'

        self.params['iop'].median_noise = os.path.join(self.masterdir, 'median_noise_master.txt')
        self.params['iop'].camera = os.path.join(self.masterdir, self.save_extension)
        self.params['iop'].fields = os.path.join(self.masterdir, self.fields_extension)

        self.fields = self.make_fields_master()

        dprint(self.params['iop'].fields)
        self.cam = Camera(params, fields=self.fields)

        if not os.path.exists(self.params['iop'].median_noise):
            self.get_median_noise()

    def get_median_noise(self, plot=False):
        wsamples = np.linspace(self.params['ap'].wvl_range[0], self.params['ap'].wvl_range[1], self.params['ap'].n_wvl_final)
        scale_list = wsamples / (self.params['ap'].wvl_range[1] - self.params['ap'].wvl_range[0])

        # if not hasattr(self.cam, 'stackcube'):
        self.cam = get_form_photons(self.fields, self.cam, comps=False)

        frame_nofc = pca.pca(self.cam.stackcube, angle_list=np.zeros((self.cam.stackcube.shape[1])),
                             scale_list=scale_list, mask_center_px=None, adimsdi='double', ncomp=7, ncomp2=None,
                             collapse='median')

        if plot:
            quick2D(frame_nofc, logZ=True)

        fwhm = self.params['mp'].lod
        mask = self.cam.QE_map == 0
        median_noise, vector_radd = noise_per_annulus(frame_nofc, separation=fwhm, fwhm=fwhm, mask=mask)
        np.savetxt(self.params['iop'].median_noise, median_noise)

    def make_fields_master(self, plot=False):
        """ The master fields file of which all the photons are seeded from according to their device

        :return:
        """

        # sim = mm.RunMedis(params=self.params, name=self.masterdir, product='fields')
        # observation = sim()
        self.params['iop'].update(self.masterdir)
        telescope = Telescope(params)
        observation = telescope()

        obs_seq = np.abs(observation['fields'][:, -1]) ** 2

        fields_master = np.zeros((self.params['sp'].numframes, self.params['ap'].n_wvl_final, 2, self.params['sp'].grid_size,
                                self.params['sp'].grid_size))
        collapse_comps = np.sum(obs_seq[:, :, 1:], axis=2)
        fields_master[:, :, 0] = obs_seq[:, :, 0]
        fields_master[:, :, 1] = collapse_comps
        if plot:
            view_spectra(fields_master[0,:,0], logZ=True, show=True, title='star cube timestep 0')
            view_spectra(fields_master[0,:,1], logZ=True, show=True, title='planets cube timestep 0')

        dprint(f"Reduced shape of obs_seq = {np.shape(fields_master)} (numframes x nwsamp x 2 x grid x grid)")

        os.rename(self.params['iop'].fields, os.path.join(self.masterdir, 'fields_planet_slices.h5'))
        telescope.save_fields(fields_master)

        return fields_master

class MetricTester():
    """ This instrument has the magical ability to quickly tune one device parameter during an observation """
    def __init__(self, obs, metric):

        self.params = obs.params

        self.wvl_range = obs.params['ap'].wvl_range
        self.n_wvl_final = obs.params['ap'].n_wvl_final
        self.lod = obs.params['mp'].lod
        self.master_cam = obs.cam
        self.master_fields = obs.fields

        self.metric = metric
        self.testdir = metric.testdir

        self.performance_data = os.path.join(self.testdir, 'performance_data.pkl')

    def __call__(self, debug=True):
        # if debug:
        #     check_contrast_contriubtions(self.metric.vals, metric_name, self.master_input, comps=False)

        dprint(self.performance_data)
        if not os.path.exists(self.performance_data):
            # self.metric.create_adapted_cams()

            comps_ = [True, False]
            pca_products = []
            for comps in comps_:
                if hasattr(self.metric, 'get_stackcubes'):
                    self.metric.get_stackcubes(self.master_fields, comps=comps)
                else:
                    self.get_stackcubes(self.master_fields, comps=comps)

                pca_products.append(self.pca_stackcubes(comps))

            maps = pca_products[0]
            rad_samps = pca_products[1][1]
            thruputs = pca_products[1][2]
            noises = pca_products[1][3]
            conts = pca_products[1][4]

            with open(self.performance_data, 'wb') as handle:
                pickle.dump((maps, rad_samps, thruputs, noises, conts, self.metric.vals), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.performance_data, 'rb') as handle:
                performance_data = pickle.load(handle)
                if len(performance_data) == 6:
                    maps, rad_samps, thruputs, noises, conts, self.metric.vals = performance_data
                else:
                    maps, rad_samps, conts, self.metric.vals = performance_data

        if debug:
            try:
                contrcurve_plot(self.metric.vals, rad_samps, thruputs, noises, conts)
                if self.metric.name != 'array_size':
                    body_spectra(maps, logZ=False, title=self.metric.name)
                else:
                    pass
            except UnboundLocalError:
                dprint('thruputs and noises not saved in old versions :(')
                # raise UnboundLocalError
                pass

            combo_performance(maps, rad_samps, conts, self.metric.vals, self.metric.name, [0,-1], savedir=self.testdir)

        return {'maps': maps, 'rad_samps':rad_samps, 'conts':conts}

    def get_stackcubes(self, master_fields, comps=True):
        obj = 'comp' if comps else 'star'
        for i, cam, metric_val in zip(range(len(self.metric.cams[obj])), self.metric.cams[obj], self.metric.vals):

            cam = get_form_photons(master_fields, cam, comps=comps)

            self.metric.cams[obj][i] = cam

    def pca_stackcubes(self, comps=True):
        wsamples = np.linspace(self.wvl_range[0], self.wvl_range[1], self.n_wvl_final)
        scale_list = wsamples / (self.wvl_range[1] - self.wvl_range[0])
        maps = []

        if comps:
            for cam in self.metric.cams['comp']:
                dprint(cam.stackcube.shape)
                SDI = pca.pca(cam.stackcube, angle_list=np.zeros((cam.stackcube.shape[1])), scale_list=scale_list,
                              mask_center_px=None, adimsdi='double', ncomp=7, ncomp2=None,
                              collapse='median')
                maps.append(SDI)
            return maps

        else:
            rad_samps, thruputs, noises, conts = [], [], [], []
            # for stackcube, dp in zip(stackcubes, dps):
            for cam in self.metric.cams['star']:
                unoccultname = os.path.join(self.testdir, f'telescope_unoccult_arraysize={cam.array_size}.pkl')
                psf_template = self.get_unoccult_psf(unoccultname)
                # star_phot = phot.contrcurve.aperture_flux(np.sum(psf_template, axis=0), [sp.grid_size // 2],
                #                                           [sp.grid_size // 2], mp.lod, 1)[0]*10**-3#1.5
                star_phot = 1.1
                dprint(star_phot)
                # body_spectra(psf_template, logZ=True)
                algo_dict = {'scale_list': scale_list}
                # temp for those older format cache files
                if hasattr(cam, 'lod'):
                    fwhm = cam.lod
                    dprint(fwhm)
                else:
                    fwhm = self.lod
                    dprint(fwhm)
                method_out = self.eval_method(cam.stackcube, pca.pca, psf_template,
                                         np.zeros((cam.stackcube.shape[1])), algo_dict,
                                         fwhm=fwhm, star_phot=star_phot, cam=cam)

                thruput, noise, cont, sigma_corr, dist = method_out[0]
                thruputs.append(thruput)
                noises.append(noise)
                conts.append(cont)
                rad_samp = cam.platescale * dist
                rad_samps.append(rad_samp)
                maps.append(method_out[1])
            plt.show(block=True)
            return maps, rad_samps, thruputs, noises, conts

    def eval_method(self, cube, algo, psf_template, angle_list, algo_dict, cam, fwhm=6, star_phot=1):
        dprint(fwhm, star_phot)
        fulloutput = contrast_curve(cube=cube, interp_order=2, angle_list=angle_list, psf_template=psf_template,
                                    fwhm=fwhm, pxscale=cam.platescale / 1000, starphot=star_phot, algo=algo, nbranch=1,
                                    adimsdi='double', ncomp=7, ncomp2=None, debug=True, plot=True, theta=0,
                                    full_output=True, fc_snr=100, # wedge=(-45, 45), int(dp.lod[0])
                                    cam=cam, **algo_dict)
        metrics_out = [fulloutput[0]['throughput'], fulloutput[0]['noise'], fulloutput[0]['sensitivity_student'],
                       fulloutput[0]['sigma corr'], fulloutput[0]['distance']]
        metrics_out = np.array(metrics_out)
        return metrics_out, fulloutput[2]

    def get_unoccult_psf(self, name):
        params = copy.deepcopy(self.params)
        params['sp'].save_fields = True
        params['ap'].companion = False
        params['tp'].cg_type = None
        params['sp'].numframes = 1
        params['ap'].sample_time = 1e-3
        params['sp'].save_list = ['detector']

        params['iop'].update(self.metric.testdir)
        params['iop'].telescope = name

        telescope = Telescope(self.params)
        observation = telescope()
        fields = observation['fields']
        psf_template = np.abs(fields[0, -1, :, 0, 1:, 1:]) ** 2
        # view_spectra(psf_template, logZ=True)

        return psf_template

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# def config_images(num_tests):
#     plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, num_tests)))

def parse_cont_data(all_cont_data, p):
    rad_samps = all_cont_data[0, p, 0]  # both repeats should be equivalent
    all_conts = np.array(all_cont_data[:, p, 1].tolist())
    mean_conts = np.mean(all_conts, axis=0)
    if len(all_conts.shape)==3:
        err_conts = np.std(all_conts, axis=0)
    else:
        err_conts = [np.std(all_conts[:,i])/np.sqrt(len(all_conts)) for i in range(len(all_conts[0]))]  #can't do std if one axis is different size (unlike mean)

    return rad_samps, mean_conts, err_conts

if __name__ == '__main__':

    # define the configuration
    repeats = 1  # number of medis runs to average over for the cont plots
    # metric_names = ['numframes', 'array_size', 'pix_yield', 'dark_bright', 'R_mean', 'g_mean']  # 'g_mean_sig']# 'star_flux', 'exp_time', 'array_size_(rebin)',
    metric_names = ['dark_bright']

    # collect the data
    all_cont_data = []
    for r in range(repeats):

        obs = ObservatoryMaster(params, iteration=r)

        dprint(params['iop'].testdir)

        comp_images, cont_data, metric_multi_list, metric_vals_list = [], [], [], []
        for metric_name in metric_names:

            # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, len(param.metric_multiplier))))

            metric_config = metrics.get_metric(metric_name, master_cam=obs.cam)
            metric_test = MetricTester(obs, metric_config)
            metric_results = metric_test()
            
            comp_images.append(metric_results['maps'])
            cont_data.append([metric_results['rad_samps'],metric_results['conts']])

            # store the mutlipliers but flip those that achieve better contrast when the metric is decreasing
            if metric_name in ['R_sig', 'g_sig', 'dark_bright']:  # 'dark_bright',
                metric_multi_list.append(metric_config.multiplier[::-1])
                metric_vals_list.append(metric_config.vals[::-1])
            else:
                metric_multi_list.append(metric_config.multiplier)
                metric_vals_list.append(metric_config.vals)

        cont_data = np.array(cont_data)
        dprint(cont_data.shape)
        all_cont_data.append(cont_data)
    all_cont_data = np.array(all_cont_data)  # (repeats x num_params x rad+cont x num_multi (changes)

    # plot the data
    three_lod_sep = 0.3
    six_lod_sep = 2 * three_lod_sep
    fhqm = 0.03
    for p, param_name in enumerate(metric_names):
        metric_multi = metric_multi_list[p]
        metric_vals = metric_vals_list[p]
        rad_samps, mean_conts, err_conts = parse_cont_data(all_cont_data, p)

        three_lod_conts = np.zeros((len(metric_multi)))
        six_lod_conts = np.zeros((len(metric_multi)))
        three_lod_errs = np.zeros((len(metric_multi)))
        six_lod_errs = np.zeros((len(metric_multi)))
        for i in range(len(mean_conts)):
            three_lod_ind = np.where(
                (np.array(rad_samps[i]) > three_lod_sep - fhqm) & (np.array(rad_samps[i]) < three_lod_sep + fhqm))
            three_lod_conts[i] = np.mean(mean_conts[i][three_lod_ind])
            three_lod_errs[i] = np.sqrt(np.sum(err_conts[i][three_lod_ind] ** 2))

            six_lod_ind = np.where(
                (np.array(rad_samps[i]) > six_lod_sep - fhqm) & (np.array(rad_samps[i]) < six_lod_sep + fhqm))
            six_lod_conts[i] = np.mean(mean_conts[i][six_lod_ind])
            six_lod_errs[i] = np.sqrt(np.sum(err_conts[i][six_lod_ind] ** 2))

        maps = comp_images[p]
        combo_performance(maps, rad_samps, mean_conts, metric_vals, param_name, [0, -1], err_conts, metric_multi,
                                 three_lod_conts, three_lod_errs, six_lod_conts, six_lod_errs)
