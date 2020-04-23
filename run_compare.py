import os, sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pickle

from vip_hci import phot, metrics, pca
from vip_hci.metrics.contrcurve import noise_per_annulus

import medis.MKIDs as mkids
import medis.medis_main as mm
from medis.telescope import Telescope
from medis.utils import dprint
from medis.plot_tools import quick2D, quicklook_wf, view_spectra

from master2 import params, get_form_photons

params['tp'].prescription = 'general_telescope'
params['sp'].save_to_disk = True
params['sp'].numframes = 1
params['ap'].n_wvl_init = 3
params['ap'].n_wvl_final = 3

investigation = 'figure3_2'


class ObservatoryMaster():
    """ Each repeat has new fields and a median device params to seed from, as well as noise data for scaling """
    def __init__(self, params, iteration=0):
        self.params = params
        self.name = str(iteration)
        self.masterdir = os.path.join(params['iop'].datadir, investigation, self.name, 'master')
        self.save_extension = 'camera.pkl'
        self.fields_extension = 'fields.pkl'

        # self.params['iop'].update(self.masterdir)
        # self.params['iop'].form_photons = os.path.join(self.masterdir, 'formatted_photons_master.pkl')
        # self.params['iop'].device = os.path.join(self.masterdir, 'deviceParams_master.pkl')
        self.params['iop'].median_noise = os.path.join(self.masterdir, 'median_noise_master.txt')
        self.params['iop'].camera = os.path.join(self.masterdir, self.save_extension)
        self.params['iop'].fields = os.path.join(self.masterdir, self.fields_extension)

        self.fields = self.make_fields_master()

        # make master dp
        # if not os.path.exists(self.params['iop'].device):
        #     self.dp = mkids.initialize()
        # else:
        #     with open(self.params['iop'].device, 'rb') as handle:
        #         self.dp = pickle.load(handle)
        dprint(self.params['iop'].fields)
        self.cam = mkids.Camera(params, fields=self.fields)

        if not os.path.exists(self.params['iop'].median_noise):
            self.get_median_noise()

    def get_median_noise(self):
        wsamples = np.linspace(self.params['ap'].wvl_range[0], self.params['ap'].wvl_range[1], self.params['ap'].n_wvl_final)
        scale_list = wsamples / (self.params['ap'].wvl_range[1] - self.params['ap'].wvl_range[0])

        # sim = mm.RunMedis(params=self.params, name=self.masterdir, product='fields')
        # observation = sim()
        # fields = observation['fields'][:,-1]  # slice out final plane
        # fields = np.abs(fields**2)
        # # fields = mkids.load_fields()

        dprint(self.params['iop'].fields, self.params['iop'].device)
        # comps = False

        # cam = mkids.Camera(self.params['iop'], self.doublecube)
        # stackcube = cam.stackcube
        # if os.path.exists(self.params['iop'].photons):
        #     dprint(f"Formatted photon data already exists at {self.params['iop'].form_photons}")
        #     with open(self.params['iop'].form_photons, 'rb') as handle:
        #         stackcube, dp = pickle.load(handle)
        # else:
        #     # stackcube, dp = self.get_form_photons(comps=comps)
        #     stackcube, dp = get_form_photons(self.fields, self.dp, os.path.join(self.masterdir,'photons.pkl'),
        #                                      self.params['mp'], comps=comps)
        if not hasattr(self.cam, 'stackcube'):
            self.cam = get_form_photons(self.fields, self.cam, comps=False)

        self.cam.stackcube /= np.sum(self.cam.stackcube)  # /sp.numframes
        self.cam.stackcube = np.transpose(self.cam.stackcube, (1, 0, 2, 3))

        frame_nofc = pca.pca(self.cam.stackcube, angle_list=np.zeros((self.cam.stackcube.shape[1])), scale_list=scale_list,
                      mask_center_px=None, adimsdi='double', ncomp=7, ncomp2=None,
                      collapse='median')

        # quick2D(frame_nofc, logAmp=True)
        fwhm = self.params['mp'].lod
        # with open(self.params['iop'].device, 'rb') as handle:
        #     dp = pickle.load(handle)

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
    def __init__(self, obs, metric, debug=True):
        self.params = obs.params
        self.master_cam = obs.cam
        self.master_fields = obs.fields

        self.metric = metric

    def __call__(self, debug=True):
        # if debug:
        #     check_contrast_contriubtions(self.metric.vals, metric_name, self.master_input, comps=False)

        self.params['iop'].performance_data = os.path.join(self.params['iop'].testdir, 'performance_data.pkl')
        dprint(self.params['iop'].performance_data)
        if not os.path.exists(self.params['iop'].performance_data):
            # import importlib
            # param = importlib.import_module(self.metric.name)

            # if not os.path.exists(f"{self.params['iop'].device[:-4]}_{self.metric.name}={self.metric.vals[0]}.pkl"):
            self.metric.create_adapted_cams()

            comps_ = [True, False]
            pca_products = []
            for comps in comps_:
                if hasattr(self.metric, 'get_stackcubes'):
                # if 'get_stackcubes' in dir(param):
                    self.metric.get_stackcubes(self.master_fields, comps=comps, plot=False)
                else:
                    self.get_stackcubes(self.master_fields, comps=comps, plot=False)

                if hasattr(self.metric, 'pca_stackcubes'):
                # if 'pca_stackcubes' in dir(param):
                    pca_products.append(self.metric.pca_stackcubes(stackcubes, dps, comps))
                else:
                    pca_products.append(self.pca_stackcubes(stackcubes, dps, comps))

            maps = pca_products[0]
            rad_samps = pca_products[1][1]
            thruputs = pca_products[1][2]
            noises = pca_products[1][3]
            conts = pca_products[1][4]

            with open(self.params['iop'].performance_data, 'wb') as handle:
                pickle.dump((maps, rad_samps, thruputs, noises, conts, self.metric.vals), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.params['iop'].performance_data, 'rb') as handle:
                performance_data = pickle.load(handle)
                if len(performance_data) == 6:
                    maps, rad_samps, thruputs, noises, conts, self.metric.vals = performance_data
                else:
                    maps, rad_samps, conts, self.metric.vals = performance_data

        if debug:
            try:
                contrcurve_plot(self.metric.vals, rad_samps, thruputs, noises, conts)
                view_spectra(maps, logAmp=True, vmin=-1e-7, vmax=1e-7, show=False)
            except UnboundLocalError:
                dprint('thruputs and noises not saved in old versions :(')
                # raise UnboundLocalError
                pass

            combo_performance(maps, rad_samps, conts, self.metric.vals, self.metric.name, [0,-1])

        return {'maps': maps, 'rad_samps':rad_samps, 'conts':conts}

    def get_stackcubes(self, master_fields, comps=True, plot=False):

        for i, cam, metric_val in zip(range(len(self.metric.cams)), self.metric.cams, self.metric.vals):

            if not hasattr(cam, 'stackcube'):
                cam = get_form_photons(master_fields, cam, comps=comps)

            if plot:
                plt.figure()
                plt.hist(cam.stackcube[cam.stackcube!=0].flatten(), bins=np.linspace(0,1e4, 50))
                plt.yscale('log')
                view_spectra(cam.stackcube[0], logAmp=True, show=False)
                view_spectra(cam.stackcube[:, 0], logAmp=True, show=True)

            cam.stackcube /= np.sum(cam.stackcube)  # /sp.numframes
            cam.stackcube = np.transpose(cam.stackcube, (1, 0, 2, 3))
            self.metric.cams[i] = cam

    def pca_stackcubes(self, stackcubes, dps, comps=True):
        wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_final)
        scale_list = wsamples / (ap.wvl_range[1] - ap.wvl_range[0])
        maps = []

        if comps:
            for stackcube in stackcubes:
                dprint(stackcube.shape)
                SDI = pca.pca(stackcube, angle_list=np.zeros((stackcube.shape[1])), scale_list=scale_list,
                              mask_center_px=None, adimsdi='double', ncomp=7, ncomp2=None,
                              collapse='median')
                maps.append(SDI)
            return maps

        else:
            rad_samps, thruputs, noises, conts = [], [], [], []
            for stackcube, dp in zip(stackcubes, dps):
                psf_template = get_unoccult_psf(fields=f'/IntHyperUnOccult_arraysize={dp.array_size}.h5', plot=False, numframes=1)
                # star_phot = phot.contrcurve.aperture_flux(np.sum(psf_template, axis=0), [sp.grid_size // 2],
                #                                           [sp.grid_size // 2], mp.lod, 1)[0]*10**-3#1.5
                star_phot = 1.1
                dprint(star_phot)
                # view_spectra(psf_template, logAmp=True)
                algo_dict = {'scale_list': scale_list}
                # temp for those older format cache files
                if hasattr(dp, 'lod'):
                    fwhm = dp.lod
                    dprint(fwhm)
                else:
                    fwhm = mp.lod
                    dprint(fwhm)
                method_out = eval_method(stackcube, pca.pca, psf_template,
                                         np.zeros((stackcube.shape[1])), algo_dict,
                                         fwhm=fwhm, star_phot=star_phot, dp=dp)

                thruput, noise, cont, sigma_corr, dist = method_out[0]
                thruputs.append(thruput)
                noises.append(noise)
                conts.append(cont)
                rad_samp = dp.platescale * dist
                rad_samps.append(rad_samp)
                maps.append(method_out[1])
            plt.show(block=True)
            return maps, rad_samps, thruputs, noises, conts

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

    repeats = 1  # number of medis runs to average over for the cont plots
    metric_names = ['numframes', 'array_size', 'array_size_(rebin)', 'pix_yield', 'dark_bright', 'R_mean', 'R_sig',
                   'g_mean', 'g_sig']  # 'g_mean_sig']# 'star_flux', 'exp_time'

    all_cont_data = []
    for r in range(repeats):

        obs = ObservatoryMaster(params, iteration=r)

        dprint(params['iop'].testdir)

        comp_images, cont_data, metric_multi_list, metric_vals_list = [], [], [], []
        for metric_name in metric_names:
            metric_module = importlib.import_module(metric_name)
            if metric_name in sys.modules:  # if the module has been loaded before it would be skipped and the params not initialized
                dprint(metric_name)
                metric_module = importlib.reload(metric_module)
            # config_images(len(param.metric_multiplier))  # the line colors and map inds depend on the amount being plotted
            # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, len(param.metric_multiplier))))

            metric_config = metric_module.MetricConfigur(obs.cam,
                                                   os.path.join(params['iop'].datadir, investigation, str(r), metric_name))

            metric_test = MetricTester(obs, metric_config)
            metric_results = metric_test(debug=True)
            # param_data = form(param.metric_vals, param.metric_name,
            #                   master_cache=(obs.dp, obs.fields), debug=True)

            comp_images.append(metric_results['maps'])
            cont_data.append([metric_results['rad_samps'],metric_results['conts']])

            # store the mutlipliers but flip those that achieve better contrast when the metric is decreasing
            if metric_name in ['R_sig', 'g_sig', 'dark_bright']:  # 'dark_bright',
                metric_multi_list.append(metric_config.metric_multiplier[::-1])
                metric_vals_list.append(metric_config.metric_vals[::-1])
            else:
                metric_multi_list.append(metric_config.metric_multiplier)
                metric_vals_list.append(metric_config.metric_vals)

        cont_data = np.array(cont_data)
        dprint(cont_data.shape)
        all_cont_data.append(cont_data)
    all_cont_data = np.array(all_cont_data)  # (repeats x num_params x rad+cont x num_multi (changes)

    three_lod_sep = 0.3
    six_lod_sep = 2 * three_lod_sep
    fhqm = 0.03
    for p, param_name in enumerate(param_names):
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
        master.combo_performance(maps, rad_samps, mean_conts, metric_vals, param_name, [0, -1], err_conts, metric_multi,
                                 three_lod_conts, three_lod_errs, six_lod_conts, six_lod_errs)
