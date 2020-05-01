import os, sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
from pprint import pprint

from vip_hci import phot, metrics, pca
import vip_hci.metrics.contrcurve as contrcurve

from medis.telescope import Telescope
from medis.MKIDs import Camera
from medis.utils import dprint
from medis.plot_tools import quick2D, view_spectra, body_spectra

from master import params
import metrics
from diagrams import contrcurve_plot, combo_performance
import substitution as subs

mode = 'develop'
# mode = 'test'

if mode == 'develop':
    params['ap'].n_wvl_init = 2
    params['ap'].n_wvl_final = 2
    params['sp'].numframes = 2
else:
    params['ap'].n_wvl_init = 8
    params['ap'].n_wvl_final = 16
    params['sp'].numframes = 10

params['tp'].detector='mkid'
investigation = f"figure3_{mode}_{params['tp'].detector}"

class ObservatoryMaster():
    """ Each repeat has new fields to seed from, as well as throughput data on that median array """
    def __init__(self, params, iteration=0):
        self.params = params
        self.name = str(iteration)
        self.masterdir = os.path.join(params['iop'].datadir, investigation, self.name, 'master')

        self.nbranch = 2
        self.ncomp = 7
        self.fc_snr = 100
        self.througput_file = os.path.join(self.masterdir,
                                           f'throughput_nbranch={self.nbranch}_ncomp={self.ncomp}.pkl')

        self.params['iop'].update(self.masterdir)

        if params['sp'].verbose:
            for param in params.values():
                pprint(param.__dict__)

        self.fields = self.make_fields_master()

        dprint(self.params['iop'].fields)
        self.cam = Camera(params, fields=self.fields, usesave=True)

        if self.params['tp'].detector == 'mkid':
            self.cam = subs.get_form_photons(self.fields, self.cam, comps=False)
        elif self.params['tp'].detector == 'ideal':
            self.cam = subs.get_ideal_photons(self.fields, self.cam, comps=False)
        else:
            raise NotImplementedError

        self.wsamples = np.linspace(params['ap'].wvl_range[0], params['ap'].wvl_range[1], params['ap'].n_wvl_final)
        self.scale_list = self.wsamples / (params['ap'].wvl_range[1] - params['ap'].wvl_range[0])

        self.throughput, self.vector_radd = self.get_throughput()

    def get_throughput(self):
        if os.path.exists(self.througput_file):
            print(f'loading throughput from {self.througput_file}')
            with open(self.througput_file, 'rb') as handle:
                throughput, vector_radd = pickle.load(handle)
        else:
            # unoccultname = os.path.join(self.params['iop'].testdir,
            #                             f'telescope_unoccult_arraysize={self.cam.array_size}')
            # psf_template = self.get_unoccult_psf(self.params, unoccultname)

            algo_dict = {'scale_list': self.scale_list}

            median_fwhm = self.cam.lod if hasattr(self.cam, 'lod') else self.params['mp'].lod
            median_wave = (self.wsamples[-1] + self.wsamples[0]) / 2
            fwhm = median_fwhm * self.wsamples/median_wave

            fulloutput = contrcurve.contrast_curve(cube=self.cam.stackcube, nbranch=self.nbranch, ncomp=self.ncomp,
                                                   fc_snr=self.fc_snr, cam=self.cam,
                                                   angle_list=np.zeros((self.cam.stackcube.shape[1])),
                                                   psf_template=self.psf_template[:,1:,1:], interp_order=2, fwhm=fwhm,
                                                   pxscale=self.cam.platescale / 1000,
                                                   starphot=1.1, algo=pca.pca, adimsdi='double',
                                                   ncomp2=None, debug=False, plot=False, theta=0,
                                                   full_output=True, **algo_dict)
            throughput, vector_radd = fulloutput[0]['throughput'], fulloutput[0]['distance']
            with open(self.througput_file, 'wb') as handle:
                pickle.dump((throughput, vector_radd), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return throughput, vector_radd

    def make_fields_master(self, plot=False):
        """ The master fields file of which all the photons are seeded from according to their device

        :return:
        """

        backup_fields = os.path.join(self.masterdir, 'fields_planet_slices.h5')

        self.params['ap'].companion = False
        self.contrast = copy.deepcopy(self.params['ap'].contrast)

        telescope = Telescope(self.params, usesave=False)
        fields = telescope()['fields']

        unoccultname = os.path.join(self.params['iop'].testdir,
                                    f'telescope_unoccult')
        self.psf_template = self.get_unoccult_psf(self.params, unoccultname)
        # body_spectra(self.psf_template)

        if os.path.exists(backup_fields):
            fields_master = fields
        else:
            assert len(fields.shape) == 6

            collapse_comps = np.zeros((self.params['sp'].numframes, self.params['ap'].n_wvl_final, self.params['sp'].grid_size,
                                    self.params['sp'].grid_size))
            # body_spectra(self.psf_template)
            for (x,y), scaling in zip(np.array(params['ap'].companion_xy) * 20, self.contrast):
                cube = copy.deepcopy(self.psf_template)
                print(x,y, scaling)
                cube = np.roll(cube,-int(x),2)
                cube = np.roll(cube,-int(y),1)
                cube *= scaling
                # body_spectra(cube)
                collapse_comps+=cube

            obs_seq = np.abs(fields[:, -1]) ** 2

            fields_master = np.zeros((self.params['sp'].numframes, self.params['ap'].n_wvl_final, 2, self.params['sp'].grid_size,
                                    self.params['sp'].grid_size))
            # collapse_comps = np.sum(obs_seq[:, :, 1:], axis=2)
            fields_master[:, :, 0] = obs_seq[:, :, 0]
            fields_master[:, :, 1] = collapse_comps
            if plot:
                body_spectra(fields_master[0], logZ=True, title='star and comps cube')

            dprint(f"Reduced shape of obs_seq = {np.shape(fields_master)} (numframes x nwsamp x 2 x grid x grid)")

            os.rename(self.params['iop'].fields, backup_fields)
            telescope.save_fields(fields_master)

        return fields_master

    def get_unoccult_psf(self, params, name):
        params = copy.deepcopy(params)
        params['sp'].save_fields = True
        params['ap'].companion = False
        params['tp'].cg_type = None
        params['sp'].numframes = 1
        params['ap'].sample_time = 1e-3
        params['sp'].save_list = ['detector']
        params['sp'].save_to_disk = True
        params['iop'].telescope = name + '.pkl'
        params['iop'].fields = name + '.h5'

        telescope = Telescope(params, usesave=True)
        fields = telescope()['fields']
        # psf_template = np.abs(fields[0, -1, :, 0, 1:, 1:]) ** 2
        psf_template = np.abs(fields[0, -1, :, 0]) ** 2
        # body_spectra(psf_template, logZ=True)

        return psf_template

class MetricTester():
    """ This instrument has the magical ability to quickly tune one device parameter during an observation """
    def __init__(self, obs, metric):

        self.params = obs.params
        self.lod = obs.params['mp'].lod
        self.scale_list = obs.scale_list
        self.ncomp = obs.ncomp
        self.master_cam = obs.cam
        self.master_fields = obs.fields
        self.vector_radd = obs.vector_radd
        self.throughput = obs.throughput

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
                    body_spectra(maps, logZ=False, title=self.metric.name, nstd=15)
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

            if self.params['tp'].detector == 'mkid':
                cam = subs.get_form_photons(master_fields, cam, comps=comps)
            elif self.params['tp'].detector == 'ideal':
                cam = subs.get_ideal_photons(master_fields, cam, comps=comps)
            else:
                raise NotImplementedError

            self.metric.cams[obj][i] = cam

    def pca_stackcubes(self, comps=True):
        maps = []

        if comps:
            for cam in self.metric.cams['comp']:
                dprint(cam.stackcube.shape)
                SDI = pca.pca(cam.stackcube, angle_list=np.zeros((cam.stackcube.shape[1])), scale_list=self.scale_list,
                              mask_center_px=None, adimsdi='double', ncomp=self.ncomp, ncomp2=None,
                              collapse='median')
                maps.append(SDI)
            return maps

        else:
            rad_samps, thruputs, noises, conts = [], [], [], []
            # for stackcube, dp in zip(stackcubes, dps):
            for cam in self.metric.cams['star']:
                frame_nofc = pca.pca(cam.stackcube, angle_list=np.zeros((cam.stackcube.shape[1])),
                                     scale_list=self.scale_list, mask_center_px=None, adimsdi='double', ncomp=7,
                                     ncomp2=None, collapse='median')

                # quick2D(frame_nofc, logZ=False, title='frame_nofc', show=False)

                fwhm = cam.lod if hasattr(cam, 'lod') else self.params['mp'].lod
                mask = cam.QE_map == 0
                noise_samp, rad_samp = contrcurve.noise_per_annulus(frame_nofc, separation=1, fwhm=fwhm,
                                                                         mask=mask)
                radmin = self.vector_radd.astype(int).min()
                cutin1 = np.where(rad_samp.astype(int) == radmin)[0][0]
                noise_samp = noise_samp[cutin1:]
                rad_samp = rad_samp[cutin1:]
                radmax = self.vector_radd.astype(int).max()
                cutin2 = np.where(rad_samp.astype(int) == radmax)[0][0]
                noise_samp = noise_samp[:cutin2 + 1]
                rad_samp = rad_samp[:cutin2 + 1]

                win = min(noise_samp.shape[0] - 2, int(2 * fwhm))
                if win % 2 == 0:
                    win += 1
                from scipy.signal import savgol_filter
                noise_samp_sm = savgol_filter(noise_samp, polyorder=2, mode='nearest',
                                              window_length=win)

                starphot = 1.1
                sigma = 5
                cont_curve_samp = ((sigma * noise_samp_sm) / self.throughput) / starphot
                cont_curve_samp[cont_curve_samp < 0] = 1
                cont_curve_samp[cont_curve_samp > 1] = 1

                thruputs.append(self.throughput)
                noises.append(noise_samp_sm)
                conts.append(cont_curve_samp)
                rad_samp = cam.platescale * rad_samp
                rad_samps.append(rad_samp)
                maps.append(frame_nofc)
            plt.show(block=True)
            return maps, rad_samps, thruputs, noises, conts

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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
    repeats = 2  # number of medis runs to average over for the cont plots
    # metric_names = ['pix_yield','g_mean', 'numframes', 'array_size', 'dark_bright', 'R_mean', 'g_mean']
    metric_names = ['ideal_placeholder']

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
