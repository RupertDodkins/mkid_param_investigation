import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
from pprint import pprint
from scipy.signal import savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline

from vip_hci import phot, metrics, pca
import vip_hci.metrics.contrcurve as contrcurve

from medis.telescope import Telescope
from medis.MKIDS import Camera
from medis.utils import dprint
from medis.plot_tools import quick2D, view_spectra, grid

from master import ap, sp, tp, iop, atmp, cdip, mp
import metrics
from diagrams import contrcurve_plot, combo_performance
import substitution as subs

mode = 'develop'
# mode = 'test'

if mode == 'develop':
    ap.n_wvl_init = 8
    ap.n_wvl_final = 16
    sp.numframes = 20
else:
    ap.n_wvl_init = 8
    ap.n_wvl_final = 16
    # sp.numframes = 10

# print('redo with pix yield 0.9')
# raise ValueError

# tp.detector='mkid'
investigation = f"mkid_param_invest/figure3_{mode}_{tp.detector}"

class ObservatoryMaster():
    """ Each repeat has new fields to seed from, as well as throughput data on that median array """
    def __init__(self, iteration=0, name=investigation):
        self.name = str(iteration)
        self.masterdir = os.path.join(iop.datadir, name, self.name, 'master')

        self.nbranch = 2
        self.ncomp = 7
        self.fc_snr = 100
        self.througput_file = os.path.join(self.masterdir,
                                           f'throughput_nbranch={self.nbranch}_ncomp={self.ncomp}.pkl')

        params = {'ap': ap, 'tp': tp, 'atmp': atmp, 'cdip': cdip, 'iop': iop, 'sp': sp, 'mp': mp}
        iop.update_testname(self.masterdir)

        if sp.verbose:
            for param in params.values():
                pprint(param.__dict__)

        self.fields = self.make_fields_master()

        dprint(iop.fields)
        self.cam = Camera(fields=False, usesave=True)
        if self.cam.usesave:
            self.cam.save_instance()

        # if tp.detector == 'mkid':
        #     self.cam = subs.get_form_photons(self.fields, self.cam, comps=False)
        # elif tp.detector == 'ideal':
        #     self.cam = subs.get_ideal_photons(self.fields, self.cam, comps=False)
        # else:
        #     raise NotImplementedError

        self.wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_final)
        self.scale_list = self.wsamples / (ap.wvl_range[1] - ap.wvl_range[0])

        # self.throughput, self.vector_radd = self.get_throughput()

    def get_throughput(self):
        if os.path.exists(self.througput_file):
            print(f'loading throughput from {self.througput_file}')
            with open(self.througput_file, 'rb') as handle:
                throughput, vector_radd = pickle.load(handle)
        else:
            # unoccultname = os.path.join(iop.testdir,
            #                             f'telescope_unoccult_arraysize={self.cam.array_size}')
            # psf_template = self.get_unoccult_psf(unoccultname)

            algo_dict = {'scale_list': self.scale_list}

            median_fwhm = self.cam.lod if hasattr(self.cam, 'lod') else mp.lod
            median_wave = (self.wsamples[-1] + self.wsamples[0]) / 2
            fwhm = median_fwhm * self.wsamples/median_wave

            fulloutput = contrcurve.contrast_curve(cube=self.cam.rebinned_cube, nbranch=self.nbranch, ncomp=self.ncomp,
                                                   fc_snr=self.fc_snr, cam=self.cam,
                                                   angle_list=np.zeros((self.cam.rebinned_cube.shape[1])),
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

        ap.companion = False
        self.contrast = copy.deepcopy(ap.contrast)

        telescope = Telescope(usesave=False)
        fields = telescope()['fields']

        unoccultname = os.path.join(iop.testdir, f'telescope_unoccult')
        self.psf_template = self.get_unoccult_psf(unoccultname)
        # grid(self.psf_template)

        if plot:
            grid(fields[0], logZ=True, title='make_fields_master')

        if os.path.exists(backup_fields):
            fields_master = fields
        else:
            assert len(fields.shape) == 6

            collapse_comps = np.zeros((sp.numframes, ap.n_wvl_final, sp.grid_size,
                                    sp.grid_size))
            # grid(self.psf_template)
            for (x,y), scaling in zip(np.array(ap.companion_xy) * 20, self.contrast):
                cube = copy.deepcopy(self.psf_template)
                print(x,y, scaling)
                cube = np.roll(cube,-int(x),2)
                cube = np.roll(cube,-int(y),1)
                cube *= scaling
                # grid(cube)
                collapse_comps+=cube

            obs_seq = np.abs(fields[:, -1]) ** 2

            fields_master = np.zeros((sp.numframes, ap.n_wvl_final, 2, sp.grid_size,
                                    sp.grid_size))
            # collapse_comps = np.sum(obs_seq[:, :, 1:], axis=2)
            fields_master[:, :, 0] = obs_seq[:, :, 0]
            fields_master[:, :, 1] = collapse_comps
            if plot:
                grid(fields_master[0], logZ=True, title='star and comps cube')

            dprint(f"Reduced shape of obs_seq = {np.shape(fields_master)} (numframes x nwsamp x 2 x grid x grid)")

            os.rename(iop.fields, backup_fields)
            telescope.save_fields(fields_master)

        return fields_master

    def get_unoccult_psf(self, name):
        # sp_orig = copy.deepcopy(sp)
        # ap_orig = copy.deepcopy(ap)
        # tp_orig = copy.deepcopy(tp)
        # iop_orig = copy.deepcopy(iop)

        params = [sp,ap,tp,iop]
        save_state = [copy.deepcopy(param) for param in params]

        sp.save_fields = True
        ap.companion = False
        tp.cg_type = None
        sp.numframes = 1
        ap.sample_time = 1e-3
        sp.save_list = ['detector']
        sp.save_to_disk = True
        iop.telescope = name + '.pkl'
        iop.fields = name + '.h5'

        telescope = Telescope(usesave=True)
        fields = telescope()['fields']

        # sp.__dict__ = sp_orig.__dict__
        # ap.__dict__ = ap_orig.__dict__
        # tp.__dict__ = tp_orig.__dict__
        # iop.__dict__ = iop_orig.__dict__
        for i, param in enumerate(params):
            param.__dict__ = save_state[i].__dict__

        # psf_template = np.abs(fields[0, -1, :, 0, 1:, 1:]) ** 2
        psf_template = np.abs(fields[0, -1, :, 0]) ** 2
        # grid(psf_template, logZ=True)

        return psf_template

class MetricTester():
    """ This instrument has the magical ability to quickly tune one device parameter during an observation """
    def __init__(self, obs, metric):

        # self.params = obs.params
        self.lod = mp.lod
        self.scale_list = obs.scale_list
        self.ncomp = 5#obs.ncomp
        self.collapse = 'median'
        self.master_cam = obs.cam
        self.master_fields = obs.fields
        # self.vector_radd = obs.vector_radd
        # self.throughput = obs.throughput
        self.wsamples = obs.wsamples

        self.metric = metric
        self.testdir = metric.testdir

        self.performance_data = os.path.join(self.testdir, 'performance_data.pkl')

    def __call__(self, debug=True):

        dprint(self.performance_data)
        # self.performance_data = '/Users/dodkins/MKIDSim/mkid_param_invest/figure3_develop_mkid_lowerflux/1/max_count/performance_data.pkl'
        if not os.path.exists(self.performance_data):
            # self.metric.create_adapted_cams()

            comps_ = [True, False]
            pca_products = []
            for comps in comps_:
                if hasattr(self.metric, 'get_rebinned_cubes'):
                    self.metric.get_rebinned_cubes(self.master_fields, comps=comps)
                else:
                    self.get_rebinned_cubes(self.master_fields, comps=comps)

            #     pca_products.append(self.pca_rebinned_cubes(comps))
            #
            # maps = pca_products[0]
            # rad_samps = pca_products[1][1]
            # thruputs = pca_products[1][2]
            # noises = pca_products[1][3]
            # conts = pca_products[1][4]

            maps, rad_samps, thruputs, noises, conts = self.pca_rebinned_cubes()

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
                    grid(maps, logZ=False, title=self.metric.name)#, vlim=(-2,2)) for no-normed
                else:
                    pass
                plt.show(block=True)
            except UnboundLocalError:
                dprint('thruputs and noises not saved in old versions :(')
                # raise UnboundLocalError
                pass

            combo_performance(maps, rad_samps, conts, self.metric.vals, self.metric.name, [0,-1], savedir=self.testdir)

        return {'maps': maps, 'rad_samps':rad_samps, 'conts':conts}

    def get_rebinned_cubes(self, master_fields, comps=True):
        obj = 'comp' if comps else 'star'
        for i, cam, metric_val in zip(range(len(self.metric.cams[obj])), self.metric.cams[obj], self.metric.vals):

            if tp.detector == 'mkid':
                cam = subs.get_form_photons(master_fields, cam, comps=comps)
            elif tp.detector == 'ideal':
                cam = subs.get_ideal_photons(master_fields, cam, comps=comps)
            else:
                raise NotImplementedError

            self.metric.cams[obj][i] = cam

    def pca_rebinned_cubes(self):
        maps, rad_samps, thruputs, noises, conts = [], [], [], [], []
        cams = self.metric.cams
        colors = [f'C{i}' for i in range(len(self.metric.cams['star']))]
        # fig, axes = plt.subplots(1,6)
        fig, axes = plt.subplots(1,3)

        dprint(np.shape(axes))
        for i in range(len(cams['comp'])):
            comp_cube = cams['comp'][i].rebinned_cube
            dprint(comp_cube.shape)
            frame_comp = pca.pca(comp_cube, angle_list=np.zeros((comp_cube.shape[1])), scale_list=self.scale_list,
                          mask_center_px=None, adimsdi='double', ncomp=self.ncomp, ncomp2=None,
                          collapse=self.collapse)
            maps.append(frame_comp)

            nocomp_cube = cams['star'][i].rebinned_cube
            frame_nocomp = pca.pca(nocomp_cube, angle_list=np.zeros((nocomp_cube.shape[1])),
                                 scale_list=self.scale_list, mask_center_px=None, adimsdi='double', ncomp=self.ncomp,
                                 ncomp2=None, collapse=self.collapse)

            median_fwhm = cams['star'][i].lod if hasattr(cams['star'][i], 'lod') else mp.lod
            median_wave = (self.wsamples[-1] + self.wsamples[0]) / 2
            fwhm = median_fwhm * self.wsamples/median_wave

            mask = cams['star'][i].QE_map == 0

            print('check whether to be using mask here?!')
            noise_samp, rad_samp = contrcurve.noise_per_annulus(frame_nocomp, separation=1, fwhm=median_fwhm,
                                                                mask=mask)

            xy = np.array(ap.companion_xy) * 20 * cams['star'][i].sampling / cams['star'][i].platescale
            px, py = (cams['star'][i].array_size/2 - xy).T
            cx, cy = cams['star'][i].array_size/2
            dists = np.sqrt((py-cy)**2 + (px-cx)**2)
            # grid(np.sum(comp_cube, axis=1), title='in comp time collapse', show=False)
            # injected_flux=[contrcurve.aperture_flux(np.sum(comp_cube-nocomp_cube, axis=1)[i], py, px, fwhm[i]/1.5, ap_factor=1) for i in range(comp_cube.shape[0])]
            injected_flux=[contrcurve.aperture_flux(np.sum(comp_cube, axis=1)[i], py, px, fwhm[i]/1.5, ap_factor=1, plot=False) for i in range(comp_cube.shape[0])]
            # [axes[i+3].plot(dists, influx_wave) for influx_wave in injected_flux]
            injected_flux = np.mean(injected_flux, axis=0)

            # if i ==0 or i == len(self.metric.cams['star'])-1:
                # grid(comp_cube, title='comp', show=False)
                # grid(nocomp_cube, title='no', show=False)
                # grid(comp_cube-nocomp_cube, title='diff', show=False)
                # grid([np.sum(comp_cube, axis=(0,1))], title='sum comp', logZ=True, show=False)
                # grid([np.sum(nocomp_cube, axis=(0,1))], title='sum nocomp', logZ=True, show=False)
                # grid([np.sum(comp_cube, axis=(0,1))-np.sum(nocomp_cube, axis=(0,1))], title='sum then diff', logZ=True, show=False)
                # grid([], title='comp', logZ=True, show=False)
            grid([frame_comp, frame_nocomp, frame_comp-frame_nocomp], title='comp, nocomp, diff', logZ=False, show=False,vlim=(-2e-7,2e-7))# vlim=(-2,2))#, vlim=(-2e-7,2e-7))
            grid([np.sum(comp_cube, axis=1)[::4], np.sum(nocomp_cube, axis=1)[::4]], title='input: comp, no comp', logZ=False, show=False,vlim=(-1e-5,1e-5))# vlim=(-2,2))#, vlim=(-1e-5,1e-5))
            grid(np.sum(comp_cube-nocomp_cube, axis=1)[::4], title='diiff input cube', logZ=False, show=False)#, vlim=(-2,2))
                # grid([(frame_comp-frame_nocomp)/(np.sum(comp_cube-nocomp_cube, axis=(0,1)))], title='throughput', logZ=True, show=False)

            # recovered_flux = contrcurve.aperture_flux((frame_comp - frame_nocomp), py, px, median_fwhm/1.5, ap_factor=1, )
            recovered_flux = contrcurve.aperture_flux((frame_comp), py, px, median_fwhm/1.5, ap_factor=1, plot=False)
            # thruput = recovered_flux*1e6 #/ injected_flux
            thruput = recovered_flux/ injected_flux

            thruput[np.where(thruput < 0)] = 0


            # plt.figure()
            axes[0].plot(dists, thruput, c=colors[i])
            axes[1].plot(dists, injected_flux, c=colors[i])
            axes[2].plot(dists, recovered_flux, c=colors[i], label=f'{self.metric.vals[i]}')
            axes[2].legend()
            # plt.plot(rad_samp, noise_samp)
            # plt.show()

            win = min(noise_samp.shape[0] - 2, int(2 * median_fwhm))
            if win % 2 == 0: win += 1
            noise_samp_sm = savgol_filter(noise_samp, polyorder=2, mode='nearest', window_length=win)


            # thruput_mean_log = np.log10(thruput + 1e-5)
            # f = InterpolatedUnivariateSpline(dists, thruput_mean_log, k=2)
            # thruput_interp_log = f(rad_samp)
            # thruput_interp = 10 ** thruput_interp_log
            # thruput_interp[thruput_interp <= 0] = np.nan  # 1e-5

            # thruput_interp = np.interp(rad_samp, dists, thruput)

            from scipy.interpolate import interp1d
            # f = interp1d(dists, thruput, fill_value='extrapolate')
            # thruput_interp = f(rad_samp)
            thruput_com = thruput.reshape(4,-1, order='F').mean(axis=0)
            dists_com = dists.reshape(4,-1, order='F').mean(axis=0)

            thruput_com_log = np.log10(thruput_com + 1e-5)
            f = interp1d(dists_com, thruput_com_log, fill_value='extrapolate')
            thruput_interp_log = f(rad_samp)
            thruput_interp = 10 ** thruput_interp_log

            axes[0].plot(dists_com, thruput_com, marker='o', c=colors[i])

            print(thruput, thruput_interp)
            axes[0].plot(rad_samp, thruput_interp, c=colors[i])

            starphot = 1.1
            sigma = 5
            cont_curve_samp = ((sigma * noise_samp_sm) / thruput_interp) / starphot
            cont_curve_samp[cont_curve_samp < 0] = 1
            cont_curve_samp[cont_curve_samp > 1] = 1

            thruputs.append(thruput_interp)
            noises.append(noise_samp_sm)
            conts.append(cont_curve_samp)
            rad_samp = cams['star'][i].platescale * rad_samp
            rad_samps.append(rad_samp)
            # maps.append(frame_nocomp)
        plt.show(block=True)
        return maps, rad_samps, thruputs, noises, conts

    def pca_rebinned_cubes_old(self, comps=True):
        maps = []

        if comps:
            for cam in self.metric.cams['comp']:
                dprint(cam.rebinned_cube.shape)
                SDI = pca.pca(cam.rebinned_cube, angle_list=np.zeros((cam.rebinned_cube.shape[1])), scale_list=self.scale_list,
                              mask_center_px=None, adimsdi='double', ncomp=self.ncomp, ncomp2=None,
                              collapse='median')
                maps.append(SDI)
            return maps

        else:
            rad_samps, thruputs, noises, conts = [], [], [], []
            for cam in self.metric.cams['star']:
                frame_nofc = pca.pca(cam.rebinned_cube, angle_list=np.zeros((cam.rebinned_cube.shape[1])),
                                     scale_list=self.scale_list, mask_center_px=None, adimsdi='double', ncomp=7,
                                     ncomp2=None, collapse='median')

                # quick2D(frame_nofc, logZ=False, title='frame_nofc', show=True)

                fwhm = cam.lod if hasattr(cam, 'lod') else mp.lod
                mask = cam.QE_map == 0
                noise_samp, rad_samp = contrcurve.noise_per_annulus(frame_nofc, separation=1, fwhm=fwhm,
                                                                         mask=mask)

                if metric_name == 'array_size':
                    _, vector_radd = contrcurve.noise_per_annulus(frame_nofc, separation=fwhm,
                                                                  fwhm=fwhm, wedge=(0,360), mask=mask)

                else:
                    vector_radd = self.vector_radd

                # crop the noise and radial sampling measurements the limits of the throughput measurement
                radmin = vector_radd.astype(int).min()
                cutin1 = np.where(rad_samp.astype(int) == radmin)[0][0]
                noise_samp = noise_samp[cutin1:]
                rad_samp = rad_samp[cutin1:]
                radmax = vector_radd.astype(int).max()
                cutin2 = np.where(rad_samp.astype(int) == radmax)[0][0]
                noise_samp = noise_samp[:cutin2 + 1]
                rad_samp = rad_samp[:cutin2 + 1]

                if metric_name == 'array_size':
                    throughput = np.interp(rad_samp * cam.platescale,
                                           self.vector_radd.values * self.master_cam.platescale, self.throughput)
                # elif metric_name == 'dark_bright':
                #     throughput = self.manual_throughput(cam.rebinned_cube, frame_nofc)
                else:
                    throughput = self.throughput

                win = min(noise_samp.shape[0] - 2, int(2 * fwhm))
                if win % 2 == 0: win += 1
                noise_samp_sm = savgol_filter(noise_samp, polyorder=2, mode='nearest',
                                              window_length=win)

                starphot = 1.1
                sigma = 5
                cont_curve_samp = ((sigma * noise_samp_sm) / throughput) / starphot
                cont_curve_samp[cont_curve_samp < 0] = 1
                cont_curve_samp[cont_curve_samp > 1] = 1

                thruputs.append(throughput)
                noises.append(noise_samp_sm)
                conts.append(cont_curve_samp)
                rad_samp = cam.platescale * rad_samp
                rad_samps.append(rad_samp)
                maps.append(frame_nofc)

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
    metric_names = ['array_size', 'g_mean', 'numframes','R_mean', 'dark_bright','pix_yield']
    # metric_names = ['ideal_placeholder']
    # metric_names = ['dark_bright']
    metric_names = ['max_count']
    # metric_names = ['array_size', 'g_mean']


    # collect the data
    all_cont_data = []
    for r in range(repeats):

        obs = ObservatoryMaster(iteration=r)

        dprint(iop.testdir)

        comp_images, cont_data, metric_multi_list, metric_vals_list = [], [], [], []
        for metric_name in metric_names:

            # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, len(param.metric_multiplier))))

            metric_config = metrics.get_metric(metric_name, master_cam=obs.cam)
            metric_test = MetricTester(obs, metric_config)
            metric_results = metric_test()
            
            comp_images.append(metric_results['maps'])
            cont_data.append([metric_results['rad_samps'],metric_results['conts']])

            # # store the mutlipliers but flip those that achieve better contrast when the metric is decreasing
            # if metric_name in ['R_sig', 'g_sig', 'dark_bright']:  # 'dark_bright',
            #     metric_multi_list.append(metric_config.multiplier[::-1])
            #     metric_vals_list.append(metric_config.vals[::-1])
            # else:

            vals = metric_config.vals if metric_name != 'numframes' else metric_config.vals*sp.sample_time
            metric_multi_list.append(metric_config.multiplier)
            metric_vals_list.append(vals)

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
