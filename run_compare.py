import os, sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import pickle
import copy

from vip_hci import phot, metrics, pca
from vip_hci.metrics.contrcurve import noise_per_annulus

import medis.MKIDs as mkids
import medis.medis_main as mm
from medis.telescope import Telescope
from medis.utils import dprint
from medis.plot_tools import quick2D, view_spectra, body_spectra

from master2 import params, get_form_photons

params['sp'].numframes = 10
# params['ap'].n_wvl_init = 3
# params['ap'].n_wvl_final = 3

investigation = 'figure3_moretimesandwaves'


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
        self.cam = mkids.Camera(params, fields=self.fields)

        if not os.path.exists(self.params['iop'].median_noise):
            self.get_median_noise()

    def get_median_noise(self, plot=False):
        wsamples = np.linspace(self.params['ap'].wvl_range[0], self.params['ap'].wvl_range[1], self.params['ap'].n_wvl_final)
        scale_list = wsamples / (self.params['ap'].wvl_range[1] - self.params['ap'].wvl_range[0])

        if not hasattr(self.cam, 'stackcube'):
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
    def __init__(self, obs, metric, debug=True):

        self.params = obs.params

        self.wvl_range = obs.params['ap'].wvl_range
        self.n_wvl_final = obs.params['ap'].n_wvl_final
        self.lod = obs.params['mp'].lod
        self.platescale = obs.params['tp'].platescale
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
                self.contrcurve_plot(self.metric.vals, rad_samps, thruputs, noises, conts)
                body_spectra(maps, logZ=False, )
            except UnboundLocalError:
                dprint('thruputs and noises not saved in old versions :(')
                # raise UnboundLocalError
                pass

            combo_performance(maps, rad_samps, conts, self.metric.vals, self.metric.name, [0,-1], savedir=self.testdir)

        return {'maps': maps, 'rad_samps':rad_samps, 'conts':conts}

    def get_stackcubes(self, master_fields, comps=True, plot=False):
        obj = 'comp' if comps else 'star'
        for i, cam, metric_val in zip(range(len(self.metric.cams[obj])), self.metric.cams[obj], self.metric.vals):

            if not hasattr(cam, 'stackcube'):
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
                                         fwhm=fwhm, star_phot=star_phot, dp=cam)

                thruput, noise, cont, sigma_corr, dist = method_out[0]
                thruputs.append(thruput)
                noises.append(noise)
                conts.append(cont)
                rad_samp = self.platescale * dist
                rad_samps.append(rad_samp)
                maps.append(method_out[1])
            plt.show(block=True)
            return maps, rad_samps, thruputs, noises, conts

    def eval_method(self, cube, algo, psf_template, angle_list, algo_dict, fwhm=6, star_phot=1, dp=None):
        dprint(fwhm, star_phot)
        fulloutput = metrics.contrcurve.contrast_curve(cube=cube, interp_order=2,
                                                       angle_list=angle_list, psf_template=psf_template,
                                                       fwhm=fwhm, pxscale=self.platescale / 1000,
                                                       # wedge=(-45, 45), int(dp.lod[0])
                                                       starphot=star_phot, algo=algo, nbranch=1,
                                                       adimsdi='double', ncomp=7, ncomp2=None,
                                                       debug=False, plot=False, theta=0, full_output=True, fc_snr=100,
                                                       dp=dp, **algo_dict)
        metrics_out = [fulloutput[0]['throughput'], fulloutput[0]['noise'], fulloutput[0]['sensitivity_student'],
                       fulloutput[0]['sigma corr'], fulloutput[0]['distance']]
        metrics_out = np.array(metrics_out)
        return metrics_out, fulloutput[2]

    def get_unoccult_psf(self, name):

        params = copy.copy(self.params)
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

    def contrcurve_plot(self, metric_vals, rad_samps, thruputs, noises, conts):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 3.4))

        # plotdata[:, 2] = plotdata[:, 1]*plotdata[:, 3] / np.mean(plotdata[:, 0], axis=0)

        for rad_samp, thruput in zip(rad_samps, thruputs):
            axes[0].plot(rad_samp, thruput)
        for rad_samp, noise in zip(rad_samps, noises):
            axes[1].plot(rad_samp, noise)
        for rad_samp, cont in zip(rad_samps, conts):
            axes[2].plot(rad_samp, cont)
        for ax in axes:
            ax.set_yscale('log')
            ax.set_xlabel('Radial Separation')
            ax.tick_params(direction='in', which='both', right=True, top=True)
        axes[0].set_ylabel('Throughput')
        axes[1].set_ylabel('Noise')
        axes[2].set_ylabel('5$\sigma$ Contrast')
        axes[2].legend([str(metric_val) for metric_val in metric_vals])

def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${} e^{{{}}}$'.format(a, b)

def combo_performance(maps, rad_samps, conts, metric_vals, param_name, plot_inds=[0, 3, 6], err=None,
                      metric_multi=None, three_lod_conts=None,  three_lod_errs=None, six_lod_conts=None,
                      six_lod_errs=None, savedir=''):

        # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, len(conts))))
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        labels = ['i', 'ii', 'iii', 'iv', 'v']
        title = r'  $I / I^{*}$'
        vmin = -1e-8
        vmax = 1e-6

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 3.4))
        param_name = param_name.replace('_', ' ')
        if param_name == 'R mean': param_name = r'$R_\mu$'
        if param_name == 'R sig': param_name = r'$R_\sigma$'
        if param_name == 'g mean': param_name = r'QE$_\mu$'
        if param_name == 'g sig': param_name = r'QE$_\sigma$'
        fig.suptitle(param_name, x=0.515)

        dprint(metric_vals, plot_inds)
        for m, ax in enumerate(axes[:2]):
            im = ax.imshow(maps[plot_inds[m]], interpolation='none', origin='lower', vmin=vmin, vmax=vmax,
                           norm=SymLogNorm(linthresh=1e-8), cmap="inferno")
            ax.text(0.05, 0.05, r'$P=$' + str(metric_vals[plot_inds[m]]), transform=ax.transAxes, fontweight='bold',
                    color='w', fontsize=16)
            # anno =
            ax.text(0.04, 0.9, labels[m], transform=ax.transAxes, fontweight='bold', color='w', fontsize=22,
                    family='serif')
            ax.axis('off')

        axes[0].text(0.84, 0.9, '0.2"', transform=axes[0].transAxes, fontweight='bold', color='w', ha='center',
                     fontsize=14,
                     family='serif')
        # axes[1].plot([114, 134], [130, 130], color='w', linestyle='-', linewidth=3)
        axes[0].plot([0.76, 0.89], [0.87, 0.87], transform=axes[0].transAxes, color='w', linestyle='-', linewidth=3)

        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("left", size="5%", pad=0.05)
        # cax.yaxis.set_ticks_position('left')
        # cax.yaxis.set_label_position('left')

        cb = fig.colorbar(im, cax=cax, orientation='vertical', norm=LogNorm(), format=ticker.FuncFormatter(fmt))
        cax.yaxis.set_ticks_position("left")
        cb.ax.set_title(title, fontsize=16)  #
        # cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5, endpoint=True)
        cbar_ticks = [-1e-8, 0, 1e-8, 1e-7, 1e-6]
        cb.set_ticks(cbar_ticks)

        for f, (rad_samp, cont) in enumerate(zip(rad_samps, conts)):
            if err is not None:
                axes[2].errorbar(rad_samp, cont, yerr=err[f], label='%5.2f' % metric_vals[f])
            else:
                axes[2].plot(rad_samp, cont, label='%5.2f' % metric_vals[f])

        axes[2].set_yscale('log')
        axes[2].set_xlabel('Radial Separation')
        axes[2].tick_params(direction='in', which='both', right=True, top=True)
        axes[2].set_ylabel('5$\sigma$ Contrast')
        planet_seps = np.arange(2.5, 6.5, 0.5) * 0.1
        # contrast = np.array([[e,e] for e in np.arange(-3.5,-5.5,-0.5)]).flatten()
        contrast = np.array([-3.5, -4, -4.5, -5] * 2)
        axes[2].scatter(planet_seps, 10 ** contrast, marker='o', color='k', label='Planets')
        axes[2].legend(ncol=2, fontsize=8, loc='upper right')
        axes[2].text(0.04, 0.9, labels[2], transform=axes[2].transAxes, fontweight='bold', color='k', fontsize=22,
                     family='serif')

        colors = plt.cycler("color", plt.cm.gnuplot2(np.linspace(0, 1, 3))).by_key()["color"]
        if np.any([metric_multi, three_lod_conts, three_lod_errs, six_lod_conts, six_lod_errs]):
            from scipy.optimize import curve_fit

            def func(x, a, b, c):
                return a * np.exp(-b * x) + c

            fit = True
            try:
                if np.any(three_lod_errs == 0) or np.any(six_lod_errs == 0):
                    popt3, pcov3 = curve_fit(func, metric_multi, three_lod_conts)
                    popt6, pcov6 = curve_fit(func, metric_multi, six_lod_conts)
                else:
                    popt3, pcov3 = curve_fit(func, metric_multi, three_lod_conts, sigma=three_lod_errs)
                    popt6, pcov6 = curve_fit(func, metric_multi, six_lod_conts, sigma=six_lod_errs)
            except RuntimeError as e:
                print(e)
                print('Could not find fit')
                fit = False

            # axes[2].get_shared_y_axes().join(axes[2], axes[3])
            axes[3].set_yscale('log')
            axes[3].set_xscale('log')
            axes[3].set_xlabel(r'$P/P_\mathrm{med}$')

            axes[3].tick_params(direction='in', which='both', right=True, top=True)

            if fit:
                axes[3].plot(metric_multi, func(metric_multi, *popt3),
                             label=r'$3\lambda/D$: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt3), c=colors[0])
                axes[3].plot(metric_multi, func(metric_multi, *popt6),
                             label=r'$6\lambda/D$: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt6), c=colors[1])
            axes[3].errorbar(metric_multi, three_lod_conts, yerr=three_lod_errs, linewidth=0, linestyle=None,
                             marker='o', c=colors[0])
            axes[3].errorbar(metric_multi, six_lod_conts, yerr=six_lod_errs, linewidth=0, linestyle=None, marker='o',
                             c=colors[1])
            axes[3].legend(fontsize=8)
            axes[3].text(0.04, 0.9, labels[3], transform=axes[3].transAxes, fontweight='bold', color='k', fontsize=22,
                         family='serif')

            ax3_top = axes[3].twiny()
            ax3_top.set_xscale('log')
            ax3_top.tick_params(direction='in', which='both', right=True, top=True)
            ax3_top.set_xlabel(r'$P$')
            if fit: ax3_top.plot(metric_vals, func(metric_multi, *popt3), linewidth=0)

            # for ax in [axes[3], ax3_top]:
            #     for axis in [ax.xaxis, ax.yaxis]:
            #         axis.set_major_formatter(ScalarFormatter())
            for ax in [axes[3], ax3_top]:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))

        # plt.tight_layout()
        plt.subplots_adjust(left=0.045, bottom=0.145, right=0.985, top=0.87, wspace=0.31)
        print(os.path.join(savedir, param_name + '.pdf'))
        fig.savefig(os.path.join(savedir, param_name + '.pdf'))
        plt.show(block=True)

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

            testdir = os.path.join(params['iop'].datadir, investigation, str(r), metric_name)
            metric_config = metric_module.MetricConfig(obs.cam, testdir)

            metric_test = MetricTester(obs, metric_config)
            metric_results = metric_test(debug=True)
            # param_data = form(param.metric_vals, param.metric_name,
            #                   master_cache=(obs.dp, obs.fields), debug=True)

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
        combo_performance(maps, rad_samps, mean_conts, metric_vals, param_name, [0, -1], err_conts, metric_multi,
                                 three_lod_conts, three_lod_errs, six_lod_conts, six_lod_errs)
