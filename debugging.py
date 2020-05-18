import os
import numpy as np
import matplotlib.pyplot as plt
from vip_hci.metrics.contrcurve import noise_per_annulus, contrast_curve
from vip_hci import pca

from medis.utils import dprint
from medis.plot_tools import body_spectra, quick2D

from run_compare import ObservatoryMaster, MetricTester
import metrics
from master import params


def debugging_noise(metric_test):
    """ debugging the throughput issue by first looking at the input noise """
    comp = 'star'
    metric_test.get_rebinned_cubes(metric_test.master_fields, comps=comp != 'star')
    nmetric = len(metric_test.metric.cams[comp])
    mid_ind = int((nmetric - 1) / 2)
    dprint(mid_ind)
    cam = metric_test.metric.cams['comp'][mid_ind]  # middle ind assuming
    wavemet_cube = [metric_test.metric.cams[comp][i].rebinned_cube[:, 0] for i in range(len(metric_test.metric.vals))]
    # body_spectra(wavemet_cube)
    median_fwhm = cam.lod
    mask = cam.QE_map == 0
    wsamples = np.linspace(metric_test.wvl_range[0], metric_test.wvl_range[1], metric_test.n_wvl_final)
    median_wave = (wsamples[-1] + wsamples[0]) / 2

    for im, metcube in enumerate(wavemet_cube):
        broadband_image = np.sum(metcube, axis=0)
        # quick2D(broadband_image, title='broadband image')
        median_noise, vector_radd = noise_per_annulus(broadband_image, separation=median_fwhm, fwhm=median_fwhm,
                                                      mask=mask)
        plt.plot(vector_radd * cam.platescale, median_noise, label=f'met val: {metric_test.metric.vals[im]}')
    plt.legend()
    plt.title('broadband noise-separation curve for different metrics')
    plt.show()

    # split those lines into component wavelengths and plot the result of each metric on a new page
    for im, metcube in enumerate(wavemet_cube):
        body_spectra(metcube, title=f'met val: {metric_test.metric.vals[im]}', show=False)
        plt.figure()
        for iw, waveimage in enumerate(metcube):
            fwhm = median_fwhm  # * wsamples[iw]/median_wave
            print('wavelength=', wsamples[iw], 'met val=', metric_test.metric.vals[im], 'fwhm:', fwhm)
            median_noise, vector_radd = noise_per_annulus(waveimage, separation=fwhm, fwhm=fwhm, mask=mask)
            plt.title(f'met val: {metric_test.metric.vals[im]}')
            plt.plot(vector_radd * cam.platescale, median_noise, label=f'wave: {int(wsamples[iw] * 1e9)}')
        plt.legend()
    plt.show()

    metwave_cube = np.transpose(wavemet_cube, (1, 0, 2, 3))

    # split those lines into component wavelengths and plot the result of each common wavelength on a new page
    for iw, wavecube in enumerate(metwave_cube):
        body_spectra(wavecube, title=f'wave: {int(wsamples[iw] * 1e9)}', show=False)
        plt.figure()
        for im, metimage in enumerate(wavecube):
            fwhm = median_fwhm  # * wsamples[iw]/median_wave
            print('wavelength=', wsamples[iw], 'met val=', metric_test.metric.vals[im], 'fwhm:', fwhm)
            median_noise, vector_radd = noise_per_annulus(metimage, separation=fwhm, fwhm=fwhm, mask=mask)
            plt.title(f'wave: {int(wsamples[iw] * 1e9)}')
            plt.plot(vector_radd * cam.platescale, median_noise, label=f'wave: {int(wsamples[iw] * 1e9)}')
        plt.legend()
    plt.show()


def debugging_throughput(metric_test):
    comp = 'star'
    savefile = os.path.join(metric_test.metric.testdir, 'thruputs.pkl')

    figure, axes = plt.subplots(2, 4, figsize=(12, 7))

    # if os.path.exists(savefile):
    #     with open(savefile, 'rb') as handle:
    #         throughput_smooth, flux_in, flux_out = pickle.load(handle)
    #
    # else:
    metric_test.get_rebinned_cubes(metric_test.master_fields, comps=comp != 'star')
    wsamples = np.linspace(metric_test.wvl_range[0], metric_test.wvl_range[1], metric_test.n_wvl_final)
    scale_list = wsamples / (metric_test.wvl_range[1] - metric_test.wvl_range[0])

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    print(metric_test.metric.cams[comp][-1:])
    for i, fc_snr in enumerate([1, 10, 100, 1000, 10000]):
        for im, cam in enumerate(metric_test.metric.cams[comp][0:1]):
            unoccultname = os.path.join(metric_test.testdir, f'telescope_unoccult_arraysize={cam.array_size}.pkl')
            psf_template = metric_test.get_unoccult_psf(unoccultname)
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
                fwhm = metric_test.lod
                dprint(fwhm)

            fulloutput = contrast_curve(cube=cam.rebinned_cube, interp_order=2,
                                        angle_list=np.zeros((cam.rebinned_cube.shape[1])),
                                        psf_template=psf_template, fwhm=fwhm, pxscale=cam.platescale / 1000,
                                        starphot=star_phot, algo=pca.pca, nbranch=1, adimsdi='double', ncomp=7,
                                        ncomp2=None, debug=False, plot=False, theta=0, fc_snr=fc_snr, full_output=True,
                                        debug_output=True, cam=cam, **algo_dict)

            throughput_smooth, flux_in, flux_out = fulloutput[0]['throughput'], fulloutput[4], fulloutput[5]

            # with open(savefile, 'wb') as handle:
            #     pickle.dump((throughput_smooth, flux_in, flux_out), handle, protocol=pickle.HIGHEST_PROTOCOL)

            throughput = flux_out / flux_in
            flux_in_mean = np.mean(flux_in)
            flux_out_mean = np.mean(flux_out)
            throughput_mean = np.mean(throughput)
            color = colors[i]
            axes[0, 0].plot(flux_in, color=color)
            axes[0, 1].plot(flux_out, color=color)
            axes[0, 2].plot(throughput, color=color)
            # axes[0,3].plot(throughput_smooth, label=f'met val: {metric_test.metric.vals[im]}', color=color)
            axes[0, 3].plot(throughput_smooth, label=f'fcr_snr: {fc_snr}', color=color)
            axes[1, 0].plot(flux_in, flux_out, color=color)
            axes[1, 1].plot(flux_in, throughput, color=color)
            axes[1, 2].plot(flux_out, throughput, color=color)

            axes[1, 0].plot(flux_in_mean, flux_out_mean, marker='o', color=color)
            axes[1, 1].plot(flux_in_mean, throughput_mean, marker='o', color=color)
            axes[1, 2].plot(flux_out_mean, throughput_mean, marker='o', color=color)

    axes[0, 0].set_yscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 2].set_yscale('log')
    axes[0, 3].set_yscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xscale('log')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_xscale('log')

    axes[0, 3].legend()
    plt.show()

if __name__ == '__main__':
    obs = ObservatoryMaster(params, iteration=0)

    metric_name = 'pix_yield'

    metric_config = metrics.get_metric(metric_name, master_cam=obs.cam)
    metric_test = MetricTester(obs, metric_config)
    debugging_noise(metric_test)
    debugging_throughput(metric_test)
