import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import interpolate
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import medis.medis_main as mm
from medis.utils import dprint
from medis.MKIDS import Camera
from medis.plot_tools import grid

from master import params

params['sp'].numframes = 25  #10
params['ap'].n_wvl_init = 8 # 8
params['ap'].n_wvl_final =  params['ap'].n_wvl_init
params['sp'].save_to_disk = True
params['sp'].skip_planes = ['prop_define_entrance']

params['tp'].detector = 'ideal'  #set ideal at first then do the mkid related stuff here
params['ap'].contrast = [1e-2]
params['ap'].companion = True
params['ap'].companion_xy = [[0, -3]]
# params['ap'].star_spec = 4000
# params['ap'].planet_spec = 1500
params['ap'].spectra = [4000, 1500]
params['ap'].star_flux = 10**7.7
# params['ap'].sample_time = 0.0001
params['sp'].sample_time = 0.5
params['mp'].pix_yield = 0.8
params['tp'].satelite_speck['apply'] = True

normalize_spec = False

def get_packets_plots(datacubes, step, cam, plot=False):

    FoV_datacubes = np.zeros((2, params['ap'].n_wvl_final, params['mp'].array_size[0], params['mp'].array_size[1]))
    for d in range(2):
        datacube = datacubes[:, d]

        if params['mp'].resamp:
            nyq_sampling = params['ap'].wvl_range[0]*360*3600/(4*np.pi*params['tp'].entrance_d)
            sampling = nyq_sampling*params['sp'].beam_ratio*2  # nyq sampling happens at params['tp'].beam_ratio = 0.5
            x = np.arange(-params['sp'].grid_size*sampling/2, params['sp'].grid_size*sampling/2, sampling)
            xnew = np.arange(-cam.array_size[0]*cam.platescale/2, cam.array_size[0]*cam.platescale/2, cam.platescale)
            mkid_cube = np.zeros((len(datacube), cam.array_size[0], cam.array_size[1]))
            for s, slice in enumerate(datacube):
                f = interpolate.interp2d(x, x, slice, kind='cubic')
                mkid_cube[s] = f(xnew, xnew)
            mkid_cube = mkid_cube*np.sum(datacube)/np.sum(mkid_cube)
            datacube = mkid_cube

        datacube[datacube < 0] *= -1
        FoV_datacubes[d] = datacube

    if plot:
        fig = plt.figure(figsize=(11,6))
        ax1 = fig.add_subplot(231)
        ax1.set_ylabel(r'rows')
        ax1.set_xlabel(r'columns')
        props = dict(boxstyle='square', facecolor='k', alpha=0.5)
        # ax1.text(0.05, 0.85, 'Device FoV', transform=ax1.transAxes, fontweight='bold',
        #          color='w', fontsize=16, bbox=props)
        ax1.text(-0.11, 1.05, 'i', transform=ax1.transAxes, color='k', fontsize=21, fontname='Times New Roman')
        ax1.set_title('Step 1')
        im = ax1.imshow(np.sum(FoV_datacubes, axis=0)[0][::-1], origin='lower', norm=SymLogNorm(1e-9), cmap='inferno', vmin=1e-9, vmax = 1e-6)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax, orientation='vertical', norm=SymLogNorm(1e-9))#, format=ticker.FuncFormatter(fmt))

    if params['mp'].QE_var:
        FoV_datacubes = FoV_datacubes * cam.QE_map[np.newaxis,np.newaxis,:datacube.shape[1],:datacube.shape[1]]


    if plot:
        ax2 = fig.add_subplot(232)
        ax2.set_ylabel(r'rows')
        ax2.set_xlabel(r'columns')
        ax2.set_title('Step 2')
        ax2.text(-0.11, 1.05, 'ii', transform=ax2.transAxes, color='k', fontsize=21, fontname='Times New Roman')
        # ax2.text(0.05, 0.75, 'Responsivity\n correction', transform=ax2.transAxes, fontweight='bold',
        #          color='w', fontsize=16, bbox=props)
        im = ax2.imshow(np.sum(FoV_datacubes, axis=0)[0][::-1], origin='lower', norm=SymLogNorm(1e-9), cmap='inferno', vmin=1e-9, vmax = 1e-6)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax, orientation='vertical', norm=SymLogNorm(1e-9))

    object_photons = []
    for d in range(2):
        num_events = int(params['ap'].star_flux * params['sp'].sample_time * np.sum(FoV_datacubes[d]))
        if params['sp'].verbose:
            dprint(f"star flux: {params['ap'].star_flux}, cube sum: {np.sum(FoV_datacubes[d])}, num events: {num_events}")

        photons = cam.sample_cube(FoV_datacubes[d], num_events)
        photons = cam.calibrate_phase(photons)
        photons = cam.assign_calibtime(photons, step)

        if plot:

            colors = ['#d62728', '#1f77b4']
            alphas = [0.15, 0.95]
            zorders = [0, -1]
            if d==0:
                ax3 = fig.add_subplot(233, projection='3d')
                ax3.view_init(elev=25., azim=-45)
                ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax3.set_xlabel('$\phi$')
                ax3.set_ylabel('columns')
                ax3.set_zlabel('rows')

            ax3.scatter(photons[1], photons[3], photons[2], s=1, alpha=alphas[d], color=colors[d], zorder=zorders[d])
            ax3.set_title('Step 3', y=0, pad=163, verticalalignment="top")
            ax3.text2D(-0.11, 1.05, 'iii', transform=ax3.transAxes, color='k', fontsize=21, fontname='Times New Roman')
        object_photons.append(photons)
        # fig.tight_layout()

    if plot:
        ax4 = fig.add_subplot(234)
        ax4.hist(object_photons[0][1], bins=range(-120,0,2), histtype='stepfilled', color='#d62728',
                 alpha=0.95, label='Star')
        ax4.hist(object_photons[0][1], bins=range(-120,0,2), histtype='step', color='k', alpha=0.95)
        ax4.set_yscale('log')

    photons = np.append(object_photons[0], object_photons[1], axis=1)
    if params['mp'].dark_counts:
        dark_photons = cam.get_bad_packets(step, type='dark')
        photons = np.hstack((photons, dark_photons))

    params['mp'].hot_pix = False
    if params['mp'].hot_pix:
        hot_photons = cam.get_bad_packets(step, type='hot')
        photons = np.hstack((photons, hot_photons))

    if plot:
        ax4.hist(object_photons[1][1], bins=range(-120, 0, 2), histtype='stepfilled', color='#1f77b4',
                 alpha=0.95, label='Planet')
        ax4.hist(object_photons[1][1], bins=range(-120, 0, 2), histtype='step', color='k', alpha=0.95)

        # ax4.hist(hot_photons[1], bins=range(-120,0,2), alpha=0.5, color='m', histtype='stepfilled', label='Hot')
        # ax4.hist(hot_photons[1], bins=range(-120,0,2), histtype='step', color='k')

        ax4.hist(dark_photons[1], bins=range(-120,0,2), alpha=0.75, color='#ff7f0e',
                 histtype='stepfilled', label='Dark')
        ax4.hist(dark_photons[1], bins=range(-120,0,2), histtype='step', color='k')
        ax4.legend(loc = 'upper right')
        ax4.set_xlabel('Phase (deg)')
        ax4.set_title('Step 4')
        ax4.text(-0.11, 1.05, 'iv', transform=ax4.transAxes, color='k', fontsize=21, fontname='Times New Roman')

    if params['mp'].phase_uncertainty:
        photons[1] *= cam.responsivity_error_map[np.int_(photons[2]), np.int_(photons[3])]
        photons, idx = cam.apply_phase_offset_array(photons, cam.sigs)

    if plot:
        ax5 = fig.add_subplot(235)
        ax5.hist(photons[1], bins=range(-120,0,2), alpha=0.5, histtype='stepfilled', color='#2ca02c', label='Degraded')
        ax5.hist(photons[1], bins=range(-120,0,2), histtype='step', color='k')


    dprint(photons.shape)
    thresh =  -photons[1] > 3*cam.sigs[0,np.int_(photons[3]), np.int_(photons[2])]#cam.basesDeg[np.int_(photons[3]),np.int_(photons[2])]
    photons = photons[:, thresh]

    if plot:
        ax5.hist(photons[1], bins=range(-120, 0, 2), alpha=0.95, histtype='stepfilled', rwidth=0.9, color= '#9467bd', label='Detected')
        ax5.hist(photons[1], bins=range(-120, 0, 2), histtype='step', color='k')
        ax5.set_xlabel('Phase (deg)')
        ax5.legend(loc = 'upper right')
        ax5.set_title('Steps 5 and 7')
        ax5.set_yscale('log')
        ax5.text(-0.11, 1.05, 'v', transform=ax5.transAxes, color='k', fontsize=21, fontname='Times New Roman')

    dprint(photons.shape)


    if params['sp'].verbose: print("Completed Readout Loop")

    if plot:
        return photons.T, fig
    else:
        return photons.T

if __name__ == '__main__':  # required for multiprocessing - make sure globals are set before though

    sim = mm.RunMedis(params=params, name='mkid_param_invest/figure2_mkid_demo', product='fields')
    observation = sim()
    fields = observation['fields']
    # grid(fields)

    # form_photons = os.path.join(params['iop'].testdir, 'form_photons.pkl')
    #
    # if os.path.exists(form_photons):
    #     print(f'loading formatted photon data from {form_photons}')
    #     with open(form_photons, 'rb') as handle:
    #         cam, fig = pickle.load(handle)

    # else:
    dprint((fields.shape))
    cam = Camera(params, fields=False, usesave=True)
    if cam.usesave:
        cam.save()

    cam.photons = np.empty((0, 4))
    dprint(len(fields))
    cam.rebinned_cube = np.zeros((params['sp'].numframes, params['ap'].n_wvl_final, params['mp'].array_size[1], params['mp'].array_size[0]))
    for step in range(len(fields)):
        dprint(step, fields.shape)
        spectralcubes = np.abs(fields[step, -1, :, :]) ** 2

        if step == 0:
            step_packets, fig = get_packets_plots(spectralcubes, step, cam, plot=True)
        else:
            step_packets = get_packets_plots(spectralcubes, step, cam, plot=False)

        cube = cam.make_datacube_from_list(step_packets)

        cam.rebinned_cube[step] = cube

        cam.photons = np.vstack((cam.photons, step_packets))

    cam.stem = cam.arange_into_stem(cam.photons, (params['mp'].array_size[0], params['mp'].array_size[1]))

        # with open(form_photons, 'wb') as handle:
        #     pickle.dump((cam, fig), handle, protocol=pickle.HIGHEST_PROTOCOL)


    ax6 = fig.add_subplot(236)

    # grid(cam.rebinned_cube)

    xc, yc = 106, 75
    x = np.arange(xc-2,xc+2)
    y = np.arange(yc-2,yc+2)
    star_events = np.empty((0,2))
    for xp in x:
        for yp in y:
            if len(cam.stem[xp][yp]) > 1:
                star_events = np.vstack((star_events, np.array(cam.stem[xp][yp])))
    # events = np.array(cam.stem[58][41])
    times = np.sort(star_events[:, 0])
    dprint((np.shape(star_events)))

    lightcurve = np.histogram(times, bins=200)
    ax6.plot(lightcurve[1][:-1][:-5], lightcurve[0][:-5], label='Original')
    ax6.set_title('Step 6')

    dprint(type(cam.stem))
    dprint(len(cam.stem))
    dprint(cam.stem[48:52])

    plt.figure()
    plt.hist(np.histogram(times, bins=200)[0], bins=range(60),#np.linspace(0,len(fields)*ap.sample_time, 50))[0], bins=60,#range(40,100),
             histtype='stepfilled', label='Original', alpha=0.5)
    plt.hist(np.histogram(times, bins=200)[0], bins=range(60),#range(40,100),
             histtype='step', color='k', alpha=0.5)

    satelite_stem = []
    for row in cam.stem[x[0]:x[-1]+1]:
        cols = []
        for col in row[y[0]:y[-1]+1]:
            cols.append(col)
        satelite_stem.append(cols)

    cam.stem = cam.remove_close(satelite_stem)

    # plt.figure()
    star_events = np.empty((0, 2))
    for xp in x-x[0]:
        for yp in y-y[0]:
            if len(cam.stem[xp][yp]) > 1:
                star_events = np.vstack((star_events, np.array(cam.stem[xp][yp])))
    dprint((np.shape(star_events)))
    times = np.sort(star_events[:, 0])


    lightcurve = np.histogram(times, bins=200)
    ax6.plot(lightcurve[1][:-1][:-5], lightcurve[0][:-5], label='Observed')

    ax6.set_xlabel(r'Time (s)')
    # fig.tight_layout()
    ax6.legend(loc = 'upper right')
    ax6.text(-0.11, 1.05, 'vi', transform=ax6.transAxes, color='k', fontsize=21, fontname='Times New Roman')

    plt.hist(np.histogram(times, bins=200)[0], bins=range(60),#np.linspace(0,len(fields)*ap.sample_time, 50))[0], bins=60,#range(40,100),
             histtype='stepfilled', label='Original', alpha=0.5)
    plt.hist(np.histogram(times, bins=200)[0], bins=range(60),#range(40,100),
             histtype='step', color='k', alpha=0.5)

    # fig.tight_layout()
    plt.show(block=True)