import os
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt

from medis.plot_tools import quick2D, view_spectra, grid
from medis.utils import dprint

def get_ideal_photons(fields, cam, comps=True, plot=False):
    ntime, nwave = fields.shape[0], fields.shape[1]

    cam.rebinned_cube = np.zeros((ntime, nwave, cam.array_size[1], cam.array_size[0]))
    for step in range(len(fields)):
        print(step)
        if comps:
            spectralcube = np.sum(fields[step], axis=1)
        else:
            spectralcube = fields[step, :, 0]

        cube = cam.get_ideal_cube(spectralcube)
        cam.rebinned_cube[step] = cube

    cam.rebinned_cube /= np.sum(cam.rebinned_cube)  # /sp.numframes
    cam.rebinned_cube = np.transpose(cam.rebinned_cube, (1, 0, 2, 3))

    if plot:
        # plt.figure()
        # plt.hist(cam.rebinned_cube[cam.rebinned_cube != 0].flatten(), bins=np.linspace(0, 1e4, 50))
        # plt.yscale('log')
        grid(cam.rebinned_cube, show=True, title='get ideal photons', nstd=6)

    if cam.usesave:
        cam.save()

    return cam

def get_form_photons(fields, cam, comps=True, plot=False, collapse_time_first=False, norm=False):
    """
    Alternative to cam.__call__ that allows the user to specify whether the spectracube contains the planets

    :param fields: ndarray
    :param cam: mkids.Camera()
    :param comps: bool

    :return:
    mkids.Camera()
    """
    dprint(cam.name)
    if os.path.exists(cam.name):
        print(f'loading cam rebined_cube save at {cam.name}')
        with open(cam.name, 'rb') as handle:
            cam = pickle.load(handle)
    else:
        if comps:
            fourcube = np.sum(fields, axis=2)
        else:
            fourcube = fields[:, :, 0]

        fourcube = np.abs(fourcube) ** 2
        dprint(np.sum(fourcube))
        fourcube = cam.rescale_cube(fourcube)

        max_steps = cam.max_chunk(fourcube)
        num_chunks = int(np.ceil(len(fourcube) / max_steps))
        dprint(fourcube.shape, max_steps, len(fourcube) / max_steps, num_chunks)
        # cam.photons = np.empty((4,0))
        cam.rebinned_cube = np.zeros_like(fourcube)
        for chunk in range(num_chunks):
            photons = cam.get_photons(fourcube[chunk*max_steps:(chunk+1)*max_steps], chunk_step=chunk*max_steps)
            photons = cam.degrade_photons(photons)
            # cam.photons = np.hstack((cam.photons, photons))
            # dprint(photons.shape, cam.photons.shape)
            cam.rebinned_cube[chunk*max_steps:(chunk+1)*max_steps] = cam.rebin_list(photons, time_inds=[chunk*max_steps,(chunk+1)*max_steps])
        # cam.rebinned_cube = cam.rebin_list(cam.photons)

        cam.photons = None

        for step in range(len(fields)):
            print(step, cam.max_count)
            if cam.max_count:
                cam.rebinned_cube[step] = cam.cut_max_count(cam.rebinned_cube[step])

        if norm:
            cam.rebinned_cube /= np.sum(cam.rebinned_cube)  # /sp.numframes

        if collapse_time_first:
            grid(cam.rebinned_cube, title='comp sum', show=False, logZ=True)
            cam.rebinned_cube = np.median(cam.rebinned_cube, axis=0)[np.newaxis]
            grid(cam.rebinned_cube, title='comp sum', show=False, logZ=True)

        cam.rebinned_cube = np.transpose(cam.rebinned_cube, (1, 0, 2, 3))

        if plot:
            grid(cam.rebinned_cube, show=True, title='get form photons')

        if cam.usesave:
            cam.save_instance()

    return cam

def save_params(param_list):
    """

    :param param_list: [ap, sp, tp, iop, atmp, cdip, mp] or less
    :return:
    """
    print(f'storing params for {[param.__name__() for param in param_list]}')
    save_state = [copy.deepcopy(param) for param in param_list]
    return save_state

def restore_params(save_state, params):
    """"""
    for i, param in enumerate(params):
        param.__dict__ = save_state[i].__dict__
