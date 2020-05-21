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

def get_form_photons(fields, cam, comps=True, plot=False):
    """
    Alternative to cam.__call__ that allows the user to specify whether the spectracube contains the planets

    :param fields: ndarray
    :param cam: mkids.Camera()
    :param comps: bool

    :return:
    mkids.Camera()
    """

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

        photons = cam.get_photons(fourcube)
        photons = cam.degrade_photons(photons)
        cam.rebinned_cube = cam.rebin_list(photons)

        for step in range(len(fields)):
            print(step)
            if cam.max_count:
                cam.rebinned_cube[step] = cam.cut_max_count(cam.rebinned_cube[step])

        cam.rebinned_cube /= np.sum(cam.rebinned_cube)  # /sp.numframes
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
