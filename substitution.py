import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from medis.plot_tools import quick2D, view_spectra, body_spectra

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
        print('loading cam stackcube save at {cam.name}')
        with open(cam.name, 'rb') as handle:
            cam = pickle.load(handle)
    else:
        ntime, nwave = fields.shape[0], fields.shape[1]

        cam.stackcube = np.zeros((ntime, nwave, cam.array_size[1], cam.array_size[0]))
        for step in range(len(fields)):
            print(step)
            if comps:
                spectralcube = np.sum(fields[step], axis=1)
            else:
                spectralcube = fields[step, :, 0]

            # step_packets = mkids.get_packets(spectralcube, step, dp, mp)
            # cube = mkids.make_datacube_from_list(step_packets, (nwave,dp.array_size[0],dp.array_size[1]))
            step_packets = cam.get_packets(spectralcube, step)
            cube = cam.make_datacube_from_list(step_packets)
            cam.stackcube[step] = cube

        cam.stackcube /= np.sum(cam.stackcube)  # /sp.numframes
        cam.stackcube = np.transpose(cam.stackcube, (1, 0, 2, 3))

        if plot:
            plt.figure()
            plt.hist(cam.stackcube[cam.stackcube != 0].flatten(), bins=np.linspace(0, 1e4, 50))
            plt.yscale('log')
            body_spectra(cam.stackcube, show=True, title='stackcube')

        if cam.usesave:
            cam.save()

    return cam