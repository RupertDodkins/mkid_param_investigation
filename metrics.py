"""

Module hosting the MetricConfig Class to created adapted mkids.Camera objects and run tailored
get_stackcubes and pca_stackcubes

"""

import os
import numpy as np
import copy as copy

from medis.plot_tools import quick2D, view_spectra
from medis.utils import dprint

from substitution import get_form_photons
from master2 import params
from medis.MKIDs import Camera

def check_attributes(metric):
    atributes_requirement = ['name', 'vals', 'master_cam', 'testdir', 'cams', 'create_adapted_cams']
    assert np.all([hasattr(metric, attr) for attr in atributes_requirement])

class numframes():
    def __init__(self, name, master_cam, testdir):
        self.name = __file__.split('/')[-1].split('.')[0] if name is None else name

        median_val = 10
        self.multiplier = np.logspace(np.log10(0.2), np.log10(5), 7)
        self.vals = np.int_(np.round(median_val * self.multiplier))
        self.master_cam = master_cam
        self.testdir = testdir
        self.cams = self.create_adapted_cams()

        if not os.path.exists(self.testdir):
            os.mkdir(self.testdir)

    def create_adapted_cams(self):
        cams = {'star': [], 'comp': []}
        for obj in cams.keys():
            for i, val in enumerate(self.vals):
                new_cam = copy.copy(self.master_cam)
                new_cam.stackcube = None
                new_cam.name = os.path.join(self.testdir, f'camera_{self.name}={val}_comp={obj}.pkl')
                cams[obj].append(new_cam)

        return cams

    def get_stackcubes(self, master_fields, comps=True):
        obj = 'comp' if comps else 'star'
        for i, cam, metric_val in zip(range(len(self.cams[obj])), self.cams[obj], self.vals):
            reduced_fields = master_fields[:metric_val]
            cam = get_form_photons(reduced_fields, cam, comps=comps)

            self.cams[obj][i] = cam

        # return self.cams

class array_size():
    def __init__(self, name, master_cam, testdir):
        self.name = __file__.split('/')[-1].split('.')[0] if name is None else name

        median_val = params['mp'].array_size[0]
        self.multiplier = np.logspace(np.log10(0.25), np.log10(4), 7)
        self.vals = np.int_(np.round(median_val * np.sqrt(self.multiplier)))
        self.master_cam = master_cam
        self.testdir = testdir
        # self.cams = {'star': [], 'comp': []}
        self.params = self.master_cam.params
        self.cams = self.create_adapted_cams()

        if not os.path.exists(self.testdir):
            os.mkdir(self.testdir)

    def create_adapted_cams(self):
        """ Create adapted cams """
        cams = {'star': [], 'comp': []}
        metric_orig = getattr(self.params['mp'], self.name)
        for obj in cams.keys():
            for i, val in enumerate(self.vals):
                # mp.array_size = np.array([metric_val] * 2)
                params = copy.copy(self.params)
                params['mp'].array_size = np.array([val] * 2)
                new_cam = Camera(params, usesave=False, fields=False)  # these two args mean no fields will be produced
                new_cam.lod = (val / metric_orig[0]) * params['mp'].lod
                dprint(val, metric_orig[0], new_cam.lod)
                new_cam.platescale = params['mp'].platescale * metric_orig[0] / val
                new_cam.array_size = np.array([val, val])
                new_cam.name = os.path.join(self.testdir, f'camera_{self.name}={val}_comp={obj}.pkl')
                cams[obj].append(new_cam)

        return cams

