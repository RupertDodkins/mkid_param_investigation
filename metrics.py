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

class MetricConfig():
    def __init__(self, name, master_cam, testdir):
        self.name = __file__.split('/')[-1].split('.')[0] if name is None else name

        median_val = 10
        self.multiplier = np.logspace(np.log10(0.2), np.log10(5), 7)
        self.vals = np.int_(np.round(median_val * self.multiplier))
        self.master_cam = master_cam
        self.testdir = testdir
        self.cams = {'star': [], 'comp': []}

        if not os.path.exists(self.testdir):
            os.mkdir(self.testdir)

    def create_adapted_cams(self):
        for obj in self.cams.keys():
            for i, val in enumerate(self.vals):
                new_cam = copy.copy(self.master_cam)
                new_cam.stackcube = None
                new_cam.name = os.path.join(self.testdir, f'camera_{self.name}={val}_comp={obj}.pkl')
                self.cams[obj].append(new_cam)

    def get_stackcubes(self, master_fields, comps=True):
        obj = 'comp' if comps else 'star'
        for i, cam, metric_val in zip(range(len(self.cams[obj])), self.cams[obj], self.vals):
            reduced_fields = master_fields[:metric_val]
            cam = get_form_photons(reduced_fields, cam, comps=comps)

            self.cams[obj][i] = cam

        # return self.cams

