"""

Module hosting the MetricConfig Class to created adapted mkids.Camera objects and run tailored
get_stackcubes and pca_stackcubes

"""

import os
import numpy as np
import copy as copy
import random

from medis.plot_tools import quick2D, view_spectra
from medis.utils import dprint

from substitution import get_form_photons
from medis.MKIDs import Camera

class numframes():
    def __init__(self, master_cam):
        self.master_cam = master_cam
        median_val = 10
        self.multiplier = np.logspace(np.log10(0.2), np.log10(5), 7)
        self.vals = np.int_(np.round(median_val * self.multiplier))

    def update_device(self, new_cam, orig_cam, val, i):
        new_cam.stackcube = None
        # new_cam.name = os.path.join(self.testdir, f'camera_{self.name}={val}_comp={obj}.pkl')
        return new_cam

    def get_stackcubes(self, master_fields, comps=True):
        obj = 'comp' if comps else 'star'
        for i, cam, metric_val in zip(range(len(self.cams[obj])), self.cams[obj], self.vals):
            reduced_fields = master_fields[:metric_val]
            cam = get_form_photons(reduced_fields, cam, comps=comps)

            self.cams[obj][i] = cam

class array_size():
    def __init__(self, master_cam):
        self.master_cam = master_cam
        self.params = self.master_cam.params
        self.median_val = self.params['mp'].array_size[0]
        self.multiplier = np.logspace(np.log10(0.25), np.log10(4), 7)
        self.vals = np.int_(np.round(self.median_val * np.sqrt(self.multiplier)))

    def update_device(self, new_cam, orig_cam, val, i):
        params = copy.deepcopy(self.params)
        params['mp'].array_size = np.array([val] * 2)
        new_cam = Camera(params, usesave=False, fields=False)  # these two args mean no fields will be produced
        new_cam.lod = (val / self.median_val) * params['mp'].lod
        dprint(val, self.median_val, new_cam.lod)
        new_cam.platescale = params['mp'].platescale * self.median_val / val
        new_cam.array_size = np.array([val, val])
        # new_cam.name = os.path.join(self.testdir, f'camera_{self.name}={val}_comp={obj}.pkl')
        return new_cam

class pix_yield():
    def __init__(self, master_cam):
        self.master_cam = master_cam
        median_val = 0.8
        self.multiplier = np.linspace(0.5,1.25,4)
        self.vals = median_val * self.multiplier
        self.vals[self.vals>1] = 1
        self.params = master_cam.params
        self.bad_inds = self.get_bad_inds(self.vals)

    def update_device(self, new_cam, orig_cam, val, i):
        bad_ind = self.bad_inds[i]
        new_cam.pix_yield = val
        new_cam.QE_map = self.add_bad_pix(self.master_cam.QE_map_all, bad_ind)
        return new_cam

    def get_bad_inds(self, pix_yields):
        pix_yields = np.array(pix_yields)
        min_yield = min(pix_yields)
        max_yield = max(pix_yields)
        amount = int(self.params['mp'].array_size[0] * self.params['mp'].array_size[1] * (1. - min_yield))
        all_bad_inds = random.sample(list(range(self.params['mp'].array_size[0] * self.params['mp'].array_size[1])), amount)
        # dprint(len(all_bad_inds))
        bad_inds_inds = np.int_((1 - (pix_yields - min_yield) / (max_yield - min_yield)) * amount)
        bad_inds = []
        for bad_inds_ind in bad_inds_inds:
            bad_inds.append(all_bad_inds[:bad_inds_ind])

        return bad_inds

    def add_bad_pix(self, QE_map_all, bad_ind):
        dprint(len(bad_ind))
        QE_map = np.array(QE_map_all, copy=True)
        if len(bad_ind) > 0:
            bad_y = np.int_(np.floor(bad_ind / self.params['mp'].array_size[1]))
            bad_x = bad_ind % self.params['mp'].array_size[1]
            QE_map[bad_x, bad_y] = 0

        return QE_map

class dark_bright():
    def __init__(self, master_cam):
        self.master_cam = master_cam
        self.params = master_cam.params
        median_val = self.params['mp'].dark_bright
        self.multiplier = np.logspace(np.log10(10), np.log10(0.1), 7)
        self.vals = np.int_(np.round(median_val * self.multiplier))

    def update_device(self, new_cam, orig_cam, val, i):
        new_cam.dark_bright = val
        new_cam.dark_pix_frac = 1. / 2
        new_cam.dark_per_step = self.params['sp'].sample_time * new_cam.dark_bright
        return new_cam

class R_mean():
    def __init__(self, master_cam):
        self.master_cam = master_cam
        self.median_val = master_cam.params['mp'].R_mean
        self.multiplier = np.logspace(np.log10(0.1), np.log10(10), 7)
        self.vals = np.int_(np.round(self.median_val * self.multiplier))

    def update_device(self, new_cam, orig_cam, val, i):
        new_cam.Rs = orig_cam.Rs - self.median_val + val
        new_cam.Rs[new_cam.Rs < 0] = 0
        new_cam.Rs[orig_cam.Rs == 0] = 0
        new_cam.sigs = new_cam.get_R_hyper(new_cam.Rs)
        return new_cam

class g_mean():
    def __init__(self, master_cam):
        self.master_cam = master_cam
        self.params = self.master_cam.params
        self.median_val = self.params['mp'].g_mean
        self.multiplier = np.logspace(np.log10(0.5), np.log10(7/3), 7)
        self.vals = self.median_val * self.multiplier

    def update_device(self, new_cam, orig_cam, val, i):
        new_cam.QE_map = orig_cam.QE_map - self.median_val + val
        new_cam.QE_map[new_cam.QE_map < 0] = 0
        new_cam.QE_map[orig_cam.QE_map == 0] = 0
        return new_cam

def create_cams(metric):
    metric.cams = {'star': [], 'comp': []}
    for obj in metric.cams.keys():
        for i, val in enumerate(metric.vals):
            new_cam = copy.deepcopy(metric.master_cam)
            new_cam = metric.update_device(new_cam, metric.master_cam, val, i)
            metric.cams[obj].append(new_cam)

def get_metric(name, master_cam):
    """
    wrapper for each metric class that automates the common steps among metrics

    :param name: str
        metric_name should match the name of a class in this module
    :param master_cam: mkids.Camera
        master that all cams for each metric are derived from
    :return:
    """

    testdir = os.path.join(os.path.dirname(master_cam.params['iop'].testdir), name)
    MetricConfig = eval(name)
    metric_config = MetricConfig(master_cam)
    metric_config.testdir = testdir
    metric_config.name = name
    create_cams(metric_config)
    if not os.path.exists(testdir):
        os.mkdir(testdir)

    return metric_config

