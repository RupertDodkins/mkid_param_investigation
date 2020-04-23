import numpy as np
import pickle

from medis.params import params
import medis.MKIDs as mkids

params['sp'].show_wframe = False
params['sp'].save_obs = False
params['sp'].show_cube = False
params['sp'].num_processes = 1
params['sp'].save_fields = False
params['sp'].save_ints = True
params['sp'].cont_save = True
params['sp'].grid_size = 512
params['sp'].closed_loop = False

params['ap'].companion = True
params['ap'].star_flux = int(1e10)
# params['ap'].contrast = 10**np.array([-3.5, -4, -4.5, -5] * 2)
# params['ap'].companion_xy = [[2.5,0], [0,3], [-3.5,0], [0,-4], [4.5,0], [0,5], [-5.5,0],[0,-6]]
# params['ap'].n_wvl_init = 8
# params['ap'].n_wvl_final = 16
params['ap'].contrast = [10**-3.5]
params['ap'].companion_xy = [[2.5,0]]
params['ap'].n_wvl_init = 2
params['ap'].n_wvl_final = 2

params['tp'].prescription = 'general_telescope'
params['sp'].beam_ratio = 0.3 #0.25
# sp.save_locs = np.empty((0, 1))
params['tp'].entrance_d = 8.
params['tp'].obscure = True
params['tp'].use_ao = True
params['tp'].include_tiptilt = False
params['tp'].ao_act = 50
params['tp'].platescale = 10  # mas
params['tp'].detector = 'ideal'
params['tp'].use_atmos = True
params['tp'].use_zern_ab = False
params['tp'].occulter_type = 'Vortex'
params['tp'].aber_params = {'CPA': True,
                         'NCPA': True,
                         'QuasiStatic': False,  # or Static
                         'Phase': True,
                         'Amp': False,
                         'n_surfs': 4}
params['tp'].aber_vals = {'a': [5e-18, 1e-19],
                       'b': [2.0, 0.2],
                       'c': [3.1, 0.5]}
params['tp'].piston_error = False
params['ap'].wvl_range = np.array([800, 1500]) /1e9
params['tp'].rot_rate = 0  # deg/s
params['tp'].pix_shift = [[0, 0]]
params['tp'].satelite_speck = False
params['tp'].legs_frac = 0.03
params['tp'].f_lens = 200.0 * params['tp'].entrance_d
params['tp'].use_coronagraph = True
params['tp'].occult_loc = [0,0]
params['tp'].cg_type = 'Gaussian'
params['tp'].cg_size = 3  # physical size or lambda/D size
params['tp'].cg_size_units = "l/D"  # "m" or "l/D"
params['tp'].lyot_size = 0.75  # units are in fraction of surface blocked
params['tp'].fl_cg_lens = params['tp'].f_lens  # m

# for param in [params['sp']]:
#     pprint(param.__dict__)
params['mp'].phase_uncertainty = True
params['mp'].phase_background = False
params['mp'].QE_var = True
params['mp'].bad_pix = True
params['mp'].dark_counts = True
params['mp'].dark_pix_frac = 0.1
params['mp'].dark_bright = 20
params['mp'].hot_pix = None
params['mp'].hot_bright = 2.5 * 10 ** 3
params['mp'].R_mean = 8
params['mp'].R_sig = 2
params['mp'].R_spec = 1.
params['mp'].g_mean = 0.3
params['mp'].g_sig = 0.04
params['mp'].bg_mean = -10
params['mp'].bg_sig = 40
params['mp'].pix_yield = 0.9
params['mp'].array_size = np.array([150, 150])
params['mp'].lod = 6
params['mp'].remove_close = False
params['mp'].quantize_FCs = True
params['mp'].wavecal_coeffs = [1.e9 / 6, -250]

def get_form_photons(fields, cam, comps=True):
    """
    Alternative to cam.__call__ that allows the user to specify whether the spectracube contains the planets

    :param fields: ndarray
    :param cam: mkids.Camera()
    :param comps: bool

    :return:
    mkids.Camera()
    """
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

    cam.save()

    return cam
