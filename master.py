import numpy as np
from medis.params import sp, ap, tp, iop, mp, atmp, cdip

sp.show_wframe = False
sp.save_obs = False
sp.show_cube = False
sp.num_processes = 5
sp.save_fields = False
sp.save_ints = True
sp.cont_save = True
sp.grid_size = 512
sp.closed_loop = False
sp.save_to_disk = True
sp.memory_limit = 9  # GB
sp.sample_time = 0.5

ap.companion = True
ap.star_flux = int(1e9/50)
ap.companion_xy = [[1.5,0], [0,2], [-2.5,0], [0,-3], [3.5,0], [0,4], [-4.5,0], [0,-5], [5.5,0], [0,6], [-6.5,0], [0,-7]]
ap.contrast = 10**np.array([-3.5, -4, -4.5, -5] * 3)
ap.n_wvl_init = 8
ap.n_wvl_final = 16
sp.numframes = 50

tp.prescription = 'general_telescope'
sp.beam_ratio = 0.25 # 0.3
tp.entrance_d = 8.
tp.obscure = True
tp.use_ao = True
tp.include_tiptilt = False
tp.ao_act = 50
tp.platescale = 10  # mas
tp.detector = 'ideal'
tp.use_atmos = True
tp.use_zern_ab = False
tp.aber_params = {'Phase': True, 'Amp': False}
tp.piston_error = False
ap.wvl_range = np.array([800, 1500]) /1e9
tp.rot_rate = 0  # deg/s
tp.pix_shift = [[0, 0]]
tp.satelite_speck = {'apply': True, 'phase': np.pi / 5., 'amp': 12e-9, 'xloc': 12, 'yloc': 12}
tp.legs_frac = 0.03
tp.f_lens = 200.0 * tp.entrance_d
tp.use_coronagraph = True
tp.occult_loc = [0,0]
tp.cg_type = 'Solid'
tp.cg_size = 3  # physical size or lambda/D size
tp.cg_size_units = "l/D"  # "m" or "l/D"
tp.lyot_size = 0.75  # units are in fraction of surface blocked
tp.fl_cg_lens = tp.f_lens  # m
tp.detector = 'mkid'

mp.phase_uncertainty = True
mp.phase_background = False
mp.QE_var = True
mp.bad_pix = True
mp.dark_counts = True
mp.dark_pix_frac = 0.1
mp.dark_bright = 0.01# * 1e2
mp.hot_pix = None
mp.hot_bright = 2.5 * 10 ** 3
mp.R_mean = 8
mp.R_sig = 2
mp.R_spec = 1.
mp.g_mean = 0.3
mp.g_sig = 0.04
mp.bg_mean = -10
mp.bg_sig = 40
mp.pix_yield = 1# 0.9
mp.array_size = np.array([150, 150])
mp.lod = 6
mp.remove_close = False
mp.quantize_FCs = False #True
mp.wavecal_coeffs = [1.e9 / 6, -250]

from os.path import expanduser
home = expanduser("~")
if home == '/Users/dodkins':
    iop.update_datadir('/Users/dodkins/MKIDSim/')
    sp.num_processes = 1
    sp.numframes = 50  # 2000 #500
elif home == '/home/dodkins':
    # os.environ["DISPLAY"] = "/home/dodkins"
    iop.update_datadir('/mnt/data0/dodkins/MEDIS_photonlists/')
    sp.num_processes = 5
    sp.numframes = 10  # 2000 #500
else:
    print('System not recognised. Make sure $WORKING_DIR is set')

