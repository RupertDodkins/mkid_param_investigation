import numpy as np
from run_compare import ObservatoryMaster, MetricTester
import metrics

from medis.utils import dprint

from master import ap, sp, tp, iop, atmp, mp

metric_name = 'pix_yield'
ap.star_flux = int(1e9*1e4)
# sp.debug = True
sp.numframes = 10
sp.verbose = True
mp.array_size = np.array([144, 144])
ap.companion_xy = np.array(ap.companion_xy)
ap.companion_xy *= 140/150
# ap.contrast = 10**np.array([-3.5, -4, -4.5, -3.5] * 3)
# atmp.cn_sq = 0.5e-11
mp.dark_counts = False
# tp.prescription = 'Subaru_SCExAO'
# sp.focused_sys = True  # use this to turn scaling of beam ratio by wavelength on/off

# # tp.detector='mkid'
# tp.detector='ideal'

if __name__ == "__main__":
    obs = ObservatoryMaster(iteration=0, name=f"mec_yield_144")

    dprint(iop.testdir)

    comp_images, cont_data, metric_multi_list, metric_vals_list = [], [], [], []
    # for metric_name in metric_names:
        # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, len(param.metric_multiplier))))

    metric_config = metrics.get_metric(metric_name, master_cam=obs.cam)
    metric_test = MetricTester(obs, metric_config)
    metric_results = metric_test()

