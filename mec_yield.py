from run_compare import ObservatoryMaster, MetricTester
import metrics

from medis.utils import dprint

from master import ap, sp, tp, iop

metric_name = 'pix_yield'
ap.star_flux = int(1e9*1e4)
# sp.debug = True
sp.verbose = True

# # tp.detector='mkid'
# tp.detector='ideal'

if __name__ == "__main__":
    obs = ObservatoryMaster(iteration=0, name=f"mec_yield")

    dprint(iop.testdir)

    comp_images, cont_data, metric_multi_list, metric_vals_list = [], [], [], []
    # for metric_name in metric_names:
        # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, len(param.metric_multiplier))))

    metric_config = metrics.get_metric(metric_name, master_cam=obs.cam)
    metric_test = MetricTester(obs, metric_config)
    metric_results = metric_test()

    comp_images.append(metric_results['maps'])
    cont_data.append([metric_results['rad_samps'], metric_results['conts']])
