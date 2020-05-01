import medis.medis_main as mm
from master2 import params
params['sp'].save_to_disk = True
params['tp'].prescription = 'general_telescope'
params['sp'].numframes = 1
params['ap'].n_wvl_init = 3
params['ap'].n_wvl_final = 3
params['tp'].use_atmos = True

if __name__ == '__main__':

    sim = mm.RunMedis(params=params, name='example1', product='photons')
    observation = sim()
    print(observation.keys())
