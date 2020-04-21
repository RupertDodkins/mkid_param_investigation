"""
Prescription containing a full system that can be fully toggled using switches in the parent module

*** This module is currently unverified ***

TODO
    * verify this script using all the modes
    * reimplement tiptilt again?

"""

import proper

import medis.atmosphere as atmos
import medis.adaptive as ao
import medis.aberrations as aber
import medis.optics as opx
from medis.coronagraphy import coronagraph
from medis.plot_tools import quick2D, quicklook_wf, view_spectra


def general_telescope(empty_lamda, grid_size, PASSVALUE):
    """
    #TODO pass complex datacube for photon phases

    propagates instantaneous complex E-field through the optical system in loop over wavelength range

    this function is called as a 'prescription' by proper

    uses PyPROPER3 to generate the complex E-field at the source, then propagates it through atmosphere, then telescope, to the focal plane
    currently: optics system "hard coded" as single aperture and lens
    the AO simulator happens here
    this does not include the observation of the wavefront by the detector
    :returns spectral cube at instantaneous time
    """
    # print("Propagating Broadband Wavefront Through Telescope")

    # datacube = []

    # passpara = PASSVALUE['params']
    # tp.__dict__ = passpara['tp'].__dict__
    params = PASSVALUE['params']

    wfo = opx.Wavefronts()
    wfo.initialize_proper()

    ###################################################
    # Aperture, Atmosphere, and Secondary Obscuration
    ###################################################
    # Defines aperture (baffle-before primary)
    wfo.loop_collection(proper.prop_circular_aperture, **{'radius': params['tp'].entrance_d/2})
    # wfo.loop_collection(proper.prop_define_entrance)  # normalizes the intensity

    # Pass through a mini-atmosphere inside the telescope baffle
    #  The atmospheric model used here (as of 3/5/19) uses different scale heights,
    #  wind speeds, etc to generate an atmosphere, but then flattens it all into
    #  a single phase mask. The phase mask is a real-valued delay lengths across
    #  the array from infinity. The delay length thus corresponds to a different
    #  phase offset at a particular frequency.
    # quicklook_wf(wfo.wf_collection[0,0])
    if params['tp'].use_atmos:
        wfo.loop_collection(atmos.add_atmos, PASSVALUE['iter'], plane_name='atmosphere')
        # quicklook_wf(wfo.wf_collection[0, 0])
    wfo.abs_zeros()

    # quicklook_wf(wfo.wf_collection[0,0])
    #TODO rotate atmos not yet implementid in 2.0
    # if params['tp'].rotate_atmos:
    #     wfo.loop_collection(aber.rotate_atmos, *(PASSVALUE['iter']))

    # Both offsets and scales the companion wavefront
    if wfo.wf_collection.shape[1] > 1:
        wfo.loop_collection(opx.offset_companion)
        wfo.loop_collection(proper.prop_circular_aperture,
                            **{'radius': params['tp'].entrance_d / 2})  # clear inside, dark outside

    # TODO rotate atmos not yet implementid in 2.0
    # if params['tp'].rotate_sky:
    #     wfo.loop_collection(opx.rotate_sky, *PASSVALUE['iter'])

    ########################################
    # Telescope Primary-ish Aberrations
    #######################################
    # Abberations before AO
    wfo.loop_collection(aber.add_aber, params['tp'].entrance_d, params['tp'].aber_params, params['tp'].aber_vals,
                           step=PASSVALUE['iter'], lens_name='CPA')
    # wfo.loop_collection(proper.prop_circular_aperture, **{'radius': params['tp'].entrance_d / 2})
    # wfo.wf_collection = aber.abs_zeros(wfo.wf_collection)
    wfo.abs_zeros()

    #######################################
    # AO
    #######################################
    # quicklook_wf(wfo.wf_collection[0,0], 'before AO')
    if params['tp'].use_ao:

        if params['sp'].closed_loop:
            previous_output = ao.retro_wfs(PASSVALUE['AO_field'], wfo)  # unwrap a previous steps phase map
            wfo.loop_collection(ao.deformable_mirror, WFS_map=None, iter=PASSVALUE['iter'],
                                previous_output=previous_output, plane_name='deformable mirror')
        elif params['sp'].ao_delay > 0:
            WFS_map = ao.retro_wfs(PASSVALUE['WFS_field'], wfo)  # unwrap a previous steps phase map
            wfo.loop_collection(ao.deformable_mirror, WFS_map, iter=PASSVALUE['iter'], previous_output=None,
                                plane_name='deformable mirror')
        else:
            WFS_map = ao.open_loop_wfs(wfo)  # just uwraps this steps measured phase_map
            wfo.loop_collection(ao.deformable_mirror, WFS_map, iter=PASSVALUE['iter'], previous_output=None,
                                tp=params['tp'], plane_name='deformable mirror')

    # Obscure Baffle
    if params['tp'].obscure:
        wfo.loop_collection(opx.add_obscurations, M2_frac=1/8, d_primary=params['tp'].entrance_d, legs_frac=params['tp'].legs_frac)

    # quicklook_wf(wfo.wf_collection[0,0])
   ########################################
    # Post-AO Telescope Distortions
    # #######################################
    # Abberations after the AO Loop
    wfo.loop_collection(aber.add_aber, params['tp'].entrance_d, params['tp'].aber_params, params['tp'].aber_vals,
                           step=PASSVALUE['iter'], lens_name='NCPA')
    wfo.loop_collection(proper.prop_circular_aperture, **{'radius': params['tp'].entrance_d / 2})
    # TODO does this need to be here?
    # wfo.loop_collection(opx.add_obscurations, params['tp'].entrance_d/4, legs=False)
    # wfo.wf_collection = aber.abs_zeros(wfo.wf_collection)
    # quicklook_wf(wfo.wf_collection[0,0], title='NCPA')
    wfo.loop_collection(opx.prop_pass_lens, params['tp'].f_lens, params['tp'].f_lens, plane_name='pre_coron')
    # quicklook_wf(wfo.wf_collection[0,0], title='lens')
    ########################################
    # Coronagraph
    ########################################
    # there are additional un-aberated optics in the coronagraph module

    if params['tp'].use_coronagraph:
        wfo.loop_collection(coronagraph, occulter_mode=params['tp'].cg_type, plane_name='coronagraph')

    ########################################
    # Focal Plane
    ########################################
    cpx_planes, sampling = wfo.focal_plane()

    print(f"Finished datacube at timestep = {PASSVALUE['iter']}")

    return cpx_planes, wfo.plane_sampling
