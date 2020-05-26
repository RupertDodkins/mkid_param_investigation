import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

from medis.utils import dprint

def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${} e^{{{}}}$'.format(a, b)

def contrcurve_plot(metric_vals, rad_samps, thruputs, noises, conts):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 3.4))

    # plotdata[:, 2] = plotdata[:, 1]*plotdata[:, 3] / np.mean(plotdata[:, 0], axis=0)

    for rad_samp, thruput in zip(rad_samps, thruputs):
        axes[0].plot(rad_samp, thruput)
    for rad_samp, noise in zip(rad_samps, noises):
        axes[1].plot(rad_samp, noise)
    for rad_samp, cont in zip(rad_samps, conts):
        axes[2].plot(rad_samp, cont)
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xlabel('Radial Separation (mas)')
        ax.tick_params(direction='in', which='both', right=True, top=True)
    axes[0].set_ylabel('Throughput')
    axes[1].set_ylabel('Noise')
    axes[2].set_ylabel('5$\sigma$ Contrast')
    axes[2].legend([str(metric_val) for metric_val in metric_vals])

def combo_performance(maps, rad_samps, conts, metric_vals, param_name, plot_inds=[0, 3, 6], err=None,
                      metric_multi=None, three_lod_conts=None, three_lod_errs=None, six_lod_conts=None,
                      six_lod_errs=None, savedir=''):

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, len(conts))))
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    labels = ['i', 'ii', 'iii', 'iv', 'v']
    title = r'  $I / I^{*}$'
    # vmin = -1e-8
    # vmin = 0.001
    # # vmax = 1e-6
    # vmax = 10
    vmin, vmax = 1e-8, 1e-4
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 3.4))
    param_name = param_name.replace('_', ' ')
    if param_name == 'R mean': param_name = r'$R$'
    if param_name == 'g mean': param_name = r'QE'
    if param_name == 'array size': param_name = r'$w$'
    if param_name == 'numframes': param_name = r'$\tau$'
    if param_name == 'dark bright': param_name = r'$d$'
    if param_name == 'pix yield': param_name = r'$Y$'
    if param_name == 'max count': param_name = r'$m$'
    # fig.suptitle(param_name, x=0.515)

    dprint(metric_vals, plot_inds)
    for m, ax in enumerate(axes[:2]):
        im = ax.imshow(maps[plot_inds[m]], interpolation='none', vmin=vmin, vmax=vmax, #origin='lower',
                       norm=SymLogNorm(linthresh=1e-8), cmap="inferno")
        ax.text(0.05, 0.05, f'{param_name}$=$' + str(metric_vals[plot_inds[m]]), transform=ax.transAxes, #fontweight='bold',
                color='w', fontsize=16)
        # anno =
        ax.text(0.04, 0.9, labels[m], transform=ax.transAxes, fontweight='bold', color='w', fontsize=22,
                family='serif')
        ax.axis('off')

    axes[0].text(0.84, 0.9, '0.2"', transform=axes[0].transAxes, fontweight='bold', color='w', ha='center',
                 fontsize=14,
                 family='serif')
    # axes[1].plot([114, 134], [130, 130], color='w', linestyle='-', linewidth=3)
    axes[0].plot([0.76, 0.89], [0.87, 0.87], transform=axes[0].transAxes, color='w', linestyle='-', linewidth=3)

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("left", size="5%", pad=0.05)
    # cax.yaxis.set_ticks_position('left')
    # cax.yaxis.set_label_position('left')

    cb = fig.colorbar(im, cax=cax, orientation='vertical', norm=LogNorm(), format=ticker.FuncFormatter(fmt))
    cax.yaxis.set_ticks_position("left")
    cb.ax.set_title(title, fontsize=16)  #
    # cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5, endpoint=True)
    # cbar_ticks = [-1e-8, 0, 1e-8, 1e-7, 1e-6]
    # cbar_ticks = [0, 1e-8, 1e-7, 1e-6]
    cbar_ticks = np.logspace(np.log10(1e-8), np.log10(1e-4), 5)
    cb.set_ticks(cbar_ticks)

    for f, (rad_samp, cont) in enumerate(zip(rad_samps, conts)):
        if err is not None:
            axes[2].errorbar(rad_samp, cont, yerr=err[f], label='%5.2f' % metric_vals[f])
        else:
            axes[2].plot(rad_samp, cont, label='%5.2f' % metric_vals[f])

    axes[2].set_yscale('log')
    axes[2].set_xlabel('Radial Separation (")')
    axes[2].tick_params(direction='in', which='both', right=True, top=True)
    axes[2].set_ylabel('5$\sigma$ Contrast')
    planet_seps = np.arange(1.5, 7.5, 0.5) * 0.1
    # contrast = np.array([[e,e] for e in np.arange(-3.5,-5.5,-0.5)]).flatten()
    contrast = np.array([-3.5, -4, -4.5, -5] * 3)
    axes[2].scatter(planet_seps, 10 ** contrast, marker='o', color='k', label='Planets')
    axes[2].legend(ncol=2, fontsize=8, loc='upper right')
    axes[2].text(0.04, 0.9, labels[2], transform=axes[2].transAxes, fontweight='bold', color='k', fontsize=22,
                 family='serif')

    colors = plt.cycler("color", plt.cm.gnuplot2(np.linspace(0, 1, 3))).by_key()["color"]
    if np.any([metric_multi, three_lod_conts, three_lod_errs, six_lod_conts, six_lod_errs]):
        from scipy.optimize import curve_fit

        def func(x, a, b, c):
            return a * np.exp(-b * x) + c

        if param_name in ['R_sig', 'g_sig', r'$d$']:  # 'dark_bright',
            three_lod_conts = three_lod_conts[::-1]
            six_lod_conts = six_lod_conts[::-1]

        fit = True
        try:
            if np.any(three_lod_errs == 0) or np.any(six_lod_errs == 0):
                popt3, pcov3 = curve_fit(func, metric_multi, three_lod_conts)
                popt6, pcov6 = curve_fit(func, metric_multi, six_lod_conts)
            else:
                popt3, pcov3 = curve_fit(func, metric_multi, three_lod_conts, sigma=three_lod_errs, maxfev = 10000)
                popt6, pcov6 = curve_fit(func, metric_multi, six_lod_conts, sigma=six_lod_errs, maxfev = 10000)
        except:
            print('Could not find fit')
            fit = False

        # axes[2].get_shared_y_axes().join(axes[2], axes[3])
        # axes[3].set_yscale('log')
        axes[3].set_xscale('log')
        axes[3].set_xlabel(f'{param_name}/{param_name}'+r'$_\mathrm{med}$')

        axes[3].tick_params(direction='in', which='both', right=True, top=True)

        if fit:
            axes[3].plot(metric_multi, func(metric_multi, *popt3), label=r'$3\lambda/D$', c=colors[0])  #r'$3\lambda/D$: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt3)
            axes[3].plot(metric_multi, func(metric_multi, *popt6), label=r'$6\lambda/D$', c=colors[1])
        dprint(three_lod_errs, six_lod_errs)
        axes[3].errorbar(metric_multi, three_lod_conts, yerr=three_lod_errs, linewidth=1, fmt='.',
                         c=colors[0])
        axes[3].errorbar(metric_multi, six_lod_conts, yerr=six_lod_errs, linewidth=1, fmt='.',
                         c=colors[1])
        axes[3].legend(fontsize=8)
        axes[3].text(0.04, 0.9, labels[3], transform=axes[3].transAxes, fontweight='bold', color='k', fontsize=22,
                     family='serif')

        ax3_top = axes[3].twiny()
        ax3_top.set_xscale('log')
        ax3_top.tick_params(direction='in', which='both', right=True, top=True)
        ax3_top.set_xlabel(f'{param_name}')
        if fit: ax3_top.plot(metric_vals, func(metric_multi, *popt3), linewidth=0)

        # for ax in [axes[3], ax3_top]:
        #     for axis in [ax.xaxis, ax.yaxis]:
        #         axis.set_major_formatter(ScalarFormatter())
        for ax in [axes[3], ax3_top]:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))

    # plt.tight_layout()
    plt.subplots_adjust(left=0.045, bottom=0.145, right=0.985, top=0.87, wspace=0.31)
    dprint(os.path.join(savedir, param_name.replace('$', '') + '.pdf'))
    fig.savefig(os.path.join(savedir, param_name.replace('$', '') + '.pdf'))
    plt.show(block=True)