#!/usr/bin/python
# -*- coding: latin-1 -*-

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from scipy.io import loadmat

import odog_model as om
import dakin_bex_model as dbm
import biwam
import flodog
import analyze_contour_data

def prepare_adaptors(check_pos, adaptor_type, bar_width=38, shift=0):
    y, x = check_pos
    adapt = np.ones((6 * bar_width, 6 * bar_width)) * .5
    adapt_idx = np.zeros((6 * bar_width, 6 * bar_width), dtype=bool)
    if adaptor_type == 'vertical':
        adapt_idx[y:y+bar_width, x-2-shift:x+2-shift] = True
        adapt_idx[y:y+bar_width, x-2-shift+bar_width:x+2-shift+bar_width] = True
    elif adaptor_type == 'horizontal':
        adapt_idx[y-2-shift:y+2-shift, x:x+bar_width] = True
        adapt_idx[y-2-shift+bar_width:y+2-shift+bar_width, x:x+bar_width] = True
    adapt[adapt_idx] = 0
    return adapt

def prepare_grating(grating_ori, grating_vals, check_pos, check_val,
                        bar_width=38):
    # create square wave grating
    grating = np.ones((bar_width * 6, bar_width * 6)) * grating_vals[1]
    index = [i + j for i in range(bar_width) for j in
                range(0, bar_width * 6, bar_width * 2)]
    if grating_ori == 'vertical':
        grating[:, index] = grating_vals[0]
    elif grating_ori == 'horizontal':
        grating[index, :] = grating_vals[0]
    # place test square at appropriate position
    for pos in check_pos:
        y, x = pos
        grating[y:y+bar_width, x:x+bar_width] = check_val
    return grating

def analyze_adaptation(alphas=[.5, .2, 0], mus=[.08, .01, .001], sigmas=[.04,
    .005, .0001]):
    adaptor_para = prepare_adaptors((38 * 3, 38 * 2.5), 'horizontal')
    adaptor_ortho = prepare_adaptors((38 * 3, 38 * 2.5), 'vertical')
    adaptor_ortho_shifted = np.roll(adaptor_ortho, 19, 1)
    adaptor_para_shifted = np.roll(adaptor_para, 19, 0)
    adaptors = [None, adaptor_ortho, adaptor_para, adaptor_ortho_shifted,
            adaptor_para_shifted]

    stim_dec = prepare_grating('horizontal', (.45, .55), ((38*3, 38*2.5),), .5)
    stim_inc = prepare_grating('horizontal', (.55, .45), ((38*3, 38*2.5),), .5)

    model = om.OdogModel(img_size=(512, 512), pixels_per_degree=31.277)
    result_inc = model.evaluate(stim_inc, pad_val=.5)
    result_dec = model.evaluate(stim_dec, pad_val=.5)
    results = np.empty((2, 5, len(alphas), len(betas), len(gammas)))
    #results[0, 0, ...] = result_inc[stim_inc == .5].mean()
    #results[1, 0, ...] = result_dec[stim_dec == .5].mean()

    for j, adaptor in enumerate(adaptors):
        print j
        for b, mu in enumerate(mus):
            for c, sigma in enumerate(sigmas):
                for a, max_attenuation in enumerate(alphas):
                    result_inc = model.evaluate(stim_inc, pad_val=.5,
                            adapt=adaptor, adapt_mu=mu,
                            max_attenuation=max_attenuation,
                            adapt_sigma=sigma)
                    result_dec = model.evaluate(stim_dec, pad_val=.5,
                            adapt=adaptor, adapt_mu=mu,
                            max_attenuation=max_attenuation,
                            adapt_sigma=sigma)
                    results[0, j, a, b, c] = result_inc[stim_inc == .5].mean()
                    results[1, j, a, b, c] = result_dec[stim_dec == .5].mean()
                    if j == 0:
                        results[0, j, ...] = result_inc[stim_inc == .5].mean()
                        results[1, j, ...] = result_dec[stim_dec == .5].mean()
                        break
                if j == 0: break
            if j == 0: break

    filenumber = 0
    while os.path.exists('../data/odog_results%d.npz' % filenumber):
        filenumber += 1
    np.savez('../data/odog_results%d.npz' % filenumber, results=results,
            alphas=alphas, mus=mus, sigmas=sigmas)

def flodog_effectsizes(datadir):
    effect_sizes = np.empty((5, 3, 3, 3))
    for i, adaptor in enumerate(
            ['none', 'ortho', 'para', 'ortho_shifted', 'para_shifted']):
        dec = loadmat(os.path.join(datadir, 'dec_%s.mat' % adaptor))
        inc = loadmat(os.path.join(datadir, 'inc_%s.mat' % adaptor))
        effect_sizes[i, ...] = inc['results'] - dec['results']
    return effect_sizes, np.squeeze(dec['mus']), np.squeeze(dec['sigmas'])

def biwam_effectsizes():
    datadir = '../data/explore_biwam_full'
    baseline = 0.00827578
    effect_sizes = np.empty((4, 2, 200, 8))
    for i, adaptor in enumerate(
            ['ortho', 'para', 'ortho_shifted', 'para_shifted']):
        dec = loadmat(os.path.join(datadir, 'dec_%s.mat' % adaptor))
        inc = loadmat(os.path.join(datadir, 'inc_%s.mat' % adaptor))
        effect_sizes[i, ...] = (inc['results'] - dec['results']) / baseline
    # weird as hell bug workaround (Python crashes if I try to use alphas =
    # np.squeeze(dec['alphas']) directly. And I mean, it dies.
    alphas = np.array([a for a in np.squeeze(dec['alphas'])])
    mus = np.array([m for m in np.squeeze(dec['mus'])])
    sigmas = np.array([s for s in np.squeeze(dec['sigmas'])])

    return (effect_sizes, alphas, mus, sigmas)


def analyze_biwam():
    effects, alphas, mus, sigmas = biwam_effectsizes()
    colors = plt.cm.Blues(np.linspace(.2,1,8))
    mpl.rcParams['axes.color_cycle'] = list(colors)


    # explore effect of alpha
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(121)
    ax.plot(mus, effects[0,0,...])
    ax.set_ylim((-1.5, 4.5))
    ax.set_xlim((mus.min(), mus.max()))
    ax.set_xticks((.0005, .025, .05, .075, .1))
    ax.set_xticklabels(('.0005', '', '.05', '', '.1'))
    ax.set_yticks((-1, 0, 1, 2, 3, 4))
    ax.set_title(r'$\alpha = 0.2$')
    ax = fig.add_subplot(122)
    ax.plot(mus, effects[0,1,...])
    ax.set_ylim((-1.5, 4.5))
    ax.set_yticks((-1, 0, 1, 2, 3, 4))
    ax.set_xticks((.0005, .025, .05, .075, .1))
    ax.set_xticklabels(('.0005', '', '.05', '', '.1'))
    ax.set_xlim((mus.min(), mus.max()))
    ax.set_title(r'$\alpha = 0.0$')
    fig.savefig('../figures/model_adapt/biwam/alpha_effect.pdf')
    plt.close(fig)

    #
    fig = plt.figure(figsize=(6,6))
    for i in range(4):
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(mus, effects[i,0,...])
        ax.set_ylim((-1.5, 6.5))
        ax.set_xlim((mus.min(), mus.max()))
        ax.set_xticks((.0005, .025, .05, .075, .1))
        ax.set_xticklabels(('.0005', '', '.05', '', '.1'))
        ax.set_yticks((-1, 0, 1, 2, 3, 4, 5))
        ax.set_title(['ortho', 'para', 'ortho_shifted', 'para_shifted'][i])
    fig.savefig('../figures/model_adapt/biwam/mu_effect.pdf')
    plt.close(fig)

def plot_effectsize(fig, ax, effect_sizes, mus, sigmas):
    # iterate over adapt cutoffs. skip cutoff 1 because it leads to almost no
    # adaptation
    symbols = ['o', 's', '^']
    colors = ['w', '.5', 'k']
    sizes = [9, 7, 5]
    effect_sizes = effect_sizes / effect_sizes[0, 0, 0, 0]
    for c in [0, 1, 2]:
        # iterate over max_attenuation values
        for b in [0, 1, 2]:
            x_vals = np.array([0, .075, .15]) + .25 * c
            # iterate over adapt exponents
            for a in [0, 1, 2]:
                if sigmas[c] <= mus[b] / 2:
                    ax.plot(x_vals[a] + np.arange(5), effect_sizes[:, a, b, c],
                            '-', marker=symbols[a], lw=0, color='.9', ms=sizes[c],
                            mfc=colors[b], mec='k')
    # plot individual subject data
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.get_yaxis().tick_left()
    ax.set_xlim([-.1, 5])
    ax.set_ylim([-1.5, 1.7])
    ax.set_yticks(np.arange(-1, 1.1, .5))
    ax.set_yticklabels(ax.get_yticks(), fontname='Helvetica', fontsize=10)
    ax.set_ylabel('Illusion strength', fontname='Helvetica',
            fontsize=12)
    ax.hlines(0, 0, 5.1, 'k')
    #ax.vlines(np.arange(5) + 1, -8, 8, linestyles='dashed', colors='k')

if __name__ == '__main__':
    # create effect size plots
    results_odog = np.load('../data/odog_results1.npz')
    eff_odog = np.squeeze(np.diff(results_odog['results'], axis=0))
    mus_odog = results_odog['mus']
    sigmas_odog = results_odog['sigmas']
    eff_flodog, mus_flodog, sigmas_flodog = flodog_effectsizes('../data/flodog')

    good_vps = ['ad', 'cy', 'ak', 'rh', 'vyf', 'ts'] #observers with White's
    data = analyze_contour_data.get_all_data(good_vps)
    effect_sizes, _ = analyze_contour_data.compute_effectsize(data)
    effect_sizes /= effect_sizes.mean(1)[0]
    fig = plt.figure(frameon=False, figsize=(7.5, 5.5))
    for effects, mus, sigmas, i in [
            (eff_odog, mus_odog, sigmas_odog, 0),
            (eff_flodog, mus_flodog, sigmas_flodog, 1)]:
        ax = fig.add_subplot(2,1,i)
        plot_effectsize(fig, ax, effects, mus, sigmas)
        ax.plot(np.arange(5) + .325, effect_sizes.mean(1)[:5], 'o', ms=8, mfc='r',
                mec='None', zorder=25)
    titles = ['no_adaptor', 'orthogonal', 'parallel', 'orthogonal_shifted',
            'parallel_shifted']
    for k in range(5):
        ax_icon = fig.add_axes((.15 + k * .15, 0, .10, .10), frameon=True)
        ax_icon.set_xticks([])
        ax_icon.set_yticks([])
        ax_icon.imshow(plt.imread('../figures/contour_adapt/icons/%s_icon.png' %
            titles[k]))
    fig.savefig('../figures/contour_adapt/model_adaptation.pdf',
            transparent=True)
    plt.close(fig)
