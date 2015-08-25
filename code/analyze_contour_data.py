#!/usr/bin/python
# -*- coding: latin-1 -*-

from __future__ import division
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ocupy import datamat
from scipy.stats.stats import pearsonr

def mark_outliers(data):
    """
    Create a mask that marks outliers in a datamat.
    """
    mask = np.zeros(len(data), dtype=bool)
    for adaptor in np.unique(data.adaptor_type):
        for coaxial_lum in np.unique(data.coaxial_lum):
            for test_lum in np.unique(data.test_lum):
                idx = (data.adaptor_type == adaptor) & \
                      (data.coaxial_lum == coaxial_lum) & \
                      (data.test_lum == test_lum)
                q1 = np.percentile(data.match_lum[idx], 25)
                q3 = np.percentile(data.match_lum[idx], 75)
                median = np.median(data.match_lum[idx])
                mask[idx & ((data.match_lum > median + 2 * (q3 - q1)) |
                            (data.match_lum < median - 2 * (q3 - q1)))] = 1
    return mask

def inspect_outliers(vp):
    data = get_all_data([vp], clean=False)
    outlier_idx = np.nonzero(mark_outliers(data))[0]
    for trial in outlier_idx:
        print 'value: %1.3f; start_val: %1.3f; RT: %d; RT before: %d' % \
        (data.match_lum[trial], data.match_initial[trial],
                data.response_time[trial], data.response_time[trial - 1])

    return data, outlier_idx

def compute_match_contrast(data, vp_name):
    bar_width = 38
    y_border = 768 // 2 - int(6 * bar_width)
    x_border = (1024 - 6 * bar_width) // 2
    test_coords = ((bar_width * 3, bar_width * 2),
                   (bar_width * 3, bar_width * 3),
                   (bar_width * 4, bar_width * 2),
                   (bar_width * 4, bar_width * 3))
    match_coords = ((x_border + bar_width * 2, 384 + int(bar_width * 2)),
                    (x_border + bar_width * 3, 384 + int(bar_width * 2)),
                    (x_border + bar_width * 2, 384 + int(bar_width * 1)),
                    (x_border + bar_width * 3, 384 + int(bar_width * 1)))

    datadir = '../exp_data/check_files'
    check_filenames = [fn for fn in os.listdir(datadir) if '_%s_' % vp_name in fn]
    check_bg = []
    match_contrast_near = np.empty(len(data))
    match_contrast_far = np.empty(len(data))
    for fn in check_filenames:
        check_bg.append(np.loadtxt(os.path.join(datadir, fn), delimiter=','))
    check_bg = np.vstack(check_bg)
    for trial, checks in enumerate(check_bg):
        checks = np.reshape(checks, (13, 13))
        match_bg = np.repeat(np.repeat(checks, 18, 0), 18, 1)[3:-3, 3:-3]
        check_pos = test_coords[data.test_loc[trial]]
        match_pos = match_coords[data.test_loc[trial]]
        h_shift = (check_pos[1] % 18) - 5
        match_bg = np.roll(match_bg, h_shift, axis=1)
        v_shift = ((match_pos[1] - 384) % 18) - 5
        match_bg = np.roll(match_bg, v_shift, axis=0)
        # set test patch lum to 1 to enable easy substraction
        match_bg[match_pos[1] - 384 : match_pos[1] - 384 + 38,
                match_pos[0] - x_border : match_pos[0] - x_border + 38] = 1

        surround_near = match_bg[
                match_pos[1] - 384 - 8 : match_pos[1] - 384 + 38 + 8,
                match_pos[0] - x_border - 8 : match_pos[0] - x_border + 38 + 8]
        surround_far = match_bg[
                match_pos[1] - 384 - 26 : match_pos[1] - 384 + 38 + 26,
                match_pos[0] - x_border - 26 : match_pos[0] - x_border + 64]
        mean_near = (surround_near.sum() - 38**2) / (surround_near.size
                - 38**2) * 88
        mean_far = (surround_far.sum() - 38**2) / (surround_far.size
                - 38**2) * 88

        contrast_near = (data.match_lum[trial] - mean_near) /\
                (data.match_lum[trial] + mean_near)
        contrast_far = (data.match_lum[trial] - mean_far) /\
                (data.match_lum[trial] + mean_far)
        match_contrast_near[trial] = contrast_near
        match_contrast_far[trial] = contrast_far
    return match_contrast_far, match_contrast_near

def bootstrap_effect_size(data, repeats=1000, conf_interval=.95):
    """
    Bootstraps the range in which effect size would fall with conf_interval
    probability if coaxial lum has no effect. Can be used for significance
    testing.
    """
    coaxial_lums = np.unique(data.coaxial_lum)
    means = np.empty((2, 3, repeats))
    for j, by_testlum in enumerate(data.by_field('test_lum')):
        match_lums = by_testlum.match_lum
        for sample, coaxial_lum in enumerate(coaxial_lums):
            sample_size = sum(by_testlum.coaxial_lum == coaxial_lum)
            for i in xrange(repeats):
                means[sample, j, i] = np.random.choice(match_lums,
                    sample_size).mean()
    effect_sizes = np.diff(means, axis=0).mean(1).squeeze()
    cutoff = (1 - conf_interval) / 2 * 100
    return np.percentile(effect_sizes, (cutoff, 100 - cutoff))

def bootstrap_conf_intervals(data, repeats=100, conf_interval=.95):
    """
    Bootstraps the confidence intervals of effect size.
    """
    means = np.empty((2, 3, repeats))
    for j, by_testlum in enumerate(data.by_field('test_lum')):
        for k, by_coaxial in enumerate(by_testlum.by_field('coaxial_lum')):
            match_lums = by_coaxial.match_lum
            sample_size = len(match_lums)
            for i in xrange(repeats):
                means[k, j, i] = np.random.choice(match_lums,
                                                sample_size).mean()
    effect_sizes = -np.diff(means, axis=0).mean(1).squeeze()
    cutoff = (1 - conf_interval) / 2 * 100
    return np.percentile(effect_sizes, (cutoff, 100 - cutoff))

def plot_sbc_effect(ax, data):
    data = data[data.adaptor_type == 'sbc']
    effect_sizes = np.empty((len(np.unique(data.vp))))
    conf_intervals = np.empty((2, len(np.unique(data.vp))))
    for k, by_vp in enumerate(data.by_field('vp')):
        means = np.empty((2, 3))
        for i, by_coaxial in enumerate(by_vp.by_field('coaxial_lum')):
            means[i, :] = [d.match_lum.mean() for d in by_coaxial.by_field('test_lum')]
        effect_sizes[k] = -np.diff(means, axis=0)[0].mean()
        conf_intervals[:, k] = bootstrap_conf_intervals(by_vp)
    # make confidence intervals relative to data
    conf_intervals -= effect_sizes
    conf_intervals[0, :] *= -1
    ax.errorbar(np.linspace(-.4, .4, len(effect_sizes)) + .5,
        effect_sizes, conf_intervals, None, 'ko', capsize=0,
        elinewidth=2, ms=4)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim([0, 1])
    ax.set_xticks(np.linspace(-.4, .4, len(effect_sizes)) + .5)
    ax.set_xticklabels(np.unique(data.vp), rotation=0, ha='center')
    ax.set_ylim([-8, 8])
    #ax.set_yticks(np.arange(-.2, .21, .1))
    ax.set_xlabel('adaptor type')
    ax.set_ylabel('effect size')
    ax.hlines(0, 0, 6.1, 'k')

def compute_effectsize(data, repeats=100):
    effect_sizes = np.empty((6, len(np.unique(data.vp))))
    conf_intervals = np.empty((2, 6, len(np.unique(data.vp))))
    for k, by_vp in enumerate(data.by_field('vp')):
        for by_adaptor in by_vp.by_field('adaptor_type'):
            title = by_adaptor.adaptor_type[0]
            j = 1 if title == 'none' else 2 if title == 'orthogonal' else 3 \
                if title == 'parallel' else 4 if title == 'o_shifted' \
                else 5 if title == 'p_shifted' else 6
            means = np.empty((2, 3))
            for i, by_coaxial in enumerate(by_adaptor.by_field('coaxial_lum')):
                means[i, :] = [d.match_lum.mean() for d in by_coaxial.by_field('test_lum')]
            effect_sizes[j-1, k] = -np.diff(means, axis=0)[0].mean()
            conf_intervals[:, j-1, k] = bootstrap_conf_intervals(by_adaptor,
                    repeats=repeats)
    # make confidence intervals relative to data
    conf_intervals -= effect_sizes
    conf_intervals[0, :] *= -1
    return effect_sizes, conf_intervals


def plot_effectsize(fig, ax, effect_sizes, conf_intervals):
    # define colors for individual subjects
    if effect_sizes.shape[1] == 6:
        colors = ['#c6dbef', '#9ecae1', '#6baed6', '#3182bd',
                '#08519c', '#191970']
    else:
        colors = ['#bdd7e7', '#6baed6', '#2171b5', '#191970']
    # order subjects according to effect size for White's illusion
    idx = np.argsort(effect_sizes[0, :])
    # plot connecting lines between subjects
    x_vals = np.linspace(-.25, .25, effect_sizes.shape[1]) + .5
    for subject in range(effect_sizes.shape[1]):
        plt.plot(x_vals[subject] + np.arange(6), effect_sizes[:,
            idx[subject]], '-', lw=.5, color=colors[subject])#'.9')
    # plot individual subject data
    titles = ['no_adaptor', 'orthogonal', 'parallel', 'orthogonal_shifted',
            'parallel_shifted', 'sbc']
    for k in range(6):
        ax_icon = fig.add_axes((.1422 + k * .12563, .03, .10, .10), frameon=True)
        ax_icon.set_xticks([])
        ax_icon.set_yticks([])
        ax_icon.imshow(plt.imread('../figures/contour_adapt/icons/%s_icon.png' %
            titles[k]))
        for subject in range(effect_sizes.shape[1]):
            ax.errorbar(x_vals[subject] + k, effect_sizes[k, idx[subject]],
                    conf_intervals[:, k, idx[subject]][:, np.newaxis], None, 'o', capsize=0,
                    color=colors[subject], elinewidth=2, ms=4,
                    zorder=20, mew=0, ecolor='.7')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim([0, 6.1])
    ax.set_xticks(np.arange(6) + .5)
    ax.set_xticklabels(ax.get_xticks(), fontname='Helvetica', fontsize=10)
    ax.set_ylim([-9, 8])
    ax.set_yticks(np.arange(-8, 9, 4))
    ax.set_yticklabels(ax.get_yticks(), fontname='Helvetica', fontsize=10)
    ax.set_ylabel('Illusion strength ($\Delta cd/m^2$)', fontname='Helvetica',
            fontsize=12)
    ax.hlines(0, 0, 6.1, 'k')
    #ax.vlines(np.arange(5) + 1, -8, 8, linestyles='dashed', colors='k')
    ax.plot(np.arange(6) + .5, effect_sizes.mean(1), 'o', ms=8, mfc='r',
            mec='None', zorder=25)

def plot_data(ax, data):
    for i, by_coaxial in enumerate(data.by_field('coaxial_lum')):
        color = '.75' if by_coaxial.coaxial_lum[0] == .55 else '.25'
        x_offset = -.2 if by_coaxial.coaxial_lum[0] == .55 else .2
        means = [d.match_lum.mean() for d in by_coaxial.by_field('test_lum')]
        stds = np.array([d.match_lum.std() / len(d) ** .5 for d in
                    by_coaxial.by_field('test_lum')])
        ax.plot(np.unique(by_coaxial.test_lum) + x_offset, means, 's',
                mec=color, mfc=color)
        for by_lum in by_coaxial.by_field('test_lum'):
            points = by_lum.match_lum
            x_vals = np.ones_like(points) * by_lum.test_lum[0] + 2 * x_offset
            ax.plot(x_vals, points, '.', color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim([41, 47])
    ax.set_ylim([33, 61])
    ax.set_xticks([42.24, 44, 45.76])
    ax.set_xticklabels(['42.2', '44', '45.8'], fontname='Helvetica',
            fontsize=10)
    ax.set_yticks((.45 * 88, .5 * 88, .55 * 88))
    ax.set_yticklabels((.45 * 88, 44, .55 * 88), fontname='Helvetica',
            fontsize=10)
    ax.set_ylabel(r'Match luminance ($cd/m^2$)', fontname='Helvetica',
            fontsize=12)
    ax.hlines(np.array((.45, .55, .48, .52, .5)) * 88, 41, 47,
        linestyles=['solid', 'solid', 'dotted', 'dotted', 'solid'],
        linewidths=[.5, .5, .5, .5, .5],
        color=['.8', '.8', '.2', '.2', '.1'])

def plot_contrast_data(ax, data, surround='near'):
    match_contrast = data.match_contrast_near if surround == 'near' else \
        data.match_contrast_far
    data.add_field('match_contrast', match_contrast)
    for i, by_coaxial in enumerate(data.by_field('coaxial_lum')):
        color = '.75' if by_coaxial.coaxial_lum[0] == .55 else '.25'
        x_offset = -.2 if by_coaxial.coaxial_lum[0] == .55 else .2
        means = [d.match_contrast.mean() for d in by_coaxial.by_field('test_lum')]
        stds = np.array([d.match_contrast.std() / len(d) ** .5 for d in
                    by_coaxial.by_field('test_lum')])
        ax.plot(np.unique(by_coaxial.test_lum) + x_offset, means, 's',
                mec=color, mfc=color)
        for by_lum in by_coaxial.by_field('test_lum'):
            points = by_lum.match_contrast
            x_vals = np.ones_like(points) * by_lum.test_lum[0] + 2 * x_offset
            ax.plot(x_vals, points, '.', color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim([41, 47])
    ax.set_ylim([-.1429, .162])
    ax.set_xticks([42.24, 44, 45.76])
    ax.set_xticklabels(['42.2', '44', '45.8'], fontname='Helvetica',
            fontsize=10)
    ax.set_yticks((-.1, 0, .1))
    ax.set_yticklabels((-.1, 0, .1), fontname='Helvetica',
            fontsize=10)
    ax.set_ylabel(r'Match contrast', fontname='Helvetica',
            fontsize=12)
    ax.hlines(np.array((-.0526, .0476, -.0204, .0196, 0)), 41, 47,
        linestyles=['solid', 'solid', 'dotted', 'dotted', 'solid'],
        linewidths=[.5, .5, .5, .5, .5],
        color=['.8', '.8', '.2', '.2', '.1'])

def create_contrast_plots(vp, data, surround):
    fig = plt.figure(vp, figsize=(7.5, 6.))
    adaptor_subplots_contrast(fig, data, surround)
    for obj in fig.findobj(matplotlib.lines.Line2D):
        if obj.get_marker() == '.':
            obj.set_mfc('r')
            obj.set_mec('r')
            obj.set_marker('+')
            obj.set_ms(3)
        else:
            obj.set_visible(False)
    for obj in fig.findobj(matplotlib.collections.LineCollection):
        obj.remove()

    adaptor_subplots_contrast(fig, clean_data, surround)
    fig.suptitle(r'Test luminance ($cd/m^2$)', verticalalignment='top',
            y=.02)
    fig.savefig('../figures/contour_adapt/exploration/contrast/'+
        'contrast_%s_matches_%s.pdf' % ( surround, vp),
        bbox_inches='tight', transparent=True)
    plt.close(fig)

def adaptor_subplots_contrast(fig, data, surround):
    titles = ['no_adaptor', 'orthogonal', 'parallel', 'orthogonal_shifted',
            'parallel_shifted', 'sbc']
    for by_adaptor in data.by_field('adaptor_type'):
        title = by_adaptor.adaptor_type[0]
        i = 1 if title == 'none' else 2 if title == 'orthogonal' else 3 \
            if title == 'parallel' else 4 if title == 'o_shifted' \
            else 5 if title == 'p_shifted' else 6
        ax_icon = fig.add_axes((.205 + ((i-1)%3) * .273, .85 - (i-1)//3 * .48, .075, .075),
                frameon=True)
        ax_icon.set_xticks([])
        ax_icon.set_yticks([])
        ax_icon.imshow(plt.imread('../figures/contour_adapt/icons/%s_icon.png' %
            titles[i-1]))
        ax = fig.add_subplot(2, 3, i, frameon=False)
        plot_contrast_data(ax, by_adaptor, surround)
        #ax.set_title(titles[i-1])
        if i not in [1, 4]:
            ax.set_ylabel('')
            ax.set_yticks([])

def adaptor_subplots(fig, data):
    titles = ['no_adaptor', 'orthogonal', 'parallel', 'orthogonal_shifted',
            'parallel_shifted', 'sbc']
    for by_adaptor in data.by_field('adaptor_type'):
        title = by_adaptor.adaptor_type[0]
        i = 1 if title == 'none' else 2 if title == 'orthogonal' else 3 \
            if title == 'parallel' else 4 if title == 'o_shifted' \
            else 5 if title == 'p_shifted' else 6
        ax_icon = fig.add_axes((.205 + ((i-1)%3) * .273, .85 - (i-1)//3 * .48, .075, .075),
                frameon=True)
        ax_icon.set_xticks([])
        ax_icon.set_yticks([])
        ax_icon.imshow(plt.imread('../figures/contour_adapt/icons/%s_icon.png' %
            titles[i-1]))
        ax = fig.add_subplot(2, 3, i, frameon=False)
        plot_data(ax, by_adaptor)
        #ax.set_title(titles[i-1])
        if i not in [1, 4]:
            ax.set_ylabel('')
            ax.set_yticks([])

def get_all_data(vp_names=None, clean=True):
    datadir = '../exp_data/matching'
    if vp_names is None:
        vp_names = [vp_name for vp_name in os.listdir(datadir) if os.path.isdir(
                    os.path.join(datadir, vp_name))]
    all_data = []
    for vp in vp_names:
        filenames = os.listdir(os.path.join(datadir, vp))
        data = datamat.CsvFactory(os.path.join(datadir, vp, filenames[0]))
        for filename in filenames[1:]:
            data.join(datamat.CsvFactory(os.path.join(datadir, vp, filename)))
        # encode SBC as adaptor_type
        data.adaptor_type[(data.adaptor_type == 'none') &
                (data.coaxial_lum == data.flank_lum)] = 'sbc'
        data.adaptor_type[(data.adaptor_shift != 0) & (data.adaptor_type ==
                'parallel')] = 'p_shifted'
        data.adaptor_type[(data.adaptor_shift != 0) & (data.adaptor_type ==
                'orthogonal')] = 'o_shifted'
        # convert luminance encoding to cd/m^2
        data.match_lum *= 88
        data.match_initial *= 88
        data.test_lum *= 88
        # add match contrast information
        contrast_far, contrast_near = compute_match_contrast(data, vp)
        data.add_field('match_contrast_near', contrast_near)
        data.add_field('match_contrast_far', contrast_far)
        # remove outliers
        if clean:
            data = data[~mark_outliers(data)]
        data.add_field('vp', [vp] * len(data))
        all_data.append(data)
    # combine datamats
    data = all_data[0]
    for this_data in all_data[1:]:
        data.join(this_data)
    return data

def create_exploratory_figs(vp, data):
    # create figures for near and far patches only
    for patch_loc in ['_near', '_far']:
        fig = plt.figure(vp, figsize=(7.5, 6.))
        if patch_loc == '_near':
            current_data = data[(data.test_loc == 2) |
                                (data.test_loc == 3)]
        elif patch_loc == '_far':
            current_data = data[(data.test_loc == 0) |
                                (data.test_loc == 1)]
        adaptor_subplots(fig, current_data)
        fig.savefig('../figures/contour_adapt/exploration/near_far/' +
            'lightness_matches_%s%s.pdf' % (vp, patch_loc),
            bbox_inches='tight', transparent=True)
        plt.close(fig)
    # create figures for horizontal or vertical grating only
    for grating_ori in ['horizontal', 'vertical']:
        fig = plt.figure(vp, figsize=(7.5, 6.))
        current_data = data[data.grating_ori==grating_ori]
        adaptor_subplots(fig, current_data)
        fig.savefig('../figures/contour_adapt/exploration/horz_vert/' +
            'lightness_matches_%s_%s.pdf' % (vp, grating_ori),
            bbox_inches='tight', transparent=True)
        plt.close(fig)

    # create reaction time histograms
    fig = plt.figure(figsize=(4, 3))
    edges = range(90)
    edges.append(np.inf)
    hist_vals = np.histogram(data.response_time, edges)[0]
    plt.vlines(np.arange(5, 90, 6), 0, 60, colors='r')
    plt.bar(edges[0:-1], hist_vals, width=1, linewidth=0, color='k')
    fig.savefig('../figures/contour_adapt/exploration/rt/rt_%s.pdf' % vp,
            bbox_inches='tight', transparent=True)
    plt.close(fig)

    # create start_lum vs end_lum scatter plots
    fig = plt.figure(figsize=(3,3))
    plt.plot(data.match_initial, data.match_lum, '.k')
    plt.title('r=%.3f, p=%.3f' % pearsonr(data.match_initial,
        data.match_lum))
    plt.ylim((.3*88, .7*88))
    plt.xlim((.3*88, .7*88))
    plt.plot([.3*88, .7*88], [.3*88, .7*88], 'k')
    fig.savefig(
        '../figures/contour_adapt/exploration/startvalue/startval_correlation_%s.pdf'
         % vp, bbox_inches='tight', transparent=True)
    plt.close(fig)

def separate_effectsize_plots():
    bad_vps = ['ig', 'kt', 'jp', 'sw'] #observers without White's effect
    good_vps = ['ad', 'cy', 'ak', 'rh', 'vyf', 'ts'] #observers with White's
    bad_data = get_all_data(bad_vps)
    good_data = get_all_data(good_vps)
    fig, ax = plt.subplots(frameon=False, figsize=(3.6, 3))
    effect_sizes, conf_intervals = compute_effectsize(good_data, 10000)
    plot_effectsize(fig, ax, effect_sizes, conf_intervals)
    fig.savefig('../figures/contour_adapt/effect_sizes_good.pdf',
            transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(frameon=False, figsize=(3.6, 3))
    effect_sizes, conf_intervals = compute_effectsize(bad_data, 10000)
    plot_effectsize(fig, ax, effect_sizes, conf_intervals)
    fig.savefig('../figures/contour_adapt/effect_sizes_bad.pdf',
            transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    datadir = '../exp_data/matching'
    vp_names = [vp_name for vp_name in os.listdir(datadir) if os.path.isdir(
                    os.path.join(datadir, vp_name))]
    np.setdiff1d(vp_names, ['tb'])
    #vp_names = ['ig', 'kt', 'jp', 'sw'] #observers without White's effect
    vp_names = ['ad', 'cv', 'ak', 'rh', 'vyf', 'ts'] #observers with White's

    all_data = []
    for vp in vp_names:
        filenames = os.listdir(os.path.join(datadir, vp))
        data = datamat.CsvFactory(os.path.join(datadir, vp, filenames[0]))
        for filename in filenames[1:]:
            data.join(datamat.CsvFactory(os.path.join(datadir, vp, filename)))
        # encode SBC as adaptor_type
        data.adaptor_type[(data.adaptor_type == 'none') &
                (data.coaxial_lum == data.flank_lum)] = 'sbc'
        data.adaptor_type[(data.adaptor_shift != 0) & (data.adaptor_type ==
                'parallel')] = 'p_shifted'
        data.adaptor_type[(data.adaptor_shift != 0) & (data.adaptor_type ==
                'orthogonal')] = 'o_shifted'
        # convert luminance encoding to cd/m^2
        data.match_lum *= 88
        data.match_initial *= 88
        data.test_lum *= 88

        # add match contrast information
        contrast_far, contrast_near = compute_match_contrast(data, vp)
        data.add_field('match_contrast_near', contrast_near)
        data.add_field('match_contrast_far', contrast_far)

        #create_exploratory_figs(vp, data)

        # create individual subject data plots
        # remove outliers
        clean_data = data[~mark_outliers(data)]
        print '%s: %d outliers removed' % (vp, 288 - len(clean_data))
        fig = plt.figure(vp, figsize=(7.5, 6.))
        adaptor_subplots(fig, data)
        for obj in fig.findobj(matplotlib.lines.Line2D):
            if obj.get_marker() == '.':
                obj.set_mfc('r')
                obj.set_mec('r')
                obj.set_marker('+')
                obj.set_ms(3)
            else:
                obj.set_visible(False)
        for obj in fig.findobj(matplotlib.collections.LineCollection):
            obj.remove()

        adaptor_subplots(fig, clean_data)
        fig.suptitle(r'Test luminance ($cd/m^2$)', verticalalignment='top',
                y=.02)
        fig.savefig('../figures/contour_adapt/lightness_matches_%s.pdf' % vp,
            bbox_inches='tight', transparent=True)
        plt.close(fig)

        create_contrast_plots(vp, data, 'near')
        create_contrast_plots(vp, data, 'far')

        clean_data.add_field('vp', [vp] * len(clean_data))
        all_data.append(clean_data)

    # create effect size plots
    data = all_data[0]
    for this_data in all_data[1:]:
        data.join(this_data)
    fig, ax = plt.subplots(frameon=False, figsize=(7.5, 3))
    effect_sizes, conf_intervals = compute_effectsize(data, 10000)
    plot_effectsize(fig, ax, effect_sizes, conf_intervals)
    fig.savefig('../figures/contour_adapt/effect_sizes_good.pdf',
            bbox_inches='tight', transparent=True)

    # create SBC effect plot
    fig, ax = plt.subplots(frameon=False, figsize=(8.5, 3))
    plot_sbc_effect(ax, data)
    fig.savefig('../figures/contour_adapt/exploration/SBC_effect_sizes.pdf',
            bbox_inches='tight', transparent=True)
    plt.close(fig)

