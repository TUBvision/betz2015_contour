#!/usr/bin/python
# -*- coding: latin-1 -*-

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt

import stimuli.lightness

def place_fixation(stimulus):
    # create 20x20px fixation circle to place in center of all images
    fixation = stimuli.lightness.disc_and_ring((1.2 , 1.2), (.5, .2), (0, 1),
                                        1, 10)
    fix_size = fixation.shape[0] / 2
    (y, x) = np.asarray(stimulus.shape) / 2
    stimulus[y-fix_size:y+fix_size, x-fix_size:x+fix_size] *= fixation

def make_movie(output_dir, filename, static, adaptor0, adaptor1):
    shape = static.shape
    for stim, name in zip((static, adaptor0, adaptor1), ('static', 'adaptor0',
                           'adaptor1')):
        place_fixation(stim)
        plt.imsave(os.path.join(output_dir, name+'.png'), stim, cmap='gray',
            vmin=0, vmax=1)
    with open(os.path.join(output_dir, 'script'), 'w') as script:
        for frame in range(25):
            for i in range(2):
                script.write(os.path.join(output_dir, 'adaptor%d.png\n' % i))
        for frame in range(10):
            script.write(os.path.join(output_dir, 'static.png\n'))

    os.system('mencoder mf://@/%s -mf' % os.path.join(output_dir, 'script') +
                ' w=%d:h=%d:fps=10:type=png' % (shape[1], shape[0]) +
                ' -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o ' +
                os.path.join(output_dir, filename))
    os.remove(os.path.join(output_dir, 'script'))
    os.remove(os.path.join(output_dir, 'static.png'))
    os.remove(os.path.join(output_dir, 'adaptor0.png'))
    os.remove(os.path.join(output_dir, 'adaptor1.png'))

def prepare_grating(grating_ori, grating_vals, check_pos, check_val,
                        bar_width=34):
    # create square wave grating
    grating = np.ones((bar_width * 8, bar_width * 8)) * grating_vals[1]
    index = [i + j for i in range(bar_width) for j in
                range(0, bar_width * 8, bar_width * 2)]
    if grating_ori == 'vertical':
        grating[:, index] = grating_vals[0]
    elif grating_ori == 'horizontal':
        grating[index, :] = grating_vals[0]
    # place test square at appropriate position
    for pos in check_pos:
        y, x = pos
        grating[y:y+bar_width, x:x+bar_width] = check_val
    return grating

def prepare_adaptors(check_pos, orientation, bar_width=34):
    adapt_dark = np.ones((8 * bar_width, 8 * bar_width)) * .5
    adapt_light = np.ones((8 * bar_width, 8 * bar_width)) * .5
    adapt_idx = np.zeros((8 * bar_width, 8 * bar_width), dtype=bool)
    for pos in check_pos:
        y, x = pos
        if orientation == 'vertical':
            adapt_idx[y:y+bar_width, x-2:x+2] = True
            adapt_idx[y:y+bar_width, x-2+bar_width:x+2+bar_width] = True
        elif orientation == 'horizontal':
            adapt_idx[y-2:y+2, x:x+bar_width] = True
            adapt_idx[y-2+bar_width:y+2+bar_width, x:x+bar_width] = True
    adapt_dark[adapt_idx] = 0
    adapt_light[adapt_idx] = 1
    return (adapt_dark, adapt_light)

if __name__ == "__main__":
    check_pos = ((34 * 3.5, 34 * 2), (34 * 3.5, 34 * 5))
    output_dir = 'videos'

    # adaptors on edges
    for adaptor_ori in ['horizontal', 'vertical']:
        adaptor0, adaptor1 = prepare_adaptors(check_pos, adaptor_ori)
        static = prepare_grating('vertical', (.47, .53), check_pos, .5)
        adaptor = 'parallel' if adaptor_ori == 'vertical' else 'orthogonal'
        filename = '%s.avi' % (adaptor)
        make_movie(output_dir, filename, static, adaptor0, adaptor1)

    # shifted adaptors
    for adaptor_ori in ['horizontal', 'vertical']:
        if adaptor_ori == 'horizontal':
            adapt_check_pos = ((34 * 3, 34 * 2), (34 * 4, 34 * 5))
        else:
            adapt_check_pos = ((34 * 3.5, 34 * 2.5), (34 * 3.5, 34 * 4.5))
        adaptor0, adaptor1 = prepare_adaptors(adapt_check_pos, adaptor_ori)
        static = prepare_grating('vertical', (.47, .53), check_pos, .5)
        adaptor = 'parallel' if adaptor_ori == 'vertical' else 'orthogonal'
        filename = 'shifted_%s.avi' % (adaptor)
        make_movie(output_dir, filename, static, adaptor0, adaptor1)
