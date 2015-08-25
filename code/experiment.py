#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Contour adaptation experiment with White's illusion stimulus.
Task: luminance adjustment of a square patch to match the brightness of an
equally sized patch embedded in a square wave grating. On different trials,
some of the edges of the test patch are contour adapted with flickering lines.
"""
### Imports ###

# Package Imports
from hrl import HRL

# Qualified Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import random
import glob
import Image, ImageFont, ImageDraw

class EndTrial(Exception):
    def __init__(self, match_lum):
        self.match_lum = match_lum

class EndExperiment(Exception):
    def __init__(self, msg=None):
        self.msg = msg

def draw_text(text, bg=.5, text_color=0, fontsize=48):
    """ create a numpy array containing the string text as an image. """

    bg *= 255
    text_color *= 255
    font = ImageFont.truetype(
            "/usr/share/fonts/truetype/msttcorefonts/arial.ttf", fontsize,
            encoding='unic')
    text_width, text_height = font.getsize(text)
    im = Image.new('L', (text_width, text_height), bg)
    draw = ImageDraw.Draw(im)
    draw.text((0,0), text, fill=text_color, font=font)
    return np.array(im) / 255.

def draw_centered(texture):
    y = (768 - texture.hght) / 2
    x = (1024 - texture.wdth) / 2
    texture.draw((x, y))

def waitButton(datapixx, to):
    """wait for a button press for a given time, and return the button identity
    as first element in a tuple to match hrl API. if no button was pressed,
    button identity is None. This function is only needed until
    hrl.inputs.readButton() is fixed"""
    t = time.time()
    while time.time() - t < to:
        btn = datapixx.readButton()
        if btn is None:
            time.sleep(max(0, min(.01, to - time.time() + t)))
            continue
        btn = btn[0]
        if btn == 2: #up
            return ('Up', 0)
        elif btn == 1: #right
            return ('Right', 0)
        elif btn == 8: #down
            return ('Down', 0)
        elif btn == 4: #left
            return ('Left', 0)
        elif btn == 16: #space
            return ('Space', 0)
    return (None, 0)

### Main ###
def create_design(design_fn):
    header = 'grating_ori grating_vals test_lum test_loc adaptor_type\r\n'
    trials = []
    total_trials = 0
    for repetition in range(1):
        for grating_ori in ['horizontal', 'vertical']:
            for test_loc in range(4):
                for test_lum in [.48, .5, .52]:
                    for adaptor_type in ['horizontal', 'vertical', 'none',
                                    'horizontal_shifted', 'vertical_shifted']:
                        for grating_vals in ['0.45,0.55', '0.55,0.45',
                                             '0.45,0.45', '0.55,0.55']:
                            # don't use adaptation for SBC control
                            if grating_vals[0:4] == grating_vals[5:] and \
                                    adaptor_type != 'none':
                                        continue
                            trials.append((grating_ori, grating_vals,
                                            str(test_lum), str(test_loc),
                                            adaptor_type))
                            total_trials += 1
    random.shuffle(trials)
    with open(design_fn, 'w') as design_file:
        design_file.write(header)
        for trial in trials:
            design_file.write(' '.join(trial) + '\r\n')
    return total_trials

def prepare_files():
    vp_id  = raw_input ('VP Initialen (z.B. mm): ')
    design_fn = 'design/matching/contour_adapt_%s.csv' % vp_id
    result_dir = 'data/matching/%s' % vp_id
    # check if we are resuming with a know subject and take appropriate action
    completed_trials = 0
    if os.access(design_fn, os.F_OK):
        reply = raw_input('Es gibt bereits Daten zu dieser VP. Weiter? (j/n)')
        if reply != 'j':
            raise EndExperiment()
        filecount = -1
        for filecount, fn in enumerate(
                    glob.glob(os.path.join(result_dir, '*%s*.csv' % vp_id))):
            with open(fn) as result_file:
                # first line in the result file is not a trial
                completed_trials -= 1
                for line in result_file:
                    completed_trials += 1
        result_fn = os.path.join(result_dir, 'contour_adapt_%s_%d.csv' %
                (vp_id, filecount + 2))
        # count number of trials in design file
        total_trials = 0
        with open(design_fn) as design_file:
            for line in design_file:
                total_trials += 1
        check_fn = 'check_files/contour_adapt_%s_%d.csv' % \
                (vp_id, filecount + 2)
    # if we are not resuming, create design file
    else:
        os.mkdir(result_dir)
        result_fn = os.path.join(result_dir, 'contour_adapt_%s_1.csv' % vp_id)
        total_trials = create_design(design_fn)
        check_fn = 'check_files/contour_adapt_%s_1.csv' % vp_id
    return (design_fn, result_fn, completed_trials, total_trials, check_fn)

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
    y, x = check_pos
    grating[y:y+bar_width, x:x+bar_width] = check_val
    return grating

def prepare_check_background(grating_vals):
    # create random checks for lower screen half
    lum_range = np.abs(np.diff(grating_vals))
    checks = np.tile(np.linspace(.5 - lum_range, .5 + lum_range, 13), (13, 1))
    map(np.random.shuffle, checks)
    check_bg = np.repeat(np.repeat(checks, 18, 0), 18, 1)[3:-3, 3:-3]
    return check_bg, checks.flatten()

def prepare_adaptors(check_pos, adaptor_type, bar_width=38, shift=0):
    y, x = check_pos
    adapt_dark = np.ones((6 * bar_width, 6 * bar_width)) * .5
    adapt_light = np.ones((6 * bar_width, 6 * bar_width)) * .5
    adapt_idx = np.zeros((6 * bar_width, 6 * bar_width), dtype=bool)
    if adaptor_type == 'vertical':
        adapt_idx[y:y+bar_width, x-2-shift:x+2-shift] = True
        adapt_idx[y:y+bar_width, x-2-shift+bar_width:x+2-shift+bar_width] = True
    elif adaptor_type == 'horizontal':
        adapt_idx[y-2-shift:y+2-shift, x:x+bar_width] = True
        adapt_idx[y-2-shift+bar_width:y+2-shift+bar_width, x:x+bar_width] = True
    adapt_dark[adapt_idx] = 0
    adapt_light[adapt_idx] = 1
    return (adapt_dark, adapt_light)

def adjust_loop(start_time, duration, match_lum, match_pos, bar_width=38):
    """ react to button presses and adjust match luminance accordingly for a
    given time period. Return the final adjusted value of match_lum.
    Raises an EndTrial exception if the center button is pressed.
    """
    smlstp = 0.005
    bgstp = 0.025
    while time.time() - start_time < duration:
        # Read the next button press
        btn, _ = hrl.inputs.readButton(
                        to=duration - time.time() + start_time)
        if btn is None:
            continue
        # Respond to the pressed button
        if btn == 'Up':
            match_lum += bgstp
        elif btn == 'Right':
            match_lum += smlstp
        elif btn == 'Down':
            match_lum -= bgstp
        elif btn == 'Left':
            match_lum -= smlstp
        elif btn == 'Space':
            raise EndTrial(match_lum)
        elif btn == 'Escape':
            raise EndExperiment('escape pressed')
        match_lum = min(max(match_lum, 0), 1)
        match_patch = hrl.graphics.newTexture(np.ones((1,1)) * match_lum)
        match_patch.draw(match_pos, (bar_width, bar_width))
        hrl.graphics.flip(clr=False)
    if hrl.inputs.checkEscape():
        raise EndExperiment('escape pressed')
    return match_lum

def show_break(trial, total_trials):
    hrl.graphics.flip(clr=True)
    lines = [u'Du kannst jetzt eine Pause machen.',
             u' ',
             u'Du hast %d von %d Durchgängen geschafft.' % (trial,
                 total_trials),
             u' ',
             u'Wenn du bereit bist, drücke die mittlere Taste.',
             u' ',
             u'Wenn du zu einem späteren Zeitpunkt weiter machen willst,',
             u'wende dich an den Versuchsleiter.']
    for line_nr, line in enumerate(lines):
        textline = hrl.graphics.newTexture(draw_text(line, fontsize=36))
        textline.draw(((1024 - textline.wdth) / 2,
                       (768 / 2 - (4 - line_nr) * (textline.hght + 10))))
    hrl.graphics.flip(clr=True)
    btn = None
    while btn != 'Space':
        btn, _ = hrl.inputs.readButton(to=3600)
        if hrl.inputs.checkEscape():
            raise EndExperiment('escape pressed')

def run_trial(dsgn):
    # prepare a test stimulus texture
    grating_vals = [float(v) for v in dsgn['grating_vals'].split(',')]
    check_pos = test_coords[int(dsgn['test_loc'])]
    match_pos = match_coords[int(dsgn['test_loc'])]
    grating = prepare_grating(dsgn['grating_ori'],
                              grating_vals,
                              check_pos,
                              float(dsgn['test_lum']),
                              bar_width=bar_width)
    grating = hrl.graphics.newTexture(grating)
    # prepare and draw the matching background for this trial
    match_bg, bg_values = prepare_check_background([.45, .55])
    # shift match bg so that match patch is centered on surrounding checks
    h_shift = (check_pos[1] % 18) - 5
    match_bg = np.roll(match_bg, h_shift, axis=1)
    v_shift = ((match_pos[1] - 384) % 18) - 5
    match_bg = np.roll(match_bg, v_shift, axis=0)

    match_bg = hrl.graphics.newTexture(match_bg)
    match_bg.draw(match_bg_loc)

    # initialize match patch with random value, save it to resultdict
    match_lum = (np.random.random() * .2) + .3
    hrl.results['match_initial'] = float(match_lum)
    match_patch = hrl.graphics.newTexture(np.ones((1,1)) * match_lum)
    # place match patch same distance below center as test patch above
    match_patch.draw(match_pos, (bar_width, bar_width))

    # prepare adaptor stimulus textures
    if dsgn['adaptor_type'].endswith('_shifted'):
        shift = int(np.sign(np.random.rand() - .5) * bar_width / 2)
        adaptor_ori = dsgn['adaptor_type'].replace('_shifted', '')
    else:
        shift = 0
        adaptor_ori = dsgn['adaptor_type']
    hrl.results['adaptor_shift'] = shift
    adapt_dark, adapt_light = prepare_adaptors(check_pos,
            adaptor_ori, bar_width=bar_width, shift=shift)
    adapt_dark = hrl.graphics.newTexture(adapt_dark)
    adapt_light = hrl.graphics.newTexture(adapt_light)
    # preload some variables to prepare for our button reading loop.
    t = time.time()
    btn = None

    ### Input Loop ####
    # Until the user finalizes their luminance choice by raising
    # EndTrial
    trial_over = False
    while not trial_over:
        try:
            # show adaptation loop 25 times (5s)
            for i in range(25):
                adapt_dark.draw(stim_loc)
                fix_outer.draw((512, 383.5), (10, 10))
                fix_inner.draw((512, 383.5), (4, 4))
                hrl.graphics.flip(clr=False)
                match_lum = adjust_loop(time.time(), .1, match_lum,
                        match_pos)

                adapt_light.draw(stim_loc)
                fix_outer.draw((512, 383.5), (10, 10))
                fix_inner.draw((512, 383.5), (4, 4))
                hrl.graphics.flip(clr=False)
                match_lum = adjust_loop(time.time(), .1, match_lum,
                        match_pos)

            # show test stimulus
            grating.draw(stim_loc)
            fix_outer.draw((512, 383.5), (10, 10))
            fix_inner.draw((512, 383.5), (4, 4))
            hrl.graphics.flip(clr=False)
            match_lum = adjust_loop(time.time(), 1, match_lum,
                    match_pos)
        except EndTrial as et:
            match_lum = et.match_lum
            # show confirmation screen
            hrl.graphics.flip(clr=True)
            draw_centered(confirmation)
            hrl.graphics.flip(clr=True)
            btn, _ = hrl.inputs.readButton(to=36000.)
            if btn == 'Space':
                trial_over = True
            else:
                match_bg.draw(match_bg_loc)
                match_patch = hrl.graphics.newTexture(
                                np.ones((1,1)) * match_lum)
                match_patch.draw(match_pos, (bar_width, bar_width))
    return match_lum, t, bg_values

def main():
    # determine design and result file name
    try:
        (design_fn, result_fn, completed_trials, total_trials, check_fn) = prepare_files()
    except EndExperiment:
        return 0
    result_headers = ['Trial', 'adaptor_type', 'coaxial_lum', 'test_lum',
                      'match_lum', 'flank_lum', 'test_loc', 'grating_ori',
                      'response_time', 'match_initial', 'adaptor_shift']
    global hrl
    hrl = HRL(graphics='datapixx',
              inputs='responsepixx',
              photometer=None,
              wdth=1024,
              hght=768,
              bg=0.5,
              dfl=design_fn,
              rfl=result_fn,
              rhds=result_headers,
              scrn=1,
              lut='lut0to88.csv',
              db = False,
              fs=True)

    # monkey patch to use non-blocking readButton function
    # should be removed once hrl.inputs.readButton is fixed
    hrl.inputs.readButton = lambda to: waitButton(hrl.datapixx, to)

    # set the bar width of the grating. this value determines all positions
    global bar_width
    bar_width = 38
    y_border = 768 // 2 - int(6 * bar_width)
    x_border = (1024 - 6 * bar_width) // 2
    # the coordinates of the four possible test check positions
    # test coordinates are (y,x) position within the grating array
    global test_coords
    test_coords = ((bar_width * 3, bar_width * 2),
                   (bar_width * 3, bar_width * 3),
                   (bar_width * 4, bar_width * 2),
                   (bar_width * 4, bar_width * 3))
    # match coordinates are (x,y) screen positions
    global match_coords
    match_coords = ((x_border + bar_width * 2, 384 + int(bar_width * 2)),
                    (x_border + bar_width * 3, 384 + int(bar_width * 2)),
                    (x_border + bar_width * 2, 384 + int(bar_width * 1)),
                    (x_border + bar_width * 3, 384 + int(bar_width * 1)))
    global match_bg_loc
    match_bg_loc = (x_border, 384)
    global stim_loc
    stim_loc = (x_border, y_border)

    # prepare fixation mark textures
    global fix_inner
    fix_inner = hrl.graphics.newTexture(np.ones((1,1)) * .5, 'circle')
    global fix_outer
    fix_outer = hrl.graphics.newTexture(np.zeros((1,1)), 'circle')

    # prepare confirmation texture
    global confirmation
    confirmation = hrl.graphics.newTexture(draw_text('Weiter?', bg=.5))

    # show instruction screen
    if completed_trials == 0:
        for i in range(6):
            instructions = plt.imread('instructions%d.png' % (i + 1))[..., 0]
            instructions = hrl.graphics.newTexture(instructions)
            instructions.draw((0, 0))
            hrl.graphics.flip(clr=True)
            btn = None
            while btn != 'Space':
                btn, _ = hrl.inputs.readButton(to=3600)

        # show test trials
        test_dsgn = [{'adaptor_type': 'none',
                      'grating_ori': 'horizontal',
                      'grating_vals': '0.45,0.55',
                      'test_loc': '0',
                      'test_lum': '0.5'},
                     {'adaptor_type': 'vertical',
                      'grating_ori': 'horizontal',
                      'grating_vals': '0.55,0.45',
                      'test_loc': '1',
                      'test_lum': '0.5'},
                     {'adaptor_type': 'horizontal',
                      'grating_ori': 'horizontal',
                      'grating_vals': '0.45,0.55',
                      'test_loc': '2',
                      'test_lum': '0.5'},
                     {'adaptor_type': 'vertical',
                      'grating_ori': 'vertical',
                      'grating_vals': '0.55,0.45',
                      'test_loc': '2',
                      'test_lum': '0.5'},
                     {'adaptor_type': 'none',
                      'grating_ori': 'vertical',
                      'grating_vals': '0.45,0.45',
                      'test_loc': '1',
                      'test_lum': '0.5'},
                     {'adaptor_type': 'vertical_shifted',
                      'grating_ori': 'vertical',
                      'grating_vals': '0.55,0.45',
                      'test_loc': '2',
                      'test_lum': '0.5'},
                     {'adaptor_type': 'none',
                      'grating_ori': 'vertical',
                      'grating_vals': '0.55,0.55',
                      'test_loc': '1',
                      'test_lum': '0.5'},
                     {'adaptor_type': 'horizontal_shifted',
                      'grating_ori': 'vertical',
                      'grating_vals': '0.45,0.55',
                      'test_loc': '1',
                      'test_lum': '0.5'},
                     {'adaptor_type': 'horizontal',
                      'grating_ori': 'vertical',
                      'grating_vals': '0.45,0.55',
                      'test_loc': '3',
                      'test_lum': '0.5'}]
        for dsgn in test_dsgn:
            run_trial(dsgn)

    # show experiment start confirmation
    hrl.graphics.flip(clr=True)
    lines = [u'Die Probedurchgänge sind fertig.',
        u'Wenn du bereit bist, drücke die mittlere Taste.',
        u' ',
        u'Wenn du noch Fragen hast, oder mehr Probedurchgänge',
        u'machen willst, wende dich an den Versuchsleiter.']
    for line_nr, line in enumerate(lines):
        textline = hrl.graphics.newTexture(draw_text(line, fontsize=36))
        textline.draw(((1024 - textline.wdth) / 2,
                       (768 / 2 - (3 - line_nr) * (textline.hght + 10))))
    hrl.graphics.flip(clr=True)
    btn = None
    while btn != 'Space':
        btn, _ = hrl.inputs.readButton(to=3600)

    ### Core Loop ###

    # hrl.designs is an iterator over all the lines in the specified design
    # matrix, which was loaded at the creation of the hrl object. Looping over
    # it in a for statement provides a nice way to run each line in a design
    # matrix. The fields of each design line (dsgn) are drawn from the design
    # matrix in the design file (design.csv).

    start_time = time.time()
    try:
        for trial, dsgn in enumerate(hrl.designs):

            # skip trials that we already had data for
            if trial < completed_trials:
                continue

            # check if we should take a break (every 15 minutes)
            if time.time() - start_time > (60 * 15):
                show_break(trial, total_trials)
                start_time = time.time()

            match_lum, t, bg_values = run_trial(dsgn)

            # convert adaptor type name to convention more useful for analysis
            adaptor_ori = dsgn['adaptor_type'].replace('_shifted', '')
            if adaptor_ori == 'none':
                adaptor_type = 'none'
            elif dsgn['grating_ori'] == adaptor_ori:
                adaptor_type = 'parallel'
            else:
                adaptor_type = 'orthogonal'
            # determine flank and coaxial luminance based on test patch position
            grating_vals = [float(v) for v in dsgn['grating_vals'].split(',')]
            loc = int(dsgn['test_loc'])
            go = dsgn['grating_ori']
            if (go == 'horizontal' and loc in [2, 3]) or \
               (go == 'vertical' and loc in [0, 2]):
                coaxial_lum, flank_lum = grating_vals
            else:
                flank_lum, coaxial_lum = grating_vals

            # Once a value has been chosen by the subject, we save all relevant
            # variables to the result file by loading it all into the hrl.results
            # dictionary, and then finally running hrl.writeResultLine().
            hrl.results['Trial'] = trial
            hrl.results['test_lum'] = dsgn['test_lum']
            hrl.results['adaptor_type'] = adaptor_type
            hrl.results['test_loc'] = dsgn['test_loc']
            hrl.results['grating_ori'] = dsgn['grating_ori']
            hrl.results['coaxial_lum'] = coaxial_lum
            hrl.results['flank_lum'] = flank_lum
            hrl.results['response_time'] = time.time() - t
            hrl.results['match_lum'] = float(match_lum)
            hrl.writeResultLine()

            # write match bg values to file
            with open(check_fn, 'a') as f:
                f.write('%s\n' %','.join('%0.4f' % x for x in bg_values))

            # We print the trial number simply to keep track during an experiment
            print hrl.results['Trial']

    # catch EndExperiment exception raised by pressing escp for clean exit
    except EndExperiment:
        print "Experiment aborted"
    # And the experiment is over!
    hrl.close()
    print "Session complete"

### Run Main ###
if __name__ == '__main__':
    main()
