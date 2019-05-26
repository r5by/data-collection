#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# randomize color for visualization
# cmap=plt.cm.get_cmap(plt.cm.viridis,143)
# c=cmap(random.randint(1,144))

#################################
#           Functions
#################################
# Available gradient for cmaps:
# 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
#


def gradientbars_0(bars):
    grad = np.atleast_2d(np.linspace(0, 1, 256)).T
    ax = bars[0].axes
    lim = ax.get_xlim() + ax.get_ylim()
    for bar in bars:
        bar.set_zorder(1)
        bar.set_facecolor('none')
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        ax.imshow(grad, extent=[x, x + w, y, y + h],
                  aspect="auto", zorder=0)
    ax.axis(lim)


def gradientbars(bars, cname, py, height):
    grad = np.atleast_2d(np.linspace(0, 1, 256)).T
    ax = bars[0].axes
    lim = ax.get_xlim() + ax.get_ylim()

    i = 0
    for bar in bars:
        bar.set_zorder(1)
        bar.set_facecolor('none')
        x, y = bar.get_x(), py
        w, h = bar.get_width(), height[i]
        i += 1
        ax.imshow(grad, extent=[x, x + w, y, y + h],
                  cmap=plt.get_cmap(cname), aspect="auto", zorder=0, alpha=0.8)

    ax.axis(lim)


#################################
#           Adjustments
#################################
data_path = "https://raw.githubusercontent.com/ruby-/data-collection/master/long_90_tile.csv"
width = 0.15  # the width of a bar
fig_h = 6  # figure height
fig_w = 8  # figure width
fig_ylable = 'Pigeon Speedup'  # y-lable
fig_xlable = 'load level (0~100% of total #nodes)'  # x-lable
fig_title = '90-tile plot (long)'  # figure title
# x-lables for bars
labels = ['50%', '60%', '70%', '80%', '90%']

# Gradiently fade-out ratio of sparrow(max) bar to figure top margin
fadeYTop = 0.8
# scale of sparrow above threshold, adjust this value to change fading-bar height
scale = 2
# Space left for legend plot
lscale = 0.5
# uncomment below to choose postion for legend plot
legend_pos = 'upper right'
# legend_pos = 'upper left'

# Rotation of text label above each bar
txt_rotate_n = 60

#################################
#           Data input
#################################
df = pd.read_csv(data_path)
eagle = df['eagle']
eagle_simulated = df['eagle_s']
sparrow = df['sparrow']
sparrow_s = df['sparrow_s']

#################################
#           Vars
#################################
# split sparrow up/below threshold value
ruler = max(np.maximum(eagle, eagle_simulated))
threshold = ruler * 1.5
v_sparrow = np.array(sparrow)
v_sparrow_s = np.array(sparrow_s)

# Setting the positions and width for the bars
pos = list(range(len(eagle)))
# Adjust this value to leave space for legend plot
offset = ruler * lscale

# colors of bars
c1 = 'seagreen'  # eagle
c2 = 'goldenrod'  # eagle_s
c3 = 'darkred'  # sparrow_below_threshold
c4 = 'red'  # sparrow_above_threshold
c5 = 'mediumslateblue'  # sparrow_s_below_threshold
c6 = 'slateblue'  # sparrow_s_above_threshold
c7 = 'dimgray'  # threshold line color
c8 = 'dimgray'  # text above threshold line color
c9 = 'Reds'  # gradient color for sparrow
c10 = 'Purples'  # gradient color for sparrow_s

#################################
#           Graphix
#################################
# Plotting the bars
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
# Setting axis labels and ticks
ax.set_ylabel(fig_ylable)
ax.set_xlabel(fig_xlable)
ax.set_title(fig_title)
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticklabels(labels)

eagle_bar = plt.bar(pos, eagle, width,
                    alpha=0.8,
                    color='w',
                    hatch='',
                    facecolor=c1,
                    edgecolor='w',
                    label=labels[0])

eagle_simulated_bar = plt.bar([p + width for p in pos], eagle_simulated, width,
                              alpha=0.69,
                              color='w',
                              hatch='x',
                              facecolor=c2,
                              edgecolor='w',
                              label=labels[1])

# upper
sparrow_above_threshold_tmp = np.maximum(v_sparrow - threshold, 0)
indices_up = np.nonzero(sparrow_above_threshold_tmp)
sparrow_above_threshold = sparrow_above_threshold_tmp[indices_up]

sparrow_s_above_threshold_tmp = np.maximum(v_sparrow_s - threshold, 0)
indices_up_s = np.nonzero(sparrow_s_above_threshold_tmp)
sparrow_s_above_threshold = sparrow_s_above_threshold_tmp[indices_up]
# lower
sparrow_below_threshold = np.minimum(v_sparrow, threshold)
indices_below = np.where(v_sparrow < threshold)
sparrow_below_threshold_shown = v_sparrow[indices_below]

sparrow_s_below_threshold = np.minimum(v_sparrow_s, threshold)
indices_below_s = np.where(v_sparrow_s < threshold)
sparrow_s_below_threshold_shown = v_sparrow_s[indices_below]

# sparrow bar below threshold
sparrow_bar_below = plt.bar([p + width * 2 for p in pos], sparrow_below_threshold, width,
                            alpha=0.8,
                            color='k',
                            hatch='',
                            facecolor=c3,
                            edgecolor='w',
                            ls='-',
                            lw=None,
                            label=labels[2])

sparrow_s_bar_below = plt.bar([p + width * 3 for p in pos], sparrow_s_below_threshold, width,
                              alpha=0.99,
                              color='k',
                              hatch='\\',
                              facecolor=c5,
                              edgecolor='w',
                              ls='-',
                              lw=None,
                              label=labels[2])

# sparrow bar above threshold
if len(indices_up[0]):
    sparrow_bar_upper = plt.bar([p + width * 2 for p in np.nditer(indices_up)], sparrow_above_threshold, width,
                                bottom=sparrow_below_threshold[indices_up],
                                alpha=0.8,
                                color='k',
                                hatch='',
                                facecolor=c4,
                                edgecolor='w',
                                ls='-',
                                lw=None,
                                label=labels[2])
    # draw the bars above threshold as faded
    sparrow_dist_ratio = sparrow_above_threshold / \
        (np.amax(sparrow_above_threshold))
    sparrow_dist_arr = (ruler * scale -
                        threshold) * fadeYTop * sparrow_dist_ratio
    gradientbars(sparrow_bar_upper, c9, threshold, sparrow_dist_arr.tolist())

if len(indices_up_s[0]):
    sparrow_s_bar_upper = plt.bar([p + width * 3 for p in np.nditer(indices_up_s)], sparrow_s_above_threshold, width,
                                  bottom=sparrow_s_below_threshold[indices_up_s],
                                  alpha=0.99,
                                  color='k',
                                  hatch='\\',
                                  facecolor=c6,
                                  edgecolor='w',
                                  ls='-',
                                  lw=None,
                                  label=labels[2])

    sparrow_s_dist_ratio = sparrow_s_above_threshold / \
        (np.amax(sparrow_s_above_threshold))
    sparrow_s_dist_arr = (ruler * scale -
                          threshold) * fadeYTop * sparrow_s_dist_ratio
    gradientbars(sparrow_s_bar_upper, c10,
                 threshold, sparrow_s_dist_arr.tolist())


# Setting text labels on each bar
for x, y in zip(pos, eagle):
    plt.text(x, y, '%.1f' % y, ha='center',
             va='bottom', rotation=txt_rotate_n)

for x, y in zip(pos, eagle_simulated):
    plt.text(x + width, y, '%.1f' %
             y, ha='center', va='bottom', rotation=txt_rotate_n)

if len(indices_below[0]):
    for x, y in zip(np.nditer(indices_below), np.nditer(sparrow_below_threshold_shown)):
        plt.text(x + width * 2, y, '%.1f' %
                 y, ha='center', va='bottom', rotation=txt_rotate_n)

if len(indices_up[0]):
    for x, y, yPos in zip(np.nditer(indices_up), np.nditer(sparrow_above_threshold + threshold), np.nditer(sparrow_dist_arr)):
        plt.text(x + width * 2,  threshold + yPos, '%.1f' %
                 y, ha='center', va='bottom', rotation=txt_rotate_n)

if len(indices_below_s[0]):
    for x, y in zip(np.nditer(indices_below_s), np.nditer(sparrow_s_below_threshold_shown)):
        plt.text(x + width * 3, y, '%.1f' %
                 y, ha='center', va='bottom', rotation=txt_rotate_n)

if len(indices_up_s[0]):
    for x, y, yPos in zip(np.nditer(indices_up_s), np.nditer(sparrow_s_above_threshold + threshold), np.nditer(sparrow_s_dist_arr)):
        plt.text(x + width * 3,  threshold + yPos, '%.1f' %
                 y, ha='center', va='bottom', rotation=txt_rotate_n)

# Adding the legend and showing the plot
plt.legend(['eagle', 'eagle_simulated', 'sparrow',
            'sparrow_simulated'], loc=legend_pos)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 5)

if len(indices_up[0]) or len(indices_up_s[0]):
    plt.ylim([0, ruler * 2 + offset])
    plt.yticks(np.arange(0, threshold, 10))
    # horizontal line indicating the threshold
    ax.plot([0., 4.5], [threshold, threshold], "k--", color=c7)
    plt.text(pos[0] + width, threshold,
             '> {:0.1f}x'.format(threshold), ha='center', va='bottom', color=c8)
else:
    plt.ylim([0, ruler + offset])
    plt.yticks(np.arange(0, threshold, 0.2))

# plt.grid()
plt.show()
