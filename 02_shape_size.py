import numpy as np
import pandas as pd
import os
import sys
import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from statsmodels.stats.multitest import multipletests # for two-way ANOVA
import seaborn as sns
import math
import utils

##################################################################################################################
##################################################################################################################
# put location of the annotation file FROM THE PREVIOUS STEP, define the output folder name for this script
annotation_path = '' 
output_path = ''
# load extract of annotation filtered in the previous step's script, extract cultivars and badges
badge_annotation = pd.read_excel('./subset_annotation_2021_2022.xlsx') # change path if necessary
##################################################################################################################
##################################################################################################################

if not os.path.exists(data_path):
    os.mkdir(output_path)

used_cultivars = list(badge_annotation['Cultivar_name'].unique())
used_badges = list(badge_annotation['badge'].unique())

##################################################################################################################
# Polar coordinate data - results in px
##################################################################################################################

# obtain the degree and radius data
w2021 = utils.obtain_edge_measurements(path = os.path.join('2021','waedenswil','all.badge.polar.raw.txt'))
g2021 = utils.obtain_edge_measurements(path = os.path.join('2021','grabs','all.badge.polar.raw.txt'))
w2022 = utils.obtain_edge_measurements(path = os.path.join('2022','waedenswil','all.badge.polar.raw.txt'))

# fix grabs 2022 badge mistake (same as prior script)
g2022 = pd.read_csv(os.path.join('2022','grabs','all.badge.polar.raw.txt'), sep = '\t')
g2022_badges = list(g2022['badge'])
for i in range(len(g2022_badges)):
  g2022_badges[i] = g2022_badges[i][0:4]+'0'+g2022_badges[i][4:]
g2022['badge'] = g2022_badges

g2022 = utils.obtain_edge_measurements(pre_df = g2022, use_df = True)


w2021 = {badge: df for badge, df in w2021.items() if badge in used_badges}
g2021 = {badge: df for badge, df in g2021.items() if badge in used_badges}

# make sure that only those badges are kept in the 2022 dataset that are present in 2021 as well
w2022 = {badge: df for badge, df in w2022.items() if badge in used_badges}
w2022 = {badge: df for badge, df in w2022.items() if badge in list(w2021.keys())}

g2022 = {badge: df for badge, df in g2022.items() if badge in used_badges}
g2022 = {badge: df for badge, df in g2022.items() if badge in list(g2021.keys())}

# obtain the height-width ratios for each apple in each present badge
w2021_hw_ratios_raw, w2021_heights_raw, w2021_widths_raw = {}, {}, {}
g2021_hw_ratios_raw, g2021_heights_raw, g2021_widths_raw = {}, {}, {}
w2022_hw_ratios_raw, w2022_heights_raw, w2022_widths_raw = {}, {}, {}
g2022_hw_ratios_raw, g2022_heights_raw, g2022_widths_raw = {}, {}, {}

for badge in w2021:
  w2021_hw_ratios_raw[badge], w2021_heights_raw[badge], w2021_widths_raw[badge] = utils_apple.obtain_hw_ratio(w2021[badge].loc[:,['t_deg', 'rAverage']], calc_avg = False)
  w2022_hw_ratios_raw[badge], w2022_heights_raw[badge], w2022_widths_raw[badge] = utils_apple.obtain_hw_ratio(w2022[badge].loc[:,['t_deg', 'rAverage']], calc_avg = False)

for badge in g2021:
  g2021_hw_ratios_raw[badge], g2021_heights_raw[badge], g2021_widths_raw[badge] = utils_apple.obtain_hw_ratio(g2021[badge].loc[:,['t_deg', 'rAverage']], calc_avg = False)
  g2022_hw_ratios_raw[badge], g2022_heights_raw[badge], g2022_widths_raw[badge] = utils_apple.obtain_hw_ratio(g2022[badge].loc[:,['t_deg', 'rAverage']], calc_avg = False)

w2021_badge_measure_raw_df = pd.DataFrame({'HWratio': w2021_hw_ratios_raw.values(), 'Height': w2021_heights_raw.values(), 'Width': w2021_widths_raw.values(),
                                       }, index=list(w2021_hw_ratios_raw.keys())).explode(['HWratio', 'Height', 'Width']).reset_index(drop=False)
g2021_badge_measure_raw_df = pd.DataFrame({'HWratio': g2021_hw_ratios_raw.values(), 'Height': g2021_heights_raw.values(), 'Width': g2021_widths_raw.values(),
                                       }, index=list(g2021_hw_ratios_raw.keys())).explode(['HWratio', 'Height', 'Width']).reset_index(drop=False)

w2022_badge_measure_raw_df = pd.DataFrame({'HWratio': w2022_hw_ratios_raw.values(), 'Height': w2022_heights_raw.values(), 'Width': w2022_widths_raw.values(),
                                       }, index=list(w2022_hw_ratios_raw.keys())).explode(['HWratio', 'Height', 'Width']).reset_index(drop=False)
g2022_badge_measure_raw_df = pd.DataFrame({'HWratio': g2022_hw_ratios_raw.values(), 'Height': g2022_heights_raw.values(), 'Width': g2022_widths_raw.values(),
                                       }, index=list(g2022_hw_ratios_raw.keys())).explode(['HWratio', 'Height', 'Width']).reset_index(drop=False)

w2021_badge_measure_raw_df.columns = ['badge', 'HWratio', 'Height', 'Width']
g2021_badge_measure_raw_df.columns = ['badge', 'HWratio', 'Height', 'Width']
w2022_badge_measure_raw_df.columns = ['badge', 'HWratio', 'Height', 'Width']
g2022_badge_measure_raw_df.columns = ['badge', 'HWratio', 'Height', 'Width']

# merge df with badge annotations
w2021_badge_measure_raw_df = w2021_badge_measure_raw_df.merge(badge_annotation, left_on='badge', right_on='badge', how='left')
w2022_badge_measure_raw_df = w2022_badge_measure_raw_df.merge(badge_annotation, left_on='badge', right_on='badge', how='left')
g2021_badge_measure_raw_df = g2021_badge_measure_raw_df.merge(badge_annotation, left_on='badge', right_on='badge', how='left')
g2022_badge_measure_raw_df = g2022_badge_measure_raw_df.merge(badge_annotation, left_on='badge', right_on='badge', how='left')

# plot height-width distribution of a all selected apples in each dataset
utils.generate_height_width_plot(w2021_badge_measure_raw_df, output_path, plot_title = 'Waedenswil 2021 Apple Sizes')
utils.generate_height_width_plot(g2021_badge_measure_raw_df, output_path, plot_title = 'Grabs 2021 Apple Sizes')
utils.generate_height_width_plot(w2022_badge_measure_raw_df, output_path, plot_title = 'Waedenswil 2022 Apple Sizes')
utils.generate_height_width_plot(g2022_badge_measure_raw_df, output_path, plot_title = 'Grabs 2022 Apple Sizes')

# obtain intra-cultivar distances between apple measurements
cultivar_intra_distances = {}
for cultivar in used_cultivars:
  cultivar_intra_distances[cultivar] = {
    'w2021': utils.calculate_intra_cultivar_distance(w2021_badge_measure_raw_df, cultivar),
    'w2022': utils.calculate_intra_cultivar_distance(w2022_badge_measure_raw_df, cultivar),
    'g2021': utils.calculate_intra_cultivar_distance(g2021_badge_measure_raw_df, cultivar),
    'g2022': utils.calculate_intra_cultivar_distance(g2022_badge_measure_raw_df, cultivar)

# export distances for each year and location as pandas DataFrame 
pd.DataFrame(cultivar_intra_distances).T.to_csv(os.path.join(output_path, 'intra_cultivar_distances.csv'))}

# plot histograms of intra-cultivar distances per location and per year
w2021_distances = []
w2022_distances = []
g2021_distances = []
g2022_distances = []

# Iterate through cultivars and extract distances
for cultivar_dists in cultivar_intra_distances.values():
    w2021_distances.append(cultivar_dists.get('w2021', np.nan)) #gets value, if not available, returns np.nan
    w2022_distances.append(cultivar_dists.get('w2022', np.nan))
    g2021_distances.append(cultivar_dists.get('g2021', np.nan))
    g2022_distances.append(cultivar_dists.get('g2022', np.nan))

# Remove nan values from lists
w2021_distances = [x for x in w2021_distances if x is not None]
w2022_distances = [x for x in w2022_distances if x is not None]
g2021_distances = [x for x in g2021_distances if x is not None]
g2022_distances = [x for x in g2022_distances if x is not None]

utils.plot_histogram_of_distances_duo(dfs = (w2021_distances, w2022_distances), labels = ('2021', '2022'), title = 'Intra-cultivar Distances in Wädenswil', output_name = 'intra_cultivar_distances_wae.png')
utils.plot_histogram_of_distances_duo(dfs = (g2021_distances, g2022_distances), labels = ('2021', '2022'), title = 'Intra-cultivar Distances in Grabs', output_name = 'intra_cultivar_distances_gra.png')

utils.plot_histogram_of_distances_duo(dfs = (w2021_distances, g2021_distances), labels = ('Wädenswil', 'Grabs'), title = 'Intra-cultivar Distances in 2021', output_name = 'intra_cultivar_distances_2021.png')
utils.plot_histogram_of_distances_duo(dfs = (w2022_distances, g2022_distances), labels = ('Wädenswil', 'Grabs'), title = 'Intra-cultivar Distances in 2022', output_name = 'intra_cultivar_distances_2022.png')

#Identifying cultivars in the height-width plot of locations and years

#1. merge dataset with annotation
#2. make 2x2 plot location x year
#3. highlight N cultivars with different color / shape
#4. repeat for a total of (used_cultivars / N) plots

n_highlights = 6 # number of cultivars to highlight

# for axis normalization, obtain min and max values
dataframes = [w2021_badge_measure_raw_df, w2022_badge_measure_raw_df, g2021_badge_measure_raw_df, g2022_badge_measure_raw_df]

x_min = min(df['Width'].min() for df in dataframes)
x_max = max(df['Width'].max() for df in dataframes)
y_min = min(df['Height'].min() for df in dataframes)
y_max = max(df['Height'].max() for df in dataframes)

shapes = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h'] # highlight using different shapes, max 10 because 

titles = ['Wädenswil 2021', 'Wädenswil 2022', 'Grabs 2021', 'Grabs 2022']

plt_output = os.path.join(output_path,'cultivar_size_consistency')
if not os.path.exists(plt_output):
    os.mkdir(plt_output)

cultivar_plot_data = []

n_plots = math.ceil(len(used_cultivars) / n_highlights)  # round up the number of plots needed, highlight 10 cultivars in each plot
for plot_num in range(n_plots):
    start_index = plot_num * n_highlights
    end_index = min(start_index + n_highlights, len(used_cultivars))
    cultivars_to_highlight = used_cultivars[start_index:end_index]

    for cultivar in cultivars_to_highlight:
      cultivar_plot_data.append({'Plot Number': plot_num + 1, 'Cultivar': cultivar})  


    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    
    for i, ax in enumerate(axes.flat):
      if i >= len(cultivars_to_highlight):
        ax.remove()  # remove axes if there are not enough cultivars to populate
        continue
          
      df = dataframes[i % len(dataframes)]  # Use modulo to cycle through dataframes
      sns.scatterplot(x='Width', y='Height', data=df, ax=ax, color='gray', alpha=0.05)  # Plot all points in gray

      for j, cultivar in enumerate(cultivars_to_highlight):
        subset = df[df['Cultivar_name'] == cultivar]
        ax.plot(np.linspace(0, 1000, 100), np.linspace(0, 1000, 100), color='black', linestyle='--')
        sns.scatterplot(x='Width', y='Height', data=subset, ax=ax, marker=shapes[j % len(shapes)], label=cultivar, s=30)
            

      ax.set_title(titles[i % len(titles)])  # Use modulo to cycle through titles
      ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
      # Set axis limits to be the same for all plots
      ax.set_xlim(x_min, x_max)
      ax.set_ylim(y_min, y_max)

    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.suptitle(f'[Plot {plot_num+1}] Apple size consistency across 2021 and 2022 in Wädenswil and Grabs')    
    plt.savefig(os.path.join(plt_output,f'plot_apple_{plot_num}.png'))  # save plot number, for referencing with the dataframe exported below

# export dataframe to make it easier to locate which cultivars are in which plot
cultivar_plot_df = pd.DataFrame(cultivar_plot_data)
cultivar_plot_df.to_csv(os.path.join(plt_output, 'cultivar_plot_data_appleLevel.csv'), index=False)

##################################################################################################################
# Traits data
##################################################################################################################

dims_w2021 = pd.read_csv(os.path.join('2021','waedenswil','traits.txt', sep='\t'))
dims_w2022 = pd.read_csv(os.path.join('2021','waedenswil','traits.txt', sep='\t')
dims_g2021 = pd.read_csvos.path.join('2021','waedenswil','traits.txt', sep='\t')
dims_g2022 = pd.read_csv(os.path.join('2021','waedenswil','traits.txt', sep='\t')
# filter badges
dims_w2021 = dims_w2021[dims_w2021['badge'].isin(used_badges)].merge(badge_annotation, left_on='badge', right_on='badge', how='left')
dims_w2022 = dims_w2022[dims_w2022['badge'].isin(used_badges)].merge(badge_annotation, left_on='badge', right_on='badge', how='left')
dims_g2021 = dims_g2021[dims_g2021['badge'].isin(used_badges)].merge(badge_annotation, left_on='badge', right_on='badge', how='left')
# fix grabs 2022 badge mistake
g2022_badges = list(dims_g2022['badge'])
for i in range(len(g2022_badges)):
  g2022_badges[i] = g2022_badges[i][0:4]+'0'+g2022_badges[i][4:]
dims_g2022['badge'] = g2022_badges
dims_g2022 = dims_g2022[dims_g2022['badge'].isin(used_badges)].merge(badge_annotation, left_on='badge', right_on='badge', how='left')

# obtain h-w ratio
dims_w2021['HWratio'] = dims_w2021['height_mean_cm'] / dims_w2021['width_mean_cm']
dims_w2022['HWratio'] = dims_w2022['height_mean_cm'] / dims_w2022['width_mean_cm']
dims_g2021['HWratio'] = dims_g2021['height_mean_cm'] / dims_g2021['width_mean_cm']
dims_g2022['HWratio'] = dims_g2022['height_mean_cm'] / dims_g2022['width_mean_cm']

# generate height v width plots with h-w ratio as the color gradient (should form an ellipse-like shape with majority slightly under the diagonal line)
utils.generate_height_width_plot(dims_w2021[['width_mean_cm','height_mean_cm','HWratio']], output_path,
                                       plot_title = 'Wädenswil 2021 Apple Sizes', xlim = (2,12), ylim = (2,12),
                                       sns_palette = 'viridis')
utils.generate_height_width_plot(dims_g2021[['width_mean_cm','height_mean_cm','HWratio']], output_path,
                                       plot_title = 'Grabs 2021 Apple Sizes', xlim = (2,12), ylim = (2,12),
                                       sns_palette = 'viridis')
utils.generate_height_width_plot(dims_w2022[['width_mean_cm','height_mean_cm','HWratio']], output_path,
                                       plot_title = 'Wädenswil 2022 Apple Sizes', xlim = (2,12), ylim = (2,12),
                                       sns_palette = 'viridis')
utils.generate_height_width_plot(dims_g2022[['width_mean_cm','height_mean_cm','HWratio']], output_path,
                                       plot_title = 'Grabs 2022 Apple Sizes', xlim = (2,12), ylim = (2,12),
                                       sns_palette = 'viridis')

# define subdirectory for the cultivar highlight plots (there are very many of them)
plt_output = os.path.join(output_path, 'cultivar_size_consistency')

# if n_highlights is 6 (recommended as it plots each badge's values for each highlighted cultivars),
# there should be 53 plots in total, from plot 40 onwards, the cultivars only in Grabs 2021/22 are depicted.
utils.plot_cultivar_size_consistency(dataframes=[dims_w2021[['width_mean_cm','height_mean_cm','Cultivar_name']], dims_w2022[['width_mean_cm','height_mean_cm','Cultivar_name']],
                                                       dims_g2021[['width_mean_cm','height_mean_cm','Cultivar_name']], dims_g2022[['width_mean_cm','height_mean_cm','Cultivar_name']]],
                                           n_highlights=6,
                                           plot_title_overall = 'Apple size consistency across 2021 and 2022 in Wädenswil and Grabs',
                                           used_cultivars = used_cultivars, plt_output_path = plt_output)

utils.plot_histogram_of_distances_duo(dfs = (dims_w2021['HWratio'], dims_w2022['HWratio']),
                                            labels = ('2021', '2022'), title = 'Badge-level HW ratio in Wädenswil',
                                            output_path = output_path,
                                            output_name = 'badge_HWratio_wae.png',
                                            x_axis_title = 'HW ratio',
                                            boundaries = (0.7,1.5))

utils.plot_histogram_of_distances_duo(dfs = (dims_g2021['HWratio'], dims_g2022['HWratio']),
                                            labels = ('2021', '2022'), title = 'Badge-level HW ratio in Grabs',
                                            output_path = output_path,
                                            output_name = 'badge_HWratio_gra.png',
                                            x_axis_title = 'HW ratio',
                                            boundaries = (0.7,1.5))

# obtain means and s.d. of all common badges per cultivar.
dims_w2021_hwr_agg = dims_w2021.groupby('Cultivar_name')['HWratio'].agg(['mean', 'std'])
dims_w2022_hwr_agg = dims_w2022.groupby('Cultivar_name')['HWratio'].agg(['mean', 'std'])
dims_g2021_hwr_agg = dims_g2021.groupby('Cultivar_name')['HWratio'].agg(['mean', 'std'])
dims_g2022_hwr_agg = dims_g2022.groupby('Cultivar_name')['HWratio'].agg(['mean', 'std'])

dims_w2021_hwr_agg['err'] = dims_w2021_hwr_agg['std']/dims_w2021_hwr_agg['mean']
dims_w2022_hwr_agg['err'] = dims_w2022_hwr_agg['std']/dims_w2022_hwr_agg['mean']
dims_g2021_hwr_agg['err'] = dims_g2021_hwr_agg['std']/dims_g2021_hwr_agg['mean']
dims_g2022_hwr_agg['err'] = dims_g2022_hwr_agg['std']/dims_g2022_hwr_agg['mean']

dims_w2021_hwr_agg.reset_index(inplace=True)
dims_w2022_hwr_agg.reset_index(inplace=True)
dims_g2021_hwr_agg.reset_index(inplace=True)
dims_g2022_hwr_agg.reset_index(inplace=True)

# overview of h-w ratio consistency (x axis: h-w ratio)
plt.scatter(x = dims_w2021_hwr_agg['mean'], y = dims_w2021_hwr_agg['err']*100, s=10, alpha = 0.5, color='red')
plt.scatter(x = dims_w2022_hwr_agg['mean'], y = dims_w2022_hwr_agg['err']*100, s=10, alpha = 0.5, color='orange')
plt.scatter(x = dims_g2021_hwr_agg['mean'], y = dims_g2021_hwr_agg['err']*100, s=10, alpha = 0.5, color='blue')
plt.scatter(x = dims_g2022_hwr_agg['mean'], y = dims_g2022_hwr_agg['err']*100, s=10, alpha = 0.5, color='lightblue')
plt.legend(['Wädenswil 2021', 'Wädenswil 2022', 'Grabs 2021', 'Grabs 2022'])
plt.xlabel('HW ratio')
plt.ylabel('s.d. as percent of the mean HW ratio')
plt.title('Height-width ratio of cultivars and their standard deviation')

utils.plot_cultivar_size_consistency(dataframes = [dims_w2021_hwr_agg[['mean','err','Cultivar_name']],dims_w2022_hwr_agg[['mean','err','Cultivar_name']],
                                                         dims_g2021_hwr_agg[['mean','err','Cultivar_name']], dims_g2022_hwr_agg[['mean','err','Cultivar_name']]],
                                           plot_title_overall = 'Apple height-width ratio consistency across 2021 and 2022 in Wädenswil and Grabs',
                                           used_cultivars = used_cultivars, plt_output_path = plt_output+'_hw_ratio', n_highlights=12)

dims_w2021_hwr_agg.columns = ['Cultivar_name', 'HWratio_w2021_mean', 'HWratio_w2021_std', 'HWratio_w2021_err']
dims_w2022_hwr_agg.columns = ['Cultivar_name', 'HWratio_w2022_mean', 'HWratio_w2022_std', 'HWratio_w2022_err']
dims_g2021_hwr_agg.columns = ['Cultivar_name', 'HWratio_g2021_mean', 'HWratio_g2021_std', 'HWratio_g2021_err']
dims_g2022_hwr_agg.columns = ['Cultivar_name', 'HWratio_g2022_mean', 'HWratio_g2022_std', 'HWratio_g2022_err']

# Merge the dataframes
merged_df = dims_w2021_hwr_agg.merge(dims_w2022_hwr_agg, on='Cultivar_name', how='outer') \
                           .merge(dims_g2021_hwr_agg, on='Cultivar_name', how='outer') \
                           .merge(dims_g2022_hwr_agg, on='Cultivar_name', how='outer')

merged_df['HWratio_all_mean'] = merged_df[['HWratio_w2021_mean', 'HWratio_w2022_mean', 'HWratio_g2021_mean', 'HWratio_g2022_mean']].mean(axis=1)
merged_df.to_csv(os.path.join(output_path, 'cultivar_hw_ratio_consistency_agg.csv'))

cultivar_intra_dists = [{},{},{},{}]
for cultivar in used_cultivars:
  cultivar_intra_dists[0][cultivar] = utils.calculate_intra_cultivar_distance(dims_w2021[['height_mean_cm','width_mean_cm','Cultivar_name']], cultivar = cultivar)
  cultivar_intra_dists[1][cultivar] = utils.calculate_intra_cultivar_distance(dims_w2022[['height_mean_cm','width_mean_cm','Cultivar_name']], cultivar = cultivar)
  cultivar_intra_dists[2][cultivar] = utils.calculate_intra_cultivar_distance(dims_g2021[['height_mean_cm','width_mean_cm','Cultivar_name']], cultivar = cultivar)
  cultivar_intra_dists[3][cultivar] = utils.calculate_intra_cultivar_distance(dims_g2022[['height_mean_cm','width_mean_cm','Cultivar_name']], cultivar = cultivar)

cultivar_intra_dist_df = pd.DataFrame(cultivar_intra_dists).T
cultivar_intra_dist_df.columns = ['w2021', 'w2022', 'g2021', 'g2022']
cultivar_intra_dist_df.to_csv(os.path.join(output_path, 'cultivar_intra_dist.csv'))

utils_apple.plot_histogram_of_distances_duo(dfs = (cultivar_intra_dist_df['w2021'], cultivar_intra_dist_df['w2022']),
                                            labels = ('2021', '2022'), title = 'Intra-cultivar distances in Wädenswil in 2021 and 2022',
                                            output_name = 'intra_cultivar_distances_wael.png',
                                            output_path = output_path,
                                            x_axis_title = 'Intra-cultivar distance (cm)',
                                            boundaries = (0,3))
utils_apple.plot_histogram_of_distances_duo(dfs = (cultivar_intra_dist_df['g2021'], cultivar_intra_dist_df['g2022']),
                                            labels = ('2021', '2022'), title = 'Intra-cultivar distances in Grabs in 2021 and 2022',
                                            output_name = 'intra_cultivar_distances_grabs.png',
                                            output_path = output_path,
                                            x_axis_title = 'Intra-cultivar distance (cm)',
                                            boundaries = (0,3))
# compare locations within the same year
utils_apple.plot_histogram_of_distances_duo(dfs = (cultivar_intra_dist_df['w2021'], cultivar_intra_dist_df['g2021']),
                                            labels = ('Wädenswil', 'Grabs'), title = 'Intra-cultivar distances in Wädenswil and Grabs in 2021',
                                            output_name = 'intra_cultivar_distances_2021.png',
                                            output_path = output_path,
                                            x_axis_title = 'Intra-cultivar distance (cm)',
                                            boundaries = (0,3))
utils_apple.plot_histogram_of_distances_duo(dfs = (cultivar_intra_dist_df['w2022'], cultivar_intra_dist_df['g2022']),
                                            labels = ('Wädenswil', 'Grabs'), title = 'Intra-cultivar distances in Wädenswil and Grabs in 2022',
                                            output_name = 'intra_cultivar_distances_2022.png',
                                            output_path = output_path,
                                            x_axis_title = 'Intra-cultivar distance (cm)',
                                            boundaries = (0,3))

##################################################################
# Two-way ANOVA 
##################################################################
# add year and location to all entries of the respective dataframe for concatenation
w2021_data = dims_w2021[['Cultivar_name', 'HWratio']].assign(Year=2021, Location='wae')
w2022_data = dims_w2022[['Cultivar_name', 'HWratio']].assign(Year=2022, Location='wae')
g2021_data = dims_g2021[['Cultivar_name', 'HWratio']].assign(Year=2021, Location='gra')
g2022_data = dims_g2022[['Cultivar_name', 'HWratio']].assign(Year=2022, Location='gra')

anova_data = pd.concat([w2021_data, w2022_data, g2021_data, g2022_data]).dropna()

# add year and location to all entries of the respective dataframe for concatenation
w2021_data = dims_w2021[['Cultivar_name', 'HWratio']].assign(Year=2021, Location='wae')
w2022_data = dims_w2022[['Cultivar_name', 'HWratio']].assign(Year=2022, Location='wae')
g2021_data = dims_g2021[['Cultivar_name', 'HWratio']].assign(Year=2021, Location='gra')
g2022_data = dims_g2022[['Cultivar_name', 'HWratio']].assign(Year=2022, Location='gra')

anova_data = pd.concat([w2021_data, w2022_data, g2021_data, g2022_data]).dropna()

# extract pvalues only
pval_df = pd.DataFrame()
for cultivar in used_cultivars:
  # put p-value = 1 for comparisons that could not be made (missing in wae)
  new_pvals = anova_cultivars[cultivar].loc[['Location', 'Year', 'Year:Location'], 'PR(>F)'].fillna(1) 
  new_pvals.name = cultivar
  pval_df = pd.concat([pval_df, new_pvals], axis=1)

# Benjamini-Hochberg FDR p-value adjustment for multiple testing
_, adjusted_p_values_location, _, _ = multipletests(pval_df.loc['Location'], alpha=0.05, method='fdr_bh')
_, adjusted_p_values_year, _, _ = multipletests(pval_df.loc['Year'], alpha=0.05, method='fdr_bh')
_, adjusted_p_values_location_year, _, _ = multipletests(pval_df.loc['Year:Location'], alpha=0.05, method='fdr_bh')

# replace with adjusted pvalues
pval_df.loc['Location'] = adjusted_p_values_location
pval_df.loc['Year'] = adjusted_p_values_year
pval_df.loc['Year:Location'] = adjusted_p_values_location_year

# create separate output folder for ANOVA plots, there will be hundreds.....
anova_plot_path = os.path.join(output_path, 'anova_plots')
if not os.path.exists(anova_plot_path):
  os.mkdir(anova_plot_path)

pval_df.to_csv(os.path.join(anova_plot_path, 'pvalues_cultivars.csv'))

for cultivar, anova_result in anova_cultivars.items():
  # obtain p-values
  cultivar_data = anova_data[anova_data['Cultivar_name'] == cultivar]

  plt.figure()
  sns.stripplot(x='Year', y='HWratio', hue='Location', data=cultivar_data, jitter=True, dodge=True)
  plt.title(f'Jitter Plot for {cultivar} apples in Wädenswil and Grabs, 2021-2022,\nwith two-way ANOVA significance results')

  # subset pvalues only for the relevant cutivar
  p_values_curr = pval_df[cultivar]
    
  if any(p_values_curr < 0.05):
    significant_effects = p_values_curr[p_values_curr < 0.05]
    # the tested condition is in the index, so we display the index of significant pvalues
    p_values_str = ', '.join([f'{effect}: {p:.6f}' for effect, p in zip(significant_effects.index, p_values_curr[p_values_curr < 0.05])])
    plt.text(0.05, 0.95, f'Significant ({p_values_str})', transform=plt.gca().transAxes, color='red', fontsize=10)
    plt.ylim(0.6,1.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    cultivar = cultivar.replace('/','_')
    plt.savefig(os.path.join(anova_plot_path, f'anova_jitter_plot_{cultivar}.png'))
