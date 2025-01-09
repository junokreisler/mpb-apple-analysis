import numpy as np
import pandas as pd
import os
import sys
import cv2
from google.colab.patches import cv2_imshow # google colab specific bugfix
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import math
import utils

##################################################################################################################
##################################################################################################################
# put location of the annotation file FROM THE PREVIOUS STEP, define the output folder name for this script
annotation_path = '' 
output_path = ''
##################################################################################################################
##################################################################################################################

if not os.path.exists(data_path):
    os.mkdir(output_path)

# load extract of annotation filtered in the previous step, extract cultivars and badges
badge_annotation = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/subset_annotation_2021_2022.xlsx')
used_cultivars = list(badge_annotation['Cultivar_name'].unique())
used_badges = list(badge_annotation['badge'].unique())

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
    plt.show()
    
    plt.savefig(os.path.join(plt_output,f'plot_apple_{plot_num}.png'))  # save plot number, for referencing with the dataframe exported below

# export dataframe to make it easier to locate which cultivars are in which plot
cultivar_plot_df = pd.DataFrame(cultivar_plot_data)
cultivar_plot_df.to_csv(os.path.join(plt_output, 'cultivar_plot_data_appleLevel.csv'), index=False)

### ADD: EXTRACT BADGES WITH THEIR H-W RATIO AVERAGE, SD
