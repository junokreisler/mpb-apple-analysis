
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import matplotlib.colors as mcolors
import os


def obtain_diff_eucl(ref_df, df_compare):
  ### Euclidean distance of apple mean to badge mean
  h_dist = np.linalg.norm(ref_df['H_mean'] - df_compare['H_mean'])
  s_dist = np.linalg.norm(ref_df['S_mean'] - df_compare['S_mean'])
  return h_dist, s_dist

def obtain_diff_manhattan(ref_df, df_compare):
  ### Manhattan distance of apple mean to badge mean - DEFAULT
  h_dist = np.sum(np.abs(ref_df['H_mean'] - df_compare['H_mean']))
  s_dist = np.sum(np.abs(ref_df['S_mean'] - df_compare['S_mean']))
  return h_dist, s_dist

def identify_representative_color_apple(apple_nrs, current_badge,
                                        current_apples, metric):
  """
  Main function of method #1. For a given badge, obtain its
  most representative apple through averaging of hue frequencies for all apples,
  then with a chosen distance metric (Manhattan or Euclidean distance), identify
  the "least divergent" or the "most average" apple among all apples.

  Args:
    - apple_nrs (dtype list): the number of apples (obtained in the parent function below)
    - current_badge (dtype str): the ID of the badge currently looked at (obtained by the parent function below)
    - current_apples (dtype pandas.DataFrame): the color distribution of the apple
    - metric ('manhattan' or 'euclidean'): chooses which distance metric to use for
    comparing the individual apple hue distributions to the average hue distribution.
  
  """

  apple_H_dists = []
  apple_S_dists = []

  for apple_nr in apple_nrs:
    current_apple = current_apples.loc[apple_nr]

    # 2. Obtain diff metric between reference (avg for badge) and single apple all-cam means
    if metric == 'euclidean':
      H_dist, S_dist = obtain_diff_eucl(current_badge, current_apple)
    elif metric == 'manhattan':
      H_dist, S_dist = obtain_diff_manhattan(current_badge, current_apple)
    apple_H_dists.append(H_dist)
    apple_S_dists.append(S_dist)

  min_H_apple = min(apple_H_dists)
  min_S_apple = min(apple_S_dists)

  min_H_apple_index = apple_H_dists.index(min_H_apple)
  min_S_apple_index = apple_S_dists.index(min_S_apple)

  # the apples are counted from 1 onwards, but indices are from 0 onwards.
  # therefore, return ind+1 and search within the lists with ind.

  print(f"Lowest H dist: {round(min_H_apple,5)}", "for apple nr: ", min_H_apple_index+1,
        f"with S dist: {round(apple_S_dists[min_H_apple_index],5)}",
        f"\nLowest S dist: {round(min_S_apple,5)}", "for apple nr: ", min_S_apple_index+1,
        f"with H dist: {round(apple_H_dists[min_S_apple_index],5)}")

  ### TODO: INCLUDE OUTPUT OF MAX DISTANCES, SO THAT "WORST EXAMPLES" CAN BE
  ### EXCLUDED IN CASE OF TRAINING WITH MULTIPLE IMAGES

  return min_H_apple_index+1, min_S_apple_index+1


# MAIN FUNCTION
def identify_representative_color_apple_all(metric, min_apples = 5,
                                            path_to_summary = 'allcolhist.txt'):
  """

  Args:
    - metric ('manhattan' or 'euclidean'): chooses which distance metric to use for
    comparing the individual apple hue distributions to the average hue distribution.
    - min_apples (dtype int): minimum no. of apples a badge needs to have in order not to be discarded in the analysis.
    - path_to_summary (dtype str): path to the color histogram summary file
    ('allcolhist.txt')

  """
  
  # Extract the means from the data frames
  color_summary = pd.read_csv(path_to_summary, sep='\t').loc[:,['value','H_mean','S_mean', 'appleNR','badge']]
  all_badges = color_summary['badge'].unique()

  color_byApple = color_summary.groupby(['badge','appleNR','value']).agg({'H_mean': 'mean', 'S_mean': 'mean'})
  color_byBadge = color_summary.groupby(['badge','value']).agg({'H_mean': 'mean', 'S_mean': 'mean'})

  best_badge_apple_H = {}
  best_badge_apple_S = {}

  # Per badge, obtain apple with closest-to-mean H and S
  i=0
  for badge in all_badges:
    if i%100 == 0:
      print(round(i/len(all_badges),2)*100, "% done...")
    print(f"Badge: {badge}")

    apple_nrs = color_summary.loc[color_summary['badge'] == all_badges[i]]['appleNR'].unique()
    print(apple_nrs)

    if len(apple_nrs) >= min_apples:
      current_badge = color_byBadge.loc[all_badges[i]]
      current_apples = color_byApple.loc[all_badges[i]]

      min_H_ind, min_S_ind = identify_representative_color_apple(apple_nrs,
                                                                 current_badge,
                                                                 current_apples,
                                                                 metric)

      best_badge_apple_H[badge] = min_H_ind
      best_badge_apple_S[badge] = min_S_ind
    i += 1
  return best_badge_apple_H, best_badge_apple_S

def extract_badge_mean(badge, number, source_data):
  temp = source_data[(source_data['badge'] == badge) & (source_data['appleNR'] == number)][['badge','appleNR','H_mean']]
  badge_dict = {badge: list(temp['H_mean'])}
  return pd.DataFrame(badge_dict)

def prepare_pca(feature_df, n_pc, overall_df = False, return_object = False):
  """
  EXPECTS THE FEATURES TO BE ROWS AND SAMPLES TO BE COLUMNS!
  """
  # normalize the data before PCA
  scaler = StandardScaler()
  colnames = ['PC'+str(i+1) for i in range(n_pc)] # these will be the new colnames in the output pca_df
  if not overall_df:
    feature_df_new = feature_df.T.iloc[1:]
  else:
    feature_df_new = feature_df.T
  scaled_data = scaler.fit_transform(feature_df_new)
  pca = PCA(n_components=n_pc)
  principal_components = pca.fit_transform(scaled_data)
  pc_df = pd.DataFrame(data=principal_components, columns=colnames)
  explained_variance = pca.explained_variance_ratio_

  print(f"Explained variance by each component: {explained_variance}")
  print(f"Total explained variance: {round(sum(explained_variance),5)}")

  if not overall_df:
    pc_df['badge'] = feature_df.columns[1:]
  else:
    pc_df['badge'] = feature_df.columns # doesn't have to be badge, just the names of the original vectors
  
  if return_object:
    return pca
  else:
    return pc_df

def add_color_label(pca_df, max_val_dict_df):
  pca_df['color'] = ''
  top_colors = {0.208333:'green',0.180556:'yellow/green',0.152778:'yellow',
          0.125:'red/yellow', 0.097222:'red', 0.041667:'red',
          0.069444:'red', 0.013889:'red'}
  for badge in pca_df.badge:
    for key in max_val_dict_df:
      if badge in max_val_dict_df[key][1:len(max_val_dict_df[key])]:
        pca_df['color'][pca_df.badge == badge] = top_colors[key]

  return pca_df

### EDGE AND SHAPE ANALYSIS FUNCTIONS

def obtain_edge_measurements(year = None, location = None, edges_filename = 'all.badge.polar.raw.txt',
                             path = None, pre_df = None, use_df = False):
  """
  General file loader for the edge datasets, compatible for further years' data as well
  (as long as it's in the same format). 
  Unless an explicit path to the data is given, it looks for the data in the subfolder
  structure [project dir]/YYYY/LOCATION/filename. the default filename is given but can be edited.
  
  Args:
    - year: 4-digit year (2021, 2022...) as integer
    - location: name of the folder of the location
    - edges_filename: name of the file containing the measured degrees and radians of apples
    - path: direct path to the file containing the measured degrees and radians of apples.
    - pre_df: in case the dataset has been loaded, pass on the dataframe itself.
  Returns:
    A dictionary of Pandas DataFrames whose keys are badge numbers (basically splits the full df by badge)
    Each DataFrame contains the radius/degree measurements of each camera for each apple in the badge.
  """
  if path != None:
    print('loading from path', path)
    df = pd.read_csv(path, sep = '\t')
  elif year != None or location != None:
    print('loading from year and location', year, location)
    df = pd.read_csv(year+'/'+location+'/'+edges_filename, sep = '\t')
  elif use_df:
    print('loading dataframe')
    df = pre_df
  
  badge_dfs = {}  

  for badge_id, group_data in df.groupby('badge'):
    badge_dfs[badge_id] = group_data.reset_index(drop=True)  
    badge_dfs[badge_id] = badge_dfs[badge_id].groupby(['t','apple'])['rAverage'].mean().reset_index()
    badge_dfs[badge_id]['t_deg'] = np.degrees(badge_dfs[badge_id]['t'])

  return badge_dfs

def reconstruct_edge_image(smooth_points, image_shape = (1000,1000)):
    """
    Reconstructs the edge image from smoothed polar distances. This function is intended to be used on edges extracted from image data.
    Args:
        - smooth_points (dtype pandas.DataFrame): A list of smoothed polar distance points (IN ORDER: angle, radius).
        - image_shape (dtype np.shape()): The shape of the base black image (height, width), default = (1000,1000).

    Returns:
        The reconstructed edge image that can be loaded using cv2.imshow()
    """

    # Create a blank image and add 1px padding to show px at edges
    reconstructed_image = np.zeros((image_shape[0]+1,image_shape[1]+1), dtype=np.uint8)

    # Calculate center coordinates
    center_x = image_shape[1] // 2
    center_y = image_shape[0] // 2

    # Convert polar coordinates to Cartesian and set pixels to white
    for i in range(0, len(smooth_points)):
      # some badges' apples haave no measurements
      if not (np.isnan(smooth_points.iloc[i, 0]) or np.isnan(smooth_points.iloc[i, 1])):
        # plot the points in the polar coordinate plane
        x = int(center_x + smooth_points.iloc[i, 1] * np.cos(smooth_points.iloc[i, 0]))
        y = int(center_y + smooth_points.iloc[i, 1] * np.sin(smooth_points.iloc[i, 0]))

        # Check if the coordinates are within the image bounds
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
          reconstructed_image[y, x] = 255

    return reconstructed_image

def obtain_hw_ratio(smooth_points, calc_avg = True):
  """
  FObtains the top-bottom and left-right distances for a badge and returns their ratio.
  This function is used on edge data found in the polar distance datasets OR in radius vs. angle dataframes after edge extraction.

  Args:
      - smooth_points (dtype pandas.DataFrame): A list of smoothed polar distance points (IN ORDER: angle 't_deg', radius 'rAverage').
      
      - calc_avg (dtype boolean, default True): calculate the badge average (TRUE => function outputs a tuple of three np.floats)
        or not (FALSE => function outputs tuple of three lists of values, one for each apple of the badge)

  Returns:
      (1) The ratio of average top-bottom and left-right distances;
      (2) Top-bottom distance;
      (3) Left-right distance.
  """

  if calc_avg:
    avg_l = smooth_points[smooth_points['t_deg'] < -179.90]['rAverage'].mean()
    avg_b = smooth_points[smooth_points['t_deg'].between(89.0,91.0, inclusive="both")]['rAverage'].mean()
    avg_r = smooth_points[smooth_points['t_deg'].between(-0.5,0.5, inclusive="both")]['rAverage'].mean()
    avg_t = smooth_points[smooth_points['t_deg'].between(-91.0,-89.0, inclusive="both")]['rAverage'].mean()

    return (avg_t+avg_b)/(avg_l+avg_r), avg_t+avg_b, avg_l+avg_r  #dist from center towards top and bottom / towards left and right

  else:
    all_l = smooth_points[smooth_points['t_deg'] < -179.90]['rAverage'].values
    all_b = smooth_points[smooth_points['t_deg'].between(90.0,91.0, inclusive="both")]['rAverage'].values
    all_r = smooth_points[smooth_points['t_deg'].between(0,0.5, inclusive="both")]['rAverage'].values
    all_t = smooth_points[smooth_points['t_deg'].between(-91.0,-90.0, inclusive="both")]['rAverage'].values

    return (all_t+all_b)/(all_l+all_r), all_t+all_b, all_l+all_r

def generate_height_width_plot(df, plot_destination, sns_palette = 'rainbow', plot_title = '', xlim = (200,800), ylim = (200,800)):
  """
  Generate a height (y) vs. width (x) plot of apples, with height-width ratio as the color palette.
  The height and width is expected to be in pixels.
  The following components of packages must already be loaded:
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    import matplotlib.colors as mcolors
  Args:
    - df: pd.Dataframe with the columns Width, Height as int or float, and HWRatio as float.
      the columns don't have to have these exact names, but they should be ordered 
      width -> height -> HWratio as the first three cols of the dataframe.
    - plot_destination: path to the output folder.
    - sns_palette: desired color palette that is can be found in seaborn. Default is 'rainbow'
    - plot_title: title of the plot
    - xlim: x-axis limits as dtype tuple
    - ylim: y-axis limits as dtype tuple
  Returns: (no variable output)
    - plot of apple height vs. width with height-width ratio depicted as color palette.
  """
  norm = mcolors.Normalize(vmin=0.7, vmax=1.3)
  levels = np.linspace(0.7, 1.3, num=5) # num corresponds to the 5 types of apple shapes based on HWratio
  cmap = plt.get_cmap(sns_palette, 5)
  ax = sns.scatterplot(x=df.iloc[:,0], y=df.iloc[:,1], hue=df.iloc[:,2],
                data=df, palette=sns_palette, s=15, alpha = 0.5)  
  handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'{level:.2f}', markersize=5, markerfacecolor=cmap(norm(level))) for level in levels]
  ax.legend(handles=handles, title='Height-width ratio')
  plt.plot(np.linspace(0, 1000, 100), np.linspace(0, 1000, 100), color='black', linestyle='--')
  plt.title(plot_title)
  ax.set_xlim(xlim[0], xlim[1])
  ax.set_ylim(ylim[0], ylim[1])
  plt.savefig(os.path.join(plot_destination,plot_title+'.png'))
  print('Plot saved successfully as', os.path.join(plot_destination,plot_title+'.png'))
  plt.show()



def plot_cultivar_size_consistency(dataframes, plot_title_overall, used_cultivars,
                                   plt_output_path = '.', n_highlights=6,
                                   titles = ['Wädenswil 2021', 'Wädenswil 2022', 'Grabs 2021', 'Grabs 2022'],
                                   include_guidelines = True):
  """
  Plots apple size consistency across years and locations, highlighting a defined number of cultivars at a time.
  
  Args:
    - dataframes: A list of pandas.DataFrames containing apple size data
    - plot_title_overall: The shared title for all plots. The no. of the plot is prepended to the title
    - plt_output_path:  path to save the generated plots and the highlight assignment table
    - n_highlights: The number of cultivars to highlight in each plot. 
      For showing all apples or all badges per cultivar, the recommended n_highlights is 6.
      For showing one datapoint per cultivar (cultivar averages), the recommended n_highlights is 12.
    - titles: list of subplot titles, with length equal to the length of dataframes list
    - include_guidelines: add dashed black lines indicating the 1:1 height-width ratio and 5% (s.d. / height-width ratio).
  Returns: (no variable output)
    - (len(total cultivars highlighted) / n highligths per plot) plots as .png files saved in the provided output path
    - assignment table of highlighted cultivars to their respective plots as .csv file in the same output path
      for ease of specific cultivar location
  """
  # for axis normalization, obtain min and max values
  x_min = min(df.iloc[:,0].min() for df in dataframes)
  x_max = max(df.iloc[:,0].max() for df in dataframes)
  y_min = min(df.iloc[:,1].min() for df in dataframes)
  y_max = max(df.iloc[:,1].max() for df in dataframes)

  shapes = ['o', 's', '^', 'v', '<', '>', 'D', 'P', 'X', '*', '3', '+']

  if not os.path.exists(plt_output_path):
    os.mkdir(plt_output_path)
    print('Created folder', plt_output_path)

  cultivar_plot_data = []

  n_plots = math.ceil(len(used_cultivars) / n_highlights) # make sure leftover cultivars are shown in their own plots

  for plot_num in range(n_plots): # one .png file
    start_index = plot_num * n_highlights
    end_index = min(start_index + n_highlights, len(used_cultivars))
    cultivars_to_highlight = used_cultivars[start_index:end_index]

    for cultivar in cultivars_to_highlight: # same n_highlights in the file
      cultivar_plot_data.append({'Plot Number': plot_num + 1, 'Cultivar': cultivar})

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flat):
      if i >= len(cultivars_to_highlight):
        ax.remove()
        continue

      df = dataframes[i % len(dataframes)]
      sns.scatterplot(x=df.iloc[:,0], y=df.iloc[:,1], data=df, ax=ax, color='gray', alpha=0.05)

      for j, cultivar in enumerate(cultivars_to_highlight): # plot each of the 2x2 subplots
        subset = df[df['Cultivar_name'] == cultivar]
        # plot reference lines - height-width ratio of 1 (x = y) and err = 0.05
        if include_guidelines:
          ax.plot(np.linspace(0, 1000, 100), np.linspace(0, 1000, 100), color='black', linestyle='--')
          ax.plot(np.linspace(0,2,100), np.linspace(0.05,0.05,100), color='black', linestyle='--')
        sns.scatterplot(x=subset.iloc[:,0], y=subset.iloc[:,1], data=subset, ax=ax, marker=shapes[j % len(shapes)], label=cultivar, s=30)

      ax.set_title(titles[i % len(titles)])
      ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
      ax.set_xlim(x_min, x_max)
      ax.set_ylim(y_min, y_max)

    plt.tight_layout(rect=[0, 0.03, 1, 0.9]) # allow space for suptitle (main title)
    plt.suptitle(f'[Plot {plot_num + 1}] ' + plot_title_overall)
    plt.savefig(os.path.join(plt_output_path, f'plot_apple_{plot_num}.png'))
    plt.show()

  cultivar_plot_df = pd.DataFrame(cultivar_plot_data)
  cultivar_plot_df.to_csv(os.path.join(plt_output_path, 'cultivar_plot_data_appleLevel.csv'), index=False) # output csv file for locating cultivars
  print('Highlighted cultivar assignment to plots saved as cultivar_plot_assignments.csv in the provided output path')

def calculate_intra_cultivar_distance(df, cultivar):
  """
  Calculates the intra-cultivar distances (distances between height-width points) and
  returns the mean and s.d. of these distances for the given cultivar, in a given dataset.
  Args:
    - df: pandas.DataFrame, annotated with cultivar in the 'Cultivar_name' variable,
    also containing 'Height' and 'Width' columns.
    - cultivar: cultivar of interest to calculate the intra-cultivar distance.

  Returns:
    a list of np.floats that are both a positive value for cultivars found in dataset,
    returns 0 for cultivars with only 1 measurement (in case intra-cultivar 
    distances are calculated from badge averages instead of all apples individually)
  """
  if df.Cultivar_name.isin([cultivar]).any():
    cultivar_data = df[df['Cultivar_name'] == cultivar].iloc[:,[0,1]].values
  else:
    print(f'Cultivar {cultivar} could not be found in the dataset.')
    return None
  if len(cultivar_data) > 1: # in case there is something to compare with
    distances = [np.linalg.norm(p1 - p2) for i, p1 in enumerate(cultivar_data) for p2 in cultivar_data[i + 1:]]
    return [np.mean(distances),np.std(distances)] if distances else 0 #returns 0 if no distances could be calculated to avoid errors
  
  else: # in case there is nothing to compare with
    return 0

def plot_histogram_of_distances_duo(dfs, labels, title, output_name, output_path = '.', x_axis_title = '',boundaries = (0,10)):
  """
  Plots the frequency of intra-cultivar distances as a histogram, comparing two conditions of choice.

  Args:
    - dfs: a tuple containing two dataframes to plot histogram of.
    - labels: a tuple containing labels to use in the legend.
    - title: the title of the plot.
    - output_name: the name of the output file.
    - x_axis_title: self-explanatory
    - boundaries: the upper and lower bound to display in the x axis.
  Returns: (no variable output)
    Plots and saves two histograms in one plot as a .png file, comparing two distributions. 
  """
  plt.hist(dfs[0], bins=40, alpha=0.5, label=labels[0])
  plt.hist(dfs[1], bins=40, alpha=0.5, label=labels[1])

  plt.xlabel(x_axis_title)
  plt.ylabel('Frequency')
  plt.title(title)
  plt.xlim(boundaries[0],boundaries[1])
  plt.legend()
  plt.savefig(os.path.join(output_path,output_name))
  plt.show()
