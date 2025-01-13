import numpy as np
import pandas as pd
import os
import sys
import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import math

script_path = '/content/drive/MyDrive/Colab Notebooks/utils_apple.py'
apple_data_path = ''

annotation_path = ''
output_dir = ''

shape_file_w2021 = ''
shape_file_w2022 = ''
shape_file_g2021 = ''
shape_file_g2022 = ''

#######################################################################

relevant_locations = [os.path.join('2021','waedenswil'),
                      os.path.join('2021','grabs'),
                      os.path.join('2022','waedenswil'),
                      os.path.join('2022','grabs')]

badge_annotation = pd.read_excel(annotation_path)
output_path = os.path.join(apple_data_path,'total_pca')
used_cultivars = list(badge_annotation['Cultivar_name'].unique())
used_badges = list(badge_annotation['badge'].unique())

# fix the grabs 2022 mistake... again
g2022_col= pd.read_csv(os.path.join(relevant_locations[3],'allcolhist.txt'), sep = '\t')
g2022_badges = list(g2022_col['badge'])
for i in range(len(g2022_badges)):
  g2022_badges[i] = g2022_badges[i][0:4]+'0'+g2022_badges[i][4:]
g2022_col['badge'] = g2022_badges
g2022_dims = pd.read_csv(os.path.join(apple_data_path, 'dims_filtered_g2022.csv'), sep = ',')
g2022 = pd.merge(g2022_col, g2022_dims, on = ['badge', 'appleNR'], how = 'inner').drop(columns=['S_cam1','S_cam2','S_cam3','S_cam4','S_cam5','S_mean','Unnamed: 0'])

if not os.path.exists(output_path):
    os.makedirs(output_path)

def prepare_color_height_width_df(path_color, path_traits):
  color_df = pd.read_csv(path_color, sep = '\t')
  trait_df = pd.read_csv(path_traits, sep = ',')
  return pd.merge(color_df, trait_df, on = ['badge', 'appleNR'], how = 'inner').drop(columns=['S_cam1','S_cam2','S_cam3','S_cam4','S_cam5','S_mean','Unnamed: 0'])

w2021 = prepare_color_height_width_df(os.path.join(relevant_locations[0],'allcolhist.txt'), os.path.join(apple_data_path, 'dims_filtered_w2021.csv'))
g2021 = prepare_color_height_width_df(os.path.join(relevant_locations[1],'allcolhist.txt'), os.path.join(apple_data_path, 'dims_filtered_g2021.csv'))
w2022 = prepare_color_height_width_df(os.path.join(relevant_locations[2],'allcolhist.txt'), os.path.join(apple_data_path, 'dims_filtered_w2022.csv'))

# fix the grabs 2022 mistake... again
g2022_col= pd.read_csv(os.path.join(relevant_locations[3],'allcolhist.txt'), sep = '\t')
g2022_badges = list(g2022_col['badge'])
for i in range(len(g2022_badges)):
  g2022_badges[i] = g2022_badges[i][0:4]+'0'+g2022_badges[i][4:]
g2022_col['badge'] = g2022_badges
g2022_dims = pd.read_csv(os.path.join(apple_data_path, 'dims_filtered_g2022.csv'), sep = ',')
g2022 = pd.merge(g2022_col, g2022_dims, on = ['badge', 'appleNR'], how = 'inner').drop(columns=['S_cam1','S_cam2','S_cam3','S_cam4','S_cam5','S_mean','Unnamed: 0'])

def hue_rearranger(df, n_unique_hues):
  for i in range(0, len(df), n_unique_hues): # Step by n_unique_hues
    range_lower, range_upper = i, i + n_unique_hues
    curr_apple_extract = df.iloc[range_lower:range_upper]
    if (i / n_unique_hues % 1000 == 0):
      print((i / n_unique_hues) / len(df) * 100, '% done')
    for j in range(3):
      curr_apple_extract.iloc[j, curr_apple_extract.columns.get_indexer(['H_cam1', 'H_cam2', 'H_cam3', 'H_cam4', 'H_cam5'])] += curr_apple_extract.iloc[n_unique_hues - j - 1, curr_apple_extract.columns.get_indexer(['H_cam1', 'H_cam2', 'H_cam3', 'H_cam4', 'H_cam5'])]
    df.iloc[range_lower:range_upper] = curr_apple_extract
  df = df[df.value < 0.33]
  print('finished sorting and adding hues')
  return df

def format_hue_df_for_pca(df_hue_filt, hue_averages):

  unique_combinations = df_hue_filt[['appleNR', 'badge']].drop_duplicates().values.tolist()

  current_cameras = {}
  for combo in unique_combinations:
    key = combo[1] + '_' + str(combo[0])
    current_cameras[key] = df_hue_filt[(df_hue_filt['appleNR'] == combo[0]) & (df_hue_filt['badge'] == combo[1])][['Cultivar_name','badge','appleNR','value','H_cam1','H_cam2','H_cam3','H_cam4','H_cam5','height_mean_cm','width_mean_cm','HWratio']]

  pca_df = pd.DataFrame()
  for key in current_cameras:
    cam_cols = ['H_cam1', 'H_cam2'  , 'H_cam3', 'H_cam4', 'H_cam5']
    cam_sums = [current_cameras[key][col].sum() for col in cam_cols]
    for col in cam_cols:
        current_cameras[key][col] = current_cameras[key][col] / cam_sums[cam_cols.index(col)]

    for col in cam_cols:
      cam_means = [sum(current_cameras[key]['value']*current_cameras[key][col]) for col in cam_cols]
      cam_data = list(zip(cam_means, cam_cols)) 

      # sort zipped list by camera means
      sorted_cam_data = sorted(cam_data, key=lambda item: item[0])

      # for combining the vector in the correct order, obtain sorted means' camera names
      sorted_cam_cols = [item[1] for item in sorted_cam_data]

      if hue_averages:
        # plot sorted camera hue frequency means (from red / low to green / high)
        pca_df[key] = sorted(cam_means) + [current_cameras[key][size_col].values[0] for size_col in ['height_mean_cm','width_mean_cm','HWratio']]
      else:
        # add list of sorted camera hue values (red => green) to overal dataset with index being 
        pca_df[key] = [value for col in current_cameras[key][cam_cols] for value in current_cameras[key][col].values] + [current_cameras[key][size_col].values[0] for size_col in ['height_mean_cm','width_mean_cm','HWratio']]

  return pca_df

w2021_pca_df = format_hue_df_for_pca(w2021_filt, hue_averages = True)
g2021_pca_df = format_hue_df_for_pca(g2021_filt, hue_averages = True)
w2022_pca_df = format_hue_df_for_pca(w2022_filt, hue_averages = True)
g2022_pca_df = format_hue_df_for_pca(g2022_filt, hue_averages = True)

w2021_pca_prepared = utils_apple.prepare_pca(w2021_pca_df.fillna(0), 3, overall_df = True)
g2021_pca_prepared = utils_apple.prepare_pca(g2021_pca_df.fillna(0), 3, overall_df = True)
w2022_pca_prepared = utils_apple.prepare_pca(w2022_pca_df.fillna(0), 3, overall_df = True)
g2022_pca_prepared = utils_apple.prepare_pca(g2022_pca_df.fillna(0), 3, overall_df = True)
w2021_filt = hue_rearranger(w2021, n_unique_hues)
g2021_filt = hue_rearranger(g2021, n_unique_hues)
w2022_filt = hue_rearranger(w2022, n_unique_hues)
g2022_filt = hue_rearranger(g2022, n_unique_hues)

w2021_pca_prepared[['badge', 'appleNR']] = w2021_pca_prepared['badge'].str.rsplit('_', expand=True)
w2021_pca_prepared['appleNR'] = pd.to_numeric(w2021_pca_prepared['appleNR'], errors='coerce')
w2021_pca_prepared = pd.merge(w2021_pca_prepared, w2021[['badge','appleNR','Cultivar_name','height_mean_cm','width_mean_cm','HWratio']], on=['badge','appleNR']).drop_duplicates()

g2021_pca_prepared[['badge', 'appleNR']] = g2021_pca_prepared['badge'].str.rsplit('_', expand=True)
g2021_pca_prepared['appleNR'] = pd.to_numeric(g2021_pca_prepared['appleNR'], errors='coerce')
g2021_pca_prepared = pd.merge(g2021_pca_prepared, g2021[['badge','appleNR','Cultivar_name','height_mean_cm','width_mean_cm','HWratio']], on=['badge','appleNR']).drop_duplicates()

w2022_pca_prepared[['badge', 'appleNR']] = w2022_pca_prepared['badge'].str.rsplit('_', expand=True)
w2022_pca_prepared['appleNR'] = pd.to_numeric(w2022_pca_prepared['appleNR'], errors='coerce')
w2022_pca_prepared = pd.merge(w2022_pca_prepared, w2022[['badge','appleNR','Cultivar_name','height_mean_cm','width_mean_cm','HWratio']], on=['badge','appleNR']).drop_duplicates()

g2022_pca_prepared[['badge', 'appleNR']] = g2022_pca_prepared['badge'].str.rsplit('_', expand=True)
g2022_pca_prepared['appleNR'] = pd.to_numeric(g2022_pca_prepared['appleNR'], errors='coerce')
g2022_pca_prepared = pd.merge(g2022_pca_prepared, g2022[['badge','appleNR','Cultivar_name','height_mean_cm','width_mean_cm','HWratio']], on=['badge','appleNR']).drop_duplicates()

w2021_pca_prepared['badge_appleNR'] = w2021_pca_prepared['badge'] + '_' + w2021_pca_prepared['appleNR'].astype(str)
g2021_pca_prepared['badge_appleNR'] = g2021_pca_prepared['badge'] + '_' + g2021_pca_prepared['appleNR'].astype(str)
w2022_pca_prepared['badge_appleNR'] = w2022_pca_prepared['badge'] + '_' + w2022_pca_prepared['appleNR'].astype(str)
g2022_pca_prepared['badge_appleNR'] = g2022_pca_prepared['badge'] + '_' + g2022_pca_prepared['appleNR'].astype(str)

w2021_pca_prepared.dropna(inplace=True)
g2021_pca_prepared.dropna(inplace=True)
w2022_pca_prepared.dropna(inplace=True)
g2022_pca_prepared.dropna(inplace=True)

# export PCA plots
utils_apple.plot_cultivar_size_consistency([w2021_pca_prepared.drop('PC1', axis=1), w2022_pca_prepared.drop('PC1', axis=1),
                                            g2021_pca_prepared.drop('PC1', axis=1), g2022_pca_prepared.drop('PC1', axis=1)],
                                          'PCA of all apple data based on per camera sorted hue averages, height and width\nin Wädenswil and Grabs in 2021 and 2022.',
                                           used_cultivars, os.path.join(output_path,'pca_average_hue_pc23'), 
                                           include_guidelines = False)
utils_apple.plot_cultivar_size_consistency([w2021_pca_prepared.drop('PC2', axis=1), w2022_pca_prepared.drop('PC2', axis=1), 
                                            g2021_pca_prepared.drop('PC2', axis=1), g2022_pca_prepared.drop('PC2', axis=1)],
                                          'PCA of all apple data based on per camera sorted hue averages, height and width\nin Wädenswil and Grabs in 2021 and 2022.',
                                           used_cultivars, os.path.join(output_path,'pca_average_hue_pc13'), 
                                           include_guidelines = False)
utils_apple.plot_cultivar_size_consistency([w2021_pca_prepared, w2022_pca_prepared, g2021_pca_prepared, g2022_pca_prepared],
                                          'PCA of all apple data based on per camera sorted hue averages, height and width\nin Wädenswil and Grabs in 2021 and 2022.',
                                           used_cultivars, os.path.join(output_path,'pca_average_hue_pc12'), 
                                           include_guidelines = False)
