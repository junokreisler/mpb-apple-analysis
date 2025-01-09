import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import skimage
import sklearn
import zipfile
import os
import re
import utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

##################################################################################################################
##################################################################################################################
# write the location of the zip folder in apple_data_path, then the desired unzipping destination in data_path!
apple_data_path = ''
data_path = ''
##################################################################################################################
##################################################################################################################

if not os.path.exists(data_path):
    os.mkdir(data_path)

# TO DEBUG - DON'T PUT IN UTILS YET

def obtain_primary_color_badge(badge_averages):
  # BADGE NAMES SHOULD BE IN COLUMNS!
  max_val_dict = {}
  for i in range(0, len(badge_averages.columns)):
    curr = badge_averages.iloc[:,i].idxmax()

    ## DEBUG - MISSING BADGE COLORS
    if badge_averages.columns[i] in ['11.93.0.1000', '11.93.0.1001', '11.93.0.1002', '11.93.0.1006', '11.93.0.1047', '11.93.0.1079', '11.93.0.1092']:
      print(badge_averages.columns[i], curr)

    if curr in max_val_dict:
      max_val_dict[curr][0] += 1
      max_val_dict[curr].append(badge_averages.columns[i])
    else:
      max_val_dict[curr] = [1]

  return max_val_dict

##################################################################################################################
# unzip the data folder and distribute into folders by year, location
zip_files = [f for f in os.listdir(apple_data_path) if f.endswith('.zip')]
for zip_file in zip_files:
  zip_file_path = os.path.join(apple_data_path, zip_file)
  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(data_path) #added destination directory

# list of used locations and years (2023 and conthey excluded due to incomplete / low quality data
relevant_locations = [os.path.join(data_path,'2021','waedenswil'),
                      os.path.join(data_path,'2021','grabs'),
                      os.path.join(data_path,'2022','waedenswil'),
                      os.path.join(data_path,'2022','grabs')]

# read annotation file
badge_annotation = pd.read_excel(os.path.join(data_path,'230822_annotation.xlsx'))

color_summary_w_2021 = pd.read_csv(relevant_locations[0]+'/allcolhist.txt', sep='\t')
color_summary_g_2021 = pd.read_csv(relevant_locations[1]+'/allcolhist.txt', sep='\t')
color_summary_w_2022 = pd.read_csv(relevant_locations[2]+'/allcolhist.txt', sep='\t')
color_summary_g_2022 = pd.read_csv(relevant_locations[3]+'/allcolhist.txt', sep='\t')

# identify averages and the closest "badge's average apple color representative"

#1. Obtain average color distribution for all apple images
#2. Identify the image that is the closest to the average

# aggregate datasets to obtain average hues for all cameras per apple / for all apples per badge
color_byApple_w_2021 = color_summary_w_2021.groupby(['badge','appleNR','value']).agg({'H_mean': 'mean', 'S_mean': 'mean'})
color_byBadge_w_2021 = color_summary_w_2021.groupby(['badge','value']).agg({'H_mean': 'mean', 'S_mean': 'mean'})

color_byApple_g_2021 = color_summary_g_2021.groupby(['badge','appleNR','value']).agg({'H_mean': 'mean', 'S_mean': 'mean'})
color_byBadge_g_2021 = color_summary_g_2021.groupby(['badge','value']).agg({'H_mean': 'mean', 'S_mean': 'mean'})

color_byApple_w_2022 = color_summary_w_2022.groupby(['badge','appleNR','value']).agg({'H_mean': 'mean', 'S_mean': 'mean'})
color_byBadge_w_2022 = color_summary_w_2022.groupby(['badge','value']).agg({'H_mean': 'mean', 'S_mean': 'mean'})

color_byApple_g_2022 = color_summary_g_2022.groupby(['badge','appleNR','value']).agg({'H_mean': 'mean', 'S_mean': 'mean'})
color_byBadge_g_2022 = color_summary_g_2022.groupby(['badge','value']).agg({'H_mean': 'mean', 'S_mean': 'mean'})

# extract all badges from each dataset to loop through them and check which ones have enough samples
all_badges_w_2021 = color_summary_w_2021['badge'].unique()
all_badges_g_2021 = color_summary_g_2021['badge'].unique()
all_badges_w_2022 = color_summary_w_2022['badge'].unique()
all_badges_g_2022 = color_summary_g_2022['badge'].unique()

# filter badges that contain less than minimum necessary apples, identify ID of apple with the closest hue distribution to the badge's average
w_2021_col_df, _ = utils.identify_representative_color_apple_all(path_to_summary = relevant_locations[0], metric = 'manhattan', min_apples = 5)
g_2021_col_df, _  = utils.identify_representative_color_apple_all(path_to_summary = relevant_locations[1], metric = 'manhattan', min_apples = 5)
w_2022_col_df, _  = utils.identify_representative_color_apple_all(path_to_summary = relevant_locations[2], metric = 'manhattan', min_apples = 5)
g_2022_col_df, _  = utils.identify_representative_color_apple_all(path_to_summary = relevant_locations[3], metric = 'manhattan', min_apples = 5)

# obtain number of badges retained per dataset, extract the used badges and their representing apple nrs
len_w_2021 = len(w_2021_col_df)
used_badges_w_2021 = list(w_2021_col_df.keys())
used_nrs_w_2021 = list(w_2021_col_df.values())

len_g_2021 = len(g_2021_col_df)
used_badges_g_2021 = list(g_2021_col_df.keys())
used_nrs_g_2021 = list(g_2021_col_df.values())

len_w_2022 = len(w_2022_col_df)
used_badges_w_2022 = list(w_2022_col_df.keys())
used_nrs_w_2022 = list(w_2022_col_df.values())

# MISTAKE IN GRABS 2022 - badge IDs are lacking a 0 (50.0. ... instead of 50.00. ... ==> fix)
len_g_2022 = len(g_2022_col_df)
used_badges_g_2022 = list(g_2022_col_df.keys())
used_nrs_g_2022 = list(g_2022_col_df.values())
for b in range(0, len(used_badges_g_2022)):
  used_badges_g_2022[b] = used_badges_g_2022[b][0:4]+'0'+used_badges_g_2022[b][4:]

print('Length of Waedenswil 2021 df:', len_w_2021,
      '\nLength of Grabs 2021 df:', len_g_2021,
      '\nLength of Waedenswil 2022 df:', len_w_2022,
      '\nLength of Grabs 2022 df:', len_g_2022)

# create lists for true names of badges, extract data for using in the next step with annotation
named_badges_overall = []
intersect_w = list(set(used_badges_w_2021) & set(used_badges_w_2022))
intersect_g = list(set(used_badges_g_2021) & set(used_badges_g_2022))

all_badges_wg = list(set(intersect_w) | set(intersect_g))

named_badges_w, named_badges_g = [],[]
for badge_name in intersect_w:
  named_badges_w.append(badge_annotation.loc[badge_annotation.badge == badge_name]['Cultivar_name'].iloc[0])

for badge_name in intersect_g:
  named_badges_g.append(badge_annotation.loc[badge_annotation.badge == badge_name]['Cultivar_name'].iloc[0])

named_badges_overall = list(set(named_badges_w) & set(named_badges_g))

# export the selected overlapping badges' annotations
subset_annotation = badge_annotation[badge_annotation['badge'].isin(all_badges_wg)]
subset_annotation.to_excel('subset_annotation_2021_2022.xlsx', index=False)

# retrieve the hue values for extracting the badge mean (hue dist of the "average" apple)
extract_data_w_2021 = color_byApple_w_2021.reset_index(inplace=False)
extract_data_g_2021 = color_byApple_g_2021.reset_index(inplace=False)
extract_data_w_2022 = color_byApple_w_2022.reset_index(inplace=False)
extract_data_g_2022 = color_byApple_g_2022.reset_index(inplace=False)

w_2021_h = utils_apple.extract_badge_mean(used_badges_w_2021[0], used_nrs_w_2021[0], extract_data_w_2021)
g_2021_h = utils_apple.extract_badge_mean(used_badges_g_2021[0], used_nrs_g_2021[0], extract_data_g_2021)
w_2022_h = utils_apple.extract_badge_mean(used_badges_w_2022[0], used_nrs_w_2022[0], extract_data_w_2022)
g_2022_h = utils_apple.extract_badge_mean(used_badges_g_2022[0], used_nrs_g_2022[0], extract_data_g_2022)

for i in range(0, len(used_badges_w_2021)):
  w_2021_h = pd.concat([w_2021_h,
                       utils.extract_badge_mean(used_badges_w_2021[i], used_nrs_w_2021[i], extract_data_w_2021)],
                       axis=1)

for i in range(0, len(used_badges_g_2021)):
  g_2021_h = pd.concat([g_2021_h,
                       utils.extract_badge_mean(used_badges_g_2021[i], used_nrs_g_2021[i], extract_data_g_2021)],
                       axis=1)

for i in range(0, len(used_badges_w_2022)):
  w_2022_h = pd.concat([w_2022_h,
                       utils.extract_badge_mean(used_badges_w_2022[i], used_nrs_w_2022[i], extract_data_w_2022)],
                       axis=1)

for i in range(0, len(used_badges_g_2022)):
  g_2022_h = pd.concat([g_2022_h,
                       utils.extract_badge_mean(used_badges_g_2022[i], used_nrs_g_2022[i], extract_data_g_2022)],
                       axis=1

w_2021_h.set_index(extract_data_w_2021.value.unique(),inplace=True)
g_2021_h.set_index(extract_data_g_2021.value.unique(),inplace=True)
w_2022_h.set_index(extract_data_w_2022.value.unique(),inplace=True)
g_2022_h.set_index(extract_data_g_2022.value.unique(),inplace=True)

w_2021_h.set_index(extract_data_w_2021.value.unique(),inplace=True)
g_2021_h.set_index(extract_data_g_2021.value.unique(),inplace=True)
w_2022_h.set_index(extract_data_w_2022.value.unique(),inplace=True)
g_2022_h.set_index(extract_data_g_2022.value.unique(),inplace=True)         


w_2021_maxvaldict = obtain_primary_color_badge(w_2021_h)
g_2021_maxvaldict = obtain_primary_color_badge(g_2021_h)
w_2022_maxvaldict = obtain_primary_color_badge(w_2022_h)
g_2022_maxvaldict = obtain_primary_color_badge(g_2022_h)

print('0-1 Hue value\tW2021\tW2022\tG2021\tG2022\n')
for key in w_2021_maxvaldict:
  key_print = key
  if key == 0.125:
    key_print = '0.125000'
  print(key_print,'\t', w_2021_maxvaldict[key][0], '\t', w_2022_maxvaldict[key][0],
        '\t', g_2021_maxvaldict[key][0], '\t', g_2022_maxvaldict[key][0])

print('\nTotal top hues:\t', len(w_2021_maxvaldict.keys()), '\t', len(w_2022_maxvaldict.keys()),
      '\t', len(g_2021_maxvaldict.keys()), '\t', len(g_2022_maxvaldict.keys()))

# CARRY OUT PCA
def prepare_pca(feature_df, n_pc):
  # Standardize the data
  scaler = StandardScaler()
  colnames = ['PC'+str(i+1) for i in range(n_pc)]

  feature_df_new = feature_df.T.iloc[1:]
  scaled_data = scaler.fit_transform(feature_df_new)
  pca = PCA(n_components=n_pc)
  principal_components = pca.fit_transform(scaled_data)
  pc_df = pd.DataFrame(data=principal_components, columns=colnames)  # Adjust column names as needed
  explained_variance = pca.explained_variance_ratio_

  print(f"Explained variance by each component: {explained_variance}")
  print(f"Total explained variance: {round(sum(explained_variance),5)}")

  pc_df['badge'] = feature_df.columns[1:]
  pc_df['outlier'] = False

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

w_2021_pca = prepare_pca(w_2021_h, 3)
g_2021_pca = prepare_pca(g_2021_h, 3)
w_2022_pca = prepare_pca(w_2022_h, 3)
g_2022_pca = prepare_pca(g_2022_h, 3)

w_2021_pca = add_color_label(w_2021_pca, w_2021_maxvaldict)
w_2022_pca = add_color_label(w_2022_pca, w_2022_maxvaldict)
g_2021_pca = add_color_label(g_2021_pca, g_2021_maxvaldict)
g_2022_pca = add_color_label(g_2022_pca, g_2022_maxvaldict)

# obtaining and saving the PCA plots for each location in each year - color distribution for all badges that have at least N apples measured

plt_output = os.path.join(data_path,'pca_badge_avgs')
if not os.path.exists(plt_output):
    os.mkdir(plt_output)

top_cols = {'green': 'green', 'yellow/green':'lime','red':'red',
            'yellow':'yellow', 'red/yellow':'orange', '':'grey'}
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.scatter(w_2021_pca['PC2'], w_2021_pca['PC3'], c=w_2021_pca.color.map(top_cols))
plt.xlabel('Principal Component 2 (PC2)')
plt.ylabel('Principal Component 3 (PC3)')
plt.title('PCA Transformed Data Waedenswil 2021')
plt.savefig(os.path.join(plt_output,'PCA_W2021.png'))

plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.scatter(w_2022_pca['PC2'], w_2022_pca['PC3'], c=w_2022_pca.color.map(top_cols))
plt.xlabel('Principal Component 2 (PC2)')
plt.ylabel('Principal Component 3 (PC3)')
plt.title('PCA Transformed Data Waedenswil 2022')
plt.savefig(os.path.join(plt_output,'PCA_W2022.png'))

plt.figure(figsize=(8, 6))
plt.scatter(g_2021_pca['PC2'], g_2021_pca['PC3'], c=g_2021_pca.color.map(top_cols))
plt.xlabel('Principal Component 2)')
plt.ylabel('Principal Component 3')
plt.title('PCA Transformed Data Grabs 2021')
plt.savefig(os.path.join(plt_output,'PCA_G2021.png'))

plt.figure(figsize=(8, 6))
plt.scatter(g_2022_pca['PC2'], g_2022_pca['PC3'], c=g_2022_pca.color.map(top_cols))
plt.xlabel('Principal Component 2)')
plt.ylabel('Principal Component 3')
plt.title('PCA Transformed Data Grabs 2022')
plt.savefig(os.path.join(plt_output,'PCA_G2022.png'))
