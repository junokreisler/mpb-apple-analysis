# mpb-apple-analysis
Repository containing the code used in the analysis of fruits diversity from various apple accessions.

The report is saved as `mpb_rotation_report.pdf`.

Libraries used:
* `numpy`
* `pandas`
* `matplotlib`

Data used:

Summaries of apple colors and shapes of apples from Wädenswil and Grabs in 2021 and 2022. 
Image data would be too massive and tedious to analyze locally (and virtually) and would only help with data concerning edges/shapes. However, a script dealing with edge data, clustering apples by similarities of edge distances to center, is also provided. Instead, the shape classes obtained in prior analysis are used as additional traits/variables.


## Structure:

* `utils.py` - module containing functions used in data pre-processing
* `visualization.py` - module containing functions used in data visualization incl. plot generation
* `main1.py` - main code for the first analysis approach
* `main2.py` - main code for the second analysis approach
* folder `1_results` - plots showing results and tabular data output from the first approach
* folder `2_results` - plots showing results and tabular data output from the second approach
* `extra_image_shape.py` - extra script for (rough) edge extraction and clustering by similarities of edge distances to center (radius vs degree) curves.
* `extra_polar_clustering.py` - script for obtaining radius vs degree curves from location and year summary files, manually classifying into shape classes and outputting badges with top cluster assignment.

## Analysis approaches

### Approach 1 - overview of apple color diversity

#### Summary

Rough overview of the consistency of average badge apple fruit color appearances, using a "most average" apple as a singular representative of a badge in a given year or location. The analysis involves extracting the best average color hue representative for badges with at least N apples imaged, choosing a real apple whose color hue frequency distributions are least different from the calculated hue frequency distribution averages in the given badge+location+year.

#### Steps
1. First, the average prevalence at each "value" of HUE and SATURATION is measured. If the number of apple samples is at least N (N = 5 default),

  2. The H and S values of each individual apple are compared with the means for each H/S values of the whole badge's batch.
  3. The individual appleNR with the lowest Manhattan distance to the batch average is chosen as the "color representative apple"
  4. The badge and it's representative appleNR are returned as a dictionary.   
5. Using the badge annotation file, only common badges present in both years and locations are kept.
6. The average apple representative hue distributions are extracted and the average apple becomes the representative of the badge in the given year and location.
7. The apple is assigned an "expected" color class based on the most prevalent hue (red, red-yellow, yellow, yellow-green, green)
8. PCA is carried out on the dataset of badge-representing apples to obtain an average color distribution for each location and year. 2D (2nd, 3rd PC) and 3D (1st, 2nd, 3rd PC) PCA plots are obtained. 
   * Additional plots with colors from other years, locations on each color plot are also obtained to look at consistency.

### Approach 2 - intra- and inter-badge (genotype x environment) variation estimation

#### Summary

More detailed look at the variance of apple fruit color appearances and shapes using the shortlist of common badges obtained in the previous analysis approach. Tabular data is generated that would serve as a recommendation for badges to use for generative model learning.

#### Steps

1. The shortlisted common badges (at least 5 apples per badge in all years+locations) from the previous analysis step are used.
2. Instead of using the mean of hue distributions of 5 cameras for an apple, all individual cameras' hue distributions are transformed into one vector of 36 (number of unique hues) * 5 (number of cameras) dimensions, with each camera's groups first sorted by ascending mean hue. A "dominant color" label is given to each apple, similarily to the previous analysis step (only for visualization purposes). A variable value of `0`, `0.25`, `0.5`, `0.75` or `1.00` is given to each apple to represent one of the 5 different top shape (top lobedness) classes.
3. 3D PCA plots of apples of the shortlisted badges are obtained for each year and location, yielding:
  * multiple plots with X (X = 10 default) badges highlighted at a time;
  * plots with all dominant color labels highlighted. 
4. Using the first K PCs (representing >=80% of the variance), the intra-badge distances are calculated. 3D PCA plot of the badge centroids is obtained for each year and location. 
5. A list of badges with their intra-badge distances from above calculations is obtained for both years and locations in one file.

## Shape clustering

Two scripts were written for analyzing shape data.

* `extra_image_shape.py` deals with raw image data, which there was very little of (as the size of all image data would exceed my currently available computational capacities).
* `extra_polar_clustering.py` performs de novo clustering of recorded radius vs radian measurements, using the means normalized curves of camera 2 to 5 for each apple. Based on clustering results, S unique shape classes were identified and badges were assigned to top shape hit. This data is used here as another way to label the PCA plots, as continuous and discrete variables cannot be analyzed simultaneously the way it has been done in this project.

