results of the third analysis part.

since some folders are too huge as .zip files to load here, you can obtain them here:

https://drive.google.com/drive/folders/17JQ9v4aYSQ6ZnnGyTp7r_Xe5O2osRgpG?usp=sharing

## Data contents

### Folders of plots

#### Average sorted camera hue PCA plots (`pca_average_hue_pc[xy]`)

Average sorted camera hues:

1. From each camera view, the average hue was obtained (with highest 10% hue levels - deep violet/red - added to the lowest 10%).
2. Cameras were sorted by increasing average hue, i.e. from the reddest to the greenest camera.
3. Hence, the reddest average camera values are compared with other reddest values, greenest with greenest.

The top 3 PCs represent 92-94% (depending on year and location) of all variance. General point background seems to be quite similar for all 4 datasets. Points that look like outliers shouldn't necessarily be considered that, as they often form a significant cultivar cluster themselves.

PC1 correlates very highly with the height-width ratio, and PC3 seems to correlate with the sorted camera hues. The PCA plots for Wädenswil 2021 and Grabs 2022 for reference can be viewed in the parent folder (NOT subfolders) with names `[w2021/g2022]_[..]_pca.png`

#### Ranked camera top 12 hue PCA plots (same camera mean hue order, but all hues instead of the average) (`pca_allhues_pc[xy]`)

The top 22 PCs represent 79-81% (depending on year and location) of all variance. General point background for each PCx vs PCy plot seems to have a similar shape, however, it seems to be quite condensed due to a "trail" of outliers, some of which are individual apples and others seem to be very spread-out cultivar lines. Despite the compacted shape formed by most of the PCA data points, apples from the same cultivar cluster together.

#### `[..]_cam_hue_avg_pca`

For each of the two locations and times, the PCA dataset itself, together with the used variables and annotation.

#### `[..]_pca_cultivar_means`

For each cultivar, the centroid values and their standard deviations for the 3 PCs. The recommended cultivars to use would be the ones that show minimal SD across all 4 datasets.
