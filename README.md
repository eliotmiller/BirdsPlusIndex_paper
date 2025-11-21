# BirdsPlus Species Scores

This repository contains the data and code necessary to duplicate the **BirdsPlus Index** results that accompany our in-review paper.

**Preprint:** <https://www.biorxiv.org/content/10.1101/2025.10.14.682393v1>

------------------------------------------------------------------------

## Repository Contents

### R/

-   R/convert_Birdnet_predictions_to_detections.R
-   R/convert_Merlin_predictions_to_detections.R
-   R/holdout_parks.R
-   R/match_ebird_to_parks-current.R

### Python/

-   Python/1-prep-BPI-train&test-forReview.py
-   Python/1-prep-ebird-train&test-forReview.py
-   Python/2-run-BPI-RF&plot-forReview.py
-   Python/2-run-ebird-RF&plot-forReview.py
-   Python/3-quantify-RFmodel-performance-forReview.py
-   Python/4-evaluate-focal-sites-forReview.py

### outputs/

-   outputs/

### data/

-   **Top-level files:**
    -   data/acoustic_checklist_metadata.csv
    -   data/BirdNET_taxonomy_fixed.csv
    -   data/eBird_taxonomy_v2024.csv
    -   data/holdout_parks_8Apr2025.csv
    -   data/indexScores_v1-4.csv
    -   data/NE_acoustic_covariates.csv
    -   data/NE_species_thresholds.csv
    -   data/srd_subset_acoustic+ebird_17Oct2023.csv
-   **Subdirectories:**
    -   data/NE_BirdNET_detections_025
    -   data/NE_BirdNET_detections_05
    -   data/NE_BirdNET_detections_09
    -   data/NE_BirdNET_predictions
    -   data/NE_geofilters
    -   data/NE_Merlin_detections_025
    -   data/NE_Merlin_detections_05
    -   data/NE_Merlin_detections_09
    -   data/NE_Merlin_detections_custom
    -   data/NE_Merlin_predictions
    -   data/QGIS

------------------------------------------------------------------------

## Usage

1.  Clone the repository:

``` bash
git clone https://github.com/eliotmiller/BirdsPlusIndex_paper.git
```

2.  ddd

3.  ddd
