# start the libraries
library(sf)
library(dplyr)

# set your WD accordingly
setwd("~/Documents/Work/Research/BirdsPlusIndex/Analysis")

# load the map extent and projection
map_ext_sf <- readRDS("data/QGIS/map_ext_sf.RDS")
this_prj <- readRDS("data/QGIS/this_prj.RDS")

# set the gis data path for the parks
gis_data <- file.path("data/QGIS/just_parks.gpkg")

# load the parks data
park_dat <- sf::st_read(dsn = gis_data, layer = "parkserve_shapefiles_05212024__parkserve_parksshp") 

# there are 11 of these
invalid_geometries <- !sf::st_is_valid(park_dat)
sum(invalid_geometries)  # Number of invalid geometries

# repair
park_dat <- sf::st_make_valid(park_dat)

# define a utility function
recenter_sf <- function(x, crs) {
  stopifnot(inherits(x, c("sf", "sfc")))
  # find central meridian
  center <- as.numeric(stringr::str_extract(crs, "(?<=lon_0=)[-.0-9]+"))
  if (is.na(center) || length(center) != 1 || !is.numeric(center)) {
    stop("CRS has no central meridian term (+lon_0).")
  }
  
  # edge is offset from center by 180 degrees
  edge <- ifelse(center < 0, center + 180, center - 180)
  
  # create an very narrow sliver to clip out
  delta <- 1e-5
  clipper <- sf::st_bbox(c(xmin = edge - delta, xmax = edge + delta,
                           ymin = -90, ymax = 90),
                         crs = 4326)
  clipper <- sf::st_as_sfc(clipper)
  clipper <- suppressWarnings(smoothr::densify(clipper, max_distance = 1e-3))
  clipper <- sf::st_transform(clipper, crs = sf::st_crs(x))
  
  # cut then project
  x_proj <- sf::st_difference(x, clipper)
  sf::st_transform(x_proj, crs = crs)
}

# now process. this takes a very long time. 12-13 mins
system.time(park_dat <- park_dat %>%
  recenter_sf(crs = this_prj) %>%
  sf::st_geometry())

# now trim the park data to the extent of the bird data
trimmed_park_dat <- sf::st_intersection(park_dat, map_ext_sf)

# now load the bound train, test, and eval data
all_dat <- read.csv("data/to_match_to_parks.csv")

# convert to a spatial object            
all_dat_sf <- sf::st_as_sf(all_dat, coords = c("longitude", "latitude"), crs = 4326)

# ensure these share the same CRS. pretty much positive they do
all_dat_sf <- sf::st_transform(all_dat_sf, crs = sf::st_crs(trimmed_park_dat))

# get the class right for joining later
trimmed_park_dat <- sf::st_as_sf(trimmed_park_dat)

# add park IDs (you lost these when exporting from QGIS)
trimmed_park_dat <- trimmed_park_dat %>%
  dplyr::mutate(park_id = row_number())

# now calculate park size in square meters
trimmed_park_dat <- trimmed_park_dat %>%
  dplyr::mutate(park_size = sf::st_area(x))

# figure out which parks each point falls within (if any). takes 7.5 mins
system.time(points_in_parks <- sf::st_join(all_dat_sf, trimmed_park_dat, join = sf::st_within))

# filter to just points that are in parks
points_with_parks <- points_in_parks %>%
  dplyr::filter(!is.na(park_id))  # Exclude points outside parks

# save out this data frame with park size and ID.
# first Convert the geometry column in points_with_parks to a
# Well-Known Text (WKT) representation. this is important to make sure it opens
# correctly later
points_with_parks$wkt_geometry <- sf::st_as_text(sf::st_geometry(points_with_parks))
points_with_parks_df <- sf::st_drop_geometry(points_with_parks)
write.csv(points_with_parks_df, "data/points_with_parks_8Apr2025.csv", row.names = FALSE)
