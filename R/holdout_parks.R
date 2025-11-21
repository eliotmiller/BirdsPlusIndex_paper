# set wd accordingly
setwd("~/Documents/Work/Research/BirdsPlusIndex/Analysis")

# decide which parks and therefore assets you'll hold out
# from training so you can evaluate on them for the index
matchedParks <- read.csv("data/points_with_parks_8Apr2025.csv")

# this number comes from assuming that the park sizes are in meters squared. then
# finding the area of a 3km diameter circle in meters squared, as follows:
# pi*1.5^2*10^6 and then taking half that value to be conservative
matchedParks <- matchedParks[matchedParks$park_size <= 3534292,]

temp <- plyr::count(matchedParks$park_id)
temp <- temp[order(temp$freq, decreasing=TRUE),]
temp <- temp[temp$freq >= 20,]

# these are the park IDs to holdout. now figure out what their assets are
toSave <- matchedParks[matchedParks$park_id %in% temp$x, c("asset","checklist_id", "park_id")]

write.csv(toSave, "data/holdout_parks_8Apr2025.csv", row.names=FALSE)
