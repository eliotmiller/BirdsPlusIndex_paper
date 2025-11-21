#' Convert Merlin predictions to detections
#'
#' @param predictions.wd Path to a directory of BirdNET predictions. These csv files should have four
#' columns each: "species", "score", "start_time", "end_time".
#' @param detections.wd Path to a directory where the BirdNET detections will be written to. 
#' @param geofilters.wd Path to the directory where acceptable species per prediction
#' file are stored. These should be csv files with a single column ("species_code")
#' of allowable species, and should have names that correspond with those from
#' predictions.wd.
#' @param thresholds A single value to be used uniformly as a cutoff
#' across all species.
#' @param taxonomy A data frame with two columns: "sci_name" and "species_code".
#' BirdNET outputs predictions in the format "scientific name_common name" (with
#' spaces, not underscores between Genus and species). The detections will be
#' returned with 
#'
#' @details Fill in here
#'
#' @return Detection files (csv) written to the specified working directory.
#'
#' @author Eliot Miller
#'
#' @export
#'
#' @examples
#' 

convert_Birdnet_predictions_to_detections <- function(predictions.wd,
                                                     detections.wd,
                                                     geofilters.wd,
                                                     thresholds,
                                                     taxonomy)
{
  SR <- c()
  
  # first confirm that there is a matching geofilter for every prediction file
  geofiles <- list.files(geofilters.wd)
  predfiles <- list.files(predictions.wd)
  diffs <- setdiff(predfiles, geofiles)
  
  if(length(diffs) > 0)
  {
    stop("Some geofilter files are missing")
  }
  
  # loop over predfiles
  for(i in 1:length(predfiles))
  {
    # load the prediction file
    preds <- read.csv(paste(predictions.wd, predfiles[i], sep="/"))
    
    # if there is nothing here, skip
    if(dim(preds)[1] == 0)
    {
      SR[i] <- 0
      next()
    }
    
    # load the geofilter file
    geofilter <- read.csv(paste(geofilters.wd, predfiles[i], sep="/"))
    
    # split out the scientific name and convert to a species code
    preds$sci_name <- unlist(lapply(strsplit(preds$species, "_"), "[", 1))
    
    # any non-matching names are going to be dropped here. recommend looking over
    # union of all originally predicted scientific names to see if you're ok with
    # that
    preds <- merge(preds, taxonomy, by="sci_name")
    
    # cut the predictions to spatiotemporally acceptable species
    preds <- preds[preds$species_code %in% geofilter$species_code,]
    
    # check if there are any valid species left, skip to next if not
    if(dim(preds)[1] == 0)
    {
      SR[i] <- 0
      next()
    }
    
    # cut the predictions to acceptable thresholds
    if(is.numeric(thresholds))
    {
      preds <- preds[preds$score >= thresholds,]
    }
    
    # throw an error otherwise
    else
    {
      stop("thresholds must be a numeric value")
    }
    
    # check again if there are any valid species left
    if(dim(preds)[1] == 0)
    {
      SR[i] <- 0
      next()
    }
    
    # otherwise sort on max score, then drop to first instance of each species.
    # this should have the effect of keeping the highest detection for each species
    preds <- preds[order(preds$score, decreasing=TRUE),]
    preds <- preds[!duplicated(preds$species_code),]
    
    # sort back on order of detection, reorganize and save out
    preds <- preds[order(preds$start_time),]
    preds <- data.frame(species_code=preds$species_code,
                        score=preds$score,
                        start_offset_sec=preds$start_time,
                        end_offset_sec=preds$end_time)
    
    SR[i] <- dim(preds)[1]
    
    write.csv(preds, file=paste(detections.wd, predfiles[i], sep="/"), row.names=FALSE)
  }
  
  SR
}

# set working directory then do this to see what the species are
# setwd("")
# results <- do.call(rbind, lapply(list.files(), read.csv))
# results$sci_name <- unlist(lapply(strsplit(results$species, "_"), "[", 1))
setwd("~/Documents/Work/Research/BirdsPlusIndex/Analysis/data")
tax <- read.csv("eBird_taxonomy_v2024.csv")
tax <- tax[tax$CATEGORY=="species",]
# diffs <- setdiff(results$sci_name, tax$SCI_NAME)
# write.csv(data.frame(orig=diffs), "BirdNET_taxonomy_toFix.csv", row.names=FALSE)

# in working through this file, anything that had no business being in that
# part of North America I chose to drop, rather than working to match taxonomically.
# this taxon conversion sheet doesn't have any generalizability; specific to this
# study.
fixed <- read.csv("BirdNET_taxonomy_fixed.csv")
names(fixed) <- c("sci_name", "species_code")
boundTax <- data.frame(sci_name=tax$SCI_NAME, species_code=tax$SPECIES_CODE)
boundTax <- rbind(boundTax, fixed)

# 43s for 28,000 files
system.time(sr1 <- convert_Birdnet_predictions_to_detections(predictions.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_BirdNET_predictions",
                                         detections.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_BirdNET_detections_05",
                                         geofilters.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_geofilters",
                                         thresholds=0.5,
                                         taxonomy=boundTax))

system.time(sr2 <- convert_Birdnet_predictions_to_detections(predictions.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_BirdNET_predictions",
                                                      detections.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_BirdNET_detections_09",
                                                      geofilters.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_geofilters",
                                                      thresholds=0.9,
                                                      taxonomy=boundTax))

system.time(sr3 <- convert_Birdnet_predictions_to_detections(predictions.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_BirdNET_predictions",
                                                      detections.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_BirdNET_detections_025",
                                                      geofilters.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_geofilters",
                                                      thresholds=0.25,
                                                      taxonomy=boundTax))



