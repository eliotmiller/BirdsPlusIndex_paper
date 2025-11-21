#' Convert Merlin predictions to detections
#'
#' @param predictions.wd Path to a directory of Merlin predictions. These csv files should have four
#' columns each: "species_code", "score", "start_offset_sec", "end_offset_sec".
#' @param detections.wd Path to a directory where the Merlin detections will be written to. 
#' @param geofilters.wd Path to the directory where acceptable species per prediction
#' file are stored. These should be csv files with a single column ("species_code")
#' of allowable species, and should have names that correspond with those from
#' predictions.wd.
#' @param thresholds Either a single value, to be used uniformly as a cutoff
#' across all species, or a data frame with two columns: "species_code" and a
#' "threshold" for detection of that species.
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

convert_Merlin_predictions_to_detections <- function(predictions.wd,
                                                     detections.wd,
                                                     geofilters.wd,
                                                     thresholds)
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
    
    # load the geofilter file
    geofilter <- read.csv(paste(geofilters.wd, predfiles[i], sep="/"))
    
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
    
    # make the assumption that if thresholds is not numeric that it is in the
    # right format
    else
    {
      # merge the thresholds in and cut
      preds <- merge(preds, thresholds)
      preds <- preds[preds$score >= preds$threshold,]
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
    
    # sort back on order of detection and save out. drop the threshold column
    preds <- preds[order(preds$start_offset_sec),]
    preds$threshold <- NULL
    
    # write out the SR
    SR[i] <- dim(preds)[1]
    
    write.csv(preds, file=paste(detections.wd, predfiles[i], sep="/"), row.names=FALSE)
  }
  
  SR
}

# 38 s for 28,000 files
system.time(sr1 <- convert_Merlin_predictions_to_detections(predictions.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_Merlin_predictions",
                                         detections.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_Merlin_detections_05",
                                         geofilters.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_geofilters",
                                         thresholds=0.5))

system.time(sr2 <- convert_Merlin_predictions_to_detections(predictions.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_Merlin_predictions",
                                                     detections.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_Merlin_detections_09",
                                                     geofilters.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_geofilters",
                                                     thresholds=0.9))

system.time(sr3 <- convert_Merlin_predictions_to_detections(predictions.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_Merlin_predictions",
                                                     detections.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_Merlin_detections_025",
                                                     geofilters.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_geofilters",
                                                     thresholds=0.25))

threshes <- read.csv("~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_species_thresholds.csv")
names(threshes) <- c("species_code","threshold")

# 55s for all files
system.time(sr4 <- convert_Merlin_predictions_to_detections(predictions.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_Merlin_predictions",
                                                     detections.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_Merlin_detections_custom",
                                                     geofilters.wd="~/Documents/Work/Research/BirdsPlusIndex/Analysis/data/NE_geofilters",
                                                     thresholds=threshes))




