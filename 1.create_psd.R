# singularity run --bind /data:/data popcycle-4.12.2.sif bash

###############################
### Prochlorococcus for MPM ###
###############################
library(popcycle)
library(dplyr)
library(glue)

#' Add zero count rows to a sparse one dimensional, one population, PSD data frame
#' 
#' Removes rows where coord_col is NA.
unsparsify <- function(psd, bins) {
  coord_col <- names(psd)[endsWith(names(psd), "_coord")]
  if (length(coord_col) != 1) {
    stop("expected only one '*_coord' column in psd")
  }
  
  psd_with_zeros <- psd %>%
    dplyr::filter(!is.na(.data[[coord_col]])) %>%
    dplyr::group_by(date) %>%
    dplyr::group_modify(function(x, y) {
      date_counts <- tibble::tibble(
        !!coord_col := seq(bins),
        n = 0L
      )
      date_counts[x[[coord_col]], "n"] <- x[["n"]]
      return(date_counts)
    }) %>%
    dplyr::ungroup()
  return(psd_with_zeros)
}

get_delta_v_int_Qc_range <- function(Qc_range, bins) {
  # The growth model takes 1 / (log2 distance between bins) as an integer. Calculate the end of
  # the grid range as the closest such integer that creates a grid that contains the true grid range
  # from the previous step
  
  delta_log2 <- diff(seq(from=log2(Qc_range[1]), to=log2(Qc_range[2]), length=(bins+1)))[1]
  delta_log2_inv <- 1 / delta_log2
  delta_log2_inv_int <- as.integer(delta_log2_inv)
  # Now original range should be
  # c(Qc_range_orig[1], 2**(log2(Qc_range_orig[1]) + (bins * (1 / delta_log2_inv))))
  # Expressing delta_log2_inv as an int gives a little headroom at the top end
  result <- list(
    Qc_range = c(
      Qc_range[1],
      2**(log2(Qc_range[1]) + (bins * (1 / delta_log2_inv_int)))
    ),
    delta_log2 = delta_log2,
    delta_log2_inv = delta_log2_inv,
    delta_log2_inv_int = delta_log2_inv_int
  )
  return(result)
}

get_Qc_ranges <- function(cruise_paths, quantile_, pop, bins) {
  for (i in seq(nrow(cruise_paths))) {
    row <- cruise_paths[i, ]
    message("getting Qc_range for cruise = ", row$cruise)
    message("pop = ", pop)
    
    sfl_tbl <- get_sfl_table(row$db)
    
    flagged_dates <- sfl_tbl %>%
      dplyr::filter(flag != 0) %>%
      dplyr::pull(date)
    
    pop_refrac <- read_refraction_csv() %>%
      dplyr::filter(cruise == row$cruise) %>%
      pull(pop)
    
    data_cols <- paste0("Qc_", pop_refrac)
    
    vct_files <- list.files(row$vct_dir, "\\.parquet$", full.names=T)
    #vct_files <- head(vct_files, 72)
    
    ptm <- proc.time()
    Qc_range_orig <- popcycle::get_vct_quantile_range(
      vct_files, data_cols, quantile_, c(0.01, 0.99), pop = pop, ignore_dates = flagged_dates
    )
    
    if(!is.finite(any(Qc_range_orig))){
      print("No cells of interest for this cruise")
      next
    }
    #Qc_range_orig <- get_vct_range(vct_files, data_cols, quantile = quantile_, pop = pop, cores = cores)
    # The growth model takes 1 / (log2 distance between bins) as an integer. Calculate the end of
    # the grid range as the closest such integer that creates a grid that contains the true grid range
    # from the previous step
    newrange <- get_delta_v_int_Qc_range(Qc_range_orig, bins)
    cruise_paths[i, "Qc_range_orig_start"] <- Qc_range_orig[1]
    cruise_paths[i, "Qc_range_orig_end"] <- Qc_range_orig[2] * 2  # to make sure to have empty bins at the end of the PSD
    cruise_paths[i, "delta_log2"] <- newrange$delta_log2
    cruise_paths[i, "delta_log2_inv"] <- newrange$delta_log2_inv
    cruise_paths[i, "delta_log2_inv_int"] <- newrange$delta_log2_inv_int
    cruise_paths[i, "Qc_range_start"] <- newrange$Qc_range[1]
    cruise_paths[i, "Qc_range_end"] <- newrange$Qc_range[2]
    invisible(gc())
    
    message("population = ", pop)
    message("refractive index = ", pop_refrac)
    message("quantile = ", quantile_)
    message("data columns = ", data_cols)
    message(glue("Qc 0.1, 0.99 range = {Qc_range_orig[1]}, {Qc_range_orig[2]}"))
    message("delta_log2 = log2 distance between bins = ", newrange$delta_log2)
    message("delta_log2_inv = 1 / (log2 distance between bins) = ", newrange$delta_log2_inv)
    message(glue("Qc range with integer value for delta_log2_inv ({newrange$delta_log2_inv_int}) = {newrange$Qc_range[1]}, {newrange$Qc_range[2]}"))
    
    deltat <- proc.time() - ptm
    message("vct range in ", lubridate::duration(deltat[["elapsed"]]))
    message(row$cruise, " finished")
    message("")
  }
  
  return(cruise_paths)
}

create_model_data <- function(cruise_paths, bins, quantile_, pop, sparse = FALSE) {
  for (i in seq(nrow(cruise_paths))) {
    row <- cruise_paths[i, ]
    Qc_range <- c(row$Qc_range_start, row$Qc_range_end)
    message("processing cruise = ", row$cruise)
    message("bins = ", bins)
    message("sparse = ", sparse)
    message("pop = ", pop)
    message("Qc_range =", Qc_range)
    out <- row$output
    
    sfl_tbl <- get_sfl_table(row$db)
    
    flagged_dates <- sfl_tbl %>%
      dplyr::filter(flag != 0) %>%
      dplyr::pull(date)
    
    par <- sfl_tbl %>%
      dplyr::filter(flag == 0) %>%
      dplyr::select(date, par, lat, lon)
    
    # Correct raw PAR values
    par_calib <- popcycle::read_par_csv() %>%
      dplyr::filter(!is.na(correction), !is.infinite(correction), cruise == row$cruise)
    if (nrow(par_calib) == 1) {
      message("Applying PAR correction value " ,par_calib$correction)
      par$par <- par$par * par_calib$correction[1]
    } else {
      message("No PAR correction value found for this cruise")
    }
    
    refracs <- read_refraction_csv() %>%
      dplyr::filter(cruise == row$cruise) %>%
      select(-c(cruise))
    pop_refrac <- refracs[[pop]]
    data_cols <- paste0("Qc_", pop_refrac)
    
    vct_files <- list.files(row$vct_dir, "\\.parquet$", full.names=T)
    #vct_files <- head(vct_files, 72)
    
    if (!any(is.infinite(Qc_range)) && !any(is.na(Qc_range))) {
      ptm <- proc.time()
      
      dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
      
      # Make the grid
      Qc_range <- c(row$Qc_range_start, row$Qc_range_end)
      grid <- popcycle::create_grid(bins, log_base=2, log_answers=FALSE, Qc_range = Qc_range)
      #grid <- popcycle::create_grid(bins, log_base=2, log_answers=FALSE, Qc_range = c(0.0135, 0.11777484))
      grid <- grid["Qc"]
      grid_df <- tibble::tibble(cruise=row$cruise, Qc=grid$Qc)
      arrow::write_parquet(grid_df, paste0(out, ".psd-grid.parquet"))
      
      # Create the distribution
      psd <- popcycle::create_PSD(
        vct_files, quantile_, refracs, grid, ignore_dates = flagged_dates, pop = pop, cores = cores
      )
      
      # Remove counts out of grid range (coord is NA)
      # Remove Qc_sum column
      psd <- psd %>%
        dplyr::filter(!is.na(Qc_coord)) %>%
        dplyr::select(-c(Qc_sum))
      hourly_psd <- popcycle::group_psd_by_time(psd, time_expr = "1 hours")
      psd <- tibble::as_tibble(psd) # group_psd_by_time may convert psd to data.table
      
      # Remove population label since we only have only population
      psd$pop <- NULL
      hourly_psd$pop <- NULL
      # Add zero counts if necessary
      if (!sparse) {
        message("adding zero count rows")
        psd <- unsparsify(psd, bins)
        hourly_psd <- unsparsify(hourly_psd, bins)
      }
      # Add cruise column
      psd <- psd %>% dplyr::mutate(cruise = row$cruise, .before = 1)
      hourly_psd <- hourly_psd %>% dplyr::mutate(cruise = row$cruise, .before = 1)
      arrow::write_parquet(psd, paste0(out, ".psd-full.parquet"))
      arrow::write_parquet(hourly_psd, paste0(out, ".psd-hourly.parquet"))
      invisible(gc())
      
      deltat <- proc.time() - ptm
      message("Full PSD dim = ", stringr::str_flatten(dim(psd), " "), ", MB = ", object.size(psd) / 2**20)
      message("Hourly PSD dim = ", stringr::str_flatten(dim(hourly_psd), " "), ", MB = ", object.size(hourly_psd) / 2**20)
      message("psd in ", lubridate::duration(deltat[["elapsed"]]))
      
      # Only keep PAR dates that are in PSD
      # Average by hour
      par <- par %>%
        dplyr::filter(date %in% unique(psd$date))
      hourly_par <- par %>%
        dplyr::group_by(date = lubridate::floor_date(date, "hour")) %>%
        dplyr::summarise(
          par = mean(par, na.rm = T),
          lat = mean(lat, na.rm = T),
          lon = mean(lon, na.rm = T)
        )
      # Add cruise column
      par <- par %>% dplyr::mutate(cruise = row$cruise, .before = 1)
      hourly_par <- hourly_par %>% dplyr::mutate(cruise = row$cruise, .before = 1)
      arrow::write_parquet(par, paste0(out, ".par-full.parquet"))
      arrow::write_parquet(hourly_par, paste0(out, ".par-hourly.parquet"))
      
    } else {
      message("psd range has infinite values")
    }
    message(row$cruise, " finished")
    message("")
  }
}






#------------
# ENVIRONMENT 
#------------
cruise_dir <- "/data/google-drive-parquet"

path.to.dbs <- sub("_vct",".db",grep("_vct$", list.dirs(cruise_dir, recursive=TRUE), value = TRUE))


# remove Low Quality cruises
path.to.dbs <- path.to.dbs[-c(grep("CMOP_1",path.to.dbs),
                              grep("GP15_2",path.to.dbs),
                              grep("MV1405",path.to.dbs),
                              grep("Thompson_2",path.to.dbs),
                              grep("Thompson_7",path.to.dbs),
                              grep("Tokyo_1",path.to.dbs),
                              grep("Tokyo_2",path.to.dbs),
                              grep("Tokyo_4",path.to.dbs))]

# remove Multiple PMT settings
path.to.dbs <- path.to.dbs[-c(grep("KiloMoana_1",path.to.dbs),
                              grep("HOT300",path.to.dbs))]

# remove data from prototype SeaFlow v2
path.to.dbs <- path.to.dbs[-c(grep("KM1923_740", path.to.dbs))]






pop <- "prochloro"
quantile_ <- "2.5"
bins <- 50
global_Qc_range_flag <- TRUE
cores <- 4
out_dir <- paste0(as.Date(Sys.time()))

cruises <- basename(dirname(path.to.dbs))

cruise_paths <- tibble::tibble(
  cruise = cruises,
  vct_dir = file.path(cruise_dir, cruises, paste0(cruises, "_vct")),
  #vct_dir = "vct_test",
  db = file.path(cruise_dir, cruises, paste0(cruises, ".db")),
  output = file.path(out_dir, cruises, pop)
)


t0 <- proc.time()
# Add Qc Range information to cruise_paths
cruise_paths <- get_Qc_ranges(cruise_paths, quantile_, pop, bins)
deltat <- proc.time() - t0
message("Calculated Qc_range for all  data in ", lubridate::duration(deltat[["elapsed"]])) # taking 7 minutes for 62 cruises


cruise_paths_orig <- cruise_paths  # keep a copy before altering


cruise_paths <- cruise_paths %>% filter(!is.na(Qc_range_end))


# Set a global Qc range for all cruises
global_Qc_range_flag <- TRUE

if (global_Qc_range_flag) {
  global_Qc_range_orig <- c(
    min(cruise_paths$Qc_range_orig_start),
    max(cruise_paths$Qc_range_orig_end))
  global_Qc_range_data <- get_delta_v_int_Qc_range(global_Qc_range_orig, bins)
  print(global_Qc_range_data)
  
  # To use the global range for all cruises, overriding their per-cruise
  # Qc ranges.
  cruise_paths$Qc_range_start <- global_Qc_range_data$Qc_range[1]
  cruise_paths$Qc_range_end <- global_Qc_range_data$Qc_range[2]
}


#-----------
# CREATE PSD (~ 10 minutes for 62 cruises)
#-----------

t0 <- proc.time()
create_model_data(cruise_paths, bins, quantile_, pop, sparse = FALSE)
deltat <- proc.time() - t0
message("Calculated PSD for all  data in ", lubridate::duration(deltat[["elapsed"]]))

