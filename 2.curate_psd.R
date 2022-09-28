library(popcycle)
library(tidyverse)
library(plotly)

#---------
# READ PSD
#---------

in_dir <- "2022-09-26"

# COMBINE all PSD (+ par)
par_hourly_files <- list.files(in_dir, pattern = "*\\.par-hourly\\.parquet", full.names = TRUE, recursive = TRUE)
  id1 <- grep("curated", par_hourly_files)
  if(length(id1) == 1) par_hourly_files <- par_hourly_files[-c(id1)]
psd_hourly_files <- list.files(in_dir, pattern = "*\\.psd-hourly\\.parquet", full.names = TRUE, recursive = TRUE)
  id2 <- grep("curated", psd_hourly_files)
  if(length(id2) == 1) psd_hourly_files <- psd_hourly_files[-c(id2)]
psd_grid_files <- list.files(in_dir, pattern = "*\\.psd-grid\\.parquet", full.names = TRUE, recursive = TRUE)
  id3 <- grep("curate", psd_grid_files)
  if(length(id3) == 1) psd_grid_files <- psd_grid_files[-c(id3)]
  
  
combined_psd_hourly <- dplyr::bind_rows(lapply(psd_hourly_files, function(f) {arrow::read_parquet(f)}))
combined_par_hourly <- dplyr::bind_rows(lapply(par_hourly_files, function(f) {arrow::read_parquet(f)}))
combined_grid <- dplyr::bind_rows(lapply(psd_grid_files, function(f) {arrow::read_parquet(f)}))


----
# PAR
#----

combined_par_hourly %>% ggplot() +
    geom_line(aes(date, par)) +
    theme_bw(base_size = 10) +
    facet_wrap(. ~ cruise, scale="free_x")

# Select bad PAR cruises 
bad_par_cruises <- c("HOT321",
             "MBARI_1","MBARI_2","MBARI_3", 
             "MESO_SCOPE",
             "Thompson_12", "Thompson_9",
             "Tokyo_3")

bad_par <- combined_par_hourly %>% filter(cruise %in%bad_par_cruises)
  
#-----------------
# missing location
#-----------------
bad_location <- combined_par_hourly %>% 
  filter(is.na(lat) | is.na(lon)) 
 

#--------------------
# Keep only good data
#--------------------
curated_psd_hourly <- anti_join(combined_psd_hourly, full_join(bad_par, bad_location))
curated_par_hourly <- anti_join(combined_par_hourly, full_join(bad_par, bad_location))
curated_grid <- anti_join(combined_grid, bad_par)

print(paste("total number of good cruises: ", length(unique(curated_psd_hourly$cruise))))



# Save plots
p <- curated_psd_hourly %>% mutate(Qc = curated_grid$Qc[curated_psd_hourly$Qc_coord]) %>%
  group_by(cruise, date) %>%
  mutate(norm = max(n)) %>%
  ggplot() + 
  geom_tile(aes(date, 1000*Qc, fill= n/norm), show.legend = FALSE) +
  theme_bw(base_size = 10) +
  viridis::scale_fill_viridis(discrete = FALSE, name='count') +
  scale_y_continuous(trans = 'log10') +
  #geom_line(data = par_all %>% filter(cruise %in% good_cruises), aes(date, par/20), col = "red3") +
  facet_wrap(. ~ cruise, scale="free_x")

png(paste0(in_dir,"/all-psd.png"), width = 4800, height = 3600, res = 300)
print(p)
dev.off()




# Save to MPM folder
arrow::write_parquet(curated_psd_hourly, paste0(in_dir,"/curated.psd-hourly.parquet"))
arrow::write_parquet(curated_par_hourly, paste0(in_dir,"/curated.par-hourly.parquet"))
arrow::write_parquet(curated_grid, paste0(in_dir,"/curated.psd-grid.parquet"))




#--------------
# Run the model
#--------------

# python fit_models.py model --desc curated_cruises  --stan-file m_pmb_sigprior_v2.stan --model-name m_pmb --output-dir 2022-04-01 --psd-file curated.psd-hourly.parquet --grid-file curated.psd-grid.parquet --par-file curated.par-hourly.parquet

# singularity run seaflow-model-pystan2_0.2.0.sif  # then in singularity type: ./parallel_run_model.sh 