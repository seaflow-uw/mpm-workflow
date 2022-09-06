library(popcycle)
library(tidyverse)
library(plotly)

#--------------------------------------------------
# Create a 3D surface of particle size distribution
#--------------------------------------------------

plot_psd <- function(data = psd, z = "proportion"){

    data_wide <- data %>% pivot_wider(id_cols = date, names_from = Qc, values_from = z)

    cruise <- unique(data$cruise)
    quotas <- unique(data$Qc) 
    time <- unique(data$date)
    value <- as.matrix(data_wide[,-c(1)])

    p <- plot_ly(x=quotas, y=time, z=value) %>% 
            add_surface() %>%
            layout(title = list(text = paste(cruise,"cruise"), y = 0.9, font = list(color = "red")),
                    scene = list(xaxis = list(title = "carbon quotas", autorange = "reversed", type="log"),
                    yaxis = list(title = "", autorange = "reversed"),
                    zaxis = list(title = z)))
    return(p)
}




#---------
# READ PSD
#---------
setwd("~/Documents/DATA/MPM")

in_dir <- "2022-09-06"

# COMBINE all PSD (+ par)
par_full_files <- list.files(in_dir, pattern = "*\\.par-full\\.parquet", full.names = TRUE, recursive = TRUE)
par_hourly_files <- list.files(in_dir, pattern = "*\\.par-hourly\\.parquet", full.names = TRUE, recursive = TRUE)
psd_full_files <- list.files(in_dir, pattern = "*\\.psd-full\\.parquet", full.names = TRUE, recursive = TRUE)
psd_hourly_files <- list.files(in_dir, pattern = "*\\.psd-hourly\\.parquet", full.names = TRUE, recursive = TRUE)
psd_grid_files <- list.files(in_dir, pattern = "*\\.psd-grid\\.parquet", full.names = TRUE, recursive = TRUE)

combined_par_hourly <- dplyr::bind_rows(lapply(par_hourly_files, function(f) {
  arrow::read_parquet(f)
}))
combined_psd_full <- dplyr::bind_rows(lapply(psd_full_files, function(f) {
  arrow::read_parquet(f)
}))
combined_psd_hourly <- dplyr::bind_rows(lapply(psd_hourly_files, function(f) {
  arrow::read_parquet(f)
}))
combined_grid <- dplyr::bind_rows(lapply(psd_grid_files, function(f) {
  arrow::read_parquet(f)
}))


#----
# PSD
#----
# Replace grid coordinates by Qc
psd_all <- combined_psd_hourly %>% mutate(Qc = combined_grid$Qc[combined_psd_hourly$Qc_coord]) %>%
                               group_by(cruise, date) %>%
                               mutate(norm = max(n))

psd_all %>% ggplot() + 
    geom_tile(aes(date, 1000*Qc, fill= n/norm)) +
    theme_bw(base_size = 10) +
    viridis::scale_fill_viridis(discrete = FALSE, name='count') +
    scale_y_continuous(trans = 'log10') +
    facet_wrap(. ~ cruise, scale="free_x")


# Select bad PSD cruises 
bad_psd <- c()


#----
# PAR
#----
# filter Bad PAR cruises
par_all <- combined_par_hourly

par_all %>% ggplot() +
    geom_line(aes(date, par)) +
    theme_bw(base_size = 10) +
    facet_wrap(. ~ cruise, scale="free_x")

# Select bad PAR cruises 
bad_par <- c("HOT321",
             "MBARI_1","MBARI_2","MBARI_3", 
             "MESO_SCOPE",
             "Thompson_12", "Thompson_9",
             "Tokyo_3")

# Combine bad cruises
all_cruises <- unique(psd_all$cruise)
bad_cruises <- c(bad_psd, bad_par)
out <- match(bad_cruises,all_cruises, nomatch = 0)
good_cruises <- all_cruises[-c(out)]

print(paste("total number of curated cruises: ", length(good_cruises)))


# Save to MPM folder
arrow::write_parquet(combined_psd_full %>% filter(cruise %in% good_cruises), paste0(in_dir, "/curated.psd-full.parquet"))
arrow::write_parquet(combined_psd_hourly %>% filter(cruise %in% good_cruises), paste0(in_dir,"/curated.psd-hourly.parquet"))
arrow::write_parquet(combined_grid %>% filter(cruise %in% good_cruises), paste0(in_dir,"/curated.psd-grid.parquet"))
arrow::write_parquet(combined_par_hourly%>% filter(cruise %in% good_cruises), paste0(in_dir,"/curated.par-hourly.parquet"))



# Save plots

p <- psd_all %>% filter(cruise %in% good_cruises) %>%
    ggplot() + 
    geom_tile(aes(date, 1000*Qc, fill= n/norm)) +
    theme_bw(base_size = 10) +
    viridis::scale_fill_viridis(discrete = FALSE, name='count') +
    scale_y_continuous(trans = 'log10') +
    facet_wrap(. ~ cruise, scale="free_x")

png(paste0(in_dir,"/all-psd.png"), width = 4800, height = 3600, res = 300)
print(p)
dev.off()



#-------------
# PLOTTING PSD
#-------------

## Select cruise
psd_curated <- psd_all %>% filter(cruise %in% good_cruises)
cruises <- unique(psd_curated$cruise)
print(cruises)

c <- cruises[33]

psd <- psd_curated %>% filter(cruise == c) %>% mutate(Qc = Qc * 1000,
                                                     proportion = n / norm)

# plot PSD
p <- plot_psd(psd, z = "proportion")

# save PSD
htmlwidgets::saveWidget(p, file = paste0(c, "-psd.html"))




#--------------
# Run the model
#--------------

# python fit_models.py model --desc curated_cruises  --stan-file m_pmb_sigprior_v2.stan --model-name m_pmb --output-dir 2022-04-01 --psd-file curated.psd-hourly.parquet --grid-file curated.psd-grid.parquet --par-file curated.par-hourly.parquet

# singularity run seaflow-model-pystan2_0.1.0.sif  # then in singularity type: ./parallel_run_model.sh 