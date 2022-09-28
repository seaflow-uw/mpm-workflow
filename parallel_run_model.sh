#!/bin/bash
pjobs=10
stanfile=m_pmb_sigprior_v2.stan
modelname=m_pmb
psdfile=curated.psd-hourly.parquet
gridfile=curated.psd-grid.parquet
parfile=curated.par-hourly.parquet
desc=Pro
outdirbase=output
logdir=logs

[[ -d "$logdir" ]] || mkdir -p "$logdir"

# Compile and cache the model once to be used by all future model runs
python3 fit_models.py compile --stan-file "$stanfile" --model-name m_pmb 2>&1 | tee $logdir/compile.log

# Create cruise-wide plots of input data for every cruise
python3 fit_models.py plot-cruise \
  --desc "$desc" \
  --output-dir cruise-plots \
  --psd-file "$psdfile" \
  --grid-file "$gridfile" \
  --par-file "$parfile" 2>&1 | tee $logdir/cruise-plots.log

# Run the model for all days
python3 fit_models.py model \
        --psd-file "$psdfile" \
        --grid-file "$gridfile" \
        --par-file "$parfile" \
        --stan-file "$stanfile" --model-name "$modelname" \
        --desc "$desc" \
        --jobs "$pjobs" \
        --output-dir ${outdirbase}-{1} 2>&1 | tee $logdir/model-run.log