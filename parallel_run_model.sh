#!/bin/bash
pjobs=10
stanfile=m_pmb_sigprior_v2.stan
modelname=m_pmb
psdfile=combined.psd-hourly.parquet
gridfile=combined.psd-grid.parquet
parfile=combined.par-hourly.parquet
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
# --colsep " " splits two space-separated arguments on single line into {1} and {2}
# e.g. if a line has 'A B' then {1} == A and {2} == B
# in this case this would be cruise and day
python3 fit_models.py days --dated-parquet-file "$psdfile" --no-partial-days \
  | awk 'NR > 1 {print $2, $3}' \
  | parallel -P "$pjobs" --colsep " " \
      sh -c "echo {1} {2}; python3 fit_models.py model \
        --psd-file "$psdfile" \
        --grid-file "$gridfile" \
        --par-file "$parfile" \
        --stan-file "$stanfile" --model-name "$modelname" \
        --desc "$desc" \
        --cruise {1} \
        --days {2} \
        --output-dir ${outdirbase}-{1} 2>&1 | tee $logdir/{1}-{2}.log"
