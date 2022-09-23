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
# --colsep " " splits two space-separated arguments on single line into {1} and {2}
# e.g. if a line has 'A B' then {1} == A and {2} == B
# in this case this would be cruise and day
# to select based on when number of hours in a day, replace line 32 by" | awk 'NR > 1 && $8 > 12 {print $2, $3, $8}' \
# to run model on a single cruise, add below line 32: grep 'CRUISENAME' \
python3 fit_models.py days --dated-parquet-file "$psdfile" \
  | awk 'NR > 1 {print $2, $3, $8}' \
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
