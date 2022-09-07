import logging
logger = logging.getLogger(__name__)
logger.propagate = False  # prevent double logging after importing pystan
logger.setLevel(logging.INFO)
logging_ch = logging.StreamHandler()
logging_ch.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )
)
logger.addHandler(logging_ch)

import datetime
from collections import deque
from hashlib import md5
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import signal
import subprocess
import sys
import time

from boltons.fileutils import atomic_save
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import netCDF4 as nc4
import numpy as np
import pandas as pd
import click
import pystan



def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    arch = "{}-{}".format(platform.system(), platform.machine())
    if model_name is None:
        cache_fn = 'cached-model-{}-{}.pkl'.format(arch, code_hash)
    else:
        cache_fn = 'cached-{}-{}-{}.pkl'.format(model_name, arch, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code, model_name=model_name, **kwargs)
        with atomic_save(cache_fn, text_mode=False) as f:
            pickle.dump(sm, f)
        logger.info("Saved cached StanModel in %s", cache_fn)
    else:
        logger.info("Using cached StanModel in %s", cache_fn)
    return sm



# Is this process a child of the CLI parent?
# Must be a dict to allow setting from within a function
process_state = {
    "is_child": False,
}

# use test data (not all data is used for fitting/training)
use_testdata = False

# save the Stan output instead a few stats (only active if filename is specified above)
save_stan_output = True

save_only_converged = False

# specify the Stan variable names to save; if set to None, all variables are saved 
# (only active if save_stan_output is True)
varnames_save = None

# the number of tries to fit each Stan model to achieve an R-hat < 1.1
num_tries = 3

# the number of chains to run
num_chains = 6

# Number of days of data to fit
limit_days = 1

# the prior_only option passed to each Stan model
prior_only = False

# Whether or not data limit is inclusive (include boundary point)
inclusive = False

# Whether to append dataset to itself to create a 96-hour dataset
# This option is used for validation experiments in rolling_window.ipynb
extend = False

# Whether to append results to an existing file or overwrite
append = False

size_units = 'fg C cell$^{-1}$'

# ---------------------------------------------------------
# load processed data
datafile = 'ProMo_A_Control.nc'

dataname = 'Control'

desc = 'ProMo Culture dataset'

# Indices of data to hold out for hold-out validation
# Uncomment desired line and set use_testdata to true
itestfile = None
# itestfile = '../data/hold_out/keep_twothirds.csv'
# itestfile = '../data/hold_out/keep_half.csv'
# itestfile = '../data/hold_out/keep_onethird.csv'

size_units = 'fg C cell$^{-1}$'


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    pass


@cli.command('days')
@click.option('--dated-parquet-file', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Parquet data file with columns for cruise and date.')
@click.option('--no-partial-days', is_flag=True, default=False, show_default=True,
            help='Exclude partial days (< 24 hours).')
def cmd_days(dated_parquet_file, no_partial_days):
    """Print a table of cruise days in this dated parquet file"""
    cruise_days = get_cruise_days(pd.read_parquet(dated_parquet_file))
    all_days = len(cruise_days)
    if no_partial_days:
        # Remove incomplete days
        cruise_days = cruise_days.groupby('cruise').apply(lambda g: g[g['hours_in_day'] == 24])
        cruise_days = cruise_days.reset_index(drop=True)
        removed_days = all_days - len(cruise_days)
        if removed_days > 0:
            logger.info('removed %d incomplete days from consideration', removed_days)
    print(cruise_days.to_string(index=False))


@cli.command('compile')
@click.option('--stan-file', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Stan file to compile and cache')
@click.option('--model-name', required=True, type=str,
            help='Stan model name.')
def cmd_compile(stan_file, model_name):
    """Compile and save a local cache of a Stan code file"""
    # ---------------------------
    # Compile the Stan model code
    # Takes about 1 minute
    # ---------------------------
    logger.info("compiling Stan code file %s", stan_file)
    with open(stan_file) as f:
        code_split = f.read().split('\n')
        model_code = '\n'.join(code_split)

    # Compile Stan code or retrieve cached model
    _ = StanModel_cache(model_code=model_code, model_name=model_name,
                        obfuscate_model_name=False)
    logger.info("compilation complete")


def parse_days_option(ctx, param, value):
    if not value:
        return
    days = []
    for item in [v.strip() for v in value.split(',')]:
        try:
            days.append(int(item))
        except ValueError:
            # Let any exceptions bubble up to stop the application
            day1, day2 = [int(day) for day in item.split('-')]
            days.extend(range(day1, day2))
    return days


@cli.command('model')
@click.option('--desc', required=True, type=str,
            help='Description common to all data sets processed.')
@click.option('--stan-file', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
            help='Stan code file.')
@click.option('--model-name', required=True, type=str,
            help='Stan model name.')
@click.option('--output-dir', required=True, type=click.Path(exists=False, dir_okay=True, file_okay=False),
            help='Output directory.')
@click.option('--psd-file', required=True, type=click.Path(exists=True, dir_okay=False, file_okay=True),
            help='Size distribution counts Parquet file.')
@click.option('--grid-file', required=True, type=click.Path(exists=True, dir_okay=False, file_okay=True),
            help='Grid Parquet file.')
@click.option('--par-file', required=True, type=click.Path(exists=True, dir_okay=False, file_okay=True),
            help='PAR Parquet file.')
@click.option('--use-model-cache/--no-use-model-cache', default=True, show_default=True,
            help='Activate Stan model cache.')
@click.option('--cruise', type=str,
              help='Name of cruise to process. If not supplied all cruises will be processed.')
@click.option('--days', type=str, callback=parse_days_option,
              help="""Cruise days to process, as a comma-separated list.
                      Items in the list may specify right-open, or [a,b), ranges, e.g. 3-5 for 3,4.""")
@click.option('--no-partial-days', is_flag=True, default=False, show_default=True,
            help='Exclude partial days (< 24 hours).')
@click.option('--jobs', type=int, default=1,
              help=f'Number of cruise days to process at a time, each using {num_chains} threads.')
@click.option('--is-child', is_flag=True, default=False, show_default=True, hidden=True,
            help='This process was called by itself, i.e. is a child prcoess.')
def cmd_run_model(desc, stan_file, model_name, output_dir, psd_file, grid_file, par_file,
                  use_model_cache, cruise, days, no_partial_days, jobs, is_child):
    """Run a Stan model for all cruises in output_dir"""
    # Read the three data files and do some basic checks
    logger.info('reading data files')
    psd = pd.read_parquet(psd_file)
    par = pd.read_parquet(par_file)
    grid = pd.read_parquet(grid_file)
    if not np.array_equal(np.sort(psd['cruise'].unique()), np.sort(par['cruise'].unique())):
        raise click.ClickException('Mismatched cruise sets in psd-file and par-file')
    if not np.array_equal(np.sort(psd['cruise'].unique()), np.sort(grid['cruise'].unique())):
        raise click.ClickException('Mismatched cruise sets in psd-file and grid-file')
    if not np.array_equal(np.sort(psd['date'].unique()), np.sort(par['date'].unique())):
        raise click.ClickException('Mismatched date sets in psd-file and par-file')

    # Construct a plan for cruises and days to process
    logger.info('constructing processing plan')
    cruise_days = get_cruise_days(psd)

    plan = {}
    if cruise:
        # Select one cruise
        cruise_days = cruise_days[cruise_days['cruise'] == cruise]
        print(cruise_days.to_string(index=False))
        if no_partial_days:
            all_days = len(cruise_days)
            # Remove incomplete days
            cruise_days = cruise_days[cruise_days['hours_in_day'] == 24]
            removed_days = all_days - len(cruise_days)
            if removed_days > 0:
                logger.info('removed %d incomplete days from consideration', removed_days)
        if days:
            # Select valid days
            diff_days = np.setdiff1d(days, cruise_days['day'])
            if len(diff_days):
                raise click.ClickException('days out of range: {:s}'.format(str(list(diff_days))))
            plan[cruise] = days
        else:
            # All days for this cruise
            plan[cruise] = list(cruise_days['day'])
            if len(plan[cruise]) == 0:
                raise click.ClickException('no days found for cruise "{:s}"'.format(cruise))
    else:
        # All cruises and days
        print(cruise_days.to_string(index=False))
        if no_partial_days:
            all_days = len(cruise_days)
            # Remove incomplete days
            cruise_days = cruise_days.groupby('cruise').apply(lambda g: g[g['hours_in_day'] == 24])
            cruise_days = cruise_days.reset_index(drop=True)
            removed_days = all_days - len(cruise_days)
            if removed_days > 0:
                logger.info('removed %d incomplete days from consideration', removed_days)
        plan = {cruise: list(group['day']) for cruise, group in cruise_days.groupby("cruise")}

    print('plan as "cruise: [days ...]"')
    for cruise in sorted(plan.keys()):
        print('  {:s}: {:s}'.format(cruise, str(plan[cruise])))
    
    total_days_to_run = sum([len(v) for v in plan.values()])
    jobs = min(total_days_to_run, jobs)

    if is_child:
        process_state["is_child"] = True

    def exit_handler(signum, frame):
        if not process_state["is_child"]:
            print(f"received signal {signum}", file=sys.stderr)
            print("All currently processing days will continue running, but no additional days will start ", file=sys.stderr)
            print("after the current batch is done.", file=sys.stderr)
            print("This is a known problem with pystan2 (https://github.com/stan-dev/pystan2/issues/506)", file=sys.stderr)
        sys.exit()

    signal.signal(signal.SIGINT, exit_handler)

    if jobs == 1:
        for cruise in sorted(plan.keys()):
            # All output files will all go in a cruise subpath
            output_dir = os.path.join(output_dir, cruise)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for day in plan[cruise]:
                results = process_cruise_day(
                    psd_file, par_file, grid_file, cruise, day,
                    model_name, stan_file, desc, use_model_cache,
                    output_dir
                )
                logger.info(results)
    else:
        todo, running, done = deque(), {}, []
        for cruise in sorted(plan.keys()):
            # All output files will all go in a cruise subpath
            output_dir = os.path.join(output_dir, cruise)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for day in plan[cruise]:
                todo.append({
                    "name": f"{cruise}-{day}",
                    "args": [
                        sys.executable, sys.argv[0], "model",
                        "--desc", desc, "--stan-file", stan_file,
                        "--model-name", model_name, "--output-dir", output_dir,
                        "--psd-file", psd_file, "--par-file", par_file,
                        "--grid-file", grid_file,
                        "--cruise", cruise, "--days", str(day),
                        "--is-child"
                    ],
                    "popen": None,
                    "logpath": os.path.join(output_dir, f"{cruise}-{day}.log"),
                    "logfile": None,
                    "start": None,
                    "end": None
                })

        while True:
            if len(done) == total_days_to_run:
                break
            # Check for finished jobs
            just_finished = []
            for k, v in running.items():
                if not (v["popen"].poll() is None):
                    v["logfile"].close()
                    v["end"] = datetime.datetime.now()
                    done.append(v)
                    just_finished.append(k)
                    logger.info(f"{k} finished with status = {v['popen'].poll()} in {v['end'] - v['start']}")
            for k in just_finished:
                del running[k]
            # Start jobs in empty slots
            while len(running) < jobs:
                try:
                    next_job = todo.popleft()
                except IndexError:
                    # No more jobs to start
                    break
                next_job["logfile"] = open(next_job["logpath"], "w")
                next_job["start"] = datetime.datetime.now()
                next_job["popen"] = subprocess.Popen(
                    next_job["args"],
                    stdout=next_job["logfile"],
                    stderr=subprocess.STDOUT
                )
                running[next_job["name"]] = next_job
                logger.info(f"starting {next_job['name']}, logging to {next_job['logpath']}")
            time.sleep(1)
        logger.info("all jobs finished")


@cli.command('plot-cruise')
@click.option('--desc', required=True, type=str,
            help='Description common to all data sets processed.')
@click.option('--output-dir', required=True, type=click.Path(exists=False, dir_okay=True, file_okay=False),
            help='Output directory.')
@click.option('--psd-file', required=True, type=click.Path(exists=True, dir_okay=False, file_okay=True),
            help='Size distribution counts Parquet file.')
@click.option('--grid-file', required=True, type=click.Path(exists=True, dir_okay=False, file_okay=True),
            help='Grid Parquet file.')
@click.option('--par-file', required=True, type=click.Path(exists=True, dir_okay=False, file_okay=True),
            help='PAR Parquet file.')
@click.option('--cruise', type=str,
              help='Name of cruise to process. If not supplied all cruises will be processed.')
def cmd_plot_cruise(desc, output_dir, psd_file, grid_file, par_file, cruise):
    """Run a Stan model for all cruises in output_dir"""
    # Read the three data files and do some basic checks
    logger.info('reading data files')
    psd = pd.read_parquet(psd_file)
    par = pd.read_parquet(par_file)
    grid = pd.read_parquet(grid_file)
    if not np.array_equal(np.sort(psd['cruise'].unique()), np.sort(par['cruise'].unique())):
        raise click.ClickException('Mismatched cruise sets in psd-file and par-file')
    if not np.array_equal(np.sort(psd['cruise'].unique()), np.sort(grid['cruise'].unique())):
        raise click.ClickException('Mismatched cruise sets in psd-file and grid-file')
    if not np.array_equal(np.sort(psd['date'].unique()), np.sort(par['date'].unique())):
        raise click.ClickException('Mismatched date sets in psd-file and par-file')

    # Construct a plan for cruises and days to process
    logger.info('selecting cruises to plot')
    cruises = list(pd.read_parquet(psd_file)['cruise'].unique())
    plan = []
    if cruise:
        if cruise not in cruises:
            raise click.ClickException('invalid cruise "{:s}"'.format(cruise))
        plan = [cruise]
    else:
        # All cruises
        plan = sorted(cruises)

    print('plotting cruises {:s}'.format(str(plan)))

    for cruise in plan:
        plot_cruise(psd_file, par_file, grid_file, cruise, desc, output_dir)


def process_cruise_day(psd_file, par_file, grid_file, cruise, day, model_name,
                       stan_file, desc, use_model_cache, output_dir):
    # ---------------------------------------
    # Compile or retrieve the Stan model code
    # Takes about 1 minute
    # ---------------------------------------
    logger.info('compiling Stan code file %s', stan_file)
    with open(stan_file) as f:
        code_split = f.read().split('\n')
        model_code = '\n'.join(code_split)
    if use_model_cache:
        # Compile Stan code or retrieve cached model
        model = StanModel_cache(model_code=model_code, model_name=model_name,
                                obfuscate_model_name=False)
    else:
        model = pystan.StanModel(model_code=model_code, model_name=model_name,
                                 obfuscate_model_name=False)

     # Get dates for this day
    cruise_days = get_cruise_days(pd.read_parquet(psd_file))
    day_row = cruise_days[(cruise_days['cruise'] == cruise) & (cruise_days['day'] == day)]
    assert len(day_row) == 1
    start_date_str = day_row.iloc[0, day_row.columns.get_loc('start')].isoformat(timespec='seconds')
    end_date_str = day_row.iloc[0, day_row.columns.get_loc('end')].isoformat(timespec='seconds')
    hours_in_day = day_row.iloc[0, day_row.columns.get_loc('hours_in_day')]

    logger.info('starting model run cruise %s, day %d, %d hours, %s - %s',
                cruise, day, hours_in_day, start_date_str, end_date_str)
    # ------------------------------
    # Get raw data for a full cruise
    # ------------------------------
    # Don't select day here, that happens later
    data_gridded, desc = get_data_parquet(psd_file, par_file, grid_file, desc,
                                          cruise=cruise)
    desc = 'day={:02d}, {:s}'.format(day, desc)

    # -----------------------------
    # Define output files path base
    # -----------------------------
   

    fname = '{:s}_day{:02d}_{:02d}hours_{:s}_{:s}'.format(
        cruise, day, hours_in_day, start_date_str, end_date_str
    )
    outfile_base = os.path.join(output_dir, fname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # netCDF output file
    savename_output = '{:s}.nc'.format(outfile_base)

    # start of the time series (in days)
    data = data_prep(data_gridded, dt=15, limit_days=limit_days, start=day*24,
                     prior_only=prior_only, inclusive=False)

    # Plot one day of processed data
    # ------------------------------
    # Processed day data figure output name
    processed_figure_output = '{:s}.processed.png'.format(outfile_base)

    nrows = 3

    v_min = data['v_min']
    delta_v = 1.0/data['delta_v_inv']
    v = v_min * 2**(np.arange(data['m'])*delta_v) 
    t = np.arange(data['nt'])*data['dt'] 

    fig,axs = plt.subplots(nrows=nrows, sharex=True, figsize=(12,4*nrows))
    fig.set_facecolor('white')  # to avoid transparent background when saving to file

    ax = axs[0]
    ax.set_title('processed '+desc, size=20)
    ax.plot(t, data['E'], color='gold')
    ax.set(ylabel='E')

    ax = axs[1]
    pc = ax.pcolormesh(data['t_obs'], v, data['obs'], shading='auto')
    ax.set(ylabel='size ({})'.format(size_units))
    add_colorbar(ax, norm=pc.norm, cmap=pc.cmap,
                label='size class proportion')
    ax.set_xlim(left=0.0)

    ax = axs[2]
    pc = ax.pcolormesh(data['t_obs'], v, data['obs_count'], shading='auto')
    ax.set(ylabel='size ({})'.format(size_units))
    add_colorbar(ax, norm=pc.norm, cmap=pc.cmap, label='counts')
    ax.set_xlim(left=0.0)
    axs[-1].set_xlabel('time (minutes)')

    fig.savefig(processed_figure_output)
    plt.close(fig)

    # Run the model on one full day of processed data
    # Save results
    # -----------------------------------------------
    try:
        run_model(model, model_name, stan_file, data, savename_output)
        status = "sucess"
    except Exception as e:
        logger.warning('model run for cruise %s, day %d failed: %s', cruise, day, e)
        status = f"error: {e}"

    return f"{cruise} {day} => {status}"


def plot_cruise(psd_file, par_file, grid_file, cruise, desc, output_dir):
    logger.info('plotting full cruise %s', cruise)
    # ------------------------------
    # Get raw data for a full cruise
    # ------------------------------
    data_gridded, desc = get_data_parquet(psd_file, par_file, grid_file, desc, cruise=cruise)

    # -----------------------------
    # Define output files path base
    # -----------------------------
    outfile_base = os.path.join(output_dir, cruise)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Plot raw data for full cruise
    # -----------------------------
    raw_figure_output = '{:s}_raw_data.png'.format(outfile_base)

    nrows = 3

    v_min = data_gridded['v_min']
    delta_v = 1.0/data_gridded['delta_v_inv']
    v = v_min * 2**(np.arange(data_gridded['m'])*delta_v)

    fig,axs = plt.subplots(nrows=nrows, sharex=True, figsize=(12,4*nrows))
    fig.set_facecolor('white')  # to avoid transparent background when saving to file

    ax = axs[0]
    ax.set_title('raw '+desc, size=20)
    ax.plot(data_gridded['time'], data_gridded['PAR'], color='gold')
    ax.set(ylabel='PAR')

    ax = axs[1]
    pc = ax.pcolormesh(data_gridded['time'], v, data_gridded['w_obs'],
                        shading='auto')
    ax.set(ylabel='size ({})'.format(size_units))
    add_colorbar(ax, norm=pc.norm, cmap=pc.cmap, label='size class proportion')

    ax = axs[2]
    pc = ax.pcolormesh(data_gridded['time'], v, data_gridded['counts'],
                        shading='auto')
    ax.set(ylabel='size ({})'.format(size_units))
    add_colorbar(ax, norm=pc.norm, cmap=pc.cmap, label='counts')
    axs[-1].set_xlabel=('time (minutes)')

    fig.savefig(raw_figure_output)
    plt.close(fig)


# Get Parquet data
def get_data_parquet(psd_file, par_file, grid_file, desc, cruise=None, day=None,
                     coord_col='Qc_coord'):
    grid_col = coord_col.split('_')[0]  # e.g. Qc from Qc_coord

    data_gridded = {}

    psd = pd.read_parquet(psd_file)
    par = pd.read_parquet(par_file)
    grid = pd.read_parquet(grid_file)

    logger.info('md5(psd["%s"]) = %s', coord_col, md5(psd[coord_col].values.tobytes()).hexdigest())
    logger.info('md5(par["par"]) = %s', md5(par['par'].values.tobytes()).hexdigest())
    logger.info('md5(grid["%s"]) = %s', grid_col, md5(grid[grid_col].values.tobytes()).hexdigest())

    # Select one cruise
    if cruise is not None:
        logger.info('selecting cruise == %s', cruise)
        psd = psd[psd['cruise'] == cruise].reset_index(drop=True)
        par = par[par['cruise'] == cruise].reset_index(drop=True)
        grid = grid[grid['cruise'] == cruise].reset_index(drop=True)
        if (len(psd) == 0 or len(par) == 0 or len(grid) == 0):
            raise Exception("incomplete data after selecting for cruise")

    logger.info('md5(psd["%s"]) = %s', coord_col, md5(psd[coord_col].values.tobytes()).hexdigest())
    logger.info('md5(par["par"]) = %s', md5(par['par'].values.tobytes()).hexdigest())
    logger.info('md5(grid["%s"]) = %s', grid_col, md5(grid[grid_col].values.tobytes()).hexdigest())

    # Collect grid information
    # All diffs should be about equal, take the first non-NA one
    delta = np.log2(grid[grid_col]).diff()[1]
    # Get inverse of single delta value
    data_gridded['delta_v_inv'] =  round(1 / delta)  # should be v close to an int to begin with
    logger.info('delta_v_inv = %f', data_gridded['delta_v_inv'])
    # Left edge of smallest bin
    data_gridded['v_min'] = grid.loc[0, grid_col]
    logger.info('v_min = %f', data_gridded['v_min'])
    # Left grid boundaries
    data_gridded['size'] = grid[grid_col][0:-1].values
    # Fence-post grid boundaries
    data_gridded['size_bounds'] = grid[grid_col].values

    # Number of bins
    data_gridded['m'] = len(grid) - 1
    logger.info("m = %d", data_gridded['m'])

    # Remove rows for data outside grid range
    psd = psd[pd.notna(psd[coord_col])].reset_index(drop=True)
    # Because NA in numpy is a float, coords may be autoconverted to floats
    # Change them back to ints
    psd[coord_col] = psd[coord_col].astype(int)
    # Make sure data is sorted by date
    psd = psd.sort_values(['date', coord_col])
    par = par.sort_values(['date'])
    # Get the timedelta for each row from the earliest time point
    psd['delta'] = psd['date'] - psd.loc[0, 'date']
    par['delta'] = par['date'] - par.loc[0, 'date']
    # Express timedelta in terms of days, starting with 0 for first day
    psd['day'] = psd['delta'].map(lambda d: d.days)
    par['day'] = par['delta'].map(lambda d: d.days)

    # Select one day
    # This assumes the dates for psd and par are the same
    if day is not None:
        logger.info('selecting day = %d', day)
        psd = psd[psd['day'] == day].reset_index(drop=True)
        par = par[par['day'] == day].reset_index(drop=True)
        if (len(psd) == 0 or len(par) == 0 or len(grid) == 0):
            raise Exception("incomplete data after selecting for day")

    data_gridded['PAR'] = par['par'].values

    # Get times
    psd_by_date = psd.groupby('date')
    dates = pd.Series([k for k, _ in psd_by_date])
    # Time since start of data in minutes
    data_gridded['time'] = np.array([tdelta.total_seconds() / 60.0 for tdelta in (dates - dates[0])])

    # Get counts and relative counts. Expand sparse data to include zero counts
    count_shape = (data_gridded['m'], len(dates))
    data_gridded['counts'] = np.zeros(count_shape, dtype=int)
    data_gridded['w_obs'] = np.zeros(count_shape, dtype=float)
    data_gridded['count'] = np.zeros(len(dates))
    time_i = 0
    for _, group in psd_by_date:
        data_gridded['counts'][group[coord_col] - 1, time_i] = group['n']   # coord_col starts at 1, not 0
        data_gridded['count'][time_i] = group['n'].sum()  # particles at time i
        data_gridded['w_obs'][:, time_i] = data_gridded['counts'][:, time_i] / data_gridded['count'][time_i]
        time_i += 1

    # add description
    desc += ' (cruise={cruise}, m={data[m]}, $\Delta_v^{{-1}}$={data[delta_v_inv]})'.format(
        cruise=cruise, data=data_gridded
    )

    logger.info("md5(data_gridded['counts']) = %s", md5(data_gridded['counts'].tobytes()).hexdigest())
    logger.info("md5(data_gridded['w_obs']) = %s", md5(data_gridded['w_obs'].tobytes()).hexdigest())

    return data_gridded, desc


def get_cruise_days(df):
    """Get a dataframe of cruise days from df
    
    df is a dataframe with columns of 'cruise' (string) and 'date' (datetime)
    """
    df = df.copy()
    cruise_days = {'cruise': [], 'day': [], 'start': [], 'end': [], 'hours_in_day': []}
    for cruise, g1 in df.groupby('cruise'):
        g1 = g1.copy()
        deltas = g1['date'] - g1.iloc[0, g1.columns.get_loc('date')]
        g1['day'] = deltas.map(lambda d: d.days)
        for day, g2 in g1.groupby('day'):
            cruise_days['cruise'].append(cruise)
            cruise_days['day'].append(day)
            cruise_days['start'].append(g2['date'].min())
            cruise_days['end'].append(g2['date'].max())
            cruise_days['hours_in_day'].append(g2['date'].unique().size)
    days = pd.DataFrame(cruise_days)
    days.insert(0, 'cumulative_day', range(len(days)))
    return days


# Prepare data for Stan model
# Can be data from NetCDF (get_data_nc()) or Parquet files (get_data_parquet())
# start is in hours
# limit_days is how many of days of data after "start" to collect
# inclusive is whether to include the final minute (closed right or open right)
def data_prep(data_gridded, dt=15, limit_days=1, start=0, prior_only=False, inclusive=False):
    data = {'dt':dt}
    for v in ('m','v_min','delta_v_inv'):
        data[v] = data_gridded[v]

    data['obs'] = data_gridded['w_obs']
    data['t_obs'] = data_gridded['time']
    par = data_gridded['PAR']

    if limit_days > 0:
        limit_minutes = limit_days*1440

        # start is in hours
        if inclusive:
            ind_obs = (start*60 <= data['t_obs']) & (data['t_obs'] <= limit_minutes+start*60)
        else:
            ind_obs = (start*60 <= data['t_obs']) & (data['t_obs'] < limit_minutes+start*60)

        if not np.all(ind_obs):
            total = data['obs'].shape[1]
            remove = total - data['obs'][:, ind_obs].shape[1]
            print('start is set to {}, limit_days is set to {}, removing {}/{} observation times'.format(start,
                                                                                                         limit_days,
                                                                                                         remove,
                                                                                                         total))

        data['t_obs'] = data['t_obs'][ind_obs] - start*60
        data['obs'] = data['obs'][:,ind_obs]

        data['nt'] = int(limit_minutes//data['dt']+1)

    data['nt_obs'] = data['t_obs'].size

    # set all indices to zero
    data['i_test'] = np.zeros(data['nt_obs'], dtype=int)

    # switch on or off data fitting
    data['prior_only'] = int(prior_only)

    # add light data
    t = np.arange(data['nt'])*data['dt'] + start*60
    data['E'] = np.interp(t, xp=data_gridded['time'][ind_obs], fp=par[ind_obs])
    #data['E'] = np.append(np.repeat(data_gridded['PAR'], 6), 150)

    # real count data
    data['obs_count'] = data_gridded['counts'][:, ind_obs]
    
    data['start'] = start

    # consistency check
    if len(data['i_test']) != data['nt_obs']:
        raise ValueError('Invalid number of testing indices (expected {}, got {}).'.format(data['nt_obs'],
                                                                                       len(data['i_test'])))
    return data


def add_colorbar(ax, **cbarargs):
    axins_cbar = inset_axes(ax, width='3%', height='90%', loc=5,
                            bbox_to_anchor=(0.05,0.0,1,1),
                            bbox_transform=ax.transAxes)
    mpl.colorbar.ColorbarBase(axins_cbar, orientation='vertical',
                              **cbarargs)


def get_max_rhat(fit):
    s = fit.summary()
    irhat = s['summary_colnames'].index("Rhat")
    return np.nanmax(s['summary'][:,irhat])


def run_model(model, model_name, stan_file, data, savename_output):
    # run a bunch of experiments -- this may take a while
    for itry in range(num_tries):
        t0 = time.time()
        mcmcs = model.sampling(data=data, iter=2000, chains=num_chains)
        sampling_time = time.time() - t0  # in seconds
        print('Model {} for {}-hour window starting at {} hours fit in {} minutes.'.format(model_name,
                                                                                        limit_days*24+2*int(inclusive),
                                                                                        data['start'],
                                                                                        np.round(sampling_time/60, 2)))
        # get max Rhat
        rhat_max = get_max_rhat(mcmcs)
        print('{}: in try {}/{} found Rhat={:.3f}'.format(model_name, itry+1, num_tries, rhat_max), end='')
        if rhat_max < 1.1 or itry == num_tries - 1:
            print()
            break
        print(', trying again')

    print('{}'.format(model_name)) 
    print('\n'.join(x for x in mcmcs.__str__().split('\n') if '[' not in x))
    print()

    with nc4.Dataset(savename_output, 'w') as nc:
        ncm = nc.createGroup(model_name)

        # write model description
        ncm.setncattr('code', stan_file)

        if save_stan_output:
            if save_only_converged and get_max_rhat(mcmcs) > 1.1:
                raise Exception('Model "{}" did not converge -- skipping.'.format(model_name))

            dimensions = {
                'obstime':int(data['nt_obs']),
                'time':int(data['nt']),
                'sizeclass':int(data['m']),
                'm_minus_j_plus_1':int(data['m']-data['delta_v_inv']),
                'm_minus_1':int(data['m']-1),
                'knots_minus_1':int(6-1),  # hardcoded, adjust for varying nknots
                'sample': mcmcs['mod_obspos'].shape[0],
                'rhat_max': 1
            }
            dimensions_inv = {v:k for k,v in dimensions.items()}

            for d in dimensions:
                if d not in ncm.dimensions:
                    ncm.createDimension(d, dimensions[d])
            
            if 'rhat_max' not in ncm.variables:
                ncm.createVariable('rhat_max', float, ('rhat_max',))
            ncm.variables['rhat_max'][:] = get_max_rhat(mcmcs)

            if 'tau[1]' in mcmcs.flatnames:
                dimensions['tau'] = mcmcs['tau'].shape[1]
                dimensions_inv[dimensions['tau']] = 'tau'
                if 'tau' not in ncm.dimensions:
                    ncm.createDimension('tau', dimensions['tau'])

            if 'time' not in ncm.variables:
                ncm.createVariable('time', int, ('time',))
            ncm.variables['time'][:] = int(data['dt']) * np.arange(data['nt'])
            ncm.variables['time'].units = 'minutes since start of experiment'

            if 'obstime' not in ncm.variables:
                ncm.createVariable('obstime', int, ('obstime',))
            ncm.variables['obstime'][:] = data['t_obs'].astype(int)
            ncm.variables['obstime'].units = 'minutes since start of experiment'
            ncm.variables['obstime'].long_name = 'time of observations'

            for v in ('dt', 'm', 'v_min', 'delta_v_inv', 'obs', 'i_test',
                        'E', 'obs_count'):
                if isinstance(data[v], int):
                    if v not in ncm.variables:
                        ncm.createVariable(v, int, zlib=True)
                    ncm.variables[v][:] = data[v]
                elif isinstance(data[v], float):
                    if v not in ncm.variables:
                        ncm.createVariable(v, float, zlib=True)
                    ncm.variables[v][:] = data[v]
                else:
                    dims = tuple(dimensions_inv[d] for d in data[v].shape)
                    if v not in ncm.variables:
                        ncm.createVariable(v, data[v].dtype, dims, zlib=True)
                    ncm.variables[v][:] = data[v]


            varnames = set(v.split('[')[0] for v in mcmcs.flatnames)
            if varnames_save is None:
                varnames_curr = varnames
            else:
                varnames_curr = varnames_save

            for v in varnames_curr:
                if v in varnames:
                    dims = tuple(dimensions_inv[d]
                                    for d in mcmcs[v].shape)
                    if v not in ncm.variables:
                        ncm.createVariable(v, float, dims, zlib=True)
                    ncm.variables[v][:] = mcmcs[v]
                else:
                    logger.warning('Cannot find variable "{}" for model "{}".'.format(v,
                                                                                    model_name))
        else:
            if 'sample' not in ncm.dimensions:
                ncm.createDimension('sample',
                                    mcmcs['divrate'].shape[0])

            if 'divrate' not in ncm.variables:
                ncm.createVariable('divrate', float, ('sample'))

            if 'sumsqdiff' not in ncm.variables:
                ncm.createVariable('sumsqdiff', float, ('sample'))

            ncm.variables['sumsqdiff'].setncattr('long_name',
                                                'sum of squared column differences')

            ncm.variables['divrate'][:] = mcmcs['divrate']

            obs = data['obs']

            tmp = mcmcs['mod_obspos']
            tmp /= np.sum(tmp, axis=1)[:, None, :]
            tmp -= obs[None, :, :]
            tmp **= 2

            if np.all(data['i_test'] == 0):
                ncm.variables['sumsqdiff'][:] = np.mean(np.sum(tmp, axis=1),
                                                        axis=1)
                ncm.variables['sumsqdiff'].setncattr('data_used',
                                                        'all data')
            else:
                ncm.variables['sumsqdiff'][:] = np.mean(np.sum(tmp[:, :, data['i_test'] == 1],
                                                            axis=1), axis=1)
                ncm.variables['sumsqdiff'].setncattr('data_used', 'testing data')

            for iv,v in enumerate(('gamma_max', 'rho_max', 'xi',
                                'xir', 'E_star')):
                if v not in ncm.variables:
                    ncm.createVariable(v, float, ('model','sample'))
                if v in mcmcs.flatnames:
                    ncm.variables[v][:] = mcmcs[v]


if __name__ == '__main__':
    # Do this to solve module not found error when creating subprocesses
    # during model run
    # https://discourse.mc-stan.org/t/new-to-pystan-always-get-this-error-when-attempting-to-sample-modulenotfounderror-no-module-named-stanfit4anon-model/19288/7
    mp.set_start_method("fork")

    cli()