# SeaFlow division rate growth model workflow repository

The basic workflow is to generate SeaFlow particle size distribution and PAR data using the R notebook `create_psd.Rmd`.
Then create plots and run the model using the Python script `fit_models.py`.
To run model days in parallel use the Bash script `parallel_run_model.sh`.

## Install dependencies

### R package popcycle

Install the R package [popcycle](https://github.com/seaflow-uw/popcycle).

### Python package PyStan

This workflow has been tested with Python 3.8 and PyStan version 2.19.1.1.
Either 1) install PyStan into a Python virtual environment or 2) build/use a Docker image.

#### Python virtual environment for PyStan

Create a Python virtual environment using whichever tool you'd like and install pystan2 requirements.
Here we'll use pyenv, but `python -m venv /envlocation`, pipenv, poetry, etc would also work.

```bash
pyenv virtualenv seaflow-model-pystan2
pyenv activate seaflow-model-pystan2
pip3 install --upgrade pip setuptools
# If starting fresh with pinned pystan verison and regenerating requirements.txt
# pip3 install matplotlib netCDF4 pandas pystan==2.19.1.1 jupyter pyarrow click
# pip3 freeze > requirements-pystan2.txt

# Install from existing requirements.txt
pip3 install -r requirements-pystan2.txt
```

#### Docker build PyStan2

To build a Docker image tagged as `seaflow-model-pystan2` with pystan2 installed

```bash
docker build -t seaflow-model-pystan2 --build-arg REQUIREMENTS_FILE=requirements-pystan2.txt -f Dockerfile .
```

You may want to tag this image with your Docker Hub account and upload to Docker Hub.

```bash
docker tag seaflow-model-pystan2 myaccount/seaflow-model-pystan2
docker push myaccount/seaflow-model-pystan2
```

## Run

### Generate model input data

Customize and run the R markdown notebook `create_psd.Rmd`.
This notebook uses `renv` to manage R package dependencies.
If you don't use `renv` then comment out the invocation near the top of the file.

### Run the model

The Stan model and data plots can be run with `fit_models.py`.
For command-line help run `fit_models.py --help`.
There are four subcommands: `days`, `compile`, `model`, and `plot-cruise`.

`days` will print a table of cruise days found in either the PSD or PAR Parquet files.

`compile` will compile the Stan model and cache it in the current working directory.
This can save up to a minute per model run.
It isn't strictly necessary to run this step separately as the default is to cache the model everytime the model is run.

`model` will run the model on some or all of the data and at the same time create a plot of input data for every day processed.

`plot-cruise` will create a plot of input data for all days in a cruise.

#### Docker run examples

##### Run `fit_models.py` in a Docker container for a single day

```bash
docker run -it --rm -v $(pwd):/data -w /data seaflow-model-pystan2:latest \
  sh -c 'python3 fit_models.py model \
    --psd-file combined.psd-hourly.parquet \
    --grid-file combined.psd-grid.parquet \
    --par-file combined.par-hourly.parquet \
    --stan-file m_pmb_sigprior_v2.stan --model-name m_pmb \
    --desc Pro \
    --cruise HOT303 \
    --days 2 \
    --output-dir out 2>&1 | tee model-run-HOT303-day2.log'
```

##### Run all cruise/day combinations in parallel using GNU parallel

Modify and run `parallel_run_model.sh` in a Docker container.
GNU parallel is already installed in the Docker container.

```bash
docker run -it --rm -v $(pwd):/data -w /data seaflow-model-pystan2:latest \
  ./parallel_run_model.sh
```

## Run in EC2

To run models in AWS EC2, use [terraform](https://www.terraform.io/) to create an EC2 instance,
then use [ansible](https://www.ansible.com/) to provision this machine,
and finally upload files and log in with SSH to run the model.

### Create EC2 instance

First install the command-line tool `terraform`.
Then optionally modify `variables.tf` in `terraform/ondemand` to change the type of EC2 instance.
Then run terraform.

```bash
cd terraform/ondemand
terraform init
terraform apply
cd ../..
```

This command will start an EC2 instance and print its public IP address.
Take this address and create an SSH host entry in `~/.ssh/config` that will be used by `ansible` to provision the EC2 instance.
In this example the host is named `psd`.

```
Host psd
    Hostname 34.209.151.135
    IdentityFile ~/.ssh/aws.pem
    User ubuntu
```

Log in to the instance with `ssh psd`

### Provision EC2 instance

Install `ansible` and make sure it's accessible in your current environment.
Then modify `ansible/inventories/psd.yml` to change the name of the Docker image containing pystan2 to match the Docker Hub location you previously pushed to.
Finally provision the instance with `ansible`. This installs Docker and pulls your pystan2 image.

```bash
ansible-playbook -i ansible/inventories/psd.yml ansible/playbook.yml
```

### Run the model

Upload files data and code files to the EC2 instance and run the model with Docker.

```bash
scp -r combined.*.parquet m_pmb_sigprior_v2.stan fit_models.py parallel_run_model.sh psd:
```

Then SSH into the `psd` EC2 instance and run `fit_models.py` following the Docker examples earlier in this document.

### Terminate the EC2 instance

```bash
cd terraform/ondemand
terraform destroy
```

It's probably a good idea to log into the EC2 web console to confirm the instance has terminated.
