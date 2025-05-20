# unisacsi

unisacsi (acsi = air-cryosphere-sea-interactions) is a collection of functions and tools to handle meteorological and oceanographical data. It is especially targeted towards students in the courses AGF-213, AGF-214, AGF-211 and AGF-212 at the University Centre in Svalbard. The toolbox contains functions to read and plot data obtained during the respective course fieldworks, e.g. CTD profiles or time series from automatic weather stations.

The oceanographic part of this toolbox was originally developed at the Geophysical Institute at the University of Bergen.

## Installation

We recommend the use of the anaconda distribution and the underlying conda package manager. It can be downloaded and installed from [here](https://www.anaconda.com/products/distribution). Furthermore, you should add the conda-forge channel. Open an anaconda-prompt/terminal window and type:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

It is advised to install this toolbox into a virtual environment, as some of the functionality depends on older versions of common packages.

Note for Apple users: The toolbox currently only works in an x86-environment (made for the old Intel chips). If you have installed anaconda recently on a Macbook with a new M1 chip, you have most likely gotten the arm64-native version (check with "conda info" and look for what is specified after "__archspec="). In that case, you need to force conda to create the new environment as x86 (see below).

Download the file requirements.txt directly from the github repository (open the file on the webpage, click on "Raw", right-click and Save as...), then type in a anaconda-prompt/terminal window (you can change *myenv* to your desired name for the enviroment):

```
conda create -n myenv --file requirements.txt
```

For Apple silicon users with arm64-native anaconda:

```
CONDA_SUBDIR=osx-64 conda create -n myenv --file requirements.txt
```

Follow the instructions and be patient, this might take some minutes!

If it doesn't work, try with the mamba solver. First, update your conda in the base environment:

```
conda update -n base conda
```

Then, install and set the new solver:

```
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Now try again to create the new environment:

```
conda create -n myenv --file requirements.txt
```

Once the environment is created, activate it with:

```
conda activate myenv
```

Two more packages can only be installed using pip. The first one you will need to process RBR data, the second one provides functionality to do tidal harmonic analyses.

```
pip install pyrsktools
pip install uptide
pip install seabird
pip install sounddevice
```

Now you can install the unisacsi toolbox with:

```
pip install git+https://github.com/UNISvalbard/unisacsi.git
```

In case we find and fix a bug or add new functionalities to the toolbox during the course, we might tell you to update your local copy of the package:

```
pip install git+https://github.com/UNISvalbard/unisacsi.git --upgrade
```

## Examples

Besides the actual toolbox code, the github repository also includes a jupyter notebook with examples how to use the different toolbox function (examples.ipynb). It should be downloaded directly from github (see instructions for the requirements.txt-file above) and saved into a local folder of your choice. Furthermore, you need to download a folder with example data from the course Onedrive page.

Start jupyter-notebook inside your newly created environment and open the example notebook. Change the path in the second box of code to the location where you have saved the example data. Then run the code.

## MET model data download

The configurations for the download of model data from AROME-Arctic or METCoOp is done via a seperate configuration file, which can be downloaded from the github repository (see instructions above) and adjusted to individual needs. When using the toolbox function to download model data, the path to the configuration file has to be specified.
