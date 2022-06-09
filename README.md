# unisacsi

unisacsi (acsi = air-cryosphere-sea-interactions) is a collection of functions and tools to handle meteorological and oceanographical data. It is especially targeted towards students in the courses AGF-213, AGF-214, AGF-211 and AGF-212 at the University Centre in Svalbard. The toolbox contains functions to read and plot data obtained during the respective course fieldworks, e.g. CTD profiles or time series from automatic weather stations.

The oceanographic part of this toolbox was originally developed at the Geophysical Institute at the University of Bergen


## Installation

It is advised to install this toolbox into a virtual environment, as some of the functionality depends on older versions of common packages. We recommend the use of the anaconda distribution and the underlying conda package manager. It can be downloaded and installed from here: https://www.anaconda.com/products/distribution. After the successful installation open an anaconda-prompt window, and create a virtual environment with Python 3.8, git and pip:

> conda create -n myenv python=3.8 git pip

Follow the instructions and once the environment is created, activate it with:

> conda activate myname

Now you can install the unisacsi toolbox with:

> pip install git+https://github.com/UNISvalbard/unisacsi.git


## Examples

Besides the actual toolbox code, the github repository also includes a jupyter notebook with examples how to use the different toolbox function. It should be downloaded directly from github and saved into a local folder of your choice. Furthermore, you need to download a folder with example data from here:

Now you can run the notebook:

> jupyter-notebook

then navigate to the file and open it. Change the path in the second box of code to the location where you have saved the example data. Then press the Execute-button in the top.

## MET model data download

The configurations for the download of model data from AROME-ARcitc or METCoOp is donw via a seperate configuration file, which can be downloaded from the github repository and adjusted to individual needs. When using the toolbox function to download model data, the path to the configuration file has to be specified.
