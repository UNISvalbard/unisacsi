from setuptools import setup, find_packages



with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ['matplotlib==3.5', 'glob', 'copy', 'seabird','numpy','scipy','pandas', 'geopandas', 'shapely', 'netCDF4','cartopy','gsw','cmocean','requests','adjusttext', 'spyder', 'xarray', 'rioxarray', 'dask', 'utm']

setup(
  name = 'unisacsi',        
  version = '0.1.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Python toolbox for reading and analysing meteorological and oceanographic data from fieldwork with UNIS AGF courses.',   # Give a short description about your library
  long_description=readme,
  long_description_content_type="text/markdown",
  author = 'Lukas Frank, Jakob Doerr',                   # Type in your name
  author_email = 'lukasf@unis.no, jakob.dorr@uib.no',      # Type in your E-Mail
  url = 'https://github.com/lfrankunis/unisacsi',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/lfrankunis/unisacsi/archive/v0.0.1.tar.gz',
  packages=find_packages(),
  keywords = ['oceanography', 'meteorology', 'UNIS'],   # Keywords that define your package best
  install_requires=requirements,
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.8',
  ],
)