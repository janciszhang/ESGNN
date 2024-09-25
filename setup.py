import requests
import tarfile

# Download and extract the file
url = "http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz"
response = requests.get(url, stream=True)
file = tarfile.open(fileobj=response.raw, mode="r|gz")
file.extractall(path=".")
#
# # Change working directory
# %cd metis-5.1.0
#
# # The remaining steps as you have shown in the question, with updated path
# !make config shared=1 prefix=~/.local/
# !make install
# !cp ~/.local/lib/libmetis.so /usr/lib/libmetis.so
# !export METIS_DLL=/usr/lib/libmetis.so
# !pip3 install metis-python
#
# import metispy as metis