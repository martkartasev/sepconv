#
# KTH Royal Institute of Technology
#

from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile

def download_davis(dest="dataset"):

    output_dir = join(dest, "DAVIS/videos")

    if not exists(output_dir):

        makedirs(dest)

        url = "..." # TODO: Add actual URL
        print("Downloading DAVIS...")

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data...")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_dir

def get_training_set():
    root_dir = download_davis()
    # TODO: ...
    return None

def get_test_set():
    root_dir = download_davis()
    # TODO: ...
    return None
