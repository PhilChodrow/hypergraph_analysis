import gzip
import shutil
import tarfile
from os import remove, rmdir
from google_drive_downloader import GoogleDriveDownloader as gdd

# This script downloads data sets kindly hosted by Austin Benson at 
# https://www.cs.cornell.edu/~arb/data/index.html
# If you use any of these data sets in scholarly publication, please make sure to cite Austin and his collaborators: 

# Simplicial closure and higher-order link prediction.
# Austin R. Benson, Rediet Abebe, Michael T. Schaub, Ali Jadbabaie, and Jon Kleinberg.
# Proceedings of the National Academy of Sciences (PNAS), 2018.

# Please also check the data page at Austin's website to find citations for the original data source.

files = {
    'tags-math-sx' : '1eDevpF6EZs19rLouNpiKGLIlFOLUfKKG',
    'email-Enron'  : '1tTVZkdpgRW47WWmsrdUCukHz0x2M6N77'
}

for name in files:
    print('downloading data from https://www.cs.cornell.edu/~arb/data/  : ' + name)
    destination = 'data/' + name + '.tar.gz'
    gdd.download_file_from_google_drive(files[name], destination)
    tar = tarfile.open(destination, "r:gz")
    
    extract_path = "data/" + name + '_tmp'
    tar.extractall(path=extract_path)
    
    # get rid of .gz: don't need it anymore
    remove(destination)
    
    # flatten folder structure
    shutil.move(extract_path + '/' + name, 'data/' + name)
    rmdir(extract_path)