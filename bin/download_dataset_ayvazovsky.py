# for google.colab
# Source: https://www.kaggle.com/general/74235

! pip install -q kaggle
from google.colab import files
files.upload() #upload kaggle.json
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets list
!kaggle datasets download -d grafstor/aivazovsky-pix2pix-dataset
!unzip aivazovsky-pix2pix-dataset.zip
