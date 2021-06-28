#Source of code: official pix2pix repository
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/facades
ZIP_FILE=./datasets/facades
TARGET_DIR=./datasets/facades/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE