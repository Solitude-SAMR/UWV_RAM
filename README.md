# UWV_RAM
Reliability assessment of deep learning driven UWV
## Installation
First of all, please set up a conda environment
```
conda create --name UWVRAM python==3.8
conda activate UWVRAM
```
This should be followed by installing software dependencies:
```
pip3 install scikit-learn torch torchsummary torchvision tqdm imgaug tensorboard terminaltables
```
## Prepare the Dataset
Download the dataset and trained model weight from server using wget:
```
wget -P ./ https://cgi.csc.liv.ac.uk/~acps/datasets/SOLITUDE/data.zip
```
Unzip the folder and add to the root directory, run the following command to start testing:
```
python uwv.py
```
