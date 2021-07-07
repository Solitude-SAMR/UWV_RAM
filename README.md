# UWV_RAM
Reliability assessment of deep learning driven UWV
## Prepare the Dataset
Download the dataset and trained model weight from server using wget:
```
wget -P ./ https://cgi.csc.liv.ac.uk/~acps/datasets/SOLITUDE/data.zip
```
Unzip the folder and add to the root directory, run the following command to start testing:
```
python uwv.py
```
