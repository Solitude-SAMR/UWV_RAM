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
pip3 install matplotlib scikit-learn torch torchsummary torchvision tqdm imgaug tensorboard terminaltables
```
## Fetch the Code
Fetch the source code for the reliabity assessment of UWV by run:
```
git clone https://github.com/Solitude-SAMR/UWV_RAM
```

## Prepare the Dataset
Download the dataset and trained model weight from server using wget:
```
wget -P ./ https://cgi.csc.liv.ac.uk/~acps/datasets/SOLITUDE/data.zip
```
Unzip the folder and add to the root directory 'UWV_RAM/'.

## Train the UWV Model and VAE Model
To train the yoloV3 model by yourself for UWV object detection, run
```
python -m pytorchyolo.train
```
To train the variantional autoencoder model by yourself to compress the UWV simulation images, run
```
python uwv_vae.py
```
## Test the Reliability of Object Detection Model 
run the following command to start testing:
```
python uwv.py
```
When the program is running, all test result for each demand (image) is saved to the output folder with format (latent_represenation, x_class, pmi). pmi is the abbreviation probabity of misclassification per input. Then you can visualize the robustness verification results of all the inputs, the update of reliability (pmi) by running
```
python plot.py
```

