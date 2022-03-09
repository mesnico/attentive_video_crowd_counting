# Video-based Counting

## Setup
Work in progress...

## Dataset

&emsp;1. Download the FDST Dataset from
Google Drive: [link](https://drive.google.com/drive/folders/19c2X529VTNjl3YL1EYweBg60G70G2D-w) 

&emsp;2.  Create symbolic links to the dataset splits in the project root:
```
ln -s [FDST_ROOT]/train_data/ train_data
ln -s [FDST_ROOT]/test_data/ test_data
```
&emsp;3. Create the hdf5 files running `python fdst_dataset/make_dataset.py`

&emsp;4. [Optional, if you want to use different train-val-test splits] Run `python fdst_dataset/create_json_divide_scenes.py` to generate the json files for the train, validation, test splits.

## Training

```
python train.py configs/fdst_XA.yaml --experiment [experiment_name]
``` 

Note that `experiment_name` is used to save the model checkpoint under the `models/` path.

## Testing
If you want to replicate the results from the paper, download the checkpoints from here: [link](https://drive.google.com/drive/folders/1I2JfYsbROGoDFqHoSTjiIRl1dhxGpu61?usp=sharing).
Then, issue the following command:
```
python test.py [checkpoint_path.pth.tar] --config configs/fdst_XA.yaml
``` 
Note that, if you want to test a model that you trained, you find the best model on the validation set at `models/model_best_[experiment_name].pth.tar`. 
Also, in this case, you do not need to add the `--config` option (in a latest update, we decided to save the configuration together with the checkpoint).

## Visualization
Work in progress...

