# pyramidnet-expert-discriminators

## Recommended Versions
- Python 3.10.12 or above
- omegaconf 2.3.0
- numpy 1.26.4
- pytorch 2.2.1
- torchvision 0.17.1
- pillow 9.0.1

## How to Use
Before running running main.py for the first time, run the script ImageDownloader.py to download the CIFAR-10 dataset in a format which is compatible with torchvision's ImageFolder API.

The only mandatory command line argument is the mode, which can be train, validate, confusion_matrix, or test. Currently, train mode trains a completely new base model and then checks the performance of each checkpoint saved during training on the validation data, validation mode checks the performance of a specific checkpoint on the validation data, confusion_matrix mode evaluates a specific checkpoint on the training data to generate the confusion matrix, and test mode checks the performance of a specific checkpoint on the test data. Keep in mind that the validation and training data will only be consistent across runs of the script if the train_split_ratio config option is kept consistent. Also, to specify which checkpoint to use in the validate, confusion_matrix, or test modes, you need to use the eval_chkpt config option.

The configuration settings are handled using OmegaConf, so you can create custom configuration files to replace the default files TrainingConf, DefaultModelConf, and DataConf (or just modify the default files directly). Just put the files in the conf/ directory and use the command line arguments --training_config, --base_model_config, and --data_config to pass in the filenames.

## Checkpoints to Reproduce Results in Report
The four checkpoints which can already be found in the eval_chkpts directory are the models which produced the validation and test accuracy results shown in the report. The checkpoints with the word "deep" in their name correspond to the model with $D = 150$ and $\alpha = 170$. The checkpoints with the word "shallow" in their name correspond to the model with $D = 110$ and $\alpha = 200$.