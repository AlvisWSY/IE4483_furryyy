# IE4483_furryyy

## Instruction
After download and unzip the project files, please create a new folder "data" under the project dirctory.

Then move the given dataset to the "data" folder, and rename dataset folder to "Cats_vs_Dogs"

Make it looks like following:

~/_PATH_/data/Cats_vs_Dogs/...

## Environment Config
There is an environment.yml used for this project

Use `conda env create -f environment.yml` to create env for this project

## Parts of This Project
`Cats_vs_Dogs.py`is the main file of this project, which is used for **Part (_a_) to (_f_)** in the report.

`CIFAR10.py`is the code used to load _CIFAR10_ dataset and train model with this dataset. This file is used for **Part (_g_) and (_h_)** in the report.

`ImageFolder.py`is a dependence of `Cats_vs_Dogs.py`. This file is based on pytorch torchvision.ImageFolder Function, and there are some changes on it for this project.

`submission.csv`is the required test result of model.

`Fault_Collection.csv`contains wrongly classified images' path.

`figure.png`shows accuracy differences of model when LearningRate settled to different values (1e-3, 1e-2, 5e-3, 5e-2)

`layers.txt`contains layer information of the model, exported from **_torchsummary_**

`Result` folder contains loss and accuracy data under different LR

`Checkpoints` folder contains saved model.


