# Speech separation in the waveform domain
This is the repository for the [DTU 02456 Deep Learning](https://kurser.dtu.dk/course/02456) project. The aim is to separate speakers in mixed speech signals using the [Demucs model](references/Defossez2019%20-%20Music%20Source%20Separation%20in%20the%20Waveform%20Domain.pdf), which was originally intended for stem separation of music. The project is supervised by Prof. Bjørn Sand Jensen (Department of Applied Mathematics and Computer Science).

Links:
* See the [synopsis](docs/synopsis.pdf) for the outline of the project
* See the [poster](docs/poster.pdf) for a summary of the results
* See the [paper](docs/paper.pdf) for the full results
* See the [developer's guidelines](docs/CONTRIBUTING.md) if you want to contribute


## Setup
* Ensure Python 3.9 is installed
* Create a virtual environment with Python 3.9 interpreter
* Install requirements from `requirements.txt`
* Generate [LibriMix dataset](https://github.com/JorisCos/LibriMix) (for training)
* Add the DTU HPC transfer server `transfer.gbar.dtu.dk` as SSH host named `dtu-hpc-transfer` (to use copy scripts)
   * See [this guide](https://www.hpc.dtu.dk/?page_id=2501) for more information on SSH access


## Usage
* Training: run `bin/librimix/train.sh` in the project directory to train locally, or submit an LSF job on the DTU HPC cluster using `bsub < bin/librimix/train_job.sh`. The trained model will be stored in `data/models/librimix/version_1`. Adjust the LibriMix folder if needed.
   * To copy your code to the HPC cluster, use `bin/copy_code_to_hpc.sh`
   * To copy trained models from the HPC cluster, use `bin/copy_model_from_hpc.sh librimix X` where `X` is the version
   * It may be necessary to restart the training from a checkpoint if the 24 h walltime limit for jobs is exceeded. In that case, add `--checkpoint-path data/models/....ckpt` to the training script.
* Prediction: run `bin/librimix/predict.sh` in the project directory to predict a single example (set by `--item`) in the LibriMix dataset. Set the checkpoint/trained model to use in the script.
   * To easily find the checkpoint path, use `find data/models/librimix/ -name *.ckpt`
* Training plot generation: run `bin/evaluate_training.sh A B C` where `A B C` are the versions you want to compare
   * If a training consists of more than one version (i.e. it has been restarted from a checkpoint), use `A+B` where `A` and `B` belong to the same training


## Development
* Format code using `black .` in project directory before committing
* Run all code in the project root directory so that paths relative to the current working directory work as intended
