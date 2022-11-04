# Speech separation in the waveform domain
This is the repository for the [DTU 02456 Deep Learning](https://kurser.dtu.dk/course/02456) project. The aim is to separate speakers in mixed speech signals using the [Demucs model](references/Defossez2019%20-%20Music%20Source%20Separation%20in%20the%20Waveform%20Domain.pdf), which was originally intended for stem separation of music. The project is supervised by Prof. Bjørn Sand Jensen (Department of Applied Mathematics and Computer Science).

Links:
* See the [synopsis](docs/synopsis.pdf) for the outline of the project
* See the [developer's guidelines](docs/CONTRIBUTING.md) if you want to contribute


## Setup
* Ensure Python 3.9 is installed
* Create a virtual environment with Python 3.9 interpreter
* Install requirements from `requirements.txt`
* Set the environment variable `LIBRIMIX_STORAGE_DIR` to the storage directory you used for `./generate_librimix.sh`


## Development
* Format code using `black .` in project directory before committing
