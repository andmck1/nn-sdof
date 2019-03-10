# Neural Nets for Structural Dynamic Response Estimation and System Identification

This project aims to replicate and improve on the neural net architecture employed by [Wu and Jahanshahi](./2018 - Deep Convolutional Neural Network for Structural Dynamic Response Estimation and System Identification.pdf).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites and Installation

These notebooks run on the keras API. GPU enabled Tensorflow was used as the backend, change the [requirements file](./pip-req.txt) if using a different backend.

The following installation notes also assume use of venv for virtual environments on Linux/Mac OSX . Change as required.

```
python3 -m pip install -U venv
python3 -m venv johnny5-wu
source johnny-wu/bin/activate
python3 -m pip install -r 'pip-req.txt'
```

## Built With

* [Keras](https://keras.io/) - High-level neural networks API.

## Contributing

Got ideas? Contribute!

## Authors

* **Alastair Hamilton** - *Initial work* - [kr4in](https://github.com/kr4in)

## License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md) file for details

## Acknowledgments

* Wu and Jahanshahi for inspiring this work!
* samph4 for motivating me into deep learning.
