# PTAP - Privacy Preserving Trigger Action Platform

Welcome to the GitHub repository for PTAP - Privacy Preserving Trigger Action Platform! PTAP aims to provide a solution for addressing privacy concerns in smart home environments, particularly those related to untrusted Trigger Action Platforms (TAPs).

![PrivacySHTAP](https://github.com/mahmoudaghvami/ptap/assets/13216089/aa05ec3c-aa58-4512-8b6a-aafa25b5361c)


## Table of Contents

- [About PTAP](#about-ptap)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About PTAP

Trigger Action Platforms (TAPs), such as Samsung SmartThings and Apple HomeKit, are becoming increasingly popular for managing smart homes. However, privacy concerns are rising as TAPs may share user data with third parties. Sensitive user-specific information can be inferred by TAPs, which include but are not limited to, user identities, occupancy status, and daily living routines.

PTAP is introduced as a solution to address the potential threat of untrusted TAPs. Unlike previous studies that focused on individual data points, PTAP considers the privacy of aggregated data sets over time, which can reveal personal behavior patterns and routines.

The PTAP design incorporates a privacy mediator, which serves as a trusted intermediary between smart home devices and the TAP server. The privacy mediator injects targeted perturbations into the data stream, confounding potentially malicious TAP classifiers. Furthermore, it also distinguishes between legitimate and fake actions sent from the TAP server, ensuring that only actions based on real data are executed.

## Usage

First you need to have Tensorflow and ART (https://github.com/Trusted-AI/adversarial-robustness-toolbox)
Then you can run the experiments  including the following Python scripts:

1. **Preprocessing:** [`preprocess-Vminute.py`](#preprocesspy)
2. **PTAP Classifier:** [`ptap_classifier_4cross_validation.py`](#ptap_classifier_4cross_validationpy)
3. **PTAP experements:** [`ptap_exp.py`](#ptap_exp.py)
4. **PTAP Utility Test:** [`ptap_utility_test.py`](#ptap_utility_testpy)

### `preprocess.py`
This script handles data preprocessing of the Orange4Home dataset (https://amiqual4home.inria.fr/orange4home/).


### `ptap_classifier_4cross_validation.py`
This script trains a classifier with 4-fold cross-validation on preprocessed data.

### `PTAP experements.py`
The ptap_exp.py script is designed to evaluate the robustness of TAP classifiers against various adversarial attacks. It contains functions for conducting different types of attacks (FGSM, JSMA, UAP), measuring their impact on model accuracy, and recording the time taken and number of events injected during each attack. Users can initiate specific experiments by calling the main_runner function with the desired attack type. The script also includes transferability tests, where adversarial examples generated for one model are tested against three target models to assess the cross-model effectiveness of the attacks.


### `ptap_utility_test.py`
This script evaluates the utility of PTAP, assuming that the user is running five real IoT applications in their smart home based on the real-world Orange4 Home dataset.


## License

PTAP is licensed under the [MIT License](./LICENSE).
