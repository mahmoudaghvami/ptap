# PTAP - Privacy Preserving Trigger Action Platform

Welcome to the GitHub repository for PTAP - Privacy Preserving Trigger Action Platform! PTAP aims to provide a solution for addressing privacy concerns in smart home environments, particularly those related to untrusted Trigger Action Platforms (TAPs).

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

## Installation

To install PTAP, please follow the instructions provided in the [installation guide](./INSTALL.md).

## Usage

For information on how to use PTAP, please refer to the [usage guide](./USAGE.md). This guide provides detailed instructions on how to deploy and configure PTAP in your smart home environment.

## License

PTAP is licensed under the [MIT License](./LICENSE).
