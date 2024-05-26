# nbeats-generic-tf

This repository contains the implementation of a generic N-BEATS (Neural Basis Expansion Analysis Time Series Forecasting) model using TensorFlow. N-BEATS is a powerful and flexible model designed for time series forecasting. This implementation includes the model architecture, configuration, and utilities for training and evaluation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/duhirwe/nbeats-generic-tf.git
    cd nbeats_generic_tensorflow
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the N-BEATS model, follow the steps below.

## Project Structure

The project structure is organized as follows:

```plaintext
nbeats-generic-tf
│
├── config
│   └── config.json            # Configuration file for the model and training parameters
│
├── data
│   └── sao_paulo_mirante.csv  # Example dataset
│
├── images
│   └── nbeats_generic.png     # Diagram or visualization of the N-BEATS model
│
├── model
│   ├── __init__.py
│   ├── basic_block.py         # Implementation of the basic block of the N-BEATS model
│   └── nbeat_gen.py           # Implementation of the N-BEATS generic model
│
├── utils
│   ├── __init__.py
│   └── utils.py                # utility functions
│
├── main.py                     # Main script for training and evaluation
└── requirements.txt           # Required Python packages
