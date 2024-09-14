[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/A2tcAnZG)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=15938310&assignment_repo_type=AssignmentRepo)
# emlov3-session-02

# PyTorch Docker Assignment

Welcome to the PyTorch Docker Assignment. This assignment is designed to help you understand and work with Docker and PyTorch.

## Assignment Overview

In this assignment, you will:

1. Create a Dockerfile for a PyTorch (CPU version) environment.
2. Keep the size of your Docker image under 1GB (uncompressed).
3. Train any model on the MNIST dataset inside the Docker container.
4. Save the trained model checkpoint to the host operating system.
5. Add an option to resume model training from a checkpoint.

## Starter Code

The provided starter code in train.py provides a basic structure for loading data, defining a model, and running training and testing loops. You will need to complete the code at locations marked by TODO: comments.

## Submission

When you have completed the assignment, push your code to your Github repository. The Github Actions workflow will automatically build your Docker image, run your training script, and check if the assignment requirements have been met. Check the Github Actions tab for the results of these checks. Make sure that all checks are passing before you submit the assignment.

## Solution

This repository contains a PyTorch implementation to train and test a neural network on the MNIST dataset. The model architecture includes convolutional and fully connected layers designed to classify images of handwritten digits (0-9). The script allows for customizable training options via command-line arguments.

### Features
- Customizable model training with configurable batch size, epochs, learning rate, and more.
- Model checkpointing for saving and resuming training from saved states.
- Logging of training progress and performance metrics during each epoch.
- Support for CUDA and macOS Metal (MPS) GPU acceleration.
- Command-line argument parsing for ease of use.

### Requirements
- Python 3.7+
- PyTorch 1.9+
- Torchvision
- argparse

Install the required dependencies with:
```bash
pip install torch torchvision
```

### Usage

#### Running the Script

To train the model from scratch, use the following command:

```bash
python train.py --batch-size 64 --epochs 14 --lr 1.0
```

#### Command-Line Arguments
The following arguments are supported to customize the training process:

- `--batch-size` (default: 64): Input batch size for training.
- `--test-batch-size` (default: 1000): Input batch size for testing.
- `--epochs` (default: 14): Number of epochs to train.
- `--lr` (default: 1.0): Learning rate for the optimizer.
- `--gamma` (default: 0.7): Learning rate step decay factor.
- `--no-cuda`: Disable CUDA (GPU) training even if CUDA is available.
- `--no-mps`: Disable macOS GPU training.
- `--dry-run`: Run a quick single batch to check if the pipeline works.
- `--log-interval` (default: 10): Number of batches to wait before logging training status.
- `--save-model` (default: True): Save the model at each epoch.
- `--resume`: Resume training from the last saved checkpoint.
- `--seed` (default: 1): Seed for random number generation.

#### Resuming Training
To resume training from a saved checkpoint, use the `--resume` flag:

```bash
python train.py --resume
```

Ensure that the `model_checkpoint.pth` file is present in the current directory.

#### Example
```bash
python train.py --batch-size 64 --epochs 10 --lr 0.1 --gamma 0.9 --log-interval 20 --save-model
```

### Model Architecture

The model consists of the following layers:
- Two convolutional layers (`conv1` and `conv2`)
- Two dropout layers (`dropout1` and `dropout2`) to prevent overfitting
- Fully connected layers (`fc1` and `fc2`)
- Softmax output for multi-class classification

### Training and Testing
The model is trained using the Negative Log-Likelihood (NLL) loss function and optimized using the Adadelta optimizer. The script implements both a training loop and a testing loop to evaluate model performance on the MNIST test set after each epoch.

Training logs are printed periodically based on the `--log-interval` argument, showing the progress and loss for each batch.

### Checkpointing
The model, optimizer state, and current epoch are saved after each epoch in `model_checkpoint.pth`. This allows you to resume training from where you left off using the `--resume` flag.

### References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
