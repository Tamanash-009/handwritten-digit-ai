# Handwritten Digit Recognizer (PyTorch)

A simple computer vision project that trains a neural network on the MNIST dataset to recognize handwritten digits (0-9).

## Features
-   **Data Loading**: Automatically downloads and normalizes the MNIST dataset.
-   **Model**: A simple Feed-Forward Neural Network (Sequential) with one hidden layer.
-   **Training**: Trains for 5 epochs using the Adam optimizer.
-   **Visualization**: Displays a random test image and the model's prediction.

## Requirements
-   Python 3.8+
-   PyTorch
-   Torchvision
-   Matplotlib

## Installation

1.  **Clone the repository** (if applicable) or download the source code.
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install torch torchvision matplotlib
    ```

## Usage

Run the visualization script to train the model and see a prediction:

```bash
python mnist_visualizer.py
```

## Project Structure
-   `mnist_visualizer.py`: Main script with visualization.
-   `mnist_torch.py`: Basic training script (no visualization).
-   `README.md`: Project documentation.
-   `LICENSE`: MIT License.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
