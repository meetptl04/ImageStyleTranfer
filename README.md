# ImageStyleTranfer

Here’s a complete Markdown documentation for your GitHub repository:

---

# Image Style Transfer

This project demonstrates how to apply neural style transfer using TensorFlow and a pre-trained VGG19 model. The goal is to combine the content of one image with the style of another to create a stylized image. The repository includes code for image processing, loss computation, style transfer, and visualization.

## Overview

The project consists of several components:

1. **Image Processing**: Methods to load, process, and de-process images for neural network input and visualization.
2. **Loss Functions**: Functions to compute content and style losses, including the gram matrix for style loss.
3. **VGG Model**: A class for initializing the VGG19 model and extracting content and style features.
4. **Style Transfer**: Core class for performing the style transfer, including loss computation and gradient updates.
5. **Visualization**: Methods for displaying images using Matplotlib.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Classes](#classes)
  - [ImageProcessing](#imageprocessing)
  - [LossFunctions](#lossfunctions)
  - [VGGModel](#vggmodel)
  - [StyleTransfer](#styletransfer)
  - [Visualization](#visualization)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/meetptl04/ImageStyleTranfer.git
   cd ImageStyleTranfer
   ```

2. Install the required libraries:

   ```bash
   pip install tensorflow matplotlib numpy
   ```

## Usage

1. **Initialize Classes**: Set up the VGG model and style transfer instance.

2. **Run Style Transfer**: Perform the style transfer by specifying paths to the content and style images.

3. **Visualize Results**: Display the original images and the final stylized image.

   Example:

   ```python
   # Initialize the classes
   vgg_model = VGGModel()
   style_transfer = StyleTransfer(vgg_model)
   visualizer = Visualization()

   # Paths to images
   original_image = '/content/the-starry-night.jpg'
   style_image = '/content/Fantasy-Garden.png'

   # Run the style transfer
   best_img, best_loss = style_transfer.run_style_transfer(original_image, style_image)

   # Display images
   visualizer.show_image_matplotlib(original_image)
   visualizer.show_image_matplotlib(style_image)

   # Display the resulting image
   visualizer.show_results(best_img)
   ```

## Classes

### ImageProcessing

The `ImageProcessing` class provides static methods for loading and processing images for neural network input. It resizes and normalizes images for VGG19 and performs de-normalization for display.

```python
class ImageProcessing:
    @staticmethod
    def load_and_process_img(path_to_img):
        # Load, resize, and preprocess image
        ...

    @staticmethod
    def deprocess_img(processed_img):
        # Convert processed image back to viewable format
        ...
```

### LossFunctions

The `LossFunctions` class defines methods to compute content loss and style loss, including the gram matrix calculation for style loss.

```python
class LossFunctions:
    @staticmethod
    def get_content_loss(base_content, target):
        # Compute content loss
        ...

    @staticmethod
    def gram_matrix(input_tensor):
        # Calculate gram matrix
        ...

    @staticmethod
    def get_style_loss(base_style, gram_target):
        # Compute style loss
        ...
```

### VGGModel

The `VGGModel` class initializes a VGG19 model pre-trained on ImageNet and extracts feature representations for content and style images.

```python
class VGGModel:
    def __init__(self):
        # Initialize VGG19 model
        ...

    def get_feature_representations(self, content_path, style_path):
        # Extract content and style features
        ...
```

### StyleTransfer

The `StyleTransfer` class applies neural style transfer, computing losses, gradients, and optimizing the image to blend the style of one image with the content of another.

```python
class StyleTransfer:
    def __init__(self, vgg_model, style_weight=1e-2, content_weight=1e3):
        # Initialize style transfer parameters
        ...

    def compute_loss(self, init_image, gram_style_features, content_features):
        # Compute style and content losses
        ...

    def compute_grads(self, init_image, gram_style_features, content_features):
        # Compute gradients
        ...

    def run_style_transfer(self, content_path, style_path, num_iterations=1000):
        # Perform style transfer and optimize image
        ...
```

### Visualization

The `Visualization` class provides methods to display images using Matplotlib.

```python
class Visualization:
    @staticmethod
    def show_results(best_img):
        # Display final stylized image
        ...

    @staticmethod
    def show_image_matplotlib(image_path):
        # Display image from file path
        ...
```

## Results

Example output of style transfer, showing the original images and the final stylized image.

![Original Image](path/to/original/image)
![Style Image](path/to/style/image)
![Stylized Image](path/to/stylized/image)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize the paths and content as needed for your repository!
