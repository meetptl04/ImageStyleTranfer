# Image Style Transfer

This project implements image style transfer using TensorFlow and a pre-trained VGG19 model. It features an interactive UI for uploading images and adjusting style transfer parameters, along with comprehensive image processing and visualization capabilities.

## Overview

The project consists of several key components:

1. **Image Processing**: Advanced tensor processing and image manipulation for neural network compatibility
2. **Loss Functions**: Sophisticated implementations of content and style losses using gram matrices
3. **VGG Model**: Customized VGG19 model with carefully selected layers for feature extraction
4. **Style Transfer**: Core implementation with adjustable weights and interactive parameter tuning
5. **Visualization**: Interactive UI with image upload capabilities and real-time parameter adjustment

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Components](#components)
  - [ImageProcessing](#imageprocessing)
  - [LossFunctions](#lossfunctions)
  - [VGGModel](#vggmodel)
  - [StyleTransfer](#styletransfer)
  - [Visualization](#visualization)
- [Interactive UI](#interactive-ui)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/meetptl04/ImageStyleTranfer.git
cd ImageStyleTranfer
```

2. Install dependencies:
```bash
pip install tensorflow matplotlib numpy ipywidgets
```

## Dependencies

- TensorFlow 2.x
- NumPy
- Matplotlib
- IPython/Jupyter
- ipywidgets

## Components

### ImageProcessing

Handles image loading, processing, and tensor conversion:

```python
class ImageProcessing:
    @staticmethod
    def load_and_process_tensor(image_tensor):
        # Process and normalize image tensor
        # Returns preprocessed tensor for VGG19
        pass

    @staticmethod
    def deprocess_img(processed_img):
        # Convert processed image back to viewable format
        pass
```

### LossFunctions

Implements loss calculations for style transfer:

```python
class LossFunctions:
    @staticmethod
    def gram_matrix(input_tensor):
        # Calculate gram matrix for style features
        pass

    @staticmethod
    def get_style_loss(base_style, gram_target):
        # Compute style loss between base and target
        pass
```

### VGGModel

VGG19 model setup for feature extraction:

```python
class VGGModel:
    def __init__(self):
        self.model = self._get_model()
        
    def _get_model(self):
        # Initialize VGG19 and configure layers
        # Returns model with selected outputs
        pass
```

### StyleTransfer

Core style transfer implementation:

```python
class StyleTransfer:
    def __init__(self, vgg_model):
        self.vgg_model = vgg_model
        self.style_weight = 1e-2
        self.content_weight = 1e4

    def compute_loss(self, init_image, gram_style_features, content_features):
        # Compute style and content losses
        pass

    def run_style_transfer(self, content_image, style_image, num_iterations=300):
        # Execute style transfer process
        pass
```

### Visualization

Interactive UI implementation:

```python
class Visualization:
    def __init__(self, style_transfer):
        self.style_transfer = style_transfer
        self.current_style_weight = 1e-2
        self.current_content_weight = 1e3

    def style_transfer_ui(self):
        # Create and display interactive UI elements
        pass
```

## Interactive UI Features

The project includes an interactive UI with:

1. **Image Upload Widgets**
   - Content image upload
   - Style image upload
   - Drag-and-drop support

2. **Parameter Controls**
   - Style weight slider (1e-4 to 1e0)
   - Content weight slider (1e2 to 1e4)
   - Real-time weight adjustment

3. **Progress Monitoring**
   - Loss value display
   - Iteration progress
   - Status messages

4. **Result Visualization**
   - Side-by-side image display
   - Original images
   - Generated style transfer result

## Usage Example

```python
# Initialize components
vgg_model = VGGModel()
style_transfer = StyleTransfer(vgg_model)
visualizer = Visualization(style_transfer)

# Launch interactive UI
visualizer.style_transfer_ui()
```

## Results

The style transfer process produces three images:
1. Content Image: The base image whose content will be preserved
2. Style Image: The image whose style will be transferred
3. Result Image: The final stylized image combining content and style


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
