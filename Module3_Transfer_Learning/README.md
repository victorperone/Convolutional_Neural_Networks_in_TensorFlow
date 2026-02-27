# Module 3 Transfer Learning

> **From Scratch to State-of-the-Art**: Leveraging Pretrained CNNs to Achieve High Accuracy with Limited Data. Transfer Learning with InceptionV3

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-Image_Data_Generator-red.svg)](https://keras.io/preprocessing/image/) [![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

---

## 🧭 Overview

In Module 2, we reduced overfitting using **Data Augmentation** and **Dropout**, achieving strong generalization (~82% validation accuracy).

However, training a CNN from scratch still has limitations:
- Requires large datasets
- Requires long training time
- Still limited in achievable accuracy

In **Module 3**, we solve this using **Transfer Learning**.

Instead of training a deep convolutional network from scratch, we:
- Import a pretrained **InceptionV3** model (trained on ImageNet)
- Freeze its convolutional layers
- Use it as a high-level feature extractor
- Train only a new classification head on our dataset

This allows us to reach **~95–98% validation accuracy in just a few epochs**, dramatically improving both performance and efficiency.

---

## 🎯 Learning Objectives

In this module, you will learn to:
- Load and configure a pretrained CNN (`InceptionV3`)
- Freeze and unfreeze layers properly
- Extract intermediate feature maps (e.g., `mixed7`)
- Attach a custom classifier head
- Train using feature extraction
- Apply fine-tuning safely with low learning rates
- Compare training-from-scratch vs transfer learning

---

## 🧩 Design Decisions

### Why InceptionV3?

- Strong accuracy/parameter ratio
- Proven ImageNet performance
- Balanced depth and computational efficiency
- Good intermediate layers for feature extraction (e.g., mixed7)

Alternative architectures considered:
- ResNet50 (heavier residual depth)
- MobileNet (lighter but lower accuracy)
- EfficientNet (newer but more complex scaling strategy)


---

## 💼 Why This Project Matters

This module demonstrates my ability to:

- Use pretrained architectures effectively
- Understand feature extraction vs fine-tuning
- Optimize training time and compute efficiency
- Reduce data requirements for high accuracy
- Apply callbacks for dynamic training control
- Compare architectural strategies using metrics

- This mirrors **real-world ML workflows** where:
  - Datasets are small
  - Compute resources are limited
  - Production models must converge quickly 
  - Pretrained backbones are standard practice
  - Rapid convergence is required

---

## 👥 Who This Module Is For

This module is designed for:
- Engineers working with small datasets
- Developers needing fast convergence
- Computer Vision students exploring pretrained networks
- Anyone wanting 95%+ accuracy without massive compute

---

## 🛠️ Skills Demonstrated

### 1. Applying Transfer Learning with InceptionV3

- Loaded pretrained convolutional base trained on ImageNet
- Removed top classifier (`include_top=False`)
- Reused learned feature representations
- Adapted model for binary classification
Demonstrates:
- Understanding of pretrained architectures
- Knowledge of feature reuse
- Ability to reduce data requirements

### 2. Feature Extraction Strategy

- Froze convolutional base
- Trained only classification head
- Reduced trainable parameter count
Demonstrates:
- Awareness of overfitting risk
- Efficient training strategy
- Proper use of pretrained models

### 3. Intermediate Feature Selection (`mixed7`)

- Extracted representations from internal layer
- Balanced abstraction depth with adaptability
Demonstrates:
- Architectural understanding
- Knowledge of internal CNN structure

### 4. Custom Classification Head Design

- Dense(1024) with ReLU
- Dropout regularization
- Sigmoid output layer
Demonstrates:
- Regularization strategy
- Binary classification setup
- Feature-to-label mapping

### 5. Fine-Tuning Strategy

- Selective layer unfreezing
- Reduced learning rate during fine-tuning
Demonstrates:
- Understanding of catastrophic forgetting
- Optimization control
- Model adaptation techniques

### 6. Training Optimization with Callbacks

- Implemented early stopping via custom callback
- Controlled compute usage
- Reduced unnecessary training
Demonstrates:
- Training efficiency awareness
- Practical ML engineering discipline

---

## ⚠️ Common Mistakes Explored in This Module

- **Unfreezing Too Many Layers Too Early**
  - Can destabilize training.
- **High Learning Rate During Fine-Tuning**
  - Can destroy pretrained weights.
- **Training Entire Network From Start**
  - Leads to unnecessary computation.
- **Ignoring Domain Gap**
  - Transfer Learning works best when new dataset resembles ImageNet-like data.

---

## ▶️ How to Run

This module consists of Jupyter notebooks that can be run locally or on Google Colab.

### Prerequisites

- Python 3.8 or higher
- pip
- Virtual environment support (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/victorperone/Convolutional_Neural_Networks_in_TensorFlow.git
cd Module3_Transfer_Learning
```

### 2. Create and Activate a Virtual Environment (Recommended)

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

All required packages are listed in `requirements.txt`

```python
pip install -r requirements.txt
```
This will install TensorFlow and other necessary libraries.

### 4. Launch Jupyter Notebook

You can run the notebooks **locally** or using **Google Colab**.

#### Option A: Run Locally (Jupyter Notebook)

```bash
jupyter notebook
```

or, if you prefer JupyterLab:
```bash
jupyter lab
```

### Option B: Run on Google Colab (No Local Setup Required)

1. Go to: [Google Colab](https://colab.research.google.com)
2. Click File → Open notebook
3. Select the GitHub tab
4. Paste your repository URL
5. Open  `File.ipynb`

Google Colab provides:

- Free CPU (and optional GPU) execution
- No local Python or TensorFlow installation
- Automatic dependency handling for most libraries

⚠️ Note: If requirements.txt is not automatically handled, install dependencies in a Colab cell:

```python
!pip install -r requirements.txt
```

### 5. Run the Exercises

Open the notebooks in numerical order
Run each cell sequentially
Observe how changes in model architecture, training duration, and callbacks affect results
It is recommended to run the exercises **in order**, as each one builds conceptually on the previous examples.

### Environment Notes

- TensorFlow may produce informational or warning messages during execution.
- These messages do not affect the correctness of the exercises.
- CPU execution is sufficient for all notebooks in this module, though GPU is recommended for faster training on the 300x300 images.

---

## 🧪 Reproducibility Note

Model training involves random initialization of weights.
As a result:
- Exact accuracy and loss values may vary slightly between runs
- Overall trends and conclusions should remain consistent
- Pretrained weights ensure consistent feature extraction 
- Minor accuracy variations may occur due to:
  - Random initialization of dense layers
  - Data shuffling
  - Augmentation randomness

---

## ❓ Problem Statement

In Module 2, we improved generalization but were still limited by training a CNN from scratch.

**The Challenge**:
How can we dramatically increase validation accuracy while reducing training time and data requirements?

**The Solution**:
Use a pretrained InceptionV3 network as a feature extractor and train only a lightweight classifier on top.

---

## 💾 Dataset

**Horses vs. Humans**
This dataset contains computer-generated images of horses and humans in various poses and backgrounds.
- Training set: ~1,000 images
- Validation set: ~250 images
- Challenge: The model must learn shapes and features specific to humans or horses, ignoring the complex backgrounds.
This dataset is intentionally small — ideal for demonstrating the power of Transfer Learning.

---

## 📉 Deep Dive: Why Transfer Learning Outperforms Scratch Training

With training-from-scratch:
- Many epochs required
- Risk of overfitting
- Slower convergence

With Transfer Learning:
1. Pretrained layers already detect edges, shapes, textures.
2. We reuse those representations.
3. Only top classifier adapts to new data.
4. Training converges in a few epochs.
5. Validation accuracy improves significantly.

---

## ⚙️ Technical Implementation

### 1. Data Pipeline

We use `ImageDataGenerator` for:
- Rescaling pixel values to [0,1]
- Applying augmentation (rotation, shift, zoom, shear)
- Streaming batches from disk

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
```

Validation data is only rescaled:

```python
validation_datagen = ImageDataGenerator(rescale=1./255)
```

This ensures:
- Training data is diversified
- Validation data remains realistic

### 2. Model Architecture Pipeline

The architecture follows this structure:

```SCSS
Input Image (150x150x3)
        ↓
InceptionV3 Convolutional Base (Frozen)
        ↓
Intermediate Layer Output (mixed7)
        ↓
Flatten
        ↓
Dense(1024, ReLU)
        ↓
Dropout(0.2)
        ↓
Dense(1, Sigmoid)
        ↓
Binary Prediction

```

This separates:
- Feature extraction
- Task-specific classification

### 3. Transfer Learning with InceptionV3 (Feature Extraction)

Instead of learning visual features from scratch, we reuse features learned from ImageNet (1.2M images, 1000 classes).
We initialize InceptionV3 without the top classification layer and load pretrained weights.

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet'
)

```

#### Technical Breakdown:

- `include_top=False`: Removes the original 1000-class classifier.
- `weights='imagenet'`: Loads pretrained weights.
- The convolutional layers now act as a **universal feature extractor**.

### 4. Freezing the Convolutional Base

```python
for layer in pre_trained_model.layers:
    layer.trainable = False

```
**Why Freeze?**
- Prevents overwriting pretrained weights
- Reduces trainable parameters
- Speeds up training
- Reduces overfitting risk

- This process is called **Feature Extraction**.

### 5. Extracting Intermediate Features

```python
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
```
**Why `mixed7`?**
- Deep enough to capture abstract features
- Not too specialized to ImageNet
- Balanced feature representation

### 6. Custom Classification Head

```python
from tensorflow.keras import layers
from tensorflow.keras import Model

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
```

**Why This Works:**

- Inception extracts visual features
- Dense layer learns dataset-specific patterns
- Dropout prevents neuron co-adaptation
- Sigmoid outputs probability for binary classification

### 7. Fine-Tuning (Advanced Optimization)

After training stabilizes, we can unfreeze the last layers:

```python
for layer in pre_trained_model.layers[-30:]:
    layer.trainable = True

```

Recompile with a lower learning rate:

```python
optimizer = RMSprop(learning_rate=1e-5)
```

**Why lower learning rate?**
- To **avoid catastrophic forgetting** — overwriting useful pretrained features.
- Fine-tuning allows deeper layers to adapt to the new dataset.

### 8.Training Efficiency with Callbacks

```python
class EarlyStopByAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target=0.98):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy")
        if acc and acc >= self.target:
            self.model.stop_training = True
```

**Why?**

- Stops training once target accuracy is reached
- Reduces compute cost
- Prevents overfitting
- Mimics production optimization strategies

---

## 🧠 Model Compilation

For binary classification, the optimizer and loss function change to reflect the nature of the output.

### Optimizer: RMSprop
While Adam is common, RMSprop is often preferred for automating the learning rate adjustment in recurrent or deep convolutional networks.

### Loss Function: Binary Crossentropy
Since we have only two classes and a single output neuron (0 to 1), we use binary_crossentropy instead of sparse_categorical_crossentropy.

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=1e-4),
              metrics=['accuracy'])
```

---

## 🏋️ Training the Model

### Training with Generators

In this module, training uses a **Python Generator**, differing from the standard model.fit(x, y) approach used in Course 1. This allows for **memory-efficient** streaming of data.


Training is performed using generator-based streaming to ensure memory efficiency and scalable batch processing.

The `fit` Method with Generators
Because our dataset (25,000 images) is too large to fit into RAM, we use the train_generator to stream images in batches.

```python
history = model.fit(
    train_generator,
    epochs=100, # Increased epochs because augmentation makes learning harder!
    verbose=1,
    validation_data=validation_generator
)
```

**Key Parameters Explained**
- `train_generator`: The source of the training images (yields batches of 128 images).
- `validation_data`: The source of the validation images. The model evaluates itself against this data at the end of every epoch.
- `steps_per_epoch` (_Implicit_): In modern TensorFlow, if you don't specify this, it defaults to `len(generator)`, which is `Total Images / Batch Size`.

**Note on Performance:** You will notice that training is slower than with MNIST. This is because we are processing color images (3 channels) that are 150x150 pixels (22,500 inputs per channel), compared to the tiny 28x28 grayscale images from the previous course.

---

## 📉 Model Evaluation

This measures how well the model performs on **unseen data**.

Evaluating model performance in this module goes beyond just looking at the final accuracy score. We focus on two key aspects:
Evaluation in this module is focused on detecting Overfitting. We do not just look at the final "Test Score"; we analyze the history of training.

- **Training vs. Validation Accuracy:** We plot the accuracy history to check for overfitting.
  - **Good Model:** Training and Validation accuracy increase together.
  - **Overfitting:** Training accuracy hits 99-100%, but Validation accuracy stalls or decreases (e.g., remains at 80%).
- **Visualizing Intermediate Representations:** We use specific code blocks to visualize the "internal state" of the Convolutional layers. This allows us to see exactly what features (lines, shapes, textures) the model is activating on.

```python
# Example of accessing history for evaluation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
```

---

## 📊 Model Evaluation Metrics

In this module, we track two primary metrics. It is crucial to understand why they behave differently. 

1. Accuracy (`accuracy` & `val_accuracy`)

- **Definition**: The percentage of images classified correctly.
- **Goal**: Maximize `val_accuracy` (closer to 1.0).
- **Observation**: Training accuracy converges to 1.0 (100%), while validation accuracy plateaus. This gap represents the model's inability to generalize.

2. Loss (`loss` & `val_loss`)

- **Definition**: A penalty score calculated by **Binary Crossentropy**. It punishes the model for being "confident and wrong."
- **Goal**: Minimize this (closer to 0.0).
- **Observation**:
  - If the model predicts "Human" (1.0) and it is a Human: **Low Loss**.
  - If the model predicts "Human" (1.0) and it is a Horse: **Massive Loss**.

---

## 📏 Metrics in This Module

We configure the model to track these specific metrics during compilation:

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(
    optimizer=RMSprop(learning_rate=1e-4),
    loss='binary_crossentropy', # Mandatory for 2-class problems
    metrics=['accuracy']        # We only need accuracy for classification
)
```

- **Optimizer**: We use **RMSprop** instead of Adam. RMSprop is often preferred for Recurrent Neural Networks (RNNs) but is also excellent for Convolutional networks as it allows for an adaptive learning rate, which helps navigate the complex "loss landscape" of image data.
- **Loss**: We use **Binary Crossentropy** because our final layer is a single neuron (Dense(1, activation='sigmoid')). It treats the output as a probability: P(Class=1).


---

## 🔑 Key Concepts

### 1. The "Lumberjack vs. Sailor" Problem (Overfitting)

As explained in the lessons, overfitting is like a human trying to classify "Lumberjacks" vs. "Sailors."
* If all the lumberjacks in your training photos happen to be wearing caps, and no sailors are...
* The model might conclude: **"Wearing a Cap = Lumberjack."**
* This is **Overfitting**: learning a coincidental feature (the cap) rather than the essential feature (the uniform or context). In this module, we see our model doing exactly this—getting 99% accuracy on training data but failing on the validation set.

### 2. Preprocessing & Generators
Since we cannot load 25,000 images into RAM at once, we use `ImageDataGenerator`. This tool acts as a streaming service, pulling images from the hard drive, resizing them on the fly, and feeding them to the GPU in batches.

### Model Capacity

Overfitting occurs when the model learns the training data too well but fails to generalize. Common causes explored in this module:

- Too many neurons or layers (Excessive Capacity)
- Training for too many epochs

Later exercises introduce techniques to control training behavior instead of blindly increasing epochs.

### Overfitting

Overfitting occurs when:

- The model learns the training data too well
- It fails to generalize to unseen data

Common causes explored in this module:

- Too many neurons
- Too many layers
- Too many training epochs

Later exercises introduce techniques to control training behavior instead of blindly increasing epochs.

---

## 📊 Results

| Model             | Train Accuracy | Validation Accuracy | Epochs |
|-------------------|----------------|---------------------|-------|
| From Scratch      | ~85%           | ~82%                | 100   |
| Transfer Learning | ~98-99%        | ~95-97%             | 3-10  |
| Fine Tuned Model  | ~99%           | ~97-98%             | 5-15  |

### Interpretation

Transfer Learning reduced required epochs by ~90%.
Validation accuracy improved by ~15%.
Fine-tuning provided incremental but meaningful gains.

---

## 💡 What I learned

- Training from scratch is often unnecessary
- Pretrained CNNs generalize remarkably well
- Learning rate control is critical during fine-tuning
- Feature extraction drastically reduces variance
- High validation accuracy can be achieved with limited data
- Engineering decisions (freeze/unfreeze strategy) matter more than raw model depth

---

## 📓 Notebooks & Exercises

- **Course Notebook**: Guided lesson on loading InceptionV3 and freezing layers.
- **Exercise**: Implement Transfer Learning on the Horses vs Humans dataset.
  - Freeze convolutional base
  - Add Dense + Dropout layers
  - Train classifier
  - Optionally fine-tune deeper layers
  - Compare performance with Module 2

---

## Comparison of Popular Pretrained Networks

| Model               | Parameters (Approx.) | Speed    | Accuracy (ImageNet Top-1) | Strengths                                             | Typical Use Case                           |
| ------------------- | -------------------- | -------- | ------------------------- | ----------------------------------------------------- | ------------------------------------------ |
| **VGG16**           | ~138 Million         | Slow     | ~71%                      | Simple architecture, easy to understand               | Educational purposes, baseline experiments |
| **ResNet50**        | ~25 Million          | Moderate | ~76%                      | Residual connections enable deep training             | General-purpose computer vision tasks      |
| **InceptionV3**     | ~24 Million          | Moderate | ~78%                      | Efficient factorized convolutions, strong performance | Balanced accuracy vs compute               |
| **MobileNetV2**     | ~3.4 Million         | Fast     | ~72%                      | Lightweight, depthwise separable convolutions         | Edge devices, mobile deployment            |
| **EfficientNet-B0** | ~5.3 Million         | Fast     | ~77%                      | Compound scaling (depth, width, resolution)           | Production systems, high efficiency        |
| **EfficientNet-B4** | ~19 Million          | Moderate | ~83%                      | High accuracy with optimized scaling                  | High-performance applications              |


---

## 📝 Summary

In this module, we moved from:
- Regularized CNN training (Module 2)
To:
- Efficient feature reuse with Transfer Learning.
By leveraging InceptionV3:
- Validation accuracy improved dramatically
- Training time reduced significantly
- Overfitting risk decreased
- Data requirements were minimized
Transfer Learning transforms deep learning from:
  - Train everything from scratch

Into:
- "Reuse knowledge intelligently and adapt efficiently."


> Transfer Learning is not just a performance trick — it is a compute-efficient engineering strategy.
---

## 📘 Files in This Module

```
📁 Module3_Transfer_Learning
├── 📓 Course_2_Part_6_Lesson_3_Notebook.ipynb.ipynb
├── 📓 Exercise_3_Horses_vs_humans_using_Transfer_Learning_Question-FINAL.ipynb
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🛑 Limitations

While Transfer Learning is powerful, it is not a universal solution.
- **Model Size**: InceptionV3 is large and memory-intensive.
- **Domain Shift Risk**: If the new dataset is very different from ImageNet (e.g., medical imaging), feature transfer may be less effective.
- **Bias Transfer**: Pretrained models inherit biases present in ImageNet.
- **Fine-Tuning Sensitivity**: Incorrect learning rate can destroy pretrained weights.
- **Deployment Constraints**: Large models may not be ideal for edge devices without optimization. 

Transfer Learning is highly effective for natural images similar to ImageNet, but careful evaluation is required for specialized domains. 

**Not a Magic Bullet**: If the original dataset is too small or non-representative, augmentation alone cannot fix it. For that, we need **Transfer Learning** (Module 3).

---

## 📚 Further Reading

### InceptionV3
 
- [InceptionV3 Architecture Overview (TensorFlow Docs)](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)
- [Original Inception Paper: "Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567)
- [ImageNet Dataset](http://www.image-net.org)

### Transfer Learning Concepts
 
- [Transfer Learning Guide (TensorFlow)](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [CS231n Notes on Transfer Learning (Stanford)](https://cs231n.github.io/transfer-learning/)
- [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526)

### Catastrophic Forgetting

- [Explanation of Catastrophic Forgetting](https://towardsdatascience.com/catastrophic-forgetting-in-neural-networks-df5f36e5e5f5)

### Other Popular Pretrained CNN Architectures

- [ResNet](https://arxiv.org/abs/1512.03385)
- [VGG16](https://arxiv.org/abs/1409.1556)
- [MobileNet](https://arxiv.org/abs/1704.04861)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [TensorFlow Implementations:](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
