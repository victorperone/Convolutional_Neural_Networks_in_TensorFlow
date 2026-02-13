# Module 1 - Exploring a Larger Dataset (Cats vs. Dogs)

> **From 28x28 pixels to Full Scale:** Handling raw data, file system manipulation, and the battle against Overfitting.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-Image_Data_Generator-red.svg)](https://keras.io/preprocessing/image/) [![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

---

## 🧭 Overview

In the previous course, we worked with "toy" datasets (MNIST) where images were small, uniform, and pre-packaged. In this module, we step into the real world using the **Kaggle Cats vs. Dogs dataset**.

The challenge here isn't just building a model; it's **Data Engineering**. We have thousands of images of various sizes, shapes, and backgrounds. The primary goal of this module is to build a pipeline to handle this data and to observe a critical problem in Deep Learning: **Overfitting**, where a model memorizes training data but fails to generalize to new images.

This module builds on CNN foundations and introduces:
* Programmatic data manipulation (unzipping, filtering, moving).
* Building a `TRAIN` / `TEST` split from scratch.
* Using `ImageDataGenerator` for streaming large datasets.
* Analyzing the divergence between Training and Validation accuracy.

---

## 🎯 Learning Objectives

In this module, you will learn to:

* **Handle Raw Data:** Programmatically unzip, organize, and filter a large dataset using Python's `os` and `shutil` libraries.
* **Data Splitting:** Manually split a directory of images into **Training** (90%) and **Validation** (10%) sets to ensure fair evaluation.
* **Build a Deep CNN:** Construct a Convolutional Neural Network capable of binary classification on color images.
* **Diagnose Overfitting:** Interpret loss/accuracy graphs to identify when a model stops learning and starts memorizing.

---

## 👥 Who This Module Is For
This project is designed for:
* **ML Engineers** moving from "toy" datasets to production-style data pipelines.
* **Data Scientists** who need to understand how to manipulate file systems programmatically (`os`, `shutil`) before modeling.
* **Students** struggling to understand why their model gets 99% training accuracy but fails in the real world (Overfitting).

---

## 🛠️ Skills Demonstrated

### 1. Data Engineering & Python Scripting
* **File Manipulation:** Using `os` and `shutil` to programmatically create directories, move files, and check for corruption (zero-length files).
* **Custom Splitting:** Implementing a robust algorithm to shuffle and split data into **Training (90%)** and **Validation (10%)** sets manually.

### 2. TensorFlow Image Pipeline
* **ImageDataGenerator:** Implementing a streaming pipeline that loads images from the hard drive in batches, resizes them to `150 X 150`, and normalizes pixel values (`1./255`) on the fly.

### 3. Deep Learning Architecture
* **Convolutional Design:** Building a deep network with **3 Convolutional Layers** (`16, 32, 64` filters) to capture increasingly complex features (`edges`, `shapes`, `ears/noses`).
* **Binary Classification:** Utilizing the `Sigmoid` activation function and `Binary Crossentropy` loss.

---

## ⚠️ Common Mistakes Explored in This Module

During the development of this pipeline, several common pitfalls were addressed:
1. **Data Leakage:** Failing to separate Training and Validation data *before* training. If the model sees validation data during training, the evaluation is meaningless.
2. **Corrupted Files:** Real-world datasets often have empty files. We implemented a check `if os.path.getsize(file) > 0` to prevent the training from crashing.
3. **Aspect Ratio Distortion:** We resized all images to a square $150 \times 150$. This squashes wide images (like a dachshund), potentially confusing the model. (Addressed in future modules with Augmentation).

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
cd Module1_Exploring_a_Larger_Dataset
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

1. Go to: [Google Collab](https://colab.research.google.com)
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

---

## ❓ Problem Statement

**The Real World is Messy.**
Introductory courses often provide clean data. Here, we face the challenge of **Raw Image Data**. We are given a chaotic directory of 25,000 images with no CSV labels, varying aspect ratios, and potential file corruptions.

### Key Challenges:
1.  **Feature Extraction:** Unlike simple digits, real-world images have complex backgrounds. The model must learn to ignore the grass behind the dog and focus on the dog itself.
2.  **Memory Constraints:** We cannot load 25,000 high-res images into RAM at once. We need a **Streaming Pipeline**.
3.  **Overfitting Risk:** With a complex model and limited data, the network is prone to memorizing specific pixels rather than learning general features.

> *Note: For this module, we use a filtered subset provided by Microsoft Research to ensure compatibility with free Colab instances.*

---

## 💾 Dataset
We utilize the famous **Kaggle Cats vs. Dogs Dataset**.
* **Size:** 25,000 Color Images (RGB).
* **Classes:** Binary (Cat = 0, Dog = 1).
* **Characteristics:** Unlike MNIST (which is 28x28 grayscale), these images are:
    * **High Resolution:** Up to 500x500 pixels.
    * **Varied Scenes:** Indoor, outdoor, different lighting, different angles.
    * **Varied Subjects:** Different breeds, colors, and poses.

If desired you can access full dataset [Cats vs Dogs on Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats).

> *Note: For this module, we use a filtered subset provided by Microsoft Research to ensure compatibility with free Colab instances.*

---

## 📉 Deep Dive: The Overfitting Problem
This module is a textbook example of **Overfitting**.

### The Concept (The "_Lumberjack vs. Sailor_" Analogy)
Imagine you show a child 5 photos of Lumberjacks and 5 photos of Sailors.
* All the **Lumberjacks** in the photos happen to be wearing **Grey Caps**.
* None of the **Sailors** are wearing caps.
* The child (the model) incorrectly learns: **"If it has a Grey Cap, it is a Lumberjack."**

This is **Overfitting**. The model latched onto a coincidental feature (the cap) instead of the true features (the axe, the flannel shirt, the forest background).

### Observing It in This Project


<p align="center">
  <img src="overfitting_graph.png" />
</p>

In this notebook, you will observe a specific phenomenon in the training graphs:
1.  **Training Accuracy::** Keeps going up, hitting 99% or even 100%. The model has memorized the training photos perfectly.
2.  **Validation Accuracy:** Stalls at ~70-75% and refuses to go higher.
3.  **Validation Loss:** Starts to **Explode (Go Up)**. As the model becomes more confident in its wrong assumptions (like the "Grey Cap"), its errors on the validation set become more severe.

**Why it happened here:**
We have a powerful model (lots of neurons) and no constraints. It found the easiest way to minimize loss on the training set (memorizing pixels) rather than learning general concepts of "Dog-ness."

---

### ⚙️ Technical Implementation

### 1. Data Preprocessing (`ImageDataGenerator`)
Because real-world datasets are too large for RAM, we use generators to stream data.

```python
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/tmp/cats-v-dogs/training/',
    target_size=(150, 150),  # Resizes images to 150x150
    batch_size=128,
    class_mode='binary'
)
```

### 2. Model Architecture

We use a stack of Conv2D and MaxPooling layers.

```python
model = tf.keras.models.Sequential([
    # Input: 150x150 with 3 Color Channels
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    
    # Sigmoid Output: Returns value 0-1 (Cat vs Dog)
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

#### Convolutional Neural Networks (CNNs)

CNNs preserve spatial structure and learn visual features such as edges and shapes.

- **Intuition:** Dense networks see pixels; CNNs see patterns.

#### Layer Explanation

- **Conv2D (Convolutional Layers with ReLU Activation)**: Convolutional layers are the core building blocks of CNNs. Instead of connecting every pixel to every neuron, they operate on small local regions of the image.
  - Applies multiple learnable 3×3 filters (kernels) that slide across the image
  - Each filter performs a dot product between its weights and a small patch of the image
  - The result of each filter is a feature map highlighting where a specific pattern appears
  - Early convolutional layers typically learn simple features:
    - Edges (horizontal, vertical, diagonal)
    - Corners
    - Basic textures
  - Deeper convolutional layers combine earlier features to detect more complex patterns:
    - Shapes
    - Object parts
    - High-level visual structures
  - Weight sharing means the same filter is applied across the entire image:
    - Greatly reduces the number of parameters
    - Makes the model efficient and scalable
  - ReLU activation:
    - Keeps positive feature responses
    - Removes negative values
    - Introduces non-linearity so the model can learn complex patterns
  - **🔑 Key intuition:**
    - Convolutional layers learn what to look for and where it appears in the image.

- **MaxPooling2D**: Pooling layers reduce the size of feature maps while retaining the most important information.
  - Operates on small windows (commonly 2×2) of each feature map
  - Replaces each window with its maximum value
  - Reduces spatial dimensions (width and height) by a factor of 2
  - Keeps the strongest feature activations while discarding weaker ones
  - Makes the network:
    - Faster (fewer computations)
    - Less memory-intensive
    - More resistant to small shifts or distortions in the image
  - Helps prevent the network from focusing too much on exact pixel locations
  - **🔑 Key intuition:**
    - Pooling answers whether a feature is present, not exactly where it is.

- **Flatten**
  - Converts the 2D feature maps into a 1D vector
  - Prepares convolutional features for the dense layers
- **Dense (fully connected layer with ReLU activation)**
  - Learns higher-level combinations of extracted features
  - Acts as a classifier based on convolutional features
- **Dense (1 neuron, Sigmoid output)**
  - Outputs a single probability score between 0 and 1.
  - Used specifically for binary classification.
  - **Thresholding:** Typically, values < 0.5 are predicted as Class 0, and values >= 0.5 are Class 1.

The number of convolutional filters and dense neurons is varied across experiments to demonstrate how **model capacity** affects feature learning and classification performance.

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
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
```

---

## 🏋️ Training the Model

### Training with Generators

In this module, training uses a **Python Generator**, differing from the standard model.fit(x, y) approach used in Course 1. This allows for **memory-efficient** streaming of data.


Since data is being streamed, we use the fit method with the generator object.

The `fit` Method with Generators
Because our dataset (25,000 images) is too large to fit into RAM, we use the train_generator to stream images in batches.

```python
history = model.fit(
    train_generator,
    epochs=15,
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

**The "Divergence" Phenomenon**

When you run the provided notebooks, you will see a clear pattern in the history plots:

1. **Training Accuracy** climbs steadily to 99% or 1.00.
2. **Validation Accuracy** improves for a few epochs, then plateaus at ~70-75%.
3. **Validation Loss** actually starts to increase after Epoch 5-10.

**What this tells us**: The model has stopped learning "what a dog looks like" and started memorizing "what image #424 looks like." This divergence is the mathematical proof of overfitting.

---

## 📊 Model Evaluation Metrics

In this module, we track two primary metrics. It is crucial to understand why they behave differently. 

1. Accuracy (`accuracy` & `val_accuracy`)

- **Definition**: The percentage of images classified correctly.
- **Goal**: Maximize this (closer to 1.0).
- **Observation**: Training accuracy converges to 1.0 (100%), while validation accuracy plateaus. This gap represents the model's inability to generalize.

2. Loss (`loss` & `val_loss`)

- **Definition**: A penalty score calculated by **Binary Crossentropy**. It punishes the model for being "confident and wrong."
- **Goal**: Minimize this (closer to 0.0).
- **Observation**:
  - If the model predicts "Dog" (1.0) and it is a Dog: **Low Loss**.
  - If the model predicts "Dog" (1.0) and it is a Cat: **Massive Loss**.
- **The Warning Sign**: In this module, you will see `val_loss` **explode** (go up to 1.0, 2.0, or higher) even if `val_accuracy` stays flat. This means the model is becoming **extremely confident in its wrong predictions**.

---

## 📏 Metrics in This Module

We configure the model to track these specific metrics during compilation:

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(
    optimizer=RMSprop(learning_rate=0.001),
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

## 📓 Notebooks & Exercises

### [1. Lesson Notebook: Overfitting Concepts](Course_2_Part_2_Lesson_2_Notebook.ipynb)
A guided lesson demonstrating the Cats vs. Dogs classifier.
* **Highlight:** Visualizing how the Training Accuracy skyrockets while Validation Accuracy stalls, proving the need for better strategies (like Augmentation in the next module).

### [2. Exercise: Building the Pipeline](Exercise_1_Cats_vs_Dogs_Question.ipynb)
**The Assignment:** Build the entire data processing pipeline from scratch.
* **Task 1:** Unzip the Kaggle dataset.
* **Task 2:** Write a Python function `split_data()` to shuffle images and move them into `training/` and `testing/` directories.
* **Task 3:** Train a CNN to achieve >90% accuracy (or observe why it fails to reach it without augmentation).

---

## 📝 Summary
In this module, I successfully built a full-stack image classification pipeline. I wrote a custom Python script to organize a raw dataset of 25,000 images, splitting them into training and testing sets.

I trained a Convolutional Neural Network that achieved nearly **100% accuracy on the training set**, but only **~70% on the validation set**. This clearly demonstrated the phenomenon of **Overfitting**, proving that "High Accuracy" is meaningless if the model cannot generalize to new data.

---

## 📘 Files in This Module

```
📁 Module1_Exploring_a_Larger_Dataset
├── 📓 Course_2_Part_2_Lesson_2_Notebook.ipynb
├── 📓 Exercise_1_Cats_vs_Dogs_Question-FINAL.ipynb
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🛑 Limitations
* **No Data Augmentation:** The model only saw images exactly as they were. It cannot recognize a dog if the image is rotated or zoomed in.
* **Overfitting:** The current model is not production-ready because the gap between Training and Validation accuracy is too wide.
* **Basic Architecture:** We used a standard stack of Conv2D layers without advanced techniques like Dropout or Transfer Learning.

> **Next Step:** In Module 2, we will implement **Data Augmentation** to fix the overfitting issue.

---

## 📚 Further Reading

- [Kaggle Cats vs Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats)
- [TensorFlow ImageDataGenerator Docs](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- [Understanding Overfitting (TensorFlow Guide)](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
- [Visualizing ConvNets](https://distill.pub/2017/feature-visualization/)
