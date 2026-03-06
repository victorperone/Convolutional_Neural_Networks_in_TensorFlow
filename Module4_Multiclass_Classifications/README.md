# Module 4 Multiclass Classifications

> **From Binary to Multi-Class Learning**: Building Deep CNNs for 3-Class and 26-Class Image Classification with Data Augmentation and Softmax Outputs.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-Image_Data_Generator-red.svg)](https://keras.io/preprocessing/image/) [![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/) [![NumPy](https://img.shields.io/badge/Numpy-1.x-blue.svg)](https://numpy.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-orange.svg)](https://matplotlib.org/)

---

## 🧭 Overview

In previous modules, we focused on:
- Binary classification
- Overfitting control
- Transfer learning with pretrained CNNs

In **Module 4**, we expand to **Multi-Class Classification**.
This module introduces:
- 3-class image classification (Rock–Paper–Scissors)
- 26-class classification (Sign Language MNIST)
- Softmax outputs
- Categorical and Sparse Categorical loss functions
- CSV-based dataset parsing
- Memory-efficient generators for large datasets

We move from “Is this A or B?” to:
> “Which of 26 possible classes does this image belong to?”

---

## 🎯 Learning Objectives

By the end of this module, you will understand how to:
- Build CNNs for multi-class classification
- Use `softmax` activation correctly
- Choose between `categorical_crossentropy` and `sparse_categorical_crossentropy`
- Implement data augmentation for multi-class datasets
- Load image data from directory structures
- Parse pixel data from CSV files
- Expand dimensions for grayscale CNN inputs
- Train efficiently using generators
- Evaluate multi-class performance

---

## 💼 Why This Project Matters

Most real-world problems are not binary.

Examples:
- Traffic sign recognition
- Medical image diagnosis (multiple conditions)
- Facial emotion detection
- Hand gesture recognition
- Product classification

This module demonstrates:
- Multi-class modeling skills
- Dataset preprocessing versatility
- Adaptability to different data formats
- Correct loss function selection
- Practical CNN scaling strategies

---

## 👥 Who This Module Is For

This module is designed for:
- Developers moving from binary to multi-class classification
- Engineers working with datasets containing more than two categories
- Students learning how softmax and categorical losses work
- Practitioners handling datasets stored in different formats (folders vs CSV)
- Anyone wanting to understand how CNN complexity scales with class count
If Module 3 focused on **feature reuse**, Module 4 focuses on **classification scaling**.

---

## 🛠️ Skills Demonstrated

### 1. Multi-Class CNN Architecture Design

- 3-class classifier (Rock–Paper–Scissors)
- 26-class classifier (Sign Language MNIST)
- Softmax output layer for probability distribution
Demonstrates:
- Understanding of multi-class output layers
- Correct architectural scaling

### 2. Proper Loss Function Selection

- `categorical_crossentropy` (one-hot labels)
- `sparse_categorical_crossentropy` (integer labels)
Demonstrates:
- Understanding of label encoding
- Correct loss-output pairing

### 3. Advanced Data Augmentation

Applied:
- Rotation
- Width/Height shift
- Zoom
- Shear
- Horizontal flip
- Fill modes
Demonstrates:
- Overfitting control
- Dataset expansion strategies

### 4. Generator-Based Training

- Directory-based generators
- Numpy-based generators
- Memory-efficient streaming
Demonstrates:
- Scalable training techniques
- Proper batch processing

### 5. CSV Image Dataset Parsing

- Read pixel data from CSV
- Converted strings to floats
- Reshaped 784 pixels into 28x28
- Expanded dimensions to (28, 28, 1)
Demonstrates:
- Raw dataset preprocessing
- Data transformation pipeline design

---

## ⚠️ Common Mistakes Explored in This Module

- **Using Sigmoid Instead of Softmax**
  - Sigmoid is for binary classification.
  - Multi-class problems require softmax to produce probability distributions.
- **Wrong Loss Function Selection**
  - `categorical_crossentropy` requires one-hot encoded labels.
  - `sparse_categorical_crossentropy` requires integer labels.
  - Mixing them causes incorrect training behavior.
- **Forgetting to Expand Dimensions**
  - Grayscale images must be reshaped from `(28, 28)` to `(28, 28, 1)` for Conv2D layers.
- **Overusing Augmentation**
  - Excessive augmentation can distort small images (28x28).
  - Augmentation should preserve semantic meaning.
- **Ignoring Class Imbalance**
  - Multi-class models can become biased toward dominant classes.
- **Underestimating Model Capacity Needs**
  - Increasing number of classes requires more representational power.

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
cd Module4_Multiclass_Classifications
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

In previous modules, we solved:
> “Is this image Class A or Class B?”

But real-world applications rarely involve only two categories.

**The Challenge:**
How can we design CNNs that:
- Correctly classify images into multiple categories
- Output valid probability distributions
- Scale from 3 classes to 26 classes
- Maintain generalization while increasing complexity

**The Solution:**
- Replace sigmoid with softmax
- Use appropriate multi-class loss functions
- Adapt architecture capacity to the number of classes
- Apply augmentation strategically
- Use efficient generators for large datasets

---

## 💾 Datasets

### 🪨 Rock–Paper–Scissors Dataset
- 3 classes
- Colored images (150x150x3)
- Real-world hand gestures
- Augmented training
- Separate test set
**Challenge:**
Hand orientation and lighting variation

### 🤟 Sign Language MNIST

- 26 classes (A–Z hand signs)
- Grayscale images (28x28x1)
- CSV formatted pixel dataset
- 27,455 training images
- 7,172 test images

**Challenge:**
- Small resolution + many classes

---

## 📉 Deep Dive: Why Multi-Class Classification Is Harder

### 1️⃣ Probability Distribution Constraint

In binary classification:
- The model predicts a single probability.
In multi-class classification:
- The model must distribute probability mass across all classes.
- Increasing the number of classes increases uncertainty.
Example (3 classes):
```
[0.1, 0.7, 0.2]
```

Example (26 classes):
- The model must distinguish between 26 subtle variations.
- Decision boundaries become significantly more complex.

### 2️⃣ Capacity vs Class Count

As class count increases:
- Feature representation must become more discriminative.
- The final Dense layer grows in size.
- Training becomes more sensitive to learning rate.
Higher class count → higher risk of:
- Confusion between similar classes
- Overfitting
- Slower convergence

### 3️⃣ Dataset Format Complexity

This module also introduces two dataset types:
- **Directory-based image datasets**
- **CSV-based pixel datasets**
- 
Handling both demonstrates adaptability in preprocessing pipelines — a critical production skill.

---

## ⚙️ Technical Implementation

### 1️⃣ Rock–Paper–Scissors CNN

#### Architecture

```text
Input (150x150x3)
↓
Conv2D(64) + MaxPool
↓
Conv2D(64) + MaxPool
↓
Conv2D(128) + MaxPool
↓
Conv2D(128) + MaxPool
↓
Flatten
↓
Dropout(0.5)
↓
Dense(512, ReLU)
↓
Dense(3, Softmax)
```

**🔍 Layer-by-Layer Breakdown**
- Input Layer (150x150x3)
  - RGB image resized to 150×150.
  - 3 channels represent Red, Green, Blue.

**🧩 Convolutional Blocks**
Each Conv block performs:
1. Convolution
    - Learns spatial filters (edges, shapes, textures).
    - 64 → 128 filters increase feature depth.
2. ReLU Activation
    - Introduces non-linearity.
    - Prevents vanishing gradients.
3. MaxPooling
    - Reduces spatial resolution.
    - Preserves strongest activations.
    - Reduces computational cost.

Progression:
  - Early layers → detect edges and simple patterns
  - Middle layers → detect shapes (hand contours)
  - Deeper layers → detect abstract features (gesture structure)

**🔄 Flatten Layer**

Transforms 3D feature maps into a 1D vector so it can be passed to Dense layers.

Without flattening, fully connected layers cannot process spatial tensors.

**🎯 Dropout (0.5)**
- Randomly disables 50% of neurons during training.
- Prevents co-adaptation.
- Strong regularization technique.
- Especially important since this dataset is relatively small.

**🧠 Dense(512, ReLU)**
- Learns high-level combinations of extracted features.
- Acts as the classifier head.
- 512 neurons give sufficient capacity to model gesture differences.

**🎯 Dense(3, Softmax)**

Final output layer.

**Why Softmax?**
Softmax converts raw outputs (logits) into a probability distribution:

```
[0.1, 0.7, 0.2]
```

Which represents probability distribution across 3 classes.
Properties:
- All values are between 0 and 1.
- All probabilities sum to 1.
- The highest value represents the predicted class.

This is required because the problem is **multi-class (3 categories)**.

### 2️⃣ Sign Language CNN

Constraint:
- Maximum 2 Conv2D layers
- Forces efficient architecture design.
- Encourages compact feature extraction.

#### Architecture

```text
Input (28x28x1)
↓
Conv2D(64) + MaxPool
↓
Conv2D(64) + MaxPool
↓
Flatten
↓
Dense(128, ReLU)
↓
Dense(26, Softmax)
```

#### 🔍 Key Differences from RPS Model

| Rock–Paper–Scissors  | Sign Language          |
| -------------------- | ---------------------- |
| 150x150 RGB images   | 28x28 Grayscale images |
| 4 Conv layers        | 2 Conv layers          |
| 3 classes            | 26 classes             |
| Larger feature depth | Smaller spatial input  |


**Why Sparse Categorical Crossentropy?**

Labels are integers:

```
0, 1, 2, 3 ,..., 25
```

Not one-hot encoded.

Using:

```
loss='sparse_categorical_crossentropy'
```

Advantages:
- No need for manual one-hot encoding.
- More memory efficient.
- Cleaner preprocessing pipeline.
- If labels were one-hot encoded, we would use: `categorical_crossentropy`

---

## 🧠 Model Compilation

Compilation defines:
- The optimization strategy
- The error calculation method
- The performance metrics

### Rock–Paper–Scissors

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)
```

**Why categorical_crossentropy?**

Because:
- Output layer has 3 neurons
- Labels are one-hot encoded (via class_mode='categorical')
- Loss compares predicted probability distribution to true distribution

**Why RMSprop?**
- Adaptive learning rate per parameter
- Handles noisy gradients well
- Efficient for CNN training
- Often performs well on medium-sized image datasets

### Sign Language

```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

**Why Adam?**
- Combines momentum + adaptive learning rates
- Faster convergence for smaller models
- Good default optimizer for constrained architectures

---

## 🏋️ Training the Model

Training is performed using **data generators**, not raw arrays.

### Why Use Generators?

- Memory efficiency (no need to load entire dataset into RAM)
- Real-time data augmentation
- Scalable pipeline
- Suitable for large datasets

### Training Workflow

1. Images are loaded in batches.
2. Data augmentation is applied (rotation, zoom, shifts).
3. Forward pass computes predictions.
4. Loss is calculated.
5. Backpropagation updates weights.
6. Validation is evaluated at the end of each epoch.

### Tracked During Training

- `accuracy`
- `val_accuracy`
- `loss`
- `val_loss`

These values are stored in:

```python
history.history
```

---

## 📉 Model Evaluation

Evaluation focuses on **generalization**, not just raw accuracy.

We analyze:
- Training vs Validation Accuracy
- Training vs Validation Loss
- Overfitting patterns

### 1️⃣ Detecting Overfitting

**Healthy Model:**

- Training and validation accuracy increase together.
- Loss decreases steadily.
- Small gap between training and validation curves.

**Overfitting Model:**

- Training accuracy → ~100%
- Validation accuracy plateaus or decreases
- Validation loss increases

This gap indicates the model memorized training data instead of learning general patterns.

### 2️⃣ Accessing Training History
```python
# Example of accessing history for evaluation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
```

These arrays allow us to:
- Plot learning curves
- Identify optimal stopping point
- Diagnose instability

### 3️⃣ Visualizing Intermediate Representations

By extracting outputs from convolutional layers, we can visualize:
- Edge detectors
- Shape activations
- Texture recognition

This provides interpretability — we can see what the network has learned.

---

## 📊 Model Evaluation Metrics

In this module, we track two primary metrics. It is crucial to understand why they behave differently. 

### 1️⃣ Accuracy (`accuracy`, `val_accuracy`)

- **Definition**: The percentage of images classified correctly.
- **Goal**: Maximize `val_accuracy` (closer to 1.0).
- **Observation**: Training accuracy converges to 1.0 (100%), while validation accuracy plateaus. This gap represents the model's inability to generalize.

For multi-class classification:

```
Correct Predictions / Total Samples
```

**Interpretation**

- High training accuracy + low validation accuracy → Overfitting
- Both high → Good generalization
- Both low → Underfitting

Goal:
```
Maximize val_accuracy
```


### 2️⃣ Loss (`loss`, `val_loss`)

Loss measures how wrong the model is — not just whether it is wrong.
Loss checks:

> How confident was the model in its prediction — and how wrong was it?

For multi-class problems:
- categorical_crossentropy
- sparse_categorical_crossentropy

Both penalize:
- Confident and wrong predictions heavily
- Slightly incorrect predictions moderately

Example:

True label: Rock

Prediction: `[0.99, 0.005, 0.005]` → Very low loss

Prediction: `[0.01, 0.98, 0.01]` → Very high loss

#### 🔬 Crossentropy in Multi-Class Classification

In this module, we use one of two mathematically related loss functions:
- categorical_crossentropy
- sparse_categorical_crossentropy

Both are designed for **multi-class classification with Softmax outputs**.

#### 🧠 What Crossentropy Actually Measures

Crossentropy compares:
- The true probability distribution
- The predicted probability distribution

Softmax outputs a probability vector like: `[0.1, 0.7, 0.2]`

Crossentropy measures the distance between the predicted distribution and the true distribution.

Mathematically:

$$
Loss = - \sum_{i=1}^{C} y_i \log(p_i)
$$

Where:
- $y_i$ = true distribution
- $p_i$ = predicted probability
- Logarithm heavily penalizes confident mistakes

#### 🎯 Categorical Crossentropy

**When To Use It**

Use when labels are **one-hot encoded**.

Example (3 classes):

True label = Rock

One-hot encoding:
```
[1, 0, 0]
```

If model predicts:
```
[0.99, 0.005, 0.005]

```

Loss becomes:

$$
-(1 \cdot \log(0.99) + 0 + 0)
$$

Very small loss.

**Why One-Hot Encoding?**

One-hot encoding transforms class index into a probability distribution.

For 3 classes:

| Class    | One-Hot |
| -------- | ------- |
| Rock     | [1,0,0] |
| Paper    | [0,1,0] |
| Scissors | [0,0,1] |

This matches the shape of the Softmax output.

**How It Works Internally**

Categorical crossentropy:

1. Multiplies each predicted probability by its true label value.
2. Since only one value is 1 (others are 0), it isolates the correct class.
3. Applies negative log.

This means:

Only the predicted probability of the correct class matters.

#### Sparse Categorical Crossentropy

**When To Use It**

Use when labels are **integer encoded**: `0, 1, 2, 3, ..., 25`

Instead of: `[0, 0, 0, 0, 0, 1, 0, 0, 0]`

**What Makes It Different?**

Sparse categorical crossentropy:
- Does NOT require one-hot encoding.
- Internally converts the integer label into a one-hot representation.
- Then computes the same crossentropy formula.

So mathematically:

```
Both compute the same crossentropy formula.
The only difference is how labels are represented before computation.
```

The difference is purely input format.

#### ⚖️ Key Differences

| Feature                | categorical_crossentropy     | sparse_categorical_crossentropy |
| ---------------------- | ---------------------------- | ------------------------------- |
| Label Format           | One-hot encoded              | Integer encoded                 |
| Memory Usage           | Higher (stores full vectors) | Lower (stores single integer)   |
| Preprocessing Required | Must one-hot encode          | No encoding needed              |
| Mathematical Behavior  | Identical                    | Identical                       |

#### 🚨 Why Using the Wrong One Breaks Training

If you:

- Use categorical_crossentropy with integer labels 
  - → The loss function expects a vector but receives a scalar 
  - → Results in shape mismatch or incorrect gradients

If you:

- Use sparse_categorical_crossentropy with one-hot labels 
  - → It interprets the vector as class index 
  - → Completely wrong loss calculation

Correct pairing is critical.

#### 📉 Why Loss Punishes Confident Mistakes More

Look at the log function behavior:

| Prediction | log(p)  | Loss       |
| ---------- | ------- | ---------- |
| 0.99       | ~ -0.01 | Very small |
| 0.7        | ~ -0.36 | Moderate   |
| 0.1        | ~ -2.30 | Large      |
| 0.01       | ~ -4.60 | Massive    |

If the model predicts: `[0.01, 0.98, 0.01]`

And the true label is class 0

Loss becomes: $-log(0.01)$

Very large penalty.

This forces the network to:
- Reduce overconfidence
- Calibrate probabilities properly
- Learn sharper decision boundaries

#### 🎓 Why Crossentropy + Softmax Is the Correct Pair

Softmax ensures:

$$
\sum P_i = 1
$$

Crossentropy assumes:
- The outputs form a probability distribution.
- The target is a probability distribution.
If you used:
- MSE instead of crossentropy
- Or sigmoid for multi-class

Training would be unstable or slower.

Softmax + Crossentropy is mathematically derived from **maximum likelihood estimation**.

#### 🧠 Practical Insight

In real-world systems:
- Use `sparse_categorical_crossentropy` when labels come as integers (most common case).
- Use `categorical_crossentropy` when your pipeline already outputs one-hot labels (e.g., `class_mode='categorical'` in generators).

In this module:
- Rock–Paper–Scissors → one-hot → `categorical_crossentropy`
- Sign Language → integer labels → `sparse_categorical_crossentropy`

### 🔎 Important Takeaway

Accuracy tells you:

> Did we predict correctly?

Loss tells you:

> How confidently correct or incorrect were we?

**Two models may have 90% accuracy — but the one with lower loss is better calibrated and more reliable.**

---

## 📏 Metrics in This Module

Unlike binary classification modules, this module uses multi-class metrics.

Compilation is configured as:

```python
model.compile(
    optimizer='rmsprop' or 'adam',
    loss='categorical_crossentropy' or 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Summary

- **Optimizer** → Controls how weights update.
- **Loss Function** → Measures prediction error.
- **Accuracy** → Measures classification performance.

For multi-class CNNs:
- Softmax + Crossentropy is the mathematically consistent pairing.
- Correct loss selection is critical for stable convergence.
- Validation accuracy is the most important generalization signal.

---

## 🔑 Key Concepts

- Softmax activation
- Probability distributions
- Multi-class loss functions
- Sparse vs One-hot encoding
- CSV image parsing
- High-class-count scaling challenges

---

## 📊 Results

| Model               | Classes | Train Accuracy | Validation Accuracy | Epochs |
| ------------------- | ------- | -------------- | ------------------- | ------ |
| Rock–Paper–Scissors | 3       | ~95–99%        | ~90–95%             | 25     |
| Sign Language CNN   | 26      | ~90–95%        | ~88–92%             | 2–5    |


### Interpretation

- Multi-class learning requires more capacity
- Increasing the number of classes makes the classification task harder, which typically reduces achievable validation accuracy under the same model capacity.
- Proper loss function selection is critical
- Augmentation improves generalization

---

## 💡 What I learned

- Multi-class classification requires architectural and loss-function changes. 
- Softmax transforms raw logits into meaningful probability distributions. 
- Sparse categorical crossentropy simplifies training when labels are integers. 
- Increasing the number of classes increases modeling complexity. 
- Data preprocessing is as important as model architecture. 
- Generators enable scalable and memory-efficient training. 
- Small resolution images (28x28) require careful augmentation.

Most importantly:

> Scaling from 2 classes to 26 classes is not just a minor adjustment — it fundamentally changes how the network learns decision boundaries.

---

## 📓 Notebooks & Exercises

- **Course Notebook**: Builds a 3-class CNN for Rock–Paper–Scissors using directory-based image loading and softmax classification.
- **Exercise**: Implements a 26-class CNN for Sign Language MNIST:
  - Parses pixel data from CSV
  - Reshapes images to 4D tensors
  - Applies augmentation
  - Builds constrained CNN architecture (2 Conv layers max)
  - Trains with sparse categorical crossentropy
  - Evaluates training vs validation accuracy
Each notebook reinforces:
- Correct loss-function selection
- Proper output layer design
- Multi-class probability modeling
- Generalization under increased class complexity

---

## 📝 Summary

In this module, we evolved from:
Binary classification → Multi-class classification

We learned:
- How softmax differs from sigmoid
- How loss functions change with label encoding
- How to process datasets from folders and CSV files
- How to scale CNNs for more classes
- How augmentation supports generalization in high-class problems

Module 4 completes the transition from:

> "Can this network separate two categories?"

To:

> "Can this network correctly classify across many possible outcomes?"

---

## 📘 Files in This Module

```
📁 Module4_Multiclass_Classifications
├── 📓 Course_2_Part_8_Lesson_2_Notebook_(RockPaperScissors).ipynb
├── 📓 Exercise_4_Multi_class_classifier_Question-FINAL.ipynb
├── 🗜️ sign_language_mnist.zip
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🛑 Limitations

- Limited epochs in Sign Language model
- No fine-tuning experiments
- No confusion matrix visualization
- No model compression
- No transfer learning comparison for 26-class case

---

## 📚 Further Reading

- [Sign Language Dataset](https://www.kaggle.com/datamunge/sign-language-mnist)
- [Softmax Function](https://en.wikipedia.org/wiki/Softmax_function)
- [Crossentropy Loss](https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451)
- [Multi-Class Classification](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library)
- [CNN Architectures](https://www.tensorflow.org/tutorials/images/cnn)
