# Convolutional_Neural_Networks_in_TensorFlow

> **Advanced Computer Vision:** From Overfitting to Transfer Learning.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-High%20Level%20API-red.svg)](https://keras.io/) [![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

This repository documents my implementation and analysis of convolutional neural network architectures built throughout the Coursera course **Convolutional Neural Networks in TensorFlow**.

It covers CNN fundamentals, robust image classification pipelines, data augmentation strategies, and transfer learning techniques using TensorFlow and Keras.


🔗 **Course link:**  

[Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow)

---

## 🎯 Focus of This Repository

Rather than simply completing course exercises, this repository emphasizes:

- Architectural reasoning
- Loss-function alignment
- Overfitting diagnostics
- Data augmentation strategies
- Transfer learning implementation

---

## 📘 Course Overview

This course, which is the second of the DeepLearning.AI TensorFlow Developer Specialization, focuses on advanced techniques for improving the computer vision models developed in [Course 1](https://www.coursera.org/learn/introduction-tensorflow). It covers working with real-world images of different shapes and sizes, visualizing an image's journey through convolutions to understand how a computer _"sees"_ information, trace loss and accuracy, and explore strategies to avoid overfitting, including boosting and dropping. Finally, it introduces transfer learning and how learned features can be extracted and reused across models.

Andrew Ng’s Machine Learning and Deep Learning Specialization courses provide a strong foundation in the core principles of machine learning and deep learning. Building on this foundation, the DeepLearning.AI TensorFlow Developer Specialization emphasizes the practical implementation of these principles using TensorFlow, enabling learners to design and deploy scalable models for real-world applications.

This course is recommended after completing [Introduction to TensorFlow](https://www.coursera.org/learn/introduction-tensorflow).

My work for that course is available on GitHub [here](https://github.com/victorperone/Introduction_to_TensorFlow_for_Artificial_Intelligence).

### **Course Modules**
1. **Exploring a Larger Dataset**
2. **Augmentation: A Technique to Avoid Overfitting**
3. **Transfer Learning**
4. **Multiclass Classifications**

---

## 💡 Why This Matters

Understanding CNN architecture design, overfitting behavior, and transfer learning is foundational for:
- Medical imaging
- Autonomous systems
- Visual inspection systems
- Large-scale image classification platforms

---

## 🛠️ Key Concepts Mastered

| Concept | Description |
| :--- | :--- |
| **Data Augmentation** | Artificially expanding datasets by rotating, zooming, and shifting images to prevent overfitting. |
| **Transfer Learning** | Leveraging pre-trained models (like InceptionV3) to solve tasks with limited data. |
| **Dropout** | A regularization technique to force network redundancy and robustness. |
| **Softmax & Crossentropy Alignment** | Designed multi-class classifiers using proper loss–activation pairing and label encoding strategies. |

---

## 📊 Results & Highlights

- Achieved ~98% training accuracy on augmented Cats vs Dogs dataset
- Successfully reduced overfitting gap using Dropout + Data Augmentation
- Implemented Transfer Learning with InceptionV3, reducing training time and improving validation stability.
- Built a 26-class Sign Language classifier with constrained CNN architecture

---

## 🧠 Engineering Lessons Learned

- Model capacity must scale with dataset complexity
- Validation metrics matter more than training metrics
- Loss curves reveal more than accuracy alone
- Transfer learning drastically reduces required training time
- Data pipelines are as important as model architecture

---

## 🔭 How This Prepares Me for Real-World ML

This course strengthened my ability to:

- Design scalable image classification systems
- Diagnose overfitting using training history
- Choose appropriate loss functions for different tasks
- Build memory-efficient data pipelines
- Leverage pretrained models for rapid development

---

## 📁 Repository Content

This repository includes:
- Jupyter notebooks for each module 
- Code examples using TensorFlow and Keras 
- Notes and explanations from the course 

---

## 📂 Folder Structure

<pre>
📦 Convolutional_Neural_Networks_in_TensorFlow
├── 📁 <a href="https://github.com/victorperone/Convolutional_Neural_Networks_in_TensorFlow/tree/main/Module1_Exploring_a_Larger_Dataset">Module1_Exploring_a_Larger_Dataset</a>
│   ├── 📓 Course_2_Part_2_Lesson_2_Notebook.ipynb
│   ├── 📓 Exercise_1_Cats_vs_Dogs_Question-FINAL.ipynb
│   ├── 🖼️ overfitting_graph.png
│   ├── 📄 requirements.txt
│   └── 📄 README.md
├── 📁 <a href="https://github.com/victorperone/Convolutional_Neural_Networks_in_TensorFlow/tree/main/Module2_Augmentation-A_Technique_to_Avoid_Overfitting">Module2_Augmentation-A_Technique_to_Avoid_Overfitting</a>
│   ├── 📓 Course_1_Part_4_Lesson_2_Notebook.ipynb
│   ├── 📄 requirements.txt
│   └── 📄 README.md
├── 📁 <a href="https://github.com/victorperone/Convolutional_Neural_Networks_in_TensorFlow/tree/main/Module3_Transfer_Learning">Module3_Transfer_Learning</a>
│   ├── 📓 Course_1_Part_6_Lesson_2_Notebook.ipynb
│   ├── 📓 Course_1_Part_6_Lesson_2_Notebook.ipynb
│   ├── 📓 Course_1_Part_6_Lesson_3_Notebook.ipynb
│   ├── 📄 requirements.txt
│   └── 📄 README.md
├── 📁 <a href="https://github.com/victorperone/Convolutional_Neural_Networks_in_TensorFlow/tree/main/Module4_Multiclass_Classifications">Module4_Multiclass_Classifications</a>
│   ├── 📓 Course_1_Part_8_Lesson_2_Notebook_Horses_Humans_Convet.ipynb
│   ├── 📓 Course_1_Part_8_Lesson_3_Notebook_Horses_Humans_with_Validation.ipynb
│   ├── 📓 Course_1_Part_8_Lesson_4_Notebook_Horses_Humans_Compact_Images.ipynb
│   ├── 📓 Semana_4_Exercicio.ipynb
│   ├── 📄 Exercise4-Question.json
│   ├── 📄 requirements.txt
│   └── 📄 README.md
│   └── 📁 datasets/
├── 📁 <a href="https://github.com/victorperone/Convolutional_Neural_Networks_in_TensorFlow/tree/main/What_I_Learned">What_I_Learned</a>
│   ├── 📓 Introduction_to_TensorFlow_Wrap_Up.ipynb
│   ├── 📓 Introduction_to_TensorFlow_Wrap_Up.ipynb
│   ├── 🧠🤖 my_horse_human_model.h5
│   ├── 🧠🤖 my_horse_human_model.keras
│   ├── 🖼️ Horse_test_image.png
│   ├── 🖼️ Human_test_image.png
│   └── 📄 README.md
└── 📄 README.md
</pre>

---

## 🚀 Getting Started

To run the notebooks:

```bash
# Clone the repository
git clone https://github.com/victorperone/Convolutional_Neural_Networks_in_TensorFlow.git
cd Convolutional_Neural_Networks_in_TensorFlow


# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

---

## 🛠️ Technologies Used:

- TensorFlow 2.x (Keras API)
- ImageDataGenerator
- Transfer Learning (InceptionV3)
- NumPy
- Matplotlib
- Google Colab / Jupyter Notebook

---

## 📚 References
- TensorFlow Documentation: https://www.tensorflow.org
- Coursera: Convolutional Neural Networks in TensorFlow: https://www.coursera.org/learn/convolutional-neural-networks-tensorflow
