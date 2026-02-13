# Convolutional_Neural_Networks_in_TensorFlow

> **Advanced Computer Vision:** From Overfitting to Transfer Learning.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-High%20Level%20API-red.svg)](https://keras.io/) [![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

Repository that contains my notes, excercises and explanations from the Coursera course **Convolutional Neural Networks in TensorFlow**. 
It covers CNN fundamentals, image classification, data augmentation, and building models using TensorFlow and Keras.


🔗 **Course link:**  

[Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow)

---

## 📘 Course Overview

This course, which is the second of the DeepLearning.IA TensorFlow Developer Specialization, focuses on advanced techniques for improving the computer vision models developed in [Course 1](https://www.coursera.org/learn/introduction-tensorflow). It covers working with real-world images of different shapes and sizes, visualize an image's journey through convolutions to understand how a computer _"sees"_ information, trace loss and precision, and explore strategies to avoid overfitting, including boosting and dropping. Finally, it introduces learning transfer and how learned features can be extracted and reused across models.

Andrew Ng’s Machine Learning and Deep Learning Specialization courses provide a strong foundation in the core principles of machine learning and deep learning. Building on this foundation, the DeepLearning.AI TensorFlow Developer Specialization emphasizes the practical implementation of these principles using TensorFlow, enabling learners to design and deploy scalable models for real-world applications.

This course is recommended after completing [Introduction to TensorFlow](https://www.coursera.org/learn/introduction-tensorflow).

My work for that course is available on GitHub [here](https://github.com/victorperone/Introduction_to_TensorFlow_for_Artificial_Intelligence).

### **Course Modules**
1. **Exploring a Larger Dataset**
2. **Augmentation: A Technique to Avoid Overfitting**
3. **Transfer Learning**
4. **Multiclass Classifications**

---

## 🛠️ Key Concepts Mastered

| Concept | Description |
| :--- | :--- |
| **Data Augmentation** | Artificially expanding datasets by rotating, zooming, and shifting images to prevent overfitting. |
| **Transfer Learning** | Leveraging pre-trained models (like InceptionV3) to solve tasks with limited data. |
| **Dropout** | A regularization technique to force network redundancy and robustness. |
| **Multi-Class Classification** | Moving beyond binary (Cat vs. Dog) to complex tasks (Rock, Paper, Scissors). |

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
git clone https://github.com/victorperone/Introduction_to_TensorFlow_for_Artificial_Intelligence.git
cd Introduction_to_TensorFlow_for_Artificial_Intelligence

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

---

## 🛠️ Technologies Used:
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 📚 References
- TensorFlow Documentation: https://www.tensorflow.org
- Coursera: Convolutional Neural Networks in TensorFlow: https://www.coursera.org/learn/convolutional-neural-networks-tensorflow
