AI-Generated Content Detection
![alt text](https://img.shields.io/badge/Python-3.10%2B-blue)
![alt text](https://img.shields.io/badge/TensorFlow-2.15-orange)
![alt text](https://img.shields.io/badge/Keras-API-red)
![alt text](https://img.shields.io/badge/Status-Completed-success)
Abstract
Submitted to Technological University Dublin in partial fulfilment of the requirements for the degree of B.Sc. (Hons) Computer Science focusing in Cyber Security & Digital Forensics.
In the era of rapidly advancing artificial intelligence, the proliferation of AI-generated content poses significant challenges for digital forensics, copyright, and information security. This project presents a comprehensive evaluation of Machine Learning (ML) techniques—specifically Convolutional Neural Networks (CNNs) and Transfer Learning—to detect AI-generated images.
This repository contains the source code, trained models, data processing pipelines, and the final thesis PDF detailing the development and analysis of a various models as detection tools designed to differentiate between human-generated and AI-generated imagery.

Tech Stack
Language: Python
Core Frameworks: TensorFlow, Keras
Data Manipulation: NumPy, Pandas
Image Processing: PIL (Pillow), OpenCV
Visualization: Matplotlib
Utilities: OS, Shutil, Glob, Scikit-learn
Development Hardware: Trained on NVIDIA RTX 3080 GPU (CUDA/cuDNN enabled).

Methodologies & Models
This project implemented and compared three distinct model architectures:
Refined Custom CNN: A convolutional neural network built from scratch with multiple convolutional, max-pooling, and dense layers.
Refined CNN (Early Stopping): The same custom architecture enhanced with Dropout layers and Early Stopping callbacks to prevent overfitting and improve generalization.
Transfer Learning (ResNet50): A pre-trained ResNet50 model (ImageNet weights) fine-tuned for binary classification.

Datasets Used
To ensure robustness, the models were trained and tested on a diverse collection of datasets, including:
AI Generated Images vs Real Images (BOWMAN)
ArtiFact (Awsaf et al.) - Primary large-scale dataset
CIFAKE (Lotfi & Bird) - CIFAR-10 based synthetic images
140k Real and Fake Faces (Xhlulu) - NVIDIA FFHQ & StyleGAN
AI Recognition Dataset (Koliha)

Key Findings
Detailed analysis can be found in the "Analysis" section of the PDF.
Custom CNN: Achieved high training accuracy (up to ~98%) but showed signs of overfitting on specific datasets.
Early Stopping: Successfully reduced training time and mitigated overfitting, though generalization to completely unseen generative models (e.g., switching from Stable Diffusion to Midjourney) remains a challenge.
Transfer Learning: In this specific implementation, ResNet50 struggled with binary classification of this nature, often exhibiting bias towards the "Real" class, highlighting the difficulty of applying ImageNet features to synthetic artifact detection without extensive fine-tuning.

Credits & References
Author: Sam Magee
Supervisor: Peter Alexander
Institution: Technological University Dublin
Full references for datasets and literature are available in the References section of the attached PDF.
