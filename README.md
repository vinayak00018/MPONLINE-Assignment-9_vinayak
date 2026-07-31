# Image Classification using Convolutional Neural Networks (CNN)


## Objective
The objective of this project is to develop a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify pet images into Cats and Dogs to support automated animal identification[cite: 2].

## Dataset Link
- [Kaggle: Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)[cite: 2]

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tensorflow` / `keras`
- `scikit-learn`
- `pillow`
- `kaggle`

## Methodology
1. **Data Understanding**: Analyzed binary dataset with 24,998 valid RGB images (12,499 Cats, 12,499 Dogs) and displayed sample images[cite: 2].
2. **Data Preprocessing**:
   - Filtered unreadable/corrupted files[cite: 2].
   - Resized images to $128 \times 128$ pixels[cite: 2].
   - Normalized pixel intensity values to $[0, 1]$ range[cite: 2].
   - Stratified split into 80% Training (19,998 images) and 20% Testing (5,000 images)[cite: 2].
   - Created batch image generators using `ImageDataGenerator`[cite: 2].
3. **CNN Architecture**:
   - **Conv2D Block 1**: 32 filters ($3\times3$, ReLU) + MaxPooling2D ($2\times2$)[cite: 2]
   - **Conv2D Block 2**: 64 filters ($3\times3$, ReLU) + MaxPooling2D ($2\times2$)[cite: 2]
   - **Conv2D Block 3**: 128 filters ($3\times3$, ReLU) + MaxPooling2D ($2\times2$)[cite: 2]
   - **Flatten Layer**
   - **Dense Layer**: 128 neurons (ReLU)[cite: 2]
   - **Output Layer**: 1 neuron (Sigmoid)[cite: 2]
4. **Compilation & Training**:
   - **Optimizer**: Adam[cite: 2]
   - **Loss Function**: Binary Crossentropy[cite: 2]
   - Trained for 10 epochs[cite: 2].
5. **Evaluation**: Evaluated Test Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and plotted Accuracy/Loss graphs per epoch[cite: 2].

## CNN Architecture Summary

| Layer | Type | Output Shape | Param # |
| :--- | :--- | :--- | :--- |
| `conv2d` | Conv2D (32, 3x3) | (None, 126, 126, 32) | 896 |
| `max_pooling2d` | MaxPooling2D (2x2) | (None, 63, 63, 32) | 0 |
| `conv2d_1` | Conv2D (64, 3x3) | (None, 61, 61, 64) | 18,496 |
| `max_pooling2d_1` | MaxPooling2D (2x2) | (None, 30, 30, 64) | 0 |
| `conv2d_2` | Conv2D (128, 3x3) | (None, 28, 28, 128) | 73,856 |
| `max_pooling2d_2` | MaxPooling2D (2x2) | (None, 14, 14, 128) | 0 |
| `flatten` | Flatten | (None, 25088) | 0 |
| `dense` | Dense (128, ReLU) | (None, 128) | 3,211,392 |
| `dense_1` | Dense (1, Sigmoid) | (None, 1) | 129 |

## Results
- **Test Accuracy:** 85.20%[cite: 2]
- **Test Loss:** 0.7194[cite: 2]
- **Precision:** 84.43%[cite: 2]
- **Recall:** 86.32%[cite: 2]
- **F1-Score:** 85.36%[cite: 2]

## Conclusion
The 3-layer CNN effectively extracts spatial features to classify pet images with 85.20% accuracy[cite: 2]. Adding data augmentation and dropout layers can help prevent training overfitting and improve generalization on unseen image variations[cite: 2].