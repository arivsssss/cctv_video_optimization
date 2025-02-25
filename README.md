# cctv_video_optimization
CNN



# CCTV Video Optimization Using Deep Learning

## Project Overview
With the increasing use of CCTV surveillance, optimizing video storage is crucial. This project employs a deep learning-based solution to detect human presence in CCTV footage and selectively save relevant frames, significantly reducing storage requirements.

## Objective
This project aims to optimize CCTV video storage by detecting human presence and retaining only frames containing humans, thereby reducing unnecessary storage usage while preserving essential footage.

## Dataset
- **Source:** [Kaggle - Human Detection Dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset/data)
- **Categories:**
  - `0` - No human present
  - `1` - Human detected
- **Preprocessing:**
  - Images are resized to **128x128** pixels.
  - Normalization is applied to scale pixel values between 0 and 1.
  - Data augmentation techniques (rotation, shifting, zooming, shearing, flipping) are used.
  - The dataset is split into **80% training** and **20% testing**.

## Model Architecture
A **Convolutional Neural Network (CNN)** is used for human detection, comprising the following layers:
- **Convolutional & Pooling Layers:**
  - Conv2D layers with ReLU activation.
  - Batch Normalization for stable training.
  - MaxPooling (2x2) for feature extraction.
  - Dropout (30%) for regularization.
- **Fully Connected Layer:**
  - 128 neurons with ReLU activation.
  - Dropout (50%) to reduce overfitting.
- **Output Layer:**
  - Single neuron with **Sigmoid Activation** for binary classification (Human / No Human).

## Implementation
### Data Preprocessing
```python
X, y = preprocess_images()
print(f"Processed {len(X)} images.")
```

### Data Augmentation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)
```

### Model Training
```python
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=50, callbacks=[early_stopping, reduce_lr])
```

### Model Evaluation
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
```

## Results & Performance
- **Accuracy:** The model achieved a **high validation accuracy** after training.
- **Error Analysis:**
  - **True Negatives (TN):** 45 (Correctly identified no human)
  - **False Positives (FP):** 35 (Wrongly detected human where there was none)
  - **False Negatives (FN):** 5 (Missed detecting human presence)
  - **True Positives (TP):** 100 (Correctly detected human presence)

## Testing on CCTV Footage
The model processes video frames and predicts whether a human is present.
```python
video_path = 'yourpath\cctv_video_trimmed.mp4'
predictions = model.predict(selected_frames_np)
```
Each frame is classified as **Human** or **No Human** and displayed accordingly.

## Conclusion
This project successfully optimizes CCTV storage using deep learning. By selectively saving human-detected frames, it achieves **efficient storage management** without losing critical information.

## Future Enhancements
- Improve **False Positive** detection through fine-tuning.
- Optimize the **real-time processing** of CCTV streams.
- Implement **edge AI deployment** for on-device processing.




