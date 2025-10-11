import tensorflow as tf
import numpy as np
import cv2

# Load CIFAR-10 (this ALWAYS works, no downloads needed)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Get dog images (class 5)
dog_indices = np.where(y_train == 5)[0][:3000]
images = x_train[dog_indices]

# Resize to 300x300
clean_images = []
for img in images:
    img_resized = cv2.resize(img, (300, 300))
    clean_images.append(img_resized)

clean_images = np.array(clean_images).astype('float32') / 255.0

# Add noise
noise = np.random.normal(10/255.0, 50/255.0, clean_images.shape)
noisy_images = np.clip(clean_images + noise, 0.0, 1.0)

# Save
np.save('train_clean.npy', clean_images[:2400])
np.save('train_noisy.npy', noisy_images[:2400])
np.save('test_clean.npy', clean_images[2400:])
np.save('test_noisy.npy', noisy_images[2400:])

print("✓ Dataset ready!")
print(f"Training: {2400} samples")
print(f"Testing: {600} samples")