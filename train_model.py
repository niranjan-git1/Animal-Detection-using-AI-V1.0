import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam # Import Adam optimizer
import os

# --- CONFIGURATION ---
# The input size for the model (match your plant disease size for consistency)
IMAGE_SIZE = (128, 128) 
BATCH_SIZE = 32
EPOCHS = 15 # Start with 10; you can increase this later
DATA_DIR = './animal_dataset/' 
MODEL_SAVE_PATH = 'trained_animal_detector.keras'
# --- END CONFIGURATION ---

def build_cnn_model(input_shape, num_classes):
    """
    Defines a simple Convolutional Neural Network (CNN) architecture.
    """
    model = Sequential([
        # Block 1: Feature Extraction
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Block 2: Deeper Feature Extraction
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Block 3: Further Abstraction
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten layer transitions from 2D features to 1D vector
        Flatten(),
        
        # Classification Layers
        Dense(256, activation='relu'),
        Dropout(0.5), # Prevents overfitting
        Dense(num_classes, activation='softmax') # Output layer
    ])
    
    # Use a slightly slower learning rate (0.0001) for better stability with larger datasets
    # Adam is a standard, excellent optimizer for this task.
    optimizer = Adam(learning_rate=0.0001) 
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_animal_detector():
    """Sets up data generators, trains, and saves the model."""
    
    # 1. Setup Data Generators and Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalize
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255) # Only normalize validation data

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Dataset directory not found at '{DATA_DIR}'. Please create it and structure your data.")
        return
        
    print("Loading training data...")
    # The generator automatically finds the class folders (e.g., cat, dog, bird)
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR + 'train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    print("Loading validation data...")
    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR + 'validation',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    num_classes = train_generator.num_classes
    print(f"Detected {num_classes} animal classes: {list(train_generator.class_indices.keys())}")
    
    if num_classes < 2:
        print("ERROR: Need at least 2 animal classes (subfolders) to train the model. Please check your data structure.")
        return

    # 2. Build and Train Model
    input_shape = IMAGE_SIZE + (3,) # (128, 128, 3) for RGB
    model = build_cnn_model(input_shape, num_classes)
    model.summary()

    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # 3. Save Model and Class Names
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved successfully as: {MODEL_SAVE_PATH}")
    
    # Save class names for prediction script
    class_names = list(train_generator.class_indices.keys())
    with open('animal_class_names.txt', 'w') as f:
        # Write classes separated by newlines
        f.write('\n'.join(class_names))
    print("Class names saved to animal_class_names.txt.")
    
if __name__ == '__main__':
    train_animal_detector()