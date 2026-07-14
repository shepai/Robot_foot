# Robot_foot
A neater write up of the tactile sensor development making use of trained models for classification

# Dependencies
- sklearn
- pytorch
- opencv
- numpy
- opencv-python
- torch
- scikit-learn
- joblib

# Library Tactile interface
## Overview

This module provides a unified interface for loading and using pretrained machine learning models for tactile sensing. It supports both optical and electrical tactile sensors, allowing prediction of:

- Surface texture classification
- Coefficient of friction estimation
- Tactile marker/feature point estimation (optical sensor only)

The module also includes preprocessing pipelines to convert raw sensor outputs into the same representation used during model training.

The supported tactile sensors are:

- **Optical tactile sensor**: Camera-based tactile sensing using an LSTM classifier and regression models.
- **PressTip electrical tactile sensor**: Low-resolution electrical tactile sensing using Random Forest models.

The pretrained models are automatically loaded from the `models/` directory relative to this file.

---
# Texture classes 
[
'Leather',
'Cork',
'wool',
'LacedMatt',
'Gfoam',
'Plastic',
'Carpet',
'bubble',
'Efoam',
'cotton',
'LongCarpet',
'Flat',
'felt',
'Jeans',
'Ffoam'
]
# Coding Examples

This section demonstrates how to load the pretrained tactile sensing models and perform texture classification and friction prediction.

## Importing the tactile sensor interface

```python
from tactile_sensor import opticalSensor, PressTipSensor, ImageDataPreprocessor

# Available texture classes
keys = [
    'Leather', 'Cork', 'wool', 'LacedMatt', 'Gfoam',
    'Plastic', 'Carpet', 'bubble', 'Efoam',
    'cotton', 'LongCarpet', 'Flat',
    'felt', 'Jeans', 'Ffoam'
]
```

---

# Optical Tactile Sensor

The optical tactile sensor uses a camera-based tactile image sequence. The preprocessing pipeline converts raw camera images into the representation used during training.

## Loading the model

```python
# Load preprocessing pipeline
processor = ImageDataPreprocessor()

# Load pretrained optical tactile model
optical_sensor = opticalSensor()
```

---

## Processing tactile images

Raw tactile images should be provided as a sequence of frames.

```python
# Example:
# images.shape = (timesteps, height, width, channels)

processed_images = processor.process_video(images)

print(processed_images.shape)
```

Output:

```
(timesteps, 110, 120)
```

---

## Texture classification

Predict the surface texture from a tactile image sequence.

```python
texture_id = optical_sensor.predict_texture(
    processed_images
)

print("Predicted texture:")
print(keys[texture_id])
```

Example output:

```
Predicted texture:
Carpet
```

---

## Batch texture classification

Multiple tactile sequences can be classified simultaneously.

```python
# Input shape:
# (batch_size, timesteps, 110, 120)

texture_ids = optical_sensor.predict_texture_multi(
    batch_images
)

for prediction in texture_ids:
    print(keys[prediction])
```

---

## Friction prediction

Estimate the coefficient of friction from tactile interaction.

```python
friction = optical_sensor.predict_friction(
    processed_images
)

print(
    "Estimated friction:",
    friction
)
```

Example output:

```
Estimated friction: 0.73
```

---

## Predict tactile marker positions

The optical sensor can also estimate marker locations.

```python
points = optical_sensor.predict_points(
    image
)

print(points.shape)
```

Output:

```
(133, 2)
```

Each row represents the predicted `(x,y)` location of a tactile marker.

---

# PressTip Electrical Tactile Sensor

The PressTip sensor uses low-resolution electrical tactile measurements.

## Loading the model

```python
# Load pretrained electrical tactile model
presstip_sensor = PressTipSensor()
```

---

## Texture classification

Electrical sensor readings can be classified directly.

```python
import numpy as np

# Example electrical tactile measurement
# Replace with real sensor data
sensor_data = np.random.random((1,64))

texture_id = presstip_sensor.predict_texture(
    sensor_data
)

print(
    "Predicted texture:",
    keys[texture_id]
)
```

Example output:

```
Predicted texture:
Leather
```

---

## Friction prediction

Estimate friction from electrical tactile readings.

```python
friction = presstip_sensor.predict_friction(
    sensor_data
)

print(
    "Estimated friction:",
    friction
)
```

Example output:

```
Estimated friction: 0.61
```

---

# Complete Example

A complete example using the optical tactile sensor:

```python
from tactile_sensor import (
    opticalSensor,
    ImageDataPreprocessor
)

keys = [
    'Leather', 'Cork', 'wool', 'LacedMatt',
    'Gfoam', 'Plastic', 'Carpet',
    'bubble', 'Efoam', 'cotton',
    'LongCarpet', 'Flat',
    'felt', 'Jeans', 'Ffoam'
]


# Initialise models
processor = ImageDataPreprocessor()
sensor = opticalSensor()


# Load tactile camera frames
# images = your tactile image sequence

processed = processor.process_video(images)


# Predict texture
prediction = sensor.predict_texture(processed)

print(
    "Detected surface:",
    keys[prediction]
)


# Predict friction
friction = sensor.predict_friction(processed)

print(
    "Estimated friction:",
    friction
)
```
# Mujuco 
We made simulated sensors to be used as a tool for sim2real tasks. We ave made simulations of the presstip and the tactip. The sensr itself is a separated file, that can be imported into other mujoco environment xml, making the sensors easily added to your project. 

![alt text](https://shepai.github.io/assets/tutorials/soft/tactip.png)

# Citation 

```BibTeX
@article{shepherd2025texture,
  title={Texture and Friction Classification: Optical TacTip vs. Vibrational Piezoeletric and Accelerometer Tactile Sensors},
  author={Shepherd, Dexter R and Husbands, Phil and Philippides, Andrew and Johnson, Chris},
  journal={Sensors},
  volume={25},
  number={16},
  pages={4971},
  year={2025},
  publisher={MDPI}
}

@inproceedings{shepherd2023low,
  title={Low-Resolution Sensing for Sim-to-Real Complex Terrain Robots},
  author={Shepherd, Dexter R and Knight, James C},
  booktitle={Annual Conference Towards Autonomous Robotic Systems},
  pages={190--201},
  year={2023},
  organization={Springer}
}
```

## Datasets
```BibTeX


@misc{dataset,
  author = {D. R. Shepherd},
  title = {Optical Tactile (TacTip) Dataset for texture classification},
  year = {2024},
  publisher = {University of Sussex},
  note = {Available: \url{https://doi.org/10.25377/sussex.26935696}}
}

@misc{dataset2,
  author = {D. R. Shepherd},
  title = {Electrical Tactile Dataset (Piezoelectric and Accelerometer) for textures},
  year = {2024},
  publisher = {University of Sussex},
  note = {Available: \url{https://doi.org/10.25377/sussex.28033589}}
}

```
Author:
**Dexter R. Shepherd**  
PhD Student in Artificial Intelligence  
University of Sussex  

GitHub: https://github.com/shepai  
LinkedIn: https://www.linkedin.com/in/dexter-shepherd-1a4a991b8/