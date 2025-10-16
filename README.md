# 🏐 ProJump: Real-Time Jump Detection and Height Estimation Using IMU Data

## Overview
**ProJump** is currently a Python-based system for detecting and measuring vertical jumps using data from an **inertial measurement unit (IMU)**.  
Originally designed for **beach volleyball performance analysis**, this project demonstrates a reliable, field-ready alternative to laboratory tools like force plates or motion capture systems.

## Motivation
Jump height is a key indicator of lower-limb explosive power and is widely used to assess athletic performance.  
Traditional measurement tools are accurate but often **expensive, bulky, and impractical for real-world use**.  
ProJump leverages lightweight, wearable IMUs to enable **low-cost, portable, and real-time** jump assessment directly in the field.

## Features
- **Jump detection** and **height estimation** from raw IMU data.  
- **Madgwick orientation filter** for aligning accelerometer data to the global vertical axis.  
- **Signal processing pipeline** with:
  - High-pass and low-pass Butterworth filtering
  - Gravity compensation
  - Zero-velocity updates (ZUPT)
- **Flight-time based height calculation**, supported by threshold and velocity peak logic.  
- Designed for **beach volleyball** but adaptable to other jumping sports.

## Methodology
1. **Data Acquisition:**  
   IMU sensors record 3-axis acceleration and angular velocity during controlled jumps and in-game movements.

2. **Preprocessing:**  
   - Orientation alignment via the *Madgwick* filter.  
   - Gravity subtraction and noise filtering.  
   - Detection of flight phase using a **0.5 g threshold** and velocity signatures.

3. **Jump Height Calculation:**  
   Flight time is determined from take-off to landing events and used to calculate height as:  

$$
h = \frac{1}{8} \cdot gt^2  \qquad g = 9.81~\mathrm{m/s^2}
$$


4. **Validation:**  
   Algorithm design inspired by recent studies demonstrating high reliability of IMU-based jump height estimation methods. Future validation tests to be performed in-field and against ground truth in force plates. 

## References
- [Madgwick, S.O.H. (2014). *An efficient orientation filter for inertial and inertial/magnetic sensor arrays.* University of Bristol.](https://x-io.co.uk/downloads/madgwick-phd-thesis.pdf)  
- Bishop, C. et al. (2023). *Jumping beyond the lab: A scoping review of in-field jump assessment in sport and clinical populations.* *Journal of Biomechanics*, 140, 118023.  
- Pueo, B. et al. (2023). *Accuracy of flight time and countermovement-jump height estimated from videos at different frame rates with MyJump.* *Biology of Sport*, 40(2), 595–601.

## Repository Structure
