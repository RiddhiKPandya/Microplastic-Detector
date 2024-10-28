from picamera2 import Picamera2
import cv2
import numpy as np

# Capture image (your existing code)
picam2 = Picamera2()
picam2.start()
picam2.capture_file("water_sample.jpg")
picam2.stop()

# Load and process image
image = cv2.imread("water_sample.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area to remove noise
min_particle_area = 10  # Adjust this value based on your needs
max_particle_area = 500  # Adjust this value based on your needs
valid_particles = []

for contour in contours:
    area = cv2.contourArea(contour)
    if min_particle_area <= area <= max_particle_area:
        valid_particles.append(contour)

# Draw contours on original image
result = image.copy()
cv2.drawContours(result, valid_particles, -1, (0, 255, 0), 2)

# Add particle count
particle_count = len(valid_particles)
cv2.putText(result, f'Particles: {particle_count}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Threshold', thresh)
cv2.imshow('Detected Particles', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save results
cv2.imwrite('particle_detection.jpg', result)

# Calculate and print statistics
areas = [cv2.contourArea(contour) for contour in valid_particles]
if areas:
    print(f"Total particles detected: {particle_count}")
    print(f"Average particle area: {np.mean(areas):.2f} pixels")
    print(f"Min particle area: {np.min(areas):.2f} pixels")
    print(f"Max particle area: {np.max(areas):.2f} pixels")
# Add these after particle detection to get more detailed analysis

# Calculate size distribution
def get_particle_diameters(contours):
    diameters = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Approximate diameter assuming circular particles
        diameter = np.sqrt(4 * area / np.pi)
        diameters.append(diameter)
    return np.array(diameters)

# Get particle diameters
diameters = get_particle_diameters(valid_particles)

# Basic statistics
if len(diameters) > 0:
    print(f"Average diameter: {np.mean(diameters):.2f} pixels")
    print(f"Standard deviation: {np.std(diameters):.2f} pixels")
   
    # Create size distribution histogram
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(diameters, bins=30)
    plt.title('Particle Size Distribution')
    plt.xlabel('Diameter (pixels)')
    plt.ylabel('Count')
    plt.savefig('size_distribution.png')
    plt.close()