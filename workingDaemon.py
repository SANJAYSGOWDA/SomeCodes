import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set print options for better readability
np.set_printoptions(precision=6, suppress=True)

def pixel_to_spherical(x, y, w=5000, h=2700):
    """Convert pixel coordinates (x, y) to spherical coordinates (theta, phi)."""
    fov_h = 360  # Horizontal FOV in degrees
    fov_v = 180  # Vertical FOV in degrees
    
    # Normalize coordinates relative to the image center
    x_normalized = (x - w / 2) / (w / 2)
    y_normalized = (y - h / 2) / (h / 2)
    
    # Convert to spherical coordinates (theta, phi)
    theta = x_normalized * (fov_h / 2)  # Azimuth in degrees
    phi = y_normalized * (fov_v / 2)      # Elevation in degrees
    print("theta:", theta)
    print("phi:", phi)
    return theta, phi

def spherical_to_unit_vector(theta, phi):
    """Convert spherical coordinates (theta, phi) to a unit vector in 2D."""
    # Here we ignore the elevation (phi) for 2D calculations.
    x = np.cos(np.radians(theta))
    y = -np.sin(np.radians(theta))
    return np.array([x, y])

def find_intersection(vec1, vec2, t1, t2):
    """Find intersection of two rays defined by unit vectors vec1, vec2 starting from t1, t2."""
    A = np.vstack((vec1, -vec2)).T
    b = t2 - t1
    t = np.linalg.lstsq(A, b, rcond=None)[0]  # Solve for scaling factors
    depth = t[0]
    return t1 + depth * vec1  

def triangulate_unit_vector(u1, v1, u2, v2, t1, t2, image_width, image_height):
    """
    Triangulate a 2D point from two camera images using only unit vectors computed 
    from pixel coordinates (without applying any camera heading/rotation).
    """
    # Convert pixel coordinates to spherical coordinates
    theta1, phi1 = pixel_to_spherical(u1, v1, image_width, image_height)
    theta2, phi2 = pixel_to_spherical(u2, v2, image_width, image_height)
    
    # Optionally, apply any offsets if needed (currently zero)
    theta1_offset = theta1 + 193.3681200 -90
    theta2_offset = theta2 + 176.46509 -90
    
    # Convert spherical coordinates to unit vectors
    unit_vector1 = spherical_to_unit_vector(theta1_offset, phi1)
    unit_vector2 = spherical_to_unit_vector(theta2_offset, phi2)
    
    print(f"Unit Vector 1: {unit_vector1}")
    print(f"Unit Vector 2: {unit_vector2}")
    
    # Compute intersection using the unit vectors (ignoring heading)
    unit_vector_intersection = find_intersection(unit_vector1, unit_vector2, t1, t2)
    
    return unit_vector_intersection, unit_vector1, unit_vector2

# Camera positions (Easting, Northing) -- these are the car positions
t1 = np.array([402105.858,4578416.531])  # Car 1 position
t2 = np.array([402106.814,4578426.823])    # Car 2 position

# Image points (pixel coordinates)
u1, v1 = 284,1435# Pixel location in image 1
u2, v2 = 2018,1526 # Pixel location in image 2

# Image dimensions
image_width = 5000
image_height = 2700

# Define the ground truth point (Easting, Northing)
gT = (402109.87,4578422.63)

# Triangulate using unit vectors only (no heading/rotation)
unit_vector_intersection, unit_vector1, unit_vector2 = triangulate_unit_vector(
    u1, v1, u2, v2, t1, t2, image_width, image_height
)

print(f"Unit Vector Intersection (Triangulated Point): {unit_vector_intersection}")

# Visualization
fig, ax = plt.subplots()

# Define scaling factors for visualization
unit_vector_length = 15  # Scale for displaying the unit vectors

# Scale the direction vectors for visualization
scaled_unit_vector1 = unit_vector1 * unit_vector_length
scaled_unit_vector2 = unit_vector2 * unit_vector_length

# Plot car positions
ax.scatter(t1[0], t1[1], c='r', label='Car 1 (Camera 1)')
ax.scatter(t2[0], t2[1], c='b', label='Car 2 (Camera 2)')

# Plot unit vectors (directions) from the car positions
ax.quiver(t1[0], t1[1], scaled_unit_vector1[0], scaled_unit_vector1[1], 
          angles='xy', scale_units='xy', scale=1, color='orange', label='Unit Vector 1')
ax.quiver(t2[0], t2[1], scaled_unit_vector2[0], scaled_unit_vector2[1], 
          angles='xy', scale_units='xy', scale=1, color='purple', label='Unit Vector 2')

# Plot the intersection point computed from unit vectors
ax.scatter(unit_vector_intersection[0], unit_vector_intersection[1], 
           c='g', marker='o', label='Unit Vector Intersection')

# Plot the ground truth point
ax.scatter(gT[0], gT[1], color='red', marker='x', s=100, label=f'Ground Truth ({gT[0]}, {gT[1]})')

# Draw unit circles (unit sphere in 2D) around car positions
circle1 = patches.Circle((t1[0], t1[1]), radius=1, edgecolor='black', linestyle='--',
                           facecolor='none', label='Unit Sphere at Car 1')
circle2 = patches.Circle((t2[0], t2[1]), radius=1, edgecolor='gray', linestyle='--',
                           facecolor='none', label='Unit Sphere at Car 2')
ax.add_patch(circle1)
ax.add_patch(circle2)

# Add labels for car positions
ax.text(t1[0], t1[1], f'({t1[0]:.2f}, {t1[1]:.2f})', color='r', fontsize=10, ha='right', va='bottom')
ax.text(t2[0], t2[1], f'({t2[0]:.2f}, {t2[1]:.2f})', color='b', fontsize=10, ha='right', va='bottom')

# Set labels and title
ax.set_xlabel('Easting')
ax.set_ylabel('Northing')
plt.title("Triangulation Using Unit Vectors Only (No Heading) with Ground Truth and Unit Spheres")
ax.legend()
plt.axis('equal')
plt.show()
