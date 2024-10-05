import numpy as np
from models.sample_configurations import sample_spherical_cap
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":


    # def sample_spherical_cap(R, theta_min, num_samples):
    #     # Convert theta_min from degrees to radians
    #     theta_min_rad = np.radians(theta_min)
    #
    #     samples = []
    #
    #     for _ in range(num_samples):
    #         # Sample u1 uniformly in [0, 1]
    #         u1 = np.random.uniform(0, 1)
    #         # Calculate theta using the derived formula
    #         cos_theta = (1 - np.cos(theta_min_rad)) * u1 + np.cos(theta_min_rad)
    #         theta = np.arccos(cos_theta)
    #
    #         # Sample u2 uniformly in [0, 1]
    #         u2 = np.random.uniform(0, 1)
    #         # Calculate phi
    #         phi = 2 * np.pi * u2
    #
    #         # Convert to Cartesian coordinates
    #         x = R * np.sin(theta) * np.cos(phi)
    #         y = R * np.sin(theta) * np.sin(phi)
    #         z = R * np.cos(theta)
    #
    #         samples.append((x, y, z))
    #
    #     return np.array(samples)


    # Parameters
    R = 1.0  # Radius of the sphere
    theta_min = 180  # Minimum angle of the spherical cap in degrees
    num_samples = 100_000  # Number of samples to generate

    # Generate samples
    # points = sample_spherical_cap(R, theta_min, num_samples)

    rng = np.random.default_rng(10)
    points = sample_spherical_cap(rng, params={
                "N": 1_000_000,
                "deg": 180
            })

    # Define the vector `z` and the `points` array
    z = np.array([0, 0, 1])

    # Step 1: Calculate the dot product between `z` and each row in `points`
    dot_products = np.dot(points, z)

    # Step 2: Compute the magnitude of each vector in `points`
    magnitudes = np.linalg.norm(points, axis=1)

    # Step 3: Compute the angular distance using arccos
    # Clip values to the range [-1, 1] to avoid floating point errors
    cosine_theta = dot_products / magnitudes
    angular_distances_radians = np.arccos(cosine_theta)
    # Now `angular_distances` contains the angular distance for all 1000 vectors in radians

    angular_distances_degrees = np.degrees(angular_distances_radians)
    sns.kdeplot(x=angular_distances_degrees)
    # plt.hist(angular_distances_degrees, density=True, )
    # plt.show()
    print(1)

