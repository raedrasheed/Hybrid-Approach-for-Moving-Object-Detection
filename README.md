# A Hybrid Approach for Moving Object Detection and Tracking in Event-Based Cameras 
## Ahmed S. Ghorab, Raed Rasheed, Hanan Abu-Mariah, Wesam M. Ashour

Here’s a comprehensive `README.md` template for the project, covering its description, setup, usage, and contributing guidelines:

```markdown
# Event-Based Clustering and Visualization

This project performs clustering on event-based data, primarily using the `DBSCAN` and `KMeans` algorithms, with additional support for other clustering techniques. The processed data represents event sequences that are visualized with convex hulls around clusters and saved as CSV files. This can be useful for applications in computer vision, motion tracking, and other event-driven data processing.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Algorithms Supported](#algorithms-supported)

## Features

- Processes event-based data stored in `.mat` files.
- Supports clustering algorithms such as DBSCAN, KMeans, Birch, and Gaussian Mixture Models.
- Detects noise and filters events accordingly.
- Visualizes clustered events in real-time using OpenCV.
- Saves processed event data with cluster labels to CSV files.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.8 or higher
- Libraries: `h5py`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`, `opencv-python`, `kneed`

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/raedrasheed/event_camera.git
cd event_camera
```

## Usage

1. **Data Preparation**: Place the event sequence data (`Hands_sequence.mat`) and timestamp data (`hands_all_ts.npy`) in the `events` directory.

2. **Run Clustering and Visualization**:

   ```bash
   python main.py
   ```

   This script processes the event frames, applies the specified clustering algorithm, and displays the visualization in real-time.

3. **View Results**: After processing, check the `clustering_predicted/our/{algorithm_name}/` directory for CSV files containing labeled events for each frame.

## Configuration

### Choosing the Clustering Algorithm

Open `main.py` and specify the algorithm by setting `algorithm_name` at the top of the file:

```python
algorithm_name = 'kmean'
# Available options: 'kmean', 'DBSCAN', 'birch', 'GaussianMixture', 'AgglomerativeClustering', etc.
```

### Clustering Parameters

Each clustering algorithm has different parameters (e.g., `eps` and `min_samples` for DBSCAN). Adjust these as needed within the `calculate_eps_elbow_new()` or `remove_noise_dbscan()` functions in `main.py`.

### Data and Frame Configuration

- **File Path**: Update `file_name` in `main.py` to point to the path of the `.mat` file containing event data.
- **Frame Range**: Adjust `NUMBER_OF_FRAMES` to limit or extend the number of frames processed.

## Algorithms Supported

- **DBSCAN**: Density-based spatial clustering, particularly useful for identifying noise.
- **KMeans**: Centroid-based clustering for fast clustering of fixed-size groups.
- **Birch**: Efficient for large datasets.
- **Gaussian Mixture Model (GMM)**: Assumes Gaussian distribution for clusters.
- **Agglomerative Clustering**: Hierarchical clustering, suitable for structured data.

## Project Structure

```plaintext
.
├── events/
│   ├── Hands_sequence.mat             # Event-based data file
│   ├── hands_all_ts.npy               # Timestamps for each frame
├── clustering_predicted/
│   ├── our/
│       ├── {algorithm_name}/          # Contains output CSV files for each frame
├── main.py                            # Main script for processing and visualization
├── README.md                          # Project documentation
└── requirements.txt                   # List of dependencies
```
