# Dense Ground Truth Generator

A ROS2 Humble package that generates dense ground truth data using multiple Gaussian distributions over a rectangular area. The ground truth can be sampled via a ROS2 service.

## Features

- **Configurable rectangular area**: Define the bounds of your ground truth region
- **Multiple Gaussian distributions**: Create complex, dense variations in the field
- **Lipschitz parameter**: Control the rate of variation and smoothness
- **ROS2 service interface**: Sample any 2D point and get a scalar value
- **Easy configuration**: All parameters tunable via YAML config file

## Package Structure

```
dense_ground_truth/
├── config/
│   └── ground_truth_params.yaml    # Configuration file
├── dense_ground_truth/
│   └── ground_truth_server.cpp     # Main service node
├── launch/
│   └── ground_truth_server.launch.py
├── srv/
│   └── SampleGroundTruth.srv       # Service definition
├── CMakeLists.txt
└── package.xml
```

## Installation

### Install Python Dependencies

```bash
pip3 install numpy matplotlib scikit-learn
```

Or install system packages (Ubuntu/Debian):
```bash
sudo apt install python3-numpy python3-matplotlib python3-sklearn
```

### Building

```bash
cd /home/emgym/dense-ground-truth-generator
colcon build --packages-select dense_ground_truth
source install/setup.bash
```

## Configuration

Edit `config/ground_truth_params.yaml` to tune parameters:

- **area_min_x, area_max_x, area_min_y, area_max_y**: Define the rectangular region bounds
- **num_gaussians**: Number of Gaussian distributions (higher = denser variations)
- **lipschitz_constant**: Controls rate of change (higher = steeper gradients, lower = smoother)

## Running

```bash
ros2 launch dense_ground_truth ground_truth_server.launch.py
```

## Usage

Call the service to sample a point:

```bash
ros2 service call /sample_ground_truth dense_ground_truth/srv/SampleGroundTruth "{x: 50.0, y: 50.0}"
```

### Python Client Example

```python
import rclpy
from rclpy.node import Node
from dense_ground_truth.srv import SampleGroundTruth

class GroundTruthClient(Node):
    def __init__(self):
        super().__init__('ground_truth_client')
        self.client = self.create_client(SampleGroundTruth, 'sample_ground_truth')

    def sample_point(self, x, y):
        request = SampleGroundTruth.Request()
        request.x = x
        request.y = y

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().value
```

## Service Definition

**Request:**
- `float64 x`: X coordinate
- `float64 y`: Y coordinate

**Response:**
- `float64 value`: Scalar value at the point

## Gaussian Process Regression Visualization

Run the GPR demo to visualize how well Gaussian Process Regression can learn the ground truth:

```bash
ros2 launch dense_ground_truth gpr_demo.launch.py
```

This will:
1. Start the ground truth server
2. Randomly sample 50 training points
3. Train a Gaussian Process Regressor on those points
4. Evaluate accuracy on a dense grid
5. Display a comprehensive visualization with:
   - Ground truth (3D and 2D)
   - GPR prediction (3D and 2D)
   - Absolute error visualization
   - Accuracy metrics (RMSE, MAE, Max Error, R² Score)

The visualization is saved to `/tmp/gpr_ground_truth_visualization.png`.

### Customize the Demo

You can adjust the number of training samples and grid resolution in the launch file or via command line:

```bash
ros2 run dense_ground_truth gpr_visualization.py --ros-args \
  -p num_training_points:=100 \
  -p grid_resolution:=80
```

## How It Works

1. The server generates N Gaussian distributions randomly placed within the rectangular area
2. Each Gaussian has a random amplitude and standard deviation based on the Lipschitz constant
3. When a point is sampled, the value is computed as the sum of all Gaussian contributions
4. The Lipschitz constant inversely affects the standard deviation, controlling variation density

### GPR Learning

The GPR visualization demonstrates:
- **Random Sampling**: Training points are uniformly sampled across the area
- **GP Training**: A Gaussian Process with RBF kernel learns the underlying function
- **Interpolation**: GPR provides smooth interpolation between training points
- **Uncertainty**: The visualization shows where the model is confident vs uncertain
- **Accuracy Metrics**: Quantifies how well GPR approximates the ground truth
