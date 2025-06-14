
import numpy as np

def read_trajectory(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    trajectory = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 0:
            timestamp,tx, ty, tz, qx, qy, qz, qw = map(float, parts)
            trajectory.append((timestamp, np.array([tx, ty, tz]), np.array([qx, qy, qz, qw])))
    return trajectory

def quaternion_to_euler(q):
    x, y, z, w = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return np.array([roll_x, pitch_y, yaw_z])

def normalize_angle(angle):
    angle = np.degrees(angle)
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    return angle

def get_movement_direction(delta):
    directions = ['left', 'right', 'up','down', 'backward', 'forward']
    axis = np.argmax(np.abs(delta))
    if delta[axis] < 0:
        return directions[axis * 2]
    else:
        return directions[axis * 2 + 1]

def evaluate_trajectory(trajectory):
    for i in range(0, len(trajectory)-1):
        current_position = trajectory[i][1]
        current_orientation = trajectory[i][2]
        current_euler = quaternion_to_euler(current_orientation)
        next_position = trajectory[i+1][1]
        next_orientation = trajectory[i+1][2]
        next_euler = quaternion_to_euler(next_orientation)

        delta_position = next_position - current_position
        delta_yaw = normalize_angle(next_euler[2] - current_euler[2])

        movement_direction = get_movement_direction(delta_position)
        if abs(delta_yaw) < 1:
            rotation_direction = 'original'
        elif abs(delta_yaw) > 10:
            rotation_direction = 'error'
        elif delta_yaw < 0:
            rotation_direction = 'turnleft'
        else:
            rotation_direction = 'turnright'

        print(f"Frame {i} to {i+1}: ")
        print(f"current_orientation: {current_euler}")
        print(f"next_orientation: {next_euler}")
        print(f"Move {movement_direction} ({delta_position[2]:.4f},{delta_position[0]:.4f},{delta_position[1]:.4f}), Rotate {rotation_direction} {delta_yaw:.4f}")

if __name__ == "__main__":
    trajectory = read_trajectory('../Examples/droneClient/logs/trajs/trajectorys_1201_1.txt')
    print(f"Trajectory length: {len(trajectory)}")
    evaluate_trajectory(trajectory)

"""
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <array>
#include <string>

struct Pose {
    double timestamp;
    std::array<double, 3> position;
    std::array<double, 4> orientation;
};

std::vector<Pose> read_trajectory(const std::string& file_path) {
    std::ifstream file(file_path);
    std::string line;
    std::vector<Pose> trajectory;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Pose pose;
        double _;
        iss >> pose.timestamp >> _ >> pose.position[0] >> pose.position[1] >> pose.position[2]
            >> pose.orientation[0] >> pose.orientation[1] >> pose.orientation[2] >> pose.orientation[3];
        trajectory.push_back(pose);
    }

    return trajectory;
}

std::array<double, 3> quaternion_to_euler(const std::array<double, 4>& q) {
    double x = q[0], y = q[1], z = q[2], w = q[3];
    double t0 = +2.0 * (w * x + y * z);
    double t1 = +1.0 - 2.0 * (x * x + y * y);
    double roll_x = std::atan2(t0, t1);

    double t2 = +2.0 * (w * y - z * x);
    t2 = t2 > +1.0 ? +1.0 : t2;
    t2 = t2 < -1.0 ? -1.0 : t2;
    double pitch_y = std::asin(t2);

    double t3 = +2.0 * (w * z + x * y);
    double t4 = +1.0 - 2.0 * (y * y + z * z);
    double yaw_z = std::atan2(t3, t4);

    return {roll_x, pitch_y, yaw_z};
}

double normalize_angle(double angle) {
    angle = angle * 180.0 / M_PI;
    if (angle > 180) {
        angle -= 360;
    } else if (angle < -180) {
        angle += 360;
    }
    return angle;
}

std::string get_movement_direction(const std::array<double, 3>& delta) {
    std::array<std::string, 6> directions = {"left", "right", "up", "down", "backward", "forward"};
    int axis = std::distance(delta.begin(), std::max_element(delta.begin(), delta.end(), [](double a, double b) { return std::abs(a) < std::abs(b); }));
    return delta[axis] < 0 ? directions[axis * 2] : directions[axis * 2 + 1];
}

void evaluate_trajectory(const std::vector<Pose>& trajectory) {
    for (size_t i = 0; i < trajectory.size() - 1; ++i) {
        const auto& current_position = trajectory[i].position;
        const auto& current_orientation = trajectory[i].orientation;
        double current_yaw = quaternion_to_euler(current_orientation)[2];

        const auto& next_position = trajectory[i + 1].position;
        const auto& next_orientation = trajectory[i + 1].orientation;
        double next_yaw = quaternion_to_euler(next_orientation)[2];

        std::array<double, 3> delta_position = {next_position[0] - current_position[0], next_position[1] - current_position[1], next_position[2] - current_position[2]};
        double delta_yaw = normalize_angle(next_yaw - current_yaw);

        std::string movement_direction = get_movement_direction(delta_position);
        std::string rotation_direction;
        if (std::abs(delta_yaw) < 1) {
            rotation_direction = "original";
        } else if (std::abs(delta_yaw) > 10) {
            rotation_direction = "error";
        } else if (delta_yaw < 0) {
            rotation_direction = "turnleft";
        } else {
            rotation_direction = "turnright";
        }

        std::cout << "Frame " << i << ": Move " << movement_direction << " (" << delta_position[2] << "," << delta_position[0] << "," << delta_position[1] << "), Rotate " << rotation_direction << " " << delta_yaw << " degrees" << std::endl;
    }
}

int main() {
    std::vector<Pose> trajectory = read_trajectory("../Examples/droneClient/logs/trajs/trajectorys_1201_1_kf.txt");
    std::cout << "Trajectory length: " << trajectory.size() << std::endl;
    evaluate_trajectory(trajectory);
    return 0;
}
"""