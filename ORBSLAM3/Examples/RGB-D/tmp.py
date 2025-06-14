
def adjust_camera_trajectory(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if not lines:
        return

    first_value = float(lines[0].split()[0])
    adjusted_lines = []

    for line in lines:
        parts = line.split()
        if parts:
            first_float = float(parts[0])
            adjusted_first_float = first_float - first_value
            parts[0] = f"{adjusted_first_float:.6f}"
            adjusted_lines.append(" ".join(parts))
    
    with open(file_path, 'w') as file:
        file.write("\n".join(adjusted_lines))

# Usage
adjust_camera_trajectory('/home/demo/Desktop/xr/ORB_SLAM3_detailed_comments/Examples/RGB-D/CameraTrajectory.txt')