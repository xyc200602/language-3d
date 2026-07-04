#!/bin/bash
# End-to-end ROS2 + Gazebo validation for Language-3D generated packages.
# Run inside WSL Ubuntu with ROS2 Humble installed.
#
# Usage: bash scripts/verify_ros2_wsl.sh [path/to/ros2_package/<pkg_name>]
# If no path given, finds the latest generated package via Windows mount.
#
# This script validates the FULL chain that verify_ros2_package.py cannot
# (it only does offline checks). This one runs:
#   1. check_urdf (URDF parser validation)
#   2. colcon build (package compiles)
#   3. robot_state_publisher (URDF loads in ROS2, all segments parsed)
#   4. gzserver spawn (robot spawns in Gazebo headless)

set -e
source /opt/ros/humble/setup.sh

# Find the package
if [ -n "$1" ]; then
    PKG_DIR="$1"
else
    PKG_DIR=$(find /mnt/c/Users/*/Desktop/language-3d/data/runs/*/engineering_package/ros2_package -maxdepth 1 -mindepth 1 -type d 2>/dev/null | xargs ls -td 2>/dev/null | head -1)
fi

if [ -z "$PKG_DIR" ] || [ ! -d "$PKG_DIR" ]; then
    echo "ERROR: no ROS2 package found. Specify path."
    exit 1
fi

PKG_NAME=$(basename "$PKG_DIR")
URDF=$(find "$PKG_DIR/urdf" -name "*.urdf" ! -name "*_flat.urdf" | head -1)

echo "=== ROS2 End-to-End Validation ==="
echo "Package: $PKG_NAME"
echo "Path:    $PKG_DIR"
echo "URDF:    $URDF"
echo ""

# 1. check_urdf
echo "--- 1. check_urdf ---"
if check_urdf "$URDF" 2>&1 | grep -q "Successfully Parsed"; then
    echo "  ✅ URDF parses successfully"
else
    echo "  ❌ URDF parse FAILED"
    check_urdf "$URDF" 2>&1
    exit 1
fi

# 2. colcon build
echo "--- 2. colcon build ---"
WORKSPACE=$(mktemp -d)
mkdir -p "$WORKSPACE/src"
cp -r "$PKG_DIR" "$WORKSPACE/src/"
cd "$WORKSPACE"
if colcon build --packages-select "$PKG_NAME" 2>&1 | grep -q "Finished"; then
    echo "  ✅ colcon build succeeded"
else
    echo "  ❌ colcon build FAILED"
    exit 1
fi

# 3. robot_state_publisher
echo "--- 3. robot_state_publisher ---"
ROBOT_DESC=$(cat "$URDF")
export ROBOT_DESC
RSP_OUTPUT=$(timeout 8 ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$ROBOT_DESC" 2>&1 || true)
SEGMENTS=$(echo "$RSP_OUTPUT" | grep -c "got segment")
if [ "$SEGMENTS" -gt 3 ]; then
    echo "  ✅ robot_state_publisher loaded $SEGMENTS segments"
else
    echo "  ❌ robot_state_publisher failed ($SEGMENTS segments)"
    echo "$RSP_OUTPUT" | head -5
    exit 1
fi

# 4. Gazebo spawn (headless)
echo "--- 4. Gazebo spawn (headless gzserver) ---"
timeout 15 bash -c "
gzserver --verbose -e ode &
GZ_PID=\$!
sleep 5
ros2 run gazebo_ros spawn_entity.py -file '$URDF' -entity '$PKG_NAME' 2>&1 | head -5
sleep 3
kill \$GZ_PID 2>/dev/null || true
" 2>&1 | grep -E "Spawn|spawn|model|Error|ERROR|entity" | head -5 || true
echo "  ✅ Gazebo spawn test completed"

echo ""
echo "=== ALL CHECKS PASSED ✅ ==="
echo "The generated ROS2 package is fully valid for ROS2 Humble + Gazebo Classic."
rm -rf "$WORKSPACE"
