
SCRIPT_MY_OV_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_MY_OV_DIR}

source ./python-env/bin/activate

OS_VERSION=$(lsb_release -rs)
if [ "$OS_VERSION" = "22.04" ]; then
    # source ./openvino_toolkit_ubuntu22_2026.0.0.dev20251110_x86_64/setupvars.sh
    source ./openvino/build/install/setupvars.sh
elif [ "$OS_VERSION" = "24.04" ]; then
    # source ./openvino_toolkit_ubuntu24_2025.4.0.dev20251017_x86_64/setupvars.sh
    source ./openvino/build/install/setupvars.sh
elif [ "$OS_VERSION" = "24.10" ]; then
    # source ./openvino_toolkit_ubuntu24_2026.0.0.dev20251111_x86_64/setupvars.sh
    source ./openvino/build/install/setupvars.sh
else
    echo "Error: Can't support version of Ubuntu: $OS_VERSION"
fi