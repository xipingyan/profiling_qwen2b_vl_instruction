
SCRIPT_MY_OV_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_MY_OV_DIR}

source ./python-env/bin/activate
# source ./openvino_toolkit_ubuntu24_2025.4.0.dev20250929_x86_64/setupvars.sh
source ./openvino_toolkit_ubuntu24_2025.4.0.dev20251017_x86_64/setupvars.sh