SCRIPT_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR}

source ../python-env/bin/activate
source ../openvino_toolkit_ubuntu24_2025.4.0.dev20250929_x86_64/setupvars.sh

python example.py