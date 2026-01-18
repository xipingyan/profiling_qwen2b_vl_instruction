
SCRIPT_MY_OV_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_MY_OV_DIR}

source ./python-env/bin/activate

USE_NIGHT_OV="1" # download from nightly build.

if [ $USE_NIGHT_OV = "1" ]; then
    echo "-------------- USE_NIGHTLY_OV"
    # Get OS version, and source corresponding OpenVINO setupvars.sh
    UBUNTU_VER=$(lsb_release -rs | cut -d. -f1)
    source ./openvino_toolkit_ubuntu${UBUNTU_VER}_2026.0.0.dev20260117_x86_64/setupvars.sh
else
    echo "-------------- Use my build OV"
    source ./openvino/build/install/setupvars.sh
fi

