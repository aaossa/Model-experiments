#!/bin/bash
REQUIREMENTS_TXT=${1:-requirements/dev.txt}


# Install project requirements
echo ">> Installing requirements.txt"
if [ -f "${REQUIREMENTS_TXT}" ]; then
    pip install -r "$REQUIREMENTS_TXT" --upgrade -f https://download.pytorch.org/whl/torch_stable.html
else
    echo ERROR: Failed to find "${REQUIREMENTS_TXT}". Please provide an existing path to a requirements file.
    exit 1
fi
echo

# Aditional setup
echo ">> Additional setup"
sudo apt-get install python3-gdbm
echo

# Explore GPUs if available
echo ">> Explore GPUs if available"
ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
if ! type nvidia-smi >/dev/null 2>&1; then
    echo Command 'nvidia-smi' not found
else
    nvidia-smi
fi
