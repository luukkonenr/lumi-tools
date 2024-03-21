# Our base-image. This one includes PyTorch for ROCM, FlashAttention v2.0.4, apex etc.
CONTAINER="/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif"

# Isolation from $USER/.local/ meaning that 'pythonuserbase' operates like a venv; this is where all of our installed packages go
export PYTHONUSERBASE="pythonuserbase"
mkdir -p $PYTHONUSERBASE

# Here we set which locations of file-system we are able to see from the container
BINDS="/scratch/project_462000086,/scratch/project_462000319/"

# Which packages we want to install. (You can edit pip-command down below to also accept a requirements.txt as an argument)
custom_packages="transformers"
finish_string="Installed packages: $custom_packages"

singularity exec -B $PWD -B $BINDS $CONTAINER \
	bash -c	"source /opt/miniconda3/bin/activate pytorch; \
	pip install $custom_packages;
	echo $finish_strings"
