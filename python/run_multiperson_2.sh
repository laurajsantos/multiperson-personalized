echo "Hello World"
source /root/anaconda3/etc/profile.d/conda.sh
conda activate multiperson
conda list | grep yaml
conda info --envs
which python
which python3
MESA_GL_VERSION_OVERRIDE=4.1 python3 tools/demo_single_loop.py --focal_adult=$1 --focal_child=$2 --config=configs/smpl/tune.py --image_folder=demo_images/ --output_folder=results/ --ckpt data/checkpoint.pt
