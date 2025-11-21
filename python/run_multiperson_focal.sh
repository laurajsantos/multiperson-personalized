echo "Hello World"
cd multiperson/mmdetection
source ~/anaconda3/etc/profile.d/conda.sh
conda activate multiperson
MESA_GL_VERSION_OVERRIDE=4.1 python3 tools/demo_focal_length.py --config=configs/smpl/tune.py --image_folder=demo_images/ --output_folder=results/ --ckpt data/checkpoint.pt
