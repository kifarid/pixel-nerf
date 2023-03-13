conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install scikit-image tqdm matplotlib scikit-image
echo "done with torch installation";
pip install pyhocon opencv-python dotmap tensorboard imageio imageio-ffmpeg ipdb pretrainedmodels lpips
echo "done with pixel nerf installation";
#conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath 
pip install absl-py gym==0.17.3 pybullet>=3.0.4 meshcat 
conda install tensorflow 
conda install pytorch3d -c pytorch3d 
echo "done with pytorch3d and reavens installation";
pip install wandb omegaconf 
pip install hydra-core --upgrade 
pip install pytorch-lightning 
echo "done with lightning installation";


