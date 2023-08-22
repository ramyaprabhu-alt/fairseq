pip install fairscale==0.4.0
pip install hydra-core==1.0.7 omegaconf==2.0.6
git clone --depth=1 --branch v2.6 https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
cd ..

pip install --editable ./
pip install boto3
pip install iopath
export PYTHONPATH=/ramyapra/fairseq