source activate
pip install --user boto3
git clone https://github.com/NVIDIA/apex
cd apex
pip install --user -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
