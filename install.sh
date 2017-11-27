cd ~/notebooks

git clone https://github.com/TensorMSA/tensormsa_jupyter.git

git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd python-package; sudo python setup.py install

http://xgboost.readthedocs.io/en/latest/build.html
echo "export PYTHONPATH=${PYTHONPATH}/root/notebooks/xgboost/python-package" >> ~/.bashrc
source ~/.bashrc

apt-get install graphviz
pip install seaborn 
pip install plotly
pip install pydot
pip install graphviz
pip install popen
