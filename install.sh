cd ~/notebooks


git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd python-package; sudo python setup.py install

apt-get install graphviz
pip install seaborn 
pip install plotly
pip install pydot
pip install graphviz
pip install popen
