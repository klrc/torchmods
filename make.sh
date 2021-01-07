source activate dl
version=`cat setup.py|grep "version="|tr -cd "[0-9]|\."`
echo $version
python setup.py sdist
twine upload dist/torchmods-${version}.tar.gz
pip install torchmods -U -i https://pypi.org/simple
pip install torchmods -U -i https://pypi.org/simple

