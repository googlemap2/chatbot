Kích hoạt môi trường ảo: source venv/bin/activate
Thoát venv: deactivate
pip freeze > requirements.txt

rm -rf venv
python3.12 -m venv venv
pip install --upgrade pip
pip install -r requirements.txt

pip install vllm
