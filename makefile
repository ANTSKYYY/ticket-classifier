.PHONY: help install train evaluate

help:
	@echo "Available commands:"
	@echo "  make install   : Installs the required dependencies via requirements.txt"
	@echo "  make train     : Starts the model training (train.py)"
	@echo "  make evaluate  : Starts the model evaluation (evaluate.py)"

install:
	pip install -r requirements.txt

train:
	python train.py

evaluate:
	python evaluate.py

