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

eval:
	# Évalue le modèle entraîné localement (recommandé si `make train` a été fait)
	python evaluate.py --mode local

eval-hub:
	python evaluate.py --mode hub --hub_repo "Antskyyy/ticket-classifier"
