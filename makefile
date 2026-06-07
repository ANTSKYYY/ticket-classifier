.PHONY: help install train evaluate

help:
	@echo "Available commands:"
	@echo "  make install   : Installs the required dependencies via requirements.txt"
	@echo "  make train     : Starts the model training (train.py)"
	@echo "  make eval  : Load the model locally trained"
	@echo "  make eval-hub  : Load our best model from HuggingFace pre-trained"	

install:
	pip install -r requirements.txt

train:
	python train.py

eval:
	python evaluate.py --mode local

eval-hub:
	python evaluate.py --mode hub --hub_repo "Antskyyy/ticket-classifier"
