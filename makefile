.PHONY: help install train evaluate

help:
	@echo "Commandes disponibles :"
	@echo "  make install   : Installe les dépendances requises via requirements.txt"
	@echo "  make train     : Lance l'entraînement du modèle (train.py)"
	@echo "  make evaluate  : Lance l'évaluation du modèle (evaluate.py)"

install:
	pip install -r requirements.txt

train:
	python train.py

evaluate:
	python evaluate.py

