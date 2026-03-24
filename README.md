# Projekt-110
## Projekt Beschreibung:

In diesem Projekt befassen wir uns mit dem MNIST-Datensatz zur Handschriftenerkennung von Ziffern.

Ziel des Projekts ist es, ein Convolutional Neural Network (CNN) zu entwickeln, zu trainieren und zu analysieren, um handgeschriebene Zahlen (0–9) automatisch zu klassifizieren, wo wir dann unserem Model mit den in der Vorlesung behandelten Netzwerken vergleichen.

## Team des Projekts
- Steffanie Schneider

- Jannik Pott

- Abdelhamid Jazzar

- Joen Berisha

## Projektstruktur

- src/ → Code für Modelle und Training des CNN
  
- notebooks/ → Experimente und Analyse (MNIST-Datensatz untersuchen, Modelle testen, Ergebnisse visualisieren.)
  
- results/ → Ergebnisse und Auswertung der Experimente (z. B. Trainingskurven, Accuracy und Diagramme)
  
- data/ → Datenspeicher des MNIST-Datensatz über torchvision

## Installation

Alle benötigten Python-Pakete können mit folgendem Befehl installiert werden:

  pip install -r requirements.txt

## Nutzung

Das Training des Modells kann mit folgendem Befehl gestartet werden:

  python src/train.py

## Datensatz

Der MNIST-Datensatz wird automatisch über torchvision heruntergeladen und lokal im data-Ordner gespeichert.
