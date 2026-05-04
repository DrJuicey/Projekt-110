# Projekt-110: MNIST Handschriftenerkennung

## Inhaltsverzeichnis
1. [Projekt Beschreibung](#projekt-beschreibung)
2. [Team des Projekts](#team-des-projekts)
3. [Projektverlauf & Branches](#projektverlauf--branches)
4. [Projektstruktur](#projektstruktur)
5. [Datensatz](#datensatz)
6. [Installation](#installation)
7. [Nutzung & Ausführung](#nutzung--ausführung)

---

## Projekt Beschreibung
In diesem Projekt befassen wir uns mit dem MNIST-Datensatz zur Handschriftenerkennung von Ziffern. 
Ziel des Projekts ist es, ein Convolutional Neural Network (CNN) zu entwickeln, zu trainieren und zu analysieren, um handgeschriebene Zahlen (0–9) automatisch zu klassifizieren. Im Anschluss vergleichen wir unser Modell mit den in der Vorlesung behandelten Netzwerken.

## Team des Projekts
* Steffanie Schneider
* Jannik Pott
* Abdelhamid Jazzar
* Joen Berisha

## Projektverlauf & Branches
Um die Arbeit im Team effizient aufzuteilen, lief das Projekt in mehreren Phasen ab, was sich auch in unserer Branch-Struktur widerspiegelt:
1. **Initiale Experimente:** Zu Beginn hat jedes Teammitglied in einem eigenen Branch (z. B. `first_modell_KingSteffy`, `first_modell_Jannik`,) ein erstes individuelles CNN-Modell entwickelt und getestet.
2. **Baseline-Findung & Weiterentwicklung:** Anschließend haben wir die Ergebnisse verglichen und uns auf ein gemeinsames Baseline-Modell geeinigt. Von dort aus haben manche Teammitglieder dedizierte Branches für ihre weitere Arbeit erstellt (z. B. `baseline_Abdel`, `Haupt_Hypothese`), während andere ihre ursprünglichen `first_modell`-Branches (`first_modell_Joen`) einfach erweitert haben.
3. **Zusammenführung:** Die finalen Modelle, Hypothesen und das zentrale Steuerungsskript wurden abschließend im aktuellen `main`-Branch zusammengeführt.

## Projektstruktur
Die wichtigsten Bestandteile des Projekts im `main`-Branch:

* **`data/`** – MNIST-Datensatz (automatisch geladen über `torchvision`)
* **`notebooks/`** – Experimente und Analysen *(Inhalte befinden sich teilweise in separaten Branches)*
* **`src/`** – Enthält den gesamten Code des Projekts:
    * `main.py`: Das zentrale Steuerungsskript. Hierüber können alle Hypothesen zentral ausgeführt werden.
    * `Haupt_Hypothese_Models/`: Basis-Modelle zum Vergleich (CNN, Random Forest, SVM, Logistische Regression).
    * `hypothese_eins/`, `hypothese_2/`, `hypothese_3/`: Spezifische Modelle und Trainingsskripte zur Untersuchung verschiedener Netzwerk-Tiefen, Architekturen und Parametergrößen.
* **`requirements.txt`** – Liste aller benötigten Python-Pakete

## Datensatz
Der MNIST-Datensatz wird automatisch über `torchvision` heruntergeladen und lokal im `data`-Ordner gespeichert.

## Installation
Alle benötigten Python-Pakete können mit folgendem Befehl installiert werden:

```bash
pip install -r requirements.txt
```

## Nutzung & Ausführung
Das Training und die Ausführung des Projekts erfolgen über das zentrale Skript:

```bash
python src/main.py
```
Darüber öffnet sich ein Programm im Terminal, das im Hintergrund alle nötigen Skripte ausführt. Du kannst dort ganz bequem auswählen, ob jede beliebige Hypothese einzeln oder alle auf einmal ausgeführt werden sollen. 
