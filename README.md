# Deep Armocromia

Deep Armocromia è un progetto di **Computer Vision** e **Deep Learning** per l’automazione dell’analisi armocromatica a partire da immagini del volto.

Il sistema classifica la stagione cromatica di una persona utilizzando una rete neurale convoluzionale basata su **ResNet18** e studia l’eventuale impatto del **makeup pesante** sulle predizioni del modello.

> Il progetto è ospitato nel repository [`Deep-color`](https://github.com/kaismankai-stack/Deep-color).

## Obiettivi del progetto

Il progetto affronta due problemi di classificazione:

- **P1 – Classificazione in 4 stagioni**
  - Inverno
  - Autunno
  - Estate
  - Primavera

- **P2 – Classificazione in 12 sotto-stagioni**
  - Inverno Deep
  - Inverno Cool
  - Inverno Bright
  - Autunno Deep
  - Autunno Soft
  - Autunno Warm
  - Estate Cool
  - Estate Soft
  - Estate Light
  - Primavera Bright
  - Primavera Light
  - Primavera Warm

La domanda di ricerca principale è:

> Il makeup pesante introduce un bias nella classificazione della stagione cromatica?

A questo scopo sono stati confrontati due scenari sperimentali:

- **Scenario A – ALL DATA**: utilizzo dell’intero dataset disponibile.
- **Scenario B – NO-MAKEUP**: utilizzo delle immagini senza makeup pesante, quando questa informazione è disponibile.

## Funzionalità

- Classificazione automatica in 4 stagioni cromatiche.
- Classificazione automatica in 12 sotto-stagioni.
- Utilizzo di Transfer Learning con ResNet18.
- Preprocessing e data augmentation delle immagini.
- Generazione delle matrici di confusione.
- Calcolo di accuracy, precision, recall e F1-score macro.
- Analisi dell’impatto del makeup pesante.
- Predizione della stagione cromatica di una nuova immagine.
- Stima delle emissioni di CO₂ tramite CodeCarbon.
- Possibile integrazione con un’interfaccia web per il caricamento delle immagini.

## Dataset

Il progetto utilizza il **Deep Armocromia Dataset**, composto principalmente da due sorgenti:

### Pivotal Armocromia Set

Il Pivotal Armocromia Set contiene circa 3.000 immagini raccolte da piattaforme online e annotate con una delle 12 sotto-stagioni cromatiche.

Queste immagini vengono utilizzate per:

- aumentare la diversità demografica;
- includere differenti fototipi ed età;
- considerare diverse condizioni di illuminazione;
- migliorare la copertura delle 12 classi.

Le immagini del Pivotal Set non contengono informazioni affidabili sull’utilizzo del makeup.

### CelebA Dataset

CelebA è un dataset pubblico per l’analisi dei volti che contiene immagini e attributi binari.

Per il progetto sono state utilizzate immagini CelebA annotate con la relativa stagione cromatica. L’attributo principale per lo studio del makeup è:

- `Heavy Makeup = +1`: presenza di makeup pesante;
- `Heavy Makeup = -1`: assenza di makeup pesante.

### File di annotazione

I principali file utilizzati sono:

```text
annotations.csv
list_attr_celeba.txt
```

Il file `annotations.csv` contiene informazioni relative a:

- nome dell’immagine;
- stagione cromatica;
- origine dell’immagine, tramite il campo `celeba`.

Il file `list_attr_celeba.txt` contiene i 40 attributi binari originali di CelebA.

> Il dataset non è incluso nel repository. È necessario scaricarlo separatamente e rispettare le condizioni di utilizzo e le licenze dei dataset originali.

## Architettura del modello

Il modello utilizzato è **ResNet18**, inizializzato con pesi pre-addestrati su ImageNet.

La testa completamente connessa originale è stata sostituita con:

```text
Dropout(p = 0.4)
Linear(512, numero_classi)
```

Il numero di classi è:

- `4` per il problema P1;
- `12` per il problema P2.

### Preprocessing

Le immagini vengono ridimensionate a `224 × 224` pixel e normalizzate utilizzando i valori medi e le deviazioni standard di ImageNet:

```python
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
```

Durante l’addestramento vengono applicate le seguenti trasformazioni:

- Random Horizontal Flip;
- Random Rotation fino a ±15°;
- Random Resized Crop;
- Conversione in tensore;
- Normalizzazione ImageNet.

Le alterazioni cromatiche, come `ColorJitter`, non vengono utilizzate perché i colori naturali del volto rappresentano l’informazione principale del problema.

## Tecnologie utilizzate

- Python
- PyTorch
- Torchvision
- ResNet18
- Transfer Learning
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- CodeCarbon
- Flask
- OpenCV
- HTML, CSS e JavaScript

## Installazione

Clonare il repository:

```bash
git clone https://github.com/kaismankai-stack/Deep-color.git
cd Deep-color
```

Creare e attivare un ambiente virtuale:

### Linux e macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Aggiornare `pip`:

```bash
python -m pip install --upgrade pip
```

Installare le dipendenze:

```bash
pip install -r requirements.txt
```

> Per utilizzare una GPU NVIDIA, installare la versione di PyTorch compatibile con la propria versione di CUDA seguendo le istruzioni ufficiali di PyTorch.

## Struttura dei dati

Una possibile struttura delle cartelle è la seguente:

```text
Deep-color/
├── app.py
├── filtering.py
├── train_resnet18.py
├── requirements.txt
├── annotations.csv
├── filtered_no_makeup.csv
├── data/
│   ├── images/
│   └── celeba/
│       └── list_attr_celeba.txt
└── models/
    └── best_model.pth
```

I percorsi dei file possono essere modificati direttamente negli script di filtraggio e addestramento.

## Utilizzo

### Filtraggio delle immagini senza makeup pesante

Per generare il file contenente le immagini CelebA con:

```text
Heavy Makeup = -1
```

eseguire:

```bash
python filtering.py
```

Il filtraggio viene applicato soltanto alle immagini CelebA, poiché le immagini del Pivotal Set non dispongono dell’attributo `Heavy Makeup`.

### Addestramento del modello

Per avviare l’addestramento della rete ResNet18:

```bash
python train_resnet18.py
```

Il modello può essere configurato per:

- classificazione in 4 stagioni;
- classificazione in 12 sotto-stagioni;
- Scenario A – ALL DATA;
- Scenario B – NO-MAKEUP.

I parametri principali utilizzati sono:

```text
Ottimizzatore: Adam
Learning rate: 1e-3
Weight decay: 1e-4
Scheduler: StepLR
Dropout: 0.4
Early stopping: patience = 10
Batch size: definita nello script di training
```

I pesi pre-addestrati di ResNet18 possono essere scaricati automaticamente da Torchvision durante la prima esecuzione.

### Avvio dell’interfaccia

Se la versione del repository include l’interfaccia Flask, avviare l’applicazione con:

```bash
python app.py
```

Successivamente aprire il browser all’indirizzo:

```text
http://127.0.0.1:5000
```

Da questa interfaccia è possibile caricare un’immagine e visualizzare la stagione cromatica predetta dal modello.

## Risultati sperimentali

I risultati riportati nel progetto mostrano che il modello supera il classificatore casuale in entrambi i problemi.

| Scenario | Problema | Accuracy approssimativa | Baseline casuale |
|---|---|---:|---:|
| ALL DATA | P1 – 4 stagioni | 48% | 25% |
| NO-MAKEUP | P1 – 4 stagioni | 47% | 25% |
| ALL DATA | P2 – 12 sotto-stagioni | 23% | 8,3% |
| NO-MAKEUP | P2 – 12 sotto-stagioni | 21% | 8,3% |

Le confusioni più frequenti si verificano tra sotto-stagioni cromaticamente vicine, ad esempio:

- Inverno Cool e Inverno Deep;
- Estate Light ed Estate Soft;
- Autunno Deep e Autunno Warm;
- Primavera Light e Primavera Bright.

Il confronto tra i due scenari mostra uno scarto di circa un punto percentuale nella classificazione a 4 stagioni. Tuttavia, questo risultato non è sufficiente da solo per dimostrare l’assenza di bias statistico, soprattutto a causa delle dimensioni limitate del dataset e delle differenze tra le sorgenti delle immagini.

## Impatto ambientale

Il consumo energetico del training è stato stimato utilizzando **CodeCarbon** su Google Colab con una GPU NVIDIA Tesla T4.

Nel report sono riportati indicativamente:

- circa `28,54 g CO₂eq` per i quattro esperimenti di training;
- circa `0,000935 g CO₂eq` per una singola inferenza.

Questi valori sono stime e possono variare in base a:

- hardware utilizzato;
- posizione del server;
- durata dell’addestramento;
- carico del sistema;
- intensità carbonica della rete elettrica.

## Limitazioni

Il progetto presenta alcune limitazioni:

- le differenze tra sotto-stagioni sono molto sottili;
- il numero di immagini per classe è limitato;
- le etichette armocromatiche possono essere soggettive;
- le condizioni di illuminazione non sono uniformi;
- le immagini possono contenere ritocchi o alterazioni cromatiche;
- il modello non utilizza una segmentazione specifica di pelle, occhi e capelli;
- il modello può analizzare anche sfondo, vestiti e accessori;
- l’attributo `Heavy Makeup` è disponibile soltanto per le immagini CelebA;
- le immagini del Pivotal Set non possono essere classificate in modo affidabile rispetto al makeup.

Il risultato della rete deve quindi essere considerato **indicativo** e non sostituisce una consulenza professionale di armocromia.

## Sviluppi futuri

Possibili sviluppi del progetto sono:

- utilizzo di ResNet50, EfficientNet o Vision Transformer;
- aumento del numero di immagini del dataset;
- bilanciamento delle classi;
- utilizzo di class weights;
- segmentazione automatica del volto;
- applicazione di meccanismi di attenzione spaziale;
- analisi separata di pelle, occhi e capelli;
- confronto sistematico tra immagini con e senza makeup;
- valutazione con cross-validation;
- studio statistico più approfondito dell’impatto del makeup;
- ottimizzazione del modello per dispositivi mobili.

## Riferimenti

- [Repository Deep Armocromia](https://github.com/lorenzo-stacchio/Deep-Armocromia)
- [Repository del progetto Deep-color](https://github.com/kaismankai-stack/Deep-color)
- [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- [CodeCarbon](https://github.com/mlco2/codecarbon)

## Autori

- **Kais Mankai**
- **Mohamed Ali Soussi**
- **Mohamed Yassine Dougari**

## Contesto accademico

Progetto realizzato nell’ambito del corso di:

**Computer Vision e Deep Learning**  
**Università Politecnica delle Marche**  
Anno accademico **2025–2026**

Docente: **Prof. Alessandro Galdelli**  
Tutor: **Prof. Rocco Pietrini**

