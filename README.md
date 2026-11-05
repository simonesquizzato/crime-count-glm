# 🚓 Crime Count GLM
### Numero di crimini orari ad Atlanta: confronto tra modelli per dati di conteggio

Quanti crimini avverranno nella prossima ora, in un dato quartiere di Atlanta?
Questo progetto modella e prevede il **numero orario di crimini** nelle 25 *Neighborhood Planning Unit* (NPU) di Atlanta (Georgia, USA) e confronta tre modelli per dati di conteggio adatti a una variabile risposta con **oltre il 90% di zeri**: **Poisson**, **Zero-Inflated Poisson (ZIP)** e **Hurdle Poisson**.

**Autore:** Simone Squizzato · A.A. 2025/2026

---

## 📑 Indice

- [Dataset](#-dataset)
- [Analisi esplorativa](#-analisi-esplorativa)
- [Modelli](#-modelli)
- [Risultati](#-risultati)
- [Conclusioni](#-conclusioni)
- [Struttura del repository](#-struttura-del-repository)
- [Come riprodurre l'analisi](#-come-riprodurre-lanalisi)
- [Limiti e sviluppi futuri](#-limiti-e-sviluppi-futuri)

---

## 🗄️ Dataset

| Caratteristica | Valore |
|---|---|
| Fonte | [Kaggle – Core Atlanta Burglary-Related Crimes 2021–2025](https://www.kaggle.com/datasets/joshuapina/core-atlanta-burglary-related-crimes-2021-2025), derivato da dati aggregati dell'FBI per lo stato della Georgia |
| Periodo | 01/01/2021 – 27/11/2025 |
| Granularità | Un'osservazione per ogni ora × NPU (panel denso) |
| Osservazioni | 1.074.500 |
| File | `npu_dense_panel.parquet` |

**Variabili principali**

| Variabile | Descrizione |
|---|---|
| `crime_count` | 🎯 Numero di crimini commessi nell'ora (variabile risposta) |
| `npu` | Neighborhood Planning Unit: 25 unità territoriali, indicate con le lettere dalla A alla Z |
| `npu_crime_avg_30d` | Numero medio di crimini nella NPU negli ultimi 30 giorni |
| `near_*` | Dummy di vicinanza a punti di interesse, soprattutto campus universitari (GSU, Georgia Tech, Emory, Clark, Spelman, Morehouse, SCAD, John Marshall, …) |
| `is_*` | Dummy meteorologiche e temporali (pioggia, caldo, luce diurna, festivo, weekend, …) |
| `temp_f` | Temperatura registrata (°F) |
| `hour_ts` | Timestamp orario |

---

## 🔍 Analisi esplorativa

- **974.003 ore senza crimini (90,65%)** e **100.497 ore con almeno un crimine (9,35%)**: una forte *zero inflation*.
- Quando si verifica almeno un crimine, il valore tipico è **1** e il massimo è **16** in un'ora.
- Le NPU con più crimini sono **M, E e B**.
- Circa il **10%** dei crimini avviene vicino al **Georgia Institute of Technology** (Midtown). Le zone di **Georgia State University** e **John Marshall University** (Downtown) ne raccolgono un ulteriore **17%**.

---

## 🧮 Modelli

Lo split train/test è **sequenziale** (80% / 20%) e non casuale, per rispettare la struttura temporale dei dati.

### 1. Regressione di Poisson (GLM)
$$Y \mid \mathbf{x} \sim \text{Pois}(\lambda), \qquad \log(\lambda) = \mathbf{x}'\boldsymbol\beta$$

### 2. Zero-Inflated Poisson (ZIP)
Gli zeri osservati vengono scomposti in **zeri strutturali** (con probabilità $\pi$) e **zeri casuali** generati dalla Poisson:

$$\Pr(Y=y \mid \mathbf{x}) = \begin{cases} \pi + (1-\pi)\,e^{-\lambda} & y = 0 \\ (1-\pi)\,\dfrac{\lambda^y e^{-\lambda}}{y!} & y = 1, 2, \dots \end{cases}$$

### 3. Hurdle Poisson
Un processo binario decide se si supera la "soglia" dello zero. I valori positivi seguono invece una **Poisson troncata in zero**:

$$\Pr(Y=y \mid \mathbf{x}) = \begin{cases} \pi & y = 0 \\ (1-\pi)\,\dfrac{\lambda^y e^{-\lambda}}{y!\,(1-e^{-\lambda})} & y > 0 \end{cases}$$

In ZIP e Hurdle si usano due link:
- **logit** per $\pi$: $\pi = \dfrac{e^{\mathbf{x}'\boldsymbol\beta}}{1+e^{\mathbf{x}'\boldsymbol\beta}}$
- **log** per $\lambda$: $\lambda = e^{\mathbf{z}'\boldsymbol\gamma}$

**Specificazione usata per ZIP e Hurdle**

| Componente | Predittori |
|---|---|
| Conteggio ($\lambda$) | `npu` + dummy `near_*` (GSU, GA Tech, Emory, Clark, Spelman, Morehouse, Atlanta Metro, SCAD, John Marshall) |
| Zeri ($\pi$) | intercetta + `npu_crime_avg_30d` |

---

## 📊 Risultati

### Stima dei modelli (training set)

**Poisson**
- Tutti i predittori risultano significativi, ma alcuni segni sono controintuitivi. Ad esempio, le NPU con più crimini (M, E, B) hanno coefficienti **negativi** rispetto al livello di riferimento A.
- Intercetta: $e^{-5.298} \approx 0.005$ crimini/ora.
- `npu_crime_avg_30d`: ogni unità in più aumenta il numero medio atteso di crimini orari del **≈ 17%**.
- AIC = 268.945.

**Zero-Inflated Poisson**
- $\hat\pi = \dfrac{e^{9.587}}{1+e^{9.587}} \approx 0.99993$ (componente zero data dall'intercetta)
- Gli effetti significativi delle NPU sono quasi tutti negativi e di piccola entità, con l'eccezione di B. Sono compensati dalla vicinanza ai campus, con effetti positivi e significativi: ad esempio `near_john_marshall` (+0.457), `near_ga_tech` (+0.429) e `near_morehouse` (+0.406).
- Residui di Pearson: massimo 216.56.

**Hurdle Poisson**
- $\hat\pi = \dfrac{e^{-10.067}}{1+e^{-10.067}} \approx 0.0000425$, coerente con la definizione della PMF dell'Hurdle.
- Gli effetti dei campus sono molto più marcati: ad esempio `near_ga_tech` (+1.904), `near_john_marshall` (+1.723) e `near_atlanta_metro` (+1.673).
- Residui di Pearson: massimo 344.26.

### Performance previsive (test set)

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat y_i)^2} \qquad \text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat y_i|$$

| Modello | RMSE | MAE |
|---|---|---|
| **ZIP** | **0.0915** | 0.0157 |
| **Hurdle** | 0.0928 | **0.0138** |
| Poisson | 0.1861 | 0.0484 |

- ZIP e Hurdle **dimezzano l'RMSE** e riducono il MAE di circa **3 volte** rispetto al Poisson standard.
- **ZIP**: RMSE leggermente migliore, ma **sottostima sistematica** quando i crimini superano 3 in un'ora.
- **Hurdle**: MAE migliore. Tende a **sovrastimare**, ma riesce a cogliere le ore con un numero di crimini elevato.

---

## ✅ Conclusioni

La scelta tra ZIP e Hurdle dipende dall'uso delle previsioni. Se servissero a decidere **quante pattuglie di polizia mantenere attive**:

| Modello | Effetto operativo |
|---|---|
| ZIP | Meno risorse impiegate, con il rischio di **personale insufficiente** nei picchi |
| Hurdle | Copertura più prudente, con il costo di un **dispiegamento talvolta eccessivo** |

Tra i due, la capacità di cogliere i valori elevati fa preferire il modello **Hurdle Poisson**, mentre il modello ZIP sottostima in modo sistematico.

---

## 📁 Struttura del repository

```
crime-count-glm/
├── Crime-Count-Atlanta.R         # Script completo: EDA, stima dei modelli, previsioni e metriche
├── npu_dense_panel.parquet       # Dataset (panel orario per NPU)
├── Crime_Count_ZIP_Report.pdf    # Report completo del progetto
└── README.md
```

---

## ▶️ Come riprodurre l'analisi

**Requisiti:** R ≥ 4.0

1. **Clona il repository**
   ```bash
   git clone https://github.com/simonesquizzato/crime-count-glm.git
   cd crime-count-glm
   ```

2. **Installa i pacchetti R**
   ```r
   install.packages(c("arrow", "ggplot2", "gridExtra", "dplyr", "tidyr", "pscl", "caret"))
   ```

3. **Esegui l'analisi**
   ```r
   source("Crime-Count-Atlanta.R")
   ```

> ⏱️ Con oltre un milione di osservazioni, la stima di ZIP e Hurdle (ottimizzazione BFGS) può richiedere alcuni minuti.

| Pacchetto | Utilizzo |
|---|---|
| `arrow` | Lettura del file `.parquet` |
| `dplyr`, `tidyr` | Manipolazione dei dati |
| `ggplot2`, `gridExtra` | Visualizzazione |
| `pscl` | Modelli `zeroinfl()` (ZIP) e `hurdle()` |
| `caret` | Metriche `RMSE()` e `MAE()` |

---

## 🚧 Limiti e sviluppi futuri

- **Sovradispersione**: valutare Negative Binomial, ZINB e Hurdle NB, che gestiscono anche l'eccesso di varianza nei conteggi positivi.
- **Componente zero**: il coefficiente di `npu_crime_avg_30d` nella parte zero ha un errore standard molto elevato. Questo suggerisce una quasi-separazione e merita una specificazione alternativa.
- **Dipendenza temporale e spaziale**: introdurre termini autoregressivi, stagionalità oraria e settimanale (`hour_sin`, `hour_cos`, …) o effetti spaziali tra NPU adiacenti.
- **Valutazione**: affiancare a RMSE e MAE metriche specifiche per i conteggi, come la log-likelihood predittiva, i rootogram e la calibrazione di $\Pr(Y=0)$, valutando anche su finestre temporali scorrevoli.
- **Variabili meteo**: integrare `is_*` e `temp_f` anche nei modelli ZIP e Hurdle.

---

## 👤 Autore

**Simone Squizzato** · [@simonesquizzato](https://github.com/simonesquizzato)
