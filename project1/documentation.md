# Documentation

## Dataset: Isear (emotions, 7 classes)

## Tokenizer

Train BPE on ISEAR (wss is de corpus te klein en kunnen we dit in eerste instantie doen
en dan vergelijken met een tokenizer die op een soortgelijke (dus een teks over hetzelfde 
onderwerp bijvoorbeeld) maar veel grotere corpus getraind is.)


## Current Issues:

Heel erg overfitten door kleine corpus. Moeten we wat op verzinnen. Bijv zie ## tokenizer

-----------------------------------------------------------------------------------------------------------

## Modellen 

### Probleem
De MLP haalde ~99% train aan het eind van de training, en ~55% validatie.
Doel: kijken of regularisatie of een andere architectuur de validatie verbetert.

### Opzet
Alles op BoW, BPE-tokenizer (vocab 3000, min_frequency 5), 5-fold CV, 1 seed.
Het model met de laagste validatieloss wordt bewaard.

### Resultaten (gemiddelde validatie-accuracy)
| Test | Variant | Accuracy |
|---|---|---|
| Geen regularisatie | dropout 0, wd 0 | 0.5426 |
| Dropout | 0.1 / 0.3 / 0.5 | 0.5511 / 0.5426 / 0.5517 |
| Dropout | 0.8 / 0.95 | 0.5556 / 0.5364 |
| Weight decay (dropout 0.1) | 1e-3 / 1e-2 | 0.5409 / 0.5413 |
| Weight decay (dropout 0.5) | 0.1 / 1 | 0.5494 / 0.5469 |
| Weight decay (dropout 0.5) | 10 / 100 | 0.5063 / 0.2092 |
| Lagere learning rate | 1e-4, 40 epochs | 0.5509 |
| Hidden size | 8 / 16 / 32 / 64 | 0.5286 / 0.5364 / 0.5429 / 0.5474 |
| Architectuur | MLP / mlp2 128-64 / mlp2 256-128 | 0.5559 / 0.5449 / 0.5434 |
| Vocabulaire | 1000 / 3000 / 5000 | 0.5319 / ~0.547 / 0.5572 |
| min_frequency (vocab 3000) | 2 / 5 | 0.5474 / 0.5466 |

### Conclusies
- Regularisatie geeft een breed plateau rond 0.54-0.556; de meeste verschillen vallen binnen de ruis.
- Identieke instellingen verschilden tot ~1 procentpunt, dus eindrun met meerdere seeds.
- Pas bij wd 10 en 100 zien we duidelijke underfitting.
- Een kleiner model of een tweede laag verbetert de validatie niet.
- Overfitting lijkt niet de hoofdoorzaak van de lage score. BoW mist woordvolgorde, en de emoties lijken op elkaar.

### Keuzes voor de eindrun
- MLP: hidden 128, dropout 0.5, weight decay 0.1, 20 epochs
- Derde architectuur: mlp2 met lagen 128 en 64, zelfde regularisatie, patience 3
- Voorstel tokenizer: vocab 5000, min_frequency 2
- of TDIF Tokenizer (even kijken of dat de makkelijkste is)


### Resultaten eindrun
Opzet: 5-fold cross-validation met 3 seeds (0, 1, 2), dus 15 metingen per systeem. Metric: accuracy op de validatiefolds. Tokenizer: BPE, vocab 5000, min_frequency 2.
MLP en mlp2: hidden 128 (mlp2: 128 en 64), dropout 0.5, weight decay 0.1, 20 epochs (mlp2: patience 3).

| Accuracy | Indicator | BoW | TF-IDF |
|---|---|---|---|
| Logistic | 0.5216 | 0.5237 | 0.5405 |
| MLP | 0.5548 | 0.5534 | 0.5520 |
| mlp2 | 0.5410 | 0.5358 | 0.5235 |

### Bayesiaanse vergelijking (ROPE 0.5%)
| Vergelijking | Verschil | P(eerste wint) | P(gelijk) | P(tweede wint) |
|---|---|---|---|---|
| MLP indicator vs MLP BoW | 0.0014 | 0.3035 | 0.5125 | 0.1840 |
| MLP indicator vs MLP TF-IDF | 0.0028 | 0.3572 | 0.5428 | 0.1000 |
| MLP BoW vs MLP TF-IDF | 0.0014 | 0.2288 | 0.6770 | 0.0942 |
| Logistic indicator vs MLP indicator | -0.0333 | 0.0000 | 0.0001 | 0.9998 |
| Logistic BoW vs MLP BoW | -0.0297 | 0.0002 | 0.0028 | 0.9970 |
| Logistic TF-IDF vs MLP TF-IDF | -0.0115 | 0.0075 | 0.1384 | 0.8541 |
| Logistic BoW vs Logistic TF-IDF | -0.0167 | 0.0031 | 0.0487 | 0.9482 |
| MLP BoW vs mlp2 BoW | 0.0176 | 0.9158 | 0.0738 | 0.0104 |
| MLP TF-IDF vs mlp2 TF-IDF | 0.0285 | 0.9962 | 0.0035 | 0.0003 |

### Foutanalyse (logistic + BoW, 80/20-split van de trainset, accuracy 0.524)
| Emotie | Precision | Recall | F1 |
|---|---|---|---|
| joy | 0.635 | 0.657 | 0.646 |
| fear | 0.687 | 0.644 | 0.665 |
| anger | 0.451 | 0.467 | 0.459 |
| sadness | 0.553 | 0.547 | 0.550 |
| disgust | 0.492 | 0.515 | 0.503 |
| shame | 0.413 | 0.410 | 0.412 |
| guilt | 0.436 | 0.421 | 0.429 |

Meest verwarde paren: shame/guilt (68), anger/guilt (45), anger/disgust (43), disgust/shame (40), joy/sadness (36).

### Conclusies eindrun
- De drie MLP's zijn de beste systemen en zijn onderling gelijk binnen de ROPE.
- De MLP verslaat logistic op elke representatie, het duidelijkst op indicator en BoW.
- TF-IDF helpt logistic, maar maakt voor de MLP niets uit.
- De tweede laag helpt niet: mlp2 scoort lager dan de MLP, vooral op TF-IDF.
- De fouten zitten vooral bij emoties die op elkaar lijken, zoals shame en guilt.

### Beperkingen
- De MLP-scores gebruiken de beste epoch op validatie, logistic niet. Dat kan de MLP's iets bevoordelen.
- De tokenizer en de TF-IDF zijn getraind op alle trainteksten, dus ook op de validatiefolds.
- Logistic gaf 27 keer een ConvergenceWarning bij max_iter 200.
- De testsplit is niet gebruikt; alle scores komen uit cross-validation.

## ROPE (Region Of Practical Equivalence)

We use a ROPE of ±0.005 accuracy (0.5 percentage points). If the difference between two systems is smaller than this, we treat them as practically equal. The Bayesian comparison reports three probabilities: A is better, practically equal (tie), and B is better.

We chose 0.005 for two reasons.

1. **Convention.** The NLP1 tutorial uses 0.005, because accuracy gains below about half a percentage point are usually seen as uninteresting in NLP papers. We follow this convention.

2. **Variation between runs.** In our tuning, identical settings gave accuracies that differed by up to about 1 percentage point. A difference of 0.5 percentage points is smaller than this, so it falls within the range of run-to-run variation we observed. We therefore treat systems that differ by less than this margin as practically equivalent.

## Checklist

What **we expect you to do**:

- [] Compare different systems of text classification for your choice of data/task.
- Your comparisons should cover:
    - [ ] the two representation spaces in the tutorial (token indicators and bag-of-words) and an additional one of your choice.
    - [ ] the two kinds of architectures in the tutorial (log-linear and single-hidden-layer MLP) and an additional one of your choice (such as a modification of the MLP).
- [ ] Your comparisons must be based on cross-validation and a relevant metric (accuracy or else).
- [ ] You should make use of Bayesian estimation to help you analyse results in relation to a region of practical equivalence (ROPE), and you can motivate the ROPE size yourself.

What you need to **bring to the oral assessment**:
* You will be asked questions about the points above, you can bring a slide deck with up to five slides (one for each of the following topics _data and metrics_, _features_, _architectures_, _results_, _extra_); you _will not_ give a presentation, rather, these slides are to be thought of as "backup slides" that you can use to guide your answers to the questions the TA will ask you.
* You may also be asked questions about the tutorial, for those, you don't need to prepare any deliverable (a good way to prepare is to work through the tutorial and reflect on the questions there).

What you should **keep in mind** given that you will eventually write a blog post about your work throughout the course:
* Organise your code;
* Summarise your motivations and findings;
* Document your sources and any form of AI-assistance you use.
