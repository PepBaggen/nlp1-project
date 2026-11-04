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
