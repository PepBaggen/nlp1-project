Nummers tussen haakjes verwijzen naar de stappen uit "Get to work".

## Al gedaan

- **Embedding model:** GloVe 2024 wikigiga 50d geladen (in notebook) in embeddings.py staan ook andere die we nog kunnen testen

- **Tokenizers:** `bow` (zonder leestekens/stopwoorden) en `rnn` (alles behouden, koppeltekens gesplitst)

- **Coverage:** 99.3% (bow) en 99.7% (rnn) token coverage (~95% type coverage - unique token coverage) - Bij rnn werden tokens met koppeltekens als <unk> gezien -> een tokenizer gemaakt die met regular expressions de koppeltekens vervangt voor spaties (99.3 -> 99.7)

- **Similariteit en buren:** emoties liggen dicht bij elkaar (0,53–0,87), tegengestelden ook.

- **Pipeline (tutorialinstellingen, 3 seeds × 5 folds):**
  | classifier (MLP) | 2 verborgen lagen van 50, dropout 0,5 (embed-and-pool)| 2 verborgen lagen van 50, dropout 0,1 (lstm)
  | optimizer | AdamW, lr 1e-3, weight decay 1e-4
  | batch size / epochs | 64 / 20
  | modelselectie | epoch met de laagste validatie-loss

  Met standaard instellingen (frozen embeddings):

  embed-and-pool mean=0.3871 std=0.0125 (n=15)
  lstm mean=0.4586 std=0.0118 (n=15)

  Met standaard instellingen (trainable embeddings):

  embed-and-pool mean=0.5205 std=0.0170 (n=15)
  lstm mean=0.4582 std=0.0228 (n=15)

  trainable params - dropout 0.1 -> 0.5, hidden size 50 -> 200, epochs 20 -> 35

  embed-and-pool mean=0.5545 std=0.0198 (n=15)

  MLP ------- dropout 0.1 -> 0.5, hidden size 50 -> 200, epochs 20 -> 60
  encoder --- dropout -> 0,2, hidden dim 50 -> 100

  lstm mean=0.5259 std=0.0247 (n=15) -- trainable params
  lstm mean=0.5581 std=0.0156 (n=15) -- frozen params

  (project 1: ~0,55)


Pep:

- **Modal (GPU):** project2/modal_run.py traint alle configuraties tegelijk op Modal (T4), zelfde 3 seeds × 5 folds. Logs in results/project2/modal/ (progress.txt, results.txt, per config .txt met curves per epoch + .json). Laad de logs daarna in in de training cell (veel efficienter dan in de notebook op een slechte CPU trainen)

- **Check:** pool-base op Modal geeft mean=0.5509 (hiervoor lokaal 0.5530/0.5545) -> zelfde orde, verschil is training randomness

- **Embedding model (fastText 300d vs GloVe 50d):** grootste winst van alles

  emb-lstm-fasttext300 mean=0.5993 std=0.0206 (n=15) -- frozen, beste tot nu toe
  emb-pool-fasttext300 mean=0.5776 std=0.0188 (n=15) -- trainable, beste epoch al 3-4 (overfit snel)
  r2-ft-lstm-hidden200 mean=0.5907 std=0.0180 (n=15)s
  r2-ft-pool-lr3e-4    mean=0.5710 std=0.0242 (n=15)
  r2-ft-pool-frozen    mean=0.5547 std=0.0099 (n=15)
  r2-ft-lstm-trainable mean=0.5090 std=0.0361 (n=15) -- overfit na 3 epochs

  GloVe 300d lukte niet: download van Stanford geeft 'No route to host' (ook vanaf mac) -> dimensie en model kunnen we dus niet los van elkaar testen

- **Kleine tweaks GloVe 50d:** binnen de ruis

  lstm-hidden200 mean=0.5585 | lstm-lr3e-3 mean=0.5507 | pool-sum mean=0.5436 (mean pooling iets beter)
  lstm-epochs100 mean=0.5562 = precies lstm-base -> beste epoch steeds < 44, de lstm WAS dus al geconvergeerd

- **Data size (10/25/50/100% van het traindeel per fold, validatie blijft volledig):**

  | | 10% | 25% | 50% | 100% |
  | pool GloVe frozen | 0.367 | 0.386 | 0.392 | 0.398 |
  | pool GloVe trainable | 0.400 | 0.457 | 0.504 | 0.551 |
  | pool fastText trainable | 0.443 | 0.495 | 0.535 | 0.578 |
  | lstm GloVe frozen | 0.324 | 0.404 | 0.495 | 0.556 |
  | lstm GloVe trainable | 0.302 | 0.386 | 0.461 | 0.520 |
  | lstm fastText frozen | 0.365 | 0.469 | 0.543 | 0.599 |

  -> pool met frozen embeddings plateaut al bij 10% (~0.40), alles leert dan via de embeddings
  -> lstm verliest van pool bij weinig data, haalt in vanaf ~50% (compositie vraagt meer data)
  -> alle curves (behalve pool frozen) stijgen nog bij 100% -> meer data zou nog helpen
  -> trainable embeddings: helpt pool, schaadt lstm (overfit, beste epoch ~10)

## Alle configs (Modal), gesorteerd op accuracy

Vaste settings overal: MLP 200-200, dropout 0.5, word dropout 0.2 (lstm), AdamW weight decay 1e-4, batch 64, 3 seeds × 5 folds (n=15), modelselectie op laagste val loss.

| config | model | embeddings | emb | data | lstm hidden | pooling | lr | epochs | mean | std | best epoch (mediaan) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| emb-lstm-fasttext300 | lstm | fastText 300d | frozen | 100% | 100 | - | 0.001 | 50 | 0.5993 | 0.0206 | 21 |
| r2-ft-lstm-hidden200 | lstm | fastText 300d | frozen | 100% | 200 | - | 0.001 | 50 | 0.5907 | 0.0180 | 16 |
| emb-pool-fasttext300 | pool | fastText 300d | trainable | 100% | - | mean | 0.001 | 35 | 0.5776 | 0.0188 | 3 |
| r2-ft-pool-lr3e-4 | pool | fastText 300d | trainable | 100% | - | mean | 0.0003 | 35 | 0.5710 | 0.0242 | 8 |
| lstm-hidden200 | lstm | GloVe 50d | frozen | 100% | 200 | - | 0.001 | 50 | 0.5585 | 0.0206 | 29 |
| lstm-epochs100 | lstm | GloVe 50d | frozen | 100% | 100 | - | 0.001 | 100 | 0.5562 | 0.0156 | 38 |
| lstm-base | lstm | GloVe 50d | frozen | 100% | 100 | - | 0.001 | 50 | 0.5562 | 0.0156 | 38 |
| r2-ft-pool-frozen | pool | fastText 300d | frozen | 100% | - | mean | 0.001 | 35 | 0.5547 | 0.0099 | 33 |
| pool-base | pool | GloVe 50d | trainable | 100% | - | mean | 0.001 | 35 | 0.5509 | 0.0179 | 22 |
| lstm-lr3e-3 | lstm | GloVe 50d | frozen | 100% | 100 | - | 0.003 | 50 | 0.5507 | 0.0141 | 19 |
| pool-sum | pool | GloVe 50d | trainable | 100% | - | sum | 0.001 | 35 | 0.5436 | 0.0193 | 25 |
| r2-ft-datasize-lstm-0.5 | lstm | fastText 300d | frozen | 50% | 100 | - | 0.001 | 50 | 0.5429 | 0.0206 | 20 |
| r2-ft-datasize-pool-0.5 | pool | fastText 300d | trainable | 50% | - | mean | 0.001 | 35 | 0.5348 | 0.0171 | 4 |
| datasize-lstm-trainable-1.0 | lstm | GloVe 50d | trainable | 100% | 100 | - | 0.001 | 50 | 0.5200 | 0.0207 | 10 |
| r2-ft-lstm-trainable | lstm | fastText 300d | trainable | 100% | 100 | - | 0.001 | 50 | 0.5090 | 0.0361 | 3 |
| datasize-pool-trainable-0.5 | pool | GloVe 50d | trainable | 50% | - | mean | 0.001 | 35 | 0.5044 | 0.0199 | 23 |
| r2-ft-datasize-pool-0.25 | pool | fastText 300d | trainable | 25% | - | mean | 0.001 | 35 | 0.4947 | 0.0262 | 6 |
| datasize-lstm-frozen-0.5 | lstm | GloVe 50d | frozen | 50% | 100 | - | 0.001 | 50 | 0.4946 | 0.0175 | 37 |
| r2-ft-datasize-lstm-0.25 | lstm | fastText 300d | frozen | 25% | 100 | - | 0.001 | 50 | 0.4689 | 0.0253 | 17 |
| datasize-lstm-trainable-0.5 | lstm | GloVe 50d | trainable | 50% | 100 | - | 0.001 | 50 | 0.4612 | 0.0234 | 11 |
| datasize-pool-trainable-0.25 | pool | GloVe 50d | trainable | 25% | - | mean | 0.001 | 35 | 0.4565 | 0.0204 | 25 |
| r2-ft-datasize-pool-0.1 | pool | fastText 300d | trainable | 10% | - | mean | 0.001 | 35 | 0.4428 | 0.0210 | 11 |
| datasize-lstm-frozen-0.25 | lstm | GloVe 50d | frozen | 25% | 100 | - | 0.001 | 50 | 0.4042 | 0.0193 | 28 |
| datasize-pool-trainable-0.1 | pool | GloVe 50d | trainable | 10% | - | mean | 0.001 | 35 | 0.4004 | 0.0184 | 28 |
| datasize-pool-frozen-1.0 | pool | GloVe 50d | frozen | 100% | - | mean | 0.001 | 35 | 0.3980 | 0.0130 | 30 |
| datasize-pool-frozen-0.5 | pool | GloVe 50d | frozen | 50% | - | mean | 0.001 | 35 | 0.3921 | 0.0137 | 33 |
| datasize-lstm-trainable-0.25 | lstm | GloVe 50d | trainable | 25% | 100 | - | 0.001 | 50 | 0.3864 | 0.0247 | 12 |
| datasize-pool-frozen-0.25 | pool | GloVe 50d | frozen | 25% | - | mean | 0.001 | 35 | 0.3859 | 0.0128 | 34 |
| datasize-pool-frozen-0.1 | pool | GloVe 50d | frozen | 10% | - | mean | 0.001 | 35 | 0.3665 | 0.0157 | 32 |
| r2-ft-datasize-lstm-0.1 | lstm | fastText 300d | frozen | 10% | 100 | - | 0.001 | 50 | 0.3650 | 0.0303 | 18 |
| datasize-lstm-frozen-0.1 | lstm | GloVe 50d | frozen | 10% | 100 | - | 0.001 | 50 | 0.3244 | 0.0248 | 22 |
| datasize-lstm-trainable-0.1 | lstm | GloVe 50d | trainable | 10% | 100 | - | 0.001 | 50 | 0.3019 | 0.0313 | 13 |

(project 1 MLP + indicator: CV 0.5548, test 0.5504)

## let op

- in project2.ipynb gebruiken we trained_emb_model voor analyse aan het begin maar gebruik later isear_emb_model om te trainen (isear_emb_model is het emb tabel met alleen isear tokens)

## Goede punten voor discussie

- Embedding model was de bottleneck, niet de architectuur: fastText +4 procentpunt, andere tweaks max ~0.01 (binnen ruis)

- Frozen vs trainable hangt af van het model: pool wil trainable, lstm wil frozen (te veel parameters op ~4.8k zinnen)

- De tokenizer aangepast zodat koppeltekens '-' vervangen worden door spaties ' ' zodat woorden met koppeltekens niet de embedding <unk> krijgen. Verder meer dan 99 procent coverage!

- In onze embedding tabel zijn alle tokens verwijderd die niet in de ISEAR dataset voorkomen. (embed and pool nu 3x sneller ong over 3 seeds 5 folds en lstm 15% sneller)
  -> resultaten zijn overigens natuurlijk hetzelfde

## TO DO

- ~~Smalle tweaks maken om te zien wat een beetje potentie heeft om effect te hebben (zoals rnn trainen, lijkt nog niet te zijn geconvergeerd)~~ gedaan (zie Pep), lstm was wel geconvergeerd

- ~~Vervolgens een groot trainingsscript maken zodat we alle configuraties op de gpu kunnen trainen~~ gedaan: project2/modal_run.py

- Bayesiaanse vergelijking (zoals project 1) van lstm fastText vs project 1 en vs pool fastText

- Eindsysteem kiezen (lstm + fastText frozen?) en één keer op de testset evalueren (project 1 test: 0.5504)
