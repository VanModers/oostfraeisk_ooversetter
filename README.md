# README

## Oostfräisk wersjoon

Dit repository daajt 'n oostfräisk ooversetter-KI undhollen, däi upbaauend up dat Helsinki-NLP/opus-mt-de-nl model (https://huggingface.co/Helsinki-NLP/opus-mt-de-nl) un dat Helsinki-NLP/opus-mt-en-de model (https://huggingface.co/Helsinki-NLP/opus-mt-en-de) undwikkelt wur.

In d' ```data```-ördner befinnent sük däi dóótensatsen bestóónd fan düütsk satsen un höör oostfräisk ooversettens. In däi ```pretrained_de_frs```-ördner is däi code föör däi ooversetter. Däi satsen wordent döör dat ```dataset_creator.py```-skript lóóden un nóó 'n ```torch.utils.data.Dataset``` kunwertäärt.
Dat ```trainer.py```-skript word't föör 't treenäären fan dat ooversetter-netwaark bruukt.
Mit ```main.py``` kan däi ooversetter test't worden.
Mit ```trainingset_creator_app.py``` kan däi dóótensats föör 't treenäären in ```data/krektüren``` mit düütsk satsen fan ```data/krektüren/deu.txt``` (https://www.kaggle.com/datasets/kaushal2896/english-to-german) as input ferwiidert worden. Doo däi ooversetter al gâaud ooversettens proodusäärt köönent däi ook gewoon oovernoomen worden. ```trainingset_creator_sentences.py``` daajt dat sülviğ, man dan sünner interface. Bâajd skripten fangent altiid bii 'n taufalliğ düütsk sats an.

Upstünds daajt dat repo twei ooversetter-models undhollen. 'N model um fan Düütsk nóó Oostfräisk t' ooversetten (de_frs_model) un 'n model um fan Oostfräisk nóó Düütsk t' ooversetten (frs_de_model).

## English version
This repository contains an East Frisian translation AI based on the Helsinki-NLP/opus-mt-de-nl model (https://huggingface.co/Helsinki-NLP/opus-mt-de-nl) and the Helsinki-NLP/opus-mt-en-de model (https://huggingface.co/Helsinki-NLP/opus-mt-en-de).

The ```data``` folder contains the data set consisting of German sentences and their East Frisian translations. The ```pretrained_de_frs``` folder contains the code for the translator. The sentences are loaded by the ```dataset_creator.py``` script and converted to a ```torch.utils.data.Dataset``` .
The ```trainer.py``` script is used to train the translator network.
The translator can be tested with ```main.py```.
With ```trainingset_creator_app.py```, the data set for training in ```data/krektüren``` can be expanded with German sentences from ```data/krektüren/deu.txt``` (https://www.kaggle.com/datasets/kaushal2896/english-to-german) as input. If the translator already produces good translations, these can simply be adopted. ```trainingset_creator_sentences.py``` does the same thing, but without an interface. Both scripts always start with a random German sentence.
The repo currently contains two translator models. One model for translating from German to East Frisian (de_frs_model) and one model for translating from East Frisian to German (frs_de_model).

## Model performance on validation data set
![alt text](<performance on validation data.png>)
-------------------------------------------------------------------------------------
de_frs_model_2
Token-wise Accuracy: 0.7635
Avg Loss: 0.1550

-------------------------------------------------------------------------------------
de_frs_model_3
Token-wise Accuracy: 0.7872
Avg Loss: 0.1545

-------------------------------------------------------------------------------------
de_frs_model_5
Token-wise Accuracy: 0.8049
Avg Loss: 0.1505

-------------------------------------------------------------------------------------
de_frs_model_6
Token-wise Accuracy: 0.8274
Avg Loss: 0.1409

-------------------------------------------------------------------------------------
de_frs_model_7
Token-wise Accuracy: 0.8418
Avg Loss: 0.1224

-------------------------------------------------------------------------------------
de_frs_model_8 (current model)
Token-wise Accuracy: 0.8456
Avg Loss: 0.1085