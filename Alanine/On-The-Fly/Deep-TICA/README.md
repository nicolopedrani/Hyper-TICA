# Reinforcenment Deep-TICA, note
  
## folders

**automatic**: dentro qui ci sono simulazioni diverse. Ciascuna usa un tempo riscalato diverso. A quanto pare la versione well-tempered funziona. Non chiaro perchè.  

**altre idee**: 
- usare un bias static con EXTERNAL method in plumed. Guardare come fare e settare il bias = -F(cv). Guardo file di INVE 
-  ad ogni iterazione aggiungere come descrittore anche la cv precedente. Questo in realtà è sensato. Nel senso che i descrittori devo essere sia quantità fisiche che rappresentino il sistema, ma allo stesso tempo anche delle quantità che scorrelano lentamente nel tempo. Occorrerà pensarci.    
- fino ad ora con la alanina ho sempre usato due cv per volta. Si può pensare di usarne solo 1. Per ora aspetto ma è tra le cose da provare
- è possibile ricavare la funzione di correlazione temporale ripesando la distribuzione dei dati? Occorrerebbe provare con dei ripesamenti e vedere cosa succede. Ripesare le coppie non è banale. Pensavo di poter usare dei valori medi dei descrittori (che possono essere ricavati da una simulazione biased) ma a tempi diversi.. questo non è banale. Occorrerebbe lanciare diverse simulazioni e fare una media. Ma anche questo non mi convince.    
