# simulations  

## Nota
Saranno comunque tutte da rilanciare e fatte bene. Un parametro che voglio usare per capire quando fermare una simulazione  
è il *WORK* fatto dalla cv. Questa quantitò può essere calcolata on the fly con OPES.  

- restart: 
    * 10\_sim *run* --> non efficiente. Mai una transizione in 30 iterazioni
    * 10\_sim\_shuffle --> leggi nota *run* --> non efficiente. Mai una transizione in 30 iterazioni
    * all\_data: mi porto dietro tutti i dati senza mischiarli per il training. Divido le batch usando il differenti lag time
    * all\_data\_shuffle: mi porto dietro tutti i dati e mischio tra loro le batch. Qui però la cosa importante è che ogni batch si riferisce comunque ad un singolo valore di lag time
    * all\_data\_single: mi porto dietro tutti i dati e divido le batch usando un solo lag time e senza mischiarle (ancora da lanciare)
    * all\_data\_shuffle: con la stessa idea dello shuffle, ma usando diversi lag time per formare le batch. In questa versione quindi in realtà `shuffle=False`. Semplicemente mischio tra loro le coppie di dati che si riferiscono a diversi lag time e ogni batch corrisponderà a un lag time differente ma con coppie mischiate.  (ancora da lanciare)
    * all\_data\_Micheletime (usando il tempo riscalato vero e proprio senza riscalare il bias)
    * all\_data\_shuffle\_Micheletime


A Michele non piace far ripartire una simulazione. 
** not yet **
- no\_restart: 
    * 10\_sim 
    * 10\_sim\_shuffle 
    * all\_data 
    * all\_data\_shuffle 

**nota**: simulazione shuffle: prendo solo un lag time (per ora) uguale a 1. non ha senso mischiare coppie con diverso lag time.
potrebbe avere senso mischiare coppie con stesso lag time ma provenienti da sampling diversi. 