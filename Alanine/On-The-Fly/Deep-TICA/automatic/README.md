# simulations  

- restart: 
    * 10\_sim *run* --> non efficiente. Mai una transizione in 30 iterazioni
    * 10\_sim\_shuffle --> leggi nota *run* --> non efficiente. Mai una transizione in 30 iterazioni
    * all\_data *run*
    * all\_data\_shuffle *run*
    * all\_data\_Micheletime (usando il tempo riscalato vero e proprio senza riscalare il bias)
    * all\_data\_shuffle\_Micheletime


A Michele non piace far ripartire una simulazione. 
** not yet **
- no\_restart: 
    * 10\_sim *cuda full*
    * 10\_sim\_shuffle *cuda full*
    * all\_data *cuda full*
    * all\_data\_shuffle *cuda full*

**nota**: simulazione shuffle: prendo solo un lag time (per ora) uguale a 1. non ha senso mischiare coppie con diverso lag time.
potrebbe avere senso mischiare coppie con stesso lag time ma provenienti da sampling diversi. 