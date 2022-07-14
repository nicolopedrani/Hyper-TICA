# folders  

- restart: 
    * 10\_sim *run*: riprovare. Anche se in generale non necessaria. Ci vuole una stopping rule più che altro
    * 10\_sim\_shuffle: riprovare. Come sopra
    * all\_data: mi porto dietro tutti i dati senza mischiarli per il training. Divido le batch usando il differenti lag time
    * all\_data\_shuffle: mi porto dietro tutti i dati e mischio tra loro le batch. Ho una batch per ogni lag time usato. Mischio dati ottenuti dallo stesso lag time.  
    * all\_data\_shuffle_unbias: senza riscalare il tempo. nessuna transizione
    * all\_data\_unbias: senza riscalare il tempo. Si rompe anche il sistema 
    * all\_data\_Micheletime: nessuna transizion
    * all\_data\_shuffle\_Micheletime: nessuna transizione
    * Test: alcuni tentativi per avere una transizione già dopo la prima iterazione. MetaD, Opes Explore, PCA. 
