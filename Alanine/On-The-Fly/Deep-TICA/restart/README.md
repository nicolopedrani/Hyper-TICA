# README file  
  
C'è un bug nel codice. La simulazione riparte sempre dalla posizione iniziale.. e non dall'ultima raggiunta nella precedente simulazione.  
Rincontrollare se gromacs fa quello che deve. 

### NOTE  
Può capitare che il sistema di rompi. Forse è meglio scegliere una barrier fin da subito di 30 al posto che 35.. 
per non far credere alla nuova cv trovata che si possa andare in regioni di rottura. Oppure come sto facendo ora partire  
da una certa soglia. E se il sistema di rompe riprovare con una soglia più bassa. (-3 per volta ad esempio.. si può ragionare in 
termini di percentuale).  
  
**training**: direi che non è intelligente portarsi dietro tutti i dataset. Per diversi motivi. Il primo è che si rischia di finire  sempre in un minimo della rete dettato dai troppi dati simili tra loro (la prima parte della simulazione darà sampling simili tra loro). Il secondo lo stesso training può essere davvero strano.. guardo esempio simulazione bias16 per `no\_restart`  
  
**sampling**: supponiamo di fare una simulazione e partire da un minimo e poi finire in un altro minimo. Quindi spendiamo la maggior parte del nostro tempo nel primo bacino (mettiamo il 48.5 % del tempo) e nel secondo bacino (mettiamo il 48.5 % del tempo) e quindi sullo stato di transizione solo 1%.
Se applichiamo l'analisi della deep-tica a questo data set la deeptica trova i due bacini e li identifica con un certo valore. Tipo -1 al primo bacino e 1 al secondo bacino. Ma cosa succede li in mezzo? Se non ho transizioni allora in mezzo ho il gradiente della deeptica.. ma una delta per i due stati. Questo non va bene, perchè non esco dai due stati.  
  
**cose da fare**: obiettivi:
-   non trainare la rete con tutti i dati.. ma usarne solo alcuni. All'inizio potrebbero essere solo i primi 10. Ma ci vorrebbe un criterio più intelligente. Tipo i più simili non mi interessano.  
-   capire come tenere tutti i dati per poi stimare delle proprietà statiche. Come l'energia libera o altri valori medi.  