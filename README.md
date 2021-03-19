# CUDARayTracer
![Alt Text](https://s4.gifyu.com/images/Untitled8772bc18b6640d85.gif)

Progetto realizzato per il corso di Computer Graphics.

## Features

Il progetto prende spunto da 'Raytracing in one weekend', 'Raytracing the next week' e le slide del corso ed modifica il raytracer illustrato con diverse features che permettono di sfruttare al meglio le GPU moderne.
In particolare, oltre all'enorme parallelismo permesso dall'architettura delle GPU, le features salienti sono:
* Le GPU che permettono l'esecuzione di codice CUDA forniscono diverse tipologie di memorie ed ogniuna di queste tipologie ha delle caratteristiche che la rendono adatta a dei compiti specifici. In particolare: la *constant memory* è stata utilizzata per tenere in memoria i dati riguardanti la scena da renderizzare, la *shared memory* è stata utilizzata per eseguire la comunicazione tra threads che renderizzano pixel adiacenti e la *texture memory* è stata utilizzata per contenere in memoria le textures.
* La comunicazione tra i thread ha permesso di individuare as un costo computazionalmente basso in quali porzioni dell'immagine si trovano i bordi degli ogetti e di conseguenza quali sono le porzioni dove è più critica l'applicazione dell'antialiasing.
* Sono state sfruttate le chiamate a funzione asincrone e gli stream per massimizzare l'utilizzo della CPU e della GPU in parallelo, soprattutto nel processo di caricamento dei dati sulla GPU.

Questi accorgimenti hanno permesso di creare un raytracer molto efficiente. I 100 frames che compongono la gif in alto sono stati renderizzati ad una media di 100ms per ogni frame utilizzando una GT 710 che ha 192 CUDA cores, una scheda video di fascia bassa di ultima generazione (Es.: RTX 3060, con 3584 CUDA cores) dovrebbe renderizzare ogni frame in circa 5ms. 
