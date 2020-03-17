# 02466 Projektarbejde i Kunstig Intelligens og Data LOGBOOG
* Søren Winkel Holm s183911@student.dtu.dk
* Asger Laurits Schultz s183912@student.dtu.dk 
* Anne Agathe Pedersen s174300@student.dtu.dk


## Projektmøder
### Uge 1:  
#### 6/2-2020
Mødtes og valgte projektprioriteter.

Opgaver før næste møde: 
* Alle: Test at LaTeX virker lokalt og at report.tex kan kompileres. 
* Alle: Test at `pytest` virker, at kommandoen `pytest` kan køre og giver, at alle tests accepteres på eksemplet. 
* Alle: Begynd at lede efter eksisterende litteratur, når vi får projektet tildelt på mandag. 

### Uge 2:
#### 12/2-2020
Diskuterede implementation af Rubiks terning og besluttede at begynde med en `torch`-implementation som inkluderer muligheden for en $N\times N\times N$-terning. 
Skabte overblik over projektplan. 
Skrev samarbejdskontrakt. 
Skrev læringsmål. 

Opgaver før næste møde (fredag):
* Asger: Implementér en Rubiks-klasse i `torch`, som tager handlinger og returnerer terningen + tjekker om løst. Fokusér på at få en $3\times 3\times 3$ til at virke før næste møde. 
* Anne: Find spørgsmål til møde med vejleder, som kan blive brugt til at skrive projektbeskrivelsen + find foreslag til artikel om gruppeteori. 
* Søren: Opsæt Gantt og projektlærred + opsæt TeX-system for rapport og projektbeskrivelse. 

Opgaver før næste vejledermøde: 
* Alle: Læs al materiale givet af Mikkel. 
* Alle: Fortsat søg efter litteratur. 
* Asger & Søren: Læs anbefalet artikel om gruppeteori. 

#### 14/2-2020
Anne anbefaler at læse Alex Weins artikel om gruppeteori i relation til Rubiks terning. 
Vi fandt nogle flere spørgsmål til vejledermødet. 
Asger gennemgik sin implementation af Rubiks terning. 

Opgaver før næste vejledermøde: 
* Samme som fra 12/2-2020.
* Asger & Søren: Læs Alex Wein-artikel. 
* Anne & Søren: Skriv tests til Asgers implementation. 
* Anne: Undersøg køb af rigtig Rubiks terning. 

### Uge 3:
#### 19/2-2020
Vi aftaler at skrive rapporten på dansk. 
Vi aftaler at skrive kodekommentarer på engelsk. 
Fagtermer skrives som udgangspunkt på dansk i rapporten, men skal også have den engelske oversættelse ved siden ad. 

Vi har lavet projektlærred, Gantt-plot, projektbeskrivelse og læremål. 
Det hele er blevet sendt til Mikkel for feedback. 

Opgaver inden næste møde (fredag): 
* Anne oversætter tidligere projektindlæg fra engelsk til dansk. 
* Søren sender alt i projektbeskrivelsen i en mail til Mikkel. 

#### 21/2-2020
Der blev aftalt at finde ressourcer på DQ-læring og afsøgning og diskuteret omstændigheder for agent-, træning- og model-struktur

Opgaver inden vejledermøde:
* VIGTIGT: Alle: Læs op på og find materiale til Dyb Q-læring og afsøgning
* Søren: Implenetere evalueringsmodul + agentklasse, der kan modtage state af rubiks og outputte handling + implemtér randomagent
* Asger: Påbegynd modelmodulet og træningsmodul
  
#### 27/2-2020
Vi aftaler at træne et værdi- og policynetværk på autodidaktisk iteration og tilføjer træsøgning til det bagefter. 
Mandag sender vi Mikkel en liste over papers.
Vi mødes igen fredag i næste uge.

Opgaver inden næste møde:
* Alle: Find referencer og skriv dem på en fælles liste.
* Alle: Begynd på state of the art.
* Anne: Begynd på gruppeteori.

### Uge 5
#### 6/3-2020
* Implementér agent til BFS
* Kode til ADI og træningsløkke (tage KDK som argument, evt. understøtte KeybordInterrupt) - Søren
* Kode til NN + konfigurationsdataklasse - Asger
* Skrive udkast til introduktion og begynder på data
* Skrive gruppeteoriafsnit efter stadieafkunstenafsnit - Anne

### Uge 7
#### 17/3-2020
Hvad vi skal skrive på rapporten: 
* Asger: Terningemiljøet (3.1) og fix 2.3. Opdatér terningemiljøet i koden, så den bruger den nye repræstentaion (lavprioritet). DeepCube (1.2.1)
* Søren: 3.2, 3.4, 1.3
* Anne: MCTS (3.3). Guds tal i data



## Vejledermøder
### Uge 3:
#### 19/2-2020
- Relater til andre diskrete kombinatoriske problemer, fx proteinfoldning og solcelleoptimering
- Projekt kan gå i flere retninger
- God adgang til flere GPU'er på Compute, men ellers bare skaler problemet ned
- Værdiapproksimation skal kombineres med MCTS eller anden søgning for at virke
- God idé: Reproducér DeepCubeA, men skær evt. nogle hjørner af
- Brug A*
- Se moves som 12 forskellige, ikke 6x2
- Læs om tabulær Q learning (tabel i stedet for NN)
  - Q learning har state-action-par med værdier med evt. discount
  - Ofte skal værdier approksimeres
- State space er for stort til tabulær Q, men Deep Q kan hjælpe
- Primært mål er at lære om metoder, men resultater er selvfølgeligt også godt
- Mikkel ville skrive engelsk, men dansk er en fin mulighed
- Anbefaler 40-50 sider, da der skal meget med ud over det faglige
- Konvolutioner kan køres på naboer i graf (hvis vi laver konvoer) - kræver gruppeteori
  - Spændende, men svært
- Brug kun gruppeteori, hvis det giver faglig mening
- Næste møde 27/2-2020 kl 10

### Uge 4:
#### 27/2-2020
- Det giver god mening at implementere autodidaktisk iteration først
- DFS, A*. Lav flere forskellige modeller, så vi har noget at sammenligne med. Start med DFS
- State of the art: RL brugt på diskret optimering
- Beskriv det brede felt, gå derefter ind på vores underfelt
  - Se på hvilke referencer papers har
  - Se på hvilke papers der citerer. Brug Google Scholar
- MCTS kan bruges undervejs til at søge mens den lærer
- Vi holder den nok på en 3x3, men 2x2 er også en mulighed
- Send links til referencer til Mikkel
  - Find paper, skriv et par linjer om hvad det handler om
  - Lav liste over feltet og send til Mikkel
- Næste møde d. 10. marts kl. 10


### Uge 6:
#### 10/3-2020
- Anbefaling at skrive teoretisk afsnit i nemmere termer 
- Anbefaling at fokusere på ADI + træafsøgning og så følger nærmere optimering af læringsproces senere
- Gennemgang af ADI


### Uge 7: 
#### 17/03-2020
- Vi har skrevet en del, men er også begyndt at træne policy- og værdi-netværk
- Det virker ret dårligt. Det har en kraftig tendens til at divergere
  - Prøv at træne ét eller to moves fra goal state. Det hjælper til at kunne se om de får de rigtige værdier
  - Lad være med at bruge RMSprop, lav det om til almindelig gradientnedstigning
  - Del kode for kostfunktion med Mikkel, så han kan se på det 
- Brug forskningsspørgsmål til at skrive udkastet til metodeafnitttet - hvilke metoder vil vi bruge til at undersøge spørgsmålene?
- Det er bedre at undersøge færre ting og så komme til bunds i det, i stedet for at undersøge en masse forskellige ting på overfladen



