# 02466 Project work in Artificial Intelligence and Data LOGBOOK

* Søren Winkel Holm s183911@student.dtu.dk
* Asger Laurits Schultz s183912@student.dtu.dk 
* Anne Agathe Pedersen s174300@student.dtu.dk


## Project Meetings

### Week 1:  
#### 6/2-2020
Met and chose project priorities.

Tasks before next meeting:
* All: Test that LaTeX works locally and that report.tex can be compiled.
* All: Test that `pytest` works and that the command `pytest` can be run and gives that all tests are passed on the example script.
* All: Start researching for existing literature when we get the project on monday.

### Week 2:
#### 12/2-2020
Discussed implementation of Rubiks and settled for starting with a `torch` implentation which includes possibility for $N\times N\times N$ cube. 
Getting overview of project plan. 
Wrote project contract.
Wrote learning objectives.

Tasks before next meeting (friday):
* Asger: Implement a Rubiks environment class in `torch` which takes in actions and returns the board position + checks for win. Focus on getting $3\times 3\times 3$ to work before next meeting.
* Anne: Find questions for meeting with supervisor that can be used to write the project description + find recommendation for group theory paper.
* Søren: Set up Gantt and Project Canvas + set up TeX-system for report and project description.

Tasks before supervisor meeting:
* All: Read all material given by Mikkel. 
* All: Continue search for literature.
* Asger & Søren: Read recommended group theory paper.
  
#### 14/2-2020
Anne recommends reading Alex Wein's paper on group theory in relation to the Rubik's cube.
We found more questions for the supervisor meeting. 
Asger explained his implementation of the Rubik's cube. 

Tasks before supervisor meeting: 
* Same as from meeting on 12/2-2020.
* Asger & Søren: Read Alex Wein paper.
* Anne & Søren: Write tests for Asger's implementation. 
* Anne: Buy an IRL Rubik's cube. 


## Supervisor Meetings
 
### Week 3

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

### Week 4

#### 27/2-2020




