# Litteratursøgning for Rubiks terning og RL 


## Søgeord

* Rubik's cube reinforcement learning
* Solving Rubik's cube
* Deep learning 
* Reinforcement learning discrete combinatorial problems



## Fundet litteratur

### Kernestof

*  Forest Agostinelli, Stephen McAleer, Alexander Shmakov and Pierre Baldi, "Solving the Rubik's cube
with deep reinforcement learning and search", URL: https://www.nature.com/articles/s42256-019-0070-z *Hoved-artiklen*

* McAleer, Agostinelli, Shmakov and Baldi, "Solving the Rubik's Cube with Approximate Policy Iteration", URL: https://openreview.net/forum?id=Hyfn2jCcKm. *Introducerer autodidaktisk iteration og kombinerer med MTCS for at løse Rubiks terning i max 30 træk.*

* McAleer, Agostinelli, Shmakov and Baldi, "Solving the Rubik's Cube Without Human Knowledge", URL: https://arxiv.org/abs/1805.07470. *Endnu en artikel fra McAleer, Agostinelli, Shmakob og Baldi. Vi må lige finde ud af, om der er relevante forskelle der gør at alle disse tre artikler skal med*


* Robert Brunetto and Otakar Trunda: "Deep Heuristic-learning in the Rubik’s Cube Domain:an Experimental Evaluation" http://ceur-ws.org/Vol-1885/57.pdf *Interessant artikel, der ser ud til at opnå meget af det samme som McAleeret al. bruger også A\*-variant og ser problemet som et heuristik-læringsproblem*

* Bowman, Jones and Guo, "Approach to Solving 2x2x2 Rubik’s Cubes Without Human Knowledge", URL (download): https://pdfs.semanticscholar.org/13d8/b58c8e6154d83c88ce28d903f2fd79e82e75.pdf. *Anvender autodidaktisk iteration til at løse en 2x2x2 Rubiks fire træk fra den løste tilstand.*

* Korf, Richard E., "Finding Optimal Solutions to Rubik’s Cube Using Pattern Databases", URL (download): https://www.aaai.org/Papers/AAAI/1997/AAAI97-109.pdf. *Anvender iterative-deepening-A\* til at afsøge tabeller af Rubiks terninger for at finde den korteste vej til en løst tilstand.*

* Smith, Robert: "Evolving Policies to Solve the Rubik's Cube: Experiments with Ideal and Approximate Performance Functions" http://hdl.handle.net/10222/72115 *Bruger RL i form af genetisk programmering og direkte policy-opdagelse til at løse Rubiks, men kendte algoritmer bruges i nogen omfang for at få succes her*

* Alexander Irpan: "Exploring Boosted Neural Nets for Rubik’s CubeSolvin":  https://www.alexirpan.com/public/research/nips_2016.pdf *Forsøger med en version af AdaBoost + LSTM. Ser dog ikke Rubiks-problem som RL men som sekvensiel supervised learning. Succes er begrænset og AdaBoost hjælper ikke*




### Formidling og forsøg på reproducering

* Max Lapan, "Reinforcement Learning to solve Rubik’s cube (and other complex problems!) "
URL: https://medium.com/datadriveninvestor/reinforcement-learning-to-solve-rubiks-cube-andother-complex-problems-106424cf26ff

* Szymon Matejczyk, "Learning to solve Rubik's cube",
URL:https://www.youtube.com/watch?v=5Mm6NQXVLAg) 

### Baggrund og gruppeteori

* Provenza, Hannah, "Group Theory and the Rubik's Cube", URL: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.643.2667. *Gennemgår en række aspekter ved Rubiks terning fra et gruppeteoretisk perspektiv, mere avanceret.*

* Wein, Alex, "Mathematics of the Rubik’s Cube", URL: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.433.3390. *Gennemgår en række aspekter ved Rubiks terning fra et gruppeteoretisk perspektiv, mere letlæseligt.*


* Erik D. Demaine, Martin L. Demaine, Sarah Eisenstat, Anna Lubiw, Andrew Winslow: "Algorithms for Solving Rubik's Cubes" https://arxiv.org/abs/1106.5736  *Baggrundslitteratur for ikke-ML metoder til Rubiks*

### Ekstralitteratur
 
* OpenAI, Ilge Akkaya, Marcin Andrychowicz, et al.: "Solving Rubik's Cube with a Robot Hand": https://arxiv.org/abs/1910.07113  *OpenAI's store publicitetsstunt indeholdt Rubiks-terning og brugte Reinforcement Learning. Den er dog kun overskriftsmæssigt relateret til emnet, da de brugte Kociemba-algoritmen til løsning og RL blev brugt til at bevæge hånden, der løste kuben*

* Erik D. Demaine, Sarah Eisenstat, Mikhail Rudoy: "Solving the Rubik's Cube Optimally is NP-complete" https://arxiv.org/abs/1706.06708 *Sjovt side-fact at nævne som argument for approksimative metoder som ML*


* Xinrui Zhuang, Yuexiang Li, Yifan Hu, Kai Ma, Yujiu Yang, Yefeng Zheng: "Self-supervised Feature Learning for 3D Medical Images by Playing a Rubik's Cube"  https://arxiv.org/abs/1910.02241 *En interessant sidehistorie, som ikke hænger sammen med RL og diskret optimering, men kunne nævnes kort som et eksempel på sjov Rubiks + DL-forbindelse*


### Guider og forklaringer
* PyTorch Deep Q learning: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

* Introduction to A star: https://www.redblobgames.com/pathfinding/a-star/introduction.html


## Andre fundne implementationer

* Max Lapan: https://github.com/Shmuma/rl/tree/master/articles/01_rubic

* Anton Bobrov: https://github.com/antbob/qub3rt

* PyTorch Deep Q learning: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

* Aashrey Anand: https://github.com/AashrayAnand/rubiks-cube-reinforcement-learning 

* CVxTz: https://github.com/CVxTz/rubiks_cube