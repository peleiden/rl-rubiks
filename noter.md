# FPP Projekt

- Find PDF af bogen
- Forskellige deadlines i løbet af semestret
- Midtvejsaflevering halvvejs
- Peergrade 🤮
- Ved slutningen af 13-ugers er der status
- Den mundtlige fremlæggelse er som udgangspunkt på frivilligt sprog
- Versionskontrol påkrævet
- Send prioriteret liste med tre projekter og grupper til mmor@dtu.dk senest 7/2.
- Individuel eksimination

## Rapportens opbygning
Brug IMRaD. Frivilligt om der skrives på dansk eller engelsk, men engelsk anbefalet.

- Indledning
- Metode
- Data
- Resultater
- Diskussion
- Konklusion
- Referenceliste
- Bilag

## Logbog
Alt arbejde, alle møder, overvejelser osv. gjort i løbet af FPP Projektet skal dokumenteres i logbogen 🤮. Logbogen skal vedlægges som bilag.

## Projekter

### Rubiksterning

Diskret kombinatorisk problem og stort tilstandsrum med kun én løsning. Løsninger kan være deep reinforcement learning, Monte Carlo-algoritmer eller andet. Derudover software til repræsentation af state.

Ansvarlig: Mikkel Schmidt 😁

### Generating speech from transcripts

Using AI to guide patient interviews. Use automatic speech recognition (ASR) as basis for most services. WaveNet for text to speech. Data will be provided in the form of real calls.

Phases:

- Translate transcripts using machine translation
- Generate audio using WaveNet (pretrained)
- Train ASR model to evaluate

Expectations: Many mathematics, much theory, and good coding skills.

Ansvarlig: Corti

### Predicting soil properties

Work with food analytics. A near infrared (NIR) analyzer is used to measure protein, fat, stach and other stuff. Deep learning (CNN's) is used to predict soil properties.

Ansvarlig: FOSS

### Grain quality

Using ML to inspect grain quality. Data consists of 110,000 RGB images of grain kernels. The goal is to detect defects.

Ansvarlig: FOSS

### Interpolate missing data in audiograms in a data-driven way

Full audiograms are rarely recorded. They do not cluster well. The goal is to find ML driven way to interpolate the missing values. Use semi-supervised learning to get more precise prediction of missing values, and help audiologists choose frequencies that maximize information gain.

Ansvarlig: WS Audiology 👓

### Kognitiv belastning

Simplificeret: Den energi det kræver at lytte og huske, hvad der er blevet sagt. Målinger af hjerneaktivitet (EEG). Data fra 7 normalthørende på 16 kanaler og 15 repetitioner. Neural tale-tracking og klassificer kognitiv belastning.

Ansvarlig: WS Audiology 👩

### Hearify - optimering af høreapperatsindstillinger

Mere end bare lydstyrke. A/B-tests og brugerstatistikker - 80.000 datapunkter. Målet er så at lave automatiske indstillinger.

Ansvarlig: WS Audiology 🔵

### Alpha-myretuen

Reinforcement learning bruges til at løse brætspillet Myretuen. En vis strategisk dybde og interessante beslutninger, men computationelt mere løsbart end skak og Star Craft II. Der skal foretages overvejelser om neurale repræsentationer af brættet.

1 mod 1. Bræt repræsenteret som graf og beboelsesinformation. Træk er stokastiske, state-afhængige og indeholder fra-til-beslutnigner.

Ansvarlig: Tue Herlau 🐜

### Deep voice conversion and ethical considarations 🤮

- Deep fakes 😁
- Ethical considarations 🤮

Ansvarlig: Morten Mørup

### Babelfish

Speech to text ➡ text to text translation ➡ text to speech. Technologies already exists in some form, but not optimal. This project will not be real-time. Highly dependent on open technologies and API's.

Ansvarlig: Morten Mørup

### Epilepsi og EEG

To niveauer af AI-støtte:

- On-site kvalitetskontrol og brugerfeedback
- Offline analyse og denoising

Udfordringer: Støjfyldt, ikke-stationært data. Derudover skal der være embedding vha. generative modeller baseret på store databaser med transfer learning.

Der skal være forklarbarhed og interaktionsdesign. Fancy augmenteringsteknikker. Der er også mulighed for at se på generering af EEG-data, som er et nyt felt.

Ansvarlig: Big K ✔

### Retfærdighed i klassifikation

Lægger op til medicinsk anvendelse og forskningsprojekt, der starter til sommer. COMPAS-datasættet anvendes, og der vises implikation mellem databias og resultatbias.

Ansvarlig: Aasa Feragen ↖


