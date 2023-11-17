## Phrase Processor
### Description
Processes sentences according to similarity
Works in two modes:
1. batch mode - processes all sentences in a file
2. interactive mode - processes sentences from cmd
### Installation
1. git clone  repo 
2. download model by URL to data foldea
3. `cd phrase-processor`
4. `pip install -r requirements.txt`

### Usage
* batch mode: `python main.py main.py --option batch`
* interactive mode: `python --option on_the_fly <string>`