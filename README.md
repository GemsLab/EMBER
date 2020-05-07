# EMBER

**Paper**: Di Jin*, Mark Heimann*, Tara Safavi, Mengdi Wang, Wei Lee, Lindsay Snider, Danai Koutra. Smart Roles: Inferring Professional Roles in Email Networks. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019.

*Link*: https://gemslab.github.io/papers/jin-2019-roles.pdf

<p align="center">
<img src="https://raw.githubusercontent.com/GemsLab/EMBER/master/overview.jpg" width="700"  alt="Overview of EMBER">
</p>

**Citation (bibtex)**:
```bibtex
@inproceedings{DBLP:conf/kdd/JinRHSWLS19,
  author    = {Di Jin* and
               Mark Heiamann* and
               Tara Safavi and
               Mengdi Wang and 
               Wei Lee and and
               Lindsay Snider and 
               Danai Koutra},
  title     = {Smart Roles: Inferring Professional Roles in Email Networks},
  booktitle = {Proceedings of the 25th {ACM} {SIGKDD} International Conference on
               Knowledge Discovery {\&} Data Mining, {KDD} 2019, London, UK,
               August 4-8, 2019},
  year      = {2019},
  }
```

**Code**: 
## Inputs:

Ember takes two files as input, the graph file and the lookup file used to indicate which nodes to embed.

### Input graph file
The input graph file is the static edge list in the following format separated by tab:
```
<src> <dst> <weight>
```
The edge list is assumed to be re-ordered consecutively from 0, i.e., the minimum node ID is 0, and the maximum node ID is <#node - 1>. A toy static graph is under "/graph/" directory.

### Input lookup file
The lookup file is a subset of nodes in the graph to embed. A toy example is under "/graph/" directory.

### Other input argumets
The specific configuration to pass to EMBER can be found with the ```python main -h``` command.
