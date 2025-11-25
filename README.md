<h1 align="center" style="margin-top: 0px;"> <b>Learning Parameterized Quantum Circuit with Quantum Gradient</b></h1>
<div align="center" >

[![paper](https://img.shields.io/static/v1.svg?label=Paper&message=arXiv:2409.20044&color=b31b1b)](https://arxiv.org/abs/2409.20044)
[![license](https://img.shields.io/static/v1.svg?label=License&message=GPL%20v3.0&color=green)](https://www.gnu.org/licenses/gpl-3.0.html)
</div>

## **Description**
This repository contains implementation and code examples of numerical experiments in :

- Paper : **Learning Parameterized Quantum Circuit with Quantum Gradient**
- Authors : **Keren Li, Yuanfeng Wang, Pan Gao, Shenggen Zheng**
- Date : **2024**

The numerical experiments of two example problems (namely the MAXCUT problem and the polynomial optimization problem) can be found in ./notebook folder.

A large portion of the code base is the implementation of reinforcement learning algorithm for Quantum Architecture Search. Similar to the RL algorithm by [Kuo of et al.](https://arxiv.org/abs/2406.06210), we built a customized [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environments and extend the action spaces to paramterized gates, like the single-qubit Pauli rotation gates
{ $RX(\theta)$, $RY(\theta)$, $RZ(\theta)$ }and two-quibit Pauli rotation gates { $RXX(\theta)$, $RYY(\theta)$, $RZZ(\theta) $}. The default observables are chosen to be expecation values of single qubit Pauli observables, and optionally higher order Pauli observables - for example all two-qubit Paulis. We use [Qulacs](https://docs.qulacs.org/en/latest/index.html) backend for quantum circuit simulation, ultilizing the Parameterized Quantum Circuit class implementation therein. 


## **Setup**
To install, clone this repository and execute the following commands :

```
$ cd PQC-quantum-grad
$ pip install -r requirements.txt
$ pip install -e .
# python setup.py install
```

## **Run examples**
To reproduce the results for MAX-CUT /polynomial optimization in the paper, run the notebooks (example1.ipynb  for MAX-CUT and example2.ipynb for  polynomial optimization) under directory /notebooks

For comparison with Adapt-VQE and standard (randomized) PQC methods, run

```
python notebooks/sup_chem.py
python notebooks/sup_standard_pqc
```

