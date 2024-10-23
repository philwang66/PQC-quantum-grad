<h1 align="center" style="margin-top: 0px;"> <b>Learning Parameterized Quantum Circuit with Quantum Gradient</b></h1>
<div align="center" >
[![license](https://img.shields.io/static/v1.svg?label=License&message=GPL%20v3.0&color=green)](https://www.gnu.org/licenses/gpl-3.0.html)
</div>

## **Description**
This repository contains implementation and code examples of numerical experiments in :

- Paper : **Learning Parameterized Quantum Circuit with Quantum Gradient**
- Authors : **Keren Li, Yuanfeng Wang, Pan Gao, Shenggen Zheng**
- Date : **2024**

The numerical experiments of two example problems (namely the MAXCUT problem and the polynomial optimization problem) can be found in ./notebook folder.

A large portion of the code base is the implementation of reinforcement learning algorithm for <ins>Quantum Architecture Search</ins>. Similar to the RL algorithm by [Kuo of et al.](https://arxiv.org/abs/2406.06210), we built a customized <ins>Gym</ins> environments and extend the action spaces to paramterized gates, like the single-qubit Pauli rotation gates
($RX(\theta)$, $RY(\theta)$, $RZ(\theta)$)and two-quibit Pauli rotation gates ($RXX(\theta)$, $RYY(\theta)$, $RZZ(\theta)$). We use <ins>Qulacs</ins> (https://docs.qulacs.org/en/latest/index.html) backend for quantum circuit simulation, ultilizing the Parameterized Quantum Circuit class implementation therein. 


## **Setup**
To <ins>install</ins>, clone this repository and execute the following commands :

```
$ cd PQC-quantum-grad
$ pip install -r requirements.txt
$ pip install -e .
```
