<h1 align="center" style="margin-top: 0px;"> <b>Quantum Architecture Search via Deep Reinforcement Learning</b></h1>
<div align="center" >
[![packages](https://img.shields.io/static/v1.svg?label=Made%20with&message=Cirq&color=fbc43b)](https://docs.qulacs.org/en/latest/index.html)
[![license](https://img.shields.io/static/v1.svg?label=License&message=GPL%20v3.0&color=green)](https://www.gnu.org/licenses/gpl-3.0.html)
</div>

## **Description**
This repository contains an extended version of the <ins>Quantum Architecture Search</ins> environments and its applications as in :

- Paper : **Quantum Architecture Search via Deep Reinforcement Learning**
- Authors : **En-Jui Kuo, Yao-Lung L. Fang, Samuel Yen-Chi Chen**
- Date : **2021**

The version extend the action spaces to paramterized gates, like the $RX(\theta)$, $RY(\theta)$, $RZ(\theta)$ gate. The customized <ins>Gym</ins> environments are built using <ins>qulacs</ins>, ultilizing the Parameterized Quantum Circuit class implementation .


## **Setup**
To <ins>install</ins>, clone this repository and execute the following commands :

```
$ cd quantum-arch-search-parameterized-gate
$ pip install -r requirements.txt
$ pip install -e .
```