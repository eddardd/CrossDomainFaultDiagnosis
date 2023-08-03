# Cross-Domain Fault Detection
Repository containing the code for the experiments and examples of my Bachelor Thesis: Cross Domain Fault Detection through Optimal Transport. More details coming soon!

## Dynamic Systems

Implemented benchmarks

* [x] Two Tanks System
* [x] Continuously Stirred Ractor Tank (CSTR) [1]

Model Identification

* [x] First Order Plus Time Delay
* [ ] Second Order Plus Time Delay

PID Tuning

* [x] Direct Synthesis [2]

## Implemented algorithms

Instance-based transfer

* [x] Kernel Mean Matching (KMM) [3]
* [x] Kullback-Leibler Importance Estimation Prodecude (KLIEP) [4]
* [x] Least Squares Importance Fitting (LSIF) [5]

Feature-based transfer

* [x] Transfer Component Analysis (TCA) [6]
* [x] Geodesic Flow Kernel (GFK) [7]
* [x] Principal Component Analysis (PCA) [7]
* [x] Domain Adversarial Neural Networks (DANN) [8]

Optimal Transport-based transfer

* [x] Sinkhorn Transport [9] - already implemented in the [Python Optimal Transport](https://github.com/PythonOT/POT) library
* [x] Monge Transport [10] - already implemented in the [Python Optimal Transprot](https://github.com/PythonOT/POT) library
* [x] Joint Distribution Optimal Transport (JDOT) [11] - implementation adapted from [rflamary Github repository](https://github.com/rflamary/JDOT)

## Abstract

Automatic fault diagnosis systems are an important component for fault tolerance in modern control loops. Nonetheless, the training of such diagnosis systems can be costly or even dangerous since faulty data need to be collected by driving the process to dangerous conditions. A possible solution to the said problem is training an automatic diagnosis system solely on simulation data. However, due to modeling errors, the data acquired may not reflect real process data. This is characterized by a change in the probability distributions upon which data is sampled. This problem is known in the literature as domain adaptation or cross-domain fault diagnosis in our context. Thus this work analyzes the cross-domain diagnosis problem through the point of view of optimal transport. We apply our methodology in a case study concerning the continuous stirred tank reactor (CSTR) system. Our contributions are three-fold: 1. we perform a comparative study concerning feature extraction and domain adaptation algorithms, 2. we analyze the relation between wrongful model specification and the distance between source and target distributions, and its impact on classification performance 3. we analyze the impact of modeling errors in the quality of optimal transport plans, and the influence of this latter factor into classification performance.

In summary, we found that optimal transport-based domain adaptation is the best choice for solving the distributional shift problem. In addition, we further verified that an increasing degree of modeling error is correlated with an increase in the distance between source and target distributions. Furthermore, we found experimentally that the latter distance is correlated with a decrease in classification performance, confirming previous theoretical findings. Finally, the degree of modeling error can cause the transportation plan between source and target domain to transfer mass between different classes, harming classification performance.

## Associated Publications

This repository is associated with the following Bachelor Thesis,

```
@phdthesis{montesuma2021,
    author       = {Eduardo Fernandes Montesuma}, 
    title        = {Cross-Domain Fault Diagnosis through Optimal Transport},
    school       = {Universidade Federal do Ceará},
    year         = 2021,
    type         = {Bachelor's Thesis}
}  
```

This thesis was summarized in the following conference paper,

```
@article{montesuma2022cross,
  title={Cross-domain fault diagnosis through optimal transport for a CSTR process},
  author={Montesuma, Eduardo Fernandes and Mulas, Michela and Corona, Francesco and Mboula, Fred-Maurice Ngole},
  journal={IFAC-PapersOnLine},
  volume={55},
  number={7},
  pages={946--951},
  year={2022},
  publisher={Elsevier}
}
```

If you find your code useful in your research, please consider citing these papers.

References
----------
[1] Li, Weijun, et al. "Transfer learning for process fault diagnosis: Knowledge transfer from simulation to physical processes." Computers & Chemical Engineering 139 (2020): 106904.

[2] Chen, Dan, and Dale E. Seborg. "PI/PID controller design based on direct synthesis and disturbance rejection." Industrial & engineering chemistry research 41.19 (2002): 4807-4822.

[3] Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.

[4] Sugiyama, Masashi, et al. "Direct importance estimation for covariate shift adaptation." Annals of the Institute of Statistical Mathematics 60.4 (2008): 699-746.

[5] Kanamori, Takafumi, Shohei Hido, and Masashi Sugiyama. "A least-squares approach to direct importance estimation." The Journal of Machine Learning Research 10 (2009): 1391-1445.

[6] Pan, Sinno Jialin, et al. "Domain adaptation via transfer component analysis." IEEE Transactions on Neural Networks 22.2 (2010): 199-210.

[7] Gong, Boqing, et al. "Geodesic flow kernel for unsupervised domain adaptation." 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012.

[8] Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." The journal of machine learning research 17.1 (2016): 2096-2030.

[9] Courty, Nicolas, et al. "Optimal transport for domain adaptation." IEEE transactions on pattern analysis and machine intelligence 39.9 (2016): 1853-1865.

[10] Perrot, Michaël, et al. "Mapping estimation for discrete optimal transport." Proceedings of the 30th International Conference on Neural Information Processing Systems. 2016.

[11] Courty, Nicolas, et al. "Joint distribution optimal transportation for domain adaptation." arXiv preprint arXiv:1705.08848 (2017).
