# Cross-Domain Fault Detection
Repository containing the code for the experiments and examples of my Bachelor Thesis: Cross Domain Fault Detection through Optimal Transport. More details coming soon!

## Benchmark Systems

Implemented plants

* [x] Two Tanks System
* [x] Continuously Stirred Ractor Tank (CSTR) [1]

## Implemented algorithms

Instance-based transfer

* [x] Kernel Mean Matching (KMM) [2]
* [x] Kullback-Leibler Importance Estimation Prodecude (KLIEP) [3]
* [ ] Least Squares Importance Fitting (LSIF) [4]

Feature-based transfer

* [ ] Transfer Component Analysis (TCA) [5]
* [ ] Geodesic Flow Kernel (GFK) [6]
* [ ] Domain Adversarial Neural Networks (DANN) [7]

Optimal Transport-based transfer

* [x] Sinkhorn Transport [8] - already implemented in the [Python Optimal Transport](https://github.com/PythonOT/POT) library
* [x] Monge Transport [9] - already implemented in the [Python Optimal Transprot](https://github.com/PythonOT/POT) library
* [x] Joint Distribution Optimal Transport (JDOT) [10] - implementation adapted from [rflamary Github repository](https://github.com/rflamary/JDOT)


References
----------
[1] Li, Weijun, et al. "Transfer learning for process fault diagnosis: Knowledge transfer from simulation to physical processes." Computers & Chemical Engineering 139 (2020): 106904.

[2] Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.

[3] Sugiyama, Masashi, et al. "Direct importance estimation for covariate shift adaptation." Annals of the Institute of Statistical Mathematics 60.4 (2008): 699-746.

[4] Kanamori, Takafumi, Shohei Hido, and Masashi Sugiyama. "A least-squares approach to direct importance estimation." The Journal of Machine Learning Research 10 (2009): 1391-1445.

[5] Pan, Sinno Jialin, et al. "Domain adaptation via transfer component analysis." IEEE Transactions on Neural Networks 22.2 (2010): 199-210.

[6] Gong, Boqing, et al. "Geodesic flow kernel for unsupervised domain adaptation." 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012.

[7] Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." The journal of machine learning research 17.1 (2016): 2096-2030.

[8] Courty, Nicolas, et al. "Optimal transport for domain adaptation." IEEE transactions on pattern analysis and machine intelligence 39.9 (2016): 1853-1865.

[9] Perrot, Michaël, et al. "Mapping estimation for discrete optimal transport." Proceedings of the 30th International Conference on Neural Information Processing Systems. 2016.

[10] Courty, Nicolas, et al. "Joint distribution optimal transportation for domain adaptation." arXiv preprint arXiv:1705.08848 (2017).


## Results

Still to come.