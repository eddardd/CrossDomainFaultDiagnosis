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

## Results

### Comparative Study

|   Reaction Order   |                  1.0                 |                                      |                                      |                  0.5                 |                  1.5                 |                  2.0                 |      Score      |
|:------------------:|:------------------------------------:|:------------------------------------:|:------------------------------------:|:------------------------------------:|:------------------------------------:|:------------------------------------:|:---------------:|
| Degree of Mismatch |                10%                |                15%                |                20%                |                15%                |                                      |                                      |                 |
|    Raw Features    |                                      |                                      |                                      |                                      |                                      |                                      |                 |
|      SVM     |          68.654 ± 0.769          |          64.519 ± 1.231          |          62.404 ± 1.154          |          56.923 ± 0.892          |          46.346 ± 1.709          |          32.981 ± 0.942          |      55.304     |
|      KMM     |          70.769 ± 0.638          |          64.423 ± 1.609          |          63.173 ± 1.889          |          57.212 ± 1.393          |          47.308 ± 0.490          |          32.885 ± 1.413          |      55.962     |
|     KLIEP    |          68.750 ± 0.745          |          64.519 ± 1.231          |          62.404 ± 1.153          |          56.923 ± 0.892          |          46.346 ± 1.709          |          32.981 ± 0.942          |      55.321     |
|     uLSIF    |          69.423 ± 1.346          |          63.846 ± 0.360          |          62.596 ± 1.860          |          56.731 ± 1.520          |          47.596 ± 0.527          |          27.692 ± 9.860          |      54.647     |
|      PCA     |          69.423 ± 0.490          |          64.519 ± 1.027          |          64.615 ± 0.577          |           57.212 ± 1.29          |          46.635 ± 0.804          |          36.058 ± 0.608          |      56.410     |
|      TCA     |          81.442 ± 0.990          |          71.923 ± 1.477          |          65.192 ± 0.990          |          66.346 ± 1.799          |          49.519 ± 1.747          |          40.385 ± 0.860          |      62.468     |
|      GFK     |          65.385 ± 1.290          |          59.135 ± 1.178          |          58.269 ± 0.561          |          53.173 ± 1.380          |          49.423 ± 1.027          |          42.692 ± 1.909          |      54.679     |
|     OTDA     | __89.423__ ± __0.680__ |          87.596 ± 0.769          |          80.673 ± 1.532          |          72.404 ± 1.380          |          85.769 ± 1.201          |          77.308 ± 1.113          |      82.196     |
|    Monge Mapping   |          89.327 ± 0.360          | __87.981__ ± __1.008__ | __84.038__ ± __1.757__ | __74.808__ ± __0.892__ | __86.538__ ± __0.804__ | __79.423__ ± __0.707__ | __83.686__ |
|     JDOT     |          86.346 ± 0.838          |          83.846 ± 1.654          |          76.058 ± 2.284          |          70.000 ± 1.840          |          81.923 ± 0.942          |          75.000 ± 0.527          |      78.862     |
|    ACF Features    |                                      |                                      |                                      |                                      |                                      |                                      |                 |
|      SVM     |          97.692 ± 0.360          |          97.308 ± 0.385          |          92.019 ± 0.490          |          85.096 ± 3.145          |          62.981 ± 0.912          |          59.808 ± 0.942          |      82.484     |
|      KMM     |          97.692 ± 0.360          |          97.308 ± 0.385          |          92.019 ± 0.490          |          85.096 ± 3.145          |          62.981 ± 0.912          |          59.808 ± 0.942          |      82.484     |
|     KLIEP    |          97.692 ± 0.360          |          97.308 ± 0.385          |          92.019 ± 0.490          |          85.096 ± 3.145          |          62.981 ± 0.912          |          59.808 ± 0.942          |      82.484     |
|     uLSIF    |          97.956 ± 0.527          |          97.212 ± 0.360          |          91.538 ± 0.720          |          83.750 ± 4.504          |          62.122 ± 1.508          |          61.827 ± 1.985          |      82.356     |
|      PCA     |          97.596 ± 0.527          | __97.404__ ± __0.490__ |          91.923 ± 1.071          |          84.135 ± 0.304          |          62.885 ± 0.881          |          61.154 ± 1.704          |      82.516     |
|      TCA     |          96.923 ± 0.720          |          94.808 ± 0.827          |          89.808 ± 0.769          |          84.519 ± 0.707          |          61.827 ± 1.162          |          55.385 ± 1.407          |      80.545     |
|      GFK     |          95.962 ± 1.380          |          91.154 ± 0.892          |          81.250 ± 2.803          |          69.519 ± 6.589          |          60.288 ± 2.463          |          52.692 ± 4.736          |      75.144     |
|     OTDA     |          98.365 ± 0.385          |          95.000 ± 1.162          |          89.231 ± 1.709          |          94.038 ± 0.490          | __95.577__ ± __1.439__ |          76.442 ± 2.085          | {91.442} |
|    Monge Mapping   | __99.615__ ± __0.192__ |          96.538 ± 1.704          | __92.596__ ± __1.736__ |          82.500 ± 0.720          |          90.192 ± 0.838          | __76.635__ ± __1.346__ |      89.679     |
|     JDOT     |          96.923 ± 0.892          |          95.481 ± 0.838          |          90.288 ± 0.638          | __97.788__ ± __0.490__ |          75.288 ± 2.910          |          67.115 ± 1.508          |      87.147     |
|    CNN Features    |                                      |                                      |                                      |                                      |                                      |                                      |                 |
|      {cnn}     |          80.192 ± 3.001          |          75.288 ± 2.031          |          74.135 ± 3.815          |          62.500 ± 2.544          |          69.808 ± 5.400          |          60.288 ± 5.830          |      70.000     |
|     {dann}     |          86.219 ± 2.745          |          76.594 ± 3.284          |          75.219 ± 2.331          |          71.875 ± 3.168          |          79.969 ± 1.457          |          69.844 ± 3.205          |      77.000     |
|      KMM     |          70.673 ± 4.641          |          67.981 ± 3.065          |          59.808 ± 2.878          |          35.962 ± 1.532          |          67.596 ± 5.237          |          44.712 ± 7.111          |      57.788     |
|     KLIEP    |          74.231 ± 3.019          |          70.385 ± 2.120          |          64.038 ± 1.783          |          44.327 ± 3.485          |          68.462 ± 4.392          |          47.308 ± 6.568          |      61.458     |
|     uLSIF    |          72.788 ± 2.763          |          71.154 ± 3.454          |          65.673 ± 2.679          |          49.135 ± 4.821          |          67.981 ± 5.512          |          44.615 ± 5.930          |      61.891     |
|      PCA     |          75.096 ± 5.137          |          69.904 ± 4.142          |          63.269 ± 4.242          |          43.558 ± 5.011          |          63.558 ± 4.888          |          37.308 ± 7.477          |      58.782     |
|      TCA     |          51.442 ± 9.035          |          52.308 ± 9.169          |          49.712 ± 7.446          |          39.231 ± 10.305         |          55.962 ± 5.048          |          39.519 ± 9.841          |      48.029     |
|      GFK     |          65.577 ± 5.315          |          59.615 ± 7.776          |          57.596 ± 7.646          |          38.077 ± 5.485          |          52.115 ± 5.854          |          33.846 ± 5.039          |      51.138     |
|     OTDA     | __89.327__ ± __0.360__ |          84.519 ± 3.803          | __77.981__ ± __2.810__ |          77.212 ± 2.288          |          84.135 ± 2.995          |          83.269 ± 1.113          |      82.740     |
|    Monge Mapping   |          89.808 ± 1.676          |          82.981 ± 2.862          |          78.077 ± 2.076          |          74.712 ± 2.518          | __87.115__ ± __2.605__ |          79.423 ± 2.777          |      82.019     |
|     JDOT     | __89.327__ ± __0.932__ | __85.673__ ± __3.062__ |          77.596 ± 2.076          | __77.692__ ± __1.654__ |          82.692 ± 2.107          | __84.519__ ± __0.471__ | __82.917__ |

## Citation

If you find this useful for your work/research, please consider citing this thesis:

```
@phdthesis{montesuma2021,
    author       = {Eduardo Fernandes Montesuma}, 
    title        = {Cross-Domain Fault Diagnosis through Optimal Transport},
    school       = {Universidade Federal do Ceará},
    year         = 2021,
    address      = {...},
    }  
```

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
