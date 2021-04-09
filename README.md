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

## Citation

If you find this useful for your work/research, please consider citing this thesis:

```
@thesis{montesuma2021,
    author       = {Eduardo Fernandes Montesuma}, 
    title        = {Cross-Domain Fault Diagnosis through Optimal Transport},
    school       = {Universidade Federal do Ceará},
    year         = 2021,
    type         = {Bachelor's Thesis}
}  
```

## Results

### Comparative Study

<table>
<thead>
  <tr>
    <th>Reaction Order</th>
    <th colspan="3" style="align: center;">1.0</th>
    <th>0.5</th>
    <th>1.5</th>
    <th>2.0</th>
    <th>Score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Degree of Mismatch</td>
    <td>10%</td>
    <td>15%</td>
    <td>20%</td>
    <td colspan="3" style="align: center;">15%</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="8" style="align: center;">Raw Features</td>
  </tr>
  <tr>
    <td>Baseline</td>
    <td>68.654 ± 0.769</td>
    <td>64.519 ± 1.231</td>
    <td>62.404 ± 1.154</td>
    <td>56.923 ± 0.892</td>
    <td>46.346 ± 1.709</td>
    <td>32.981 ± 0.942</td>
    <td>55.304</td>
  </tr>
  <tr>
    <td>KMM</td>
    <td>70.769 ± 0.638</td>
    <td>64.423 ± 1.609</td>
    <td>63.173 ± 1.889</td>
    <td>57.212 ± 1.393</td>
    <td>47.308 ± 0.490</td>
    <td>32.885 ± 1.413</td>
    <td>55.962</td>
  </tr>
  <tr>
    <td>KLIEP</td>
    <td>68.750 ± 0.745</td>
    <td>64.519 ± 1.231</td>
    <td>62.404 ± 1.153</td>
    <td>56.923 ± 0.892</td>
    <td>46.346 ± 1.709</td>
    <td>32.981 ± 0.942</td>
    <td>55.321</td>
  </tr>
  <tr>
    <td>uLSIF</td>
    <td>69.423 ± 1.346</td>
    <td>63.846 ± 0.360</td>
    <td>62.596 ± 1.860</td>
    <td>56.731 ± 1.520</td>
    <td>47.596 ± 0.527</td>
    <td>27.692 ± 9.860</td>
    <td>54.647</td>
  </tr>
  <tr>
    <td>PCA</td>
    <td>69.423 ± 0.490</td>
    <td>64.519 ± 1.027</td>
    <td>64.615 ± 0.577</td>
    <td>57.212 ± 1.29</td>
    <td>46.635 ± 0.804</td>
    <td>36.058 ± 0.608</td>
    <td>56.410</td>
  </tr>
  <tr>
    <td>TCA</td>
    <td>81.442 ± 0.990</td>
    <td>71.923 ± 1.477</td>
    <td>65.192 ± 0.990</td>
    <td>66.346 ± 1.799</td>
    <td>49.519 ± 1.747</td>
    <td>40.385 ± 0.860</td>
    <td>62.468</td>
  </tr>
  <tr>
    <td>GFK</td>
    <td>65.385 ± 1.290</td>
    <td>59.135 ± 1.178</td>
    <td>58.269 ± 0.561</td>
    <td>53.173 ± 1.380</td>
    <td>49.423 ± 1.027</td>
    <td>42.692 ± 1.909</td>
    <td>54.679</td>
  </tr>
  <tr>
    <td>OTDA</td>
    <td><b>89.423 ± 0.680</b></td>
    <td>87.596 ± 0.769</td>
    <td>80.673 ± 1.532</td>
    <td>72.404 ± 1.380</td>
    <td>85.769 ± 1.201</td>
    <td>77.308 ± 1.113</td>
    <td>82.196</td>
  </tr>
  <tr>
    <td>Monge Mapping</td>
    <td>89.327 ± 0.360</td>
    <td><b>87.981 ± 1.008</b></td>
    <td><b>84.038 ± 1.757</b></td>
    <td><b>74.808 ± 0.892</b></td>
    <td><b>86.538 ± 0.804</b></td>
    <td><b>79.423 ± 0.707</b></td>
    <td><b>83.686</b></td>
  </tr>
  <tr>
    <td>JDOT</td>
    <td>86.346 ± 0.838</td>
    <td>83.846 ± 1.654</td>
    <td>76.058 ± 2.284</td>
    <td>70.000 ± 1.840</td>
    <td>81.923 ± 0.942</td>
    <td>75.000 ± 0.527</td>
    <td>78.862</td>
  </tr>
  <tr>
    <td colspan="8" style="align: center;">ACF Features</td>
  </tr>
  <tr>
    <td>Baseline</td>
    <td>97.692 ± 0.360</td>
    <td>97.308 ± 0.385</td>
    <td>92.019 ± 0.490</td>
    <td>85.096 ± 3.145</td>
    <td>62.981 ± 0.912</td>
    <td>59.808 ± 0.942</td>
    <td>82.484</td>
  </tr>
  <tr>
    <td>KMM</td>
    <td>97.692 ± 0.360</td>
    <td>97.308 ± 0.385</td>
    <td>92.019 ± 0.490</td>
    <td>85.096 ± 3.145</td>
    <td>62.981 ± 0.912</td>
    <td>59.808 ± 0.942</td>
    <td>82.484</td>
  </tr>
  <tr>
    <td>KLIEP</td>
    <td>97.692 ± 0.360</td>
    <td>97.308 ± 0.385</td>
    <td>92.019 ± 0.490</td>
    <td>85.096 ± 3.145</td>
    <td>62.981 ± 0.912</td>
    <td>59.808 ± 0.942</td>
    <td>82.484</td>
  </tr>
  <tr>
    <td>uLSIF</td>
    <td>97.956 ± 0.527</td>
    <td>97.212 ± 0.360</td>
    <td>91.538 ± 0.720</td>
    <td>83.750 ± 4.504</td>
    <td>62.122 ± 1.508</td>
    <td>61.827 ± 1.985</td>
    <td>82.356</td>
  </tr>
  <tr>
    <td>PCA</td>
    <td>97.596 ± 0.527</td>
    <td><b>97.404 ± 0.490</b></td>
    <td>91.923 ± 1.071</td>
    <td>84.135 ± 0.304</td>
    <td>62.885 ± 0.881</td>
    <td>61.154 ± 1.704</td>
    <td>82.516</td>
  </tr>
  <tr>
    <td>TCA</td>
    <td>96.923 ± 0.720</td>
    <td>94.808 ± 0.827</td>
    <td>89.808 ± 0.769</td>
    <td>84.519 ± 0.707</td>
    <td>61.827 ± 1.162</td>
    <td>55.385 ± 1.407</td>
    <td>80.545</td>
  </tr>
  <tr>
    <td>GFK</td>
    <td>95.962 ± 1.380</td>
    <td>91.154 ± 0.892</td>
    <td>81.250 ± 2.803</td>
    <td>69.519 ± 6.589</td>
    <td>60.288 ± 2.463</td>
    <td>52.692 ± 4.736</td>
    <td>75.144</td>
  </tr>
  <tr>
    <td>OTDA</td>
    <td>98.365 ± 0.385</td>
    <td>95.000 ± 1.162</td>
    <td>89.231 ± 1.709</td>
    <td>94.038 ± 0.490</td>
    <td><b>95.577 ± 1.439</b></td>
    <td>76.442 ± 2.085</td>
    <td>\textbf91.442</td>
  </tr>
  <tr>
    <td>Monge Mapping</td>
    <td><b>99.615 ± 0.192</b></td>
    <td>96.538 ± 1.704</td>
    <td><b>92.596 ± 1.736</b></td>
    <td>82.500 ± 0.720</td>
    <td>90.192 ± 0.838</td>
    <td><b>76.635 ± 1.346</b></td>
    <td>89.679</td>
  </tr>
  <tr>
    <td>JDOT</td>
    <td>96.923 ± 0.892</td>
    <td>95.481 ± 0.838</td>
    <td>90.288 ± 0.638</td>
    <td><b>97.788 ± 0.490</b></td>
    <td>75.288 ± 2.910</td>
    <td>67.115 ± 1.508</td>
    <td>87.147</td>
  </tr>
  <tr>
    <td colspan="8" style="align: center;">CNN Features</td>
  </tr>
  <tr>
    <td>Baseline</td>
    <td>80.192 ± 3.001</td>
    <td>75.288 ± 2.031</td>
    <td>74.135 ± 3.815</td>
    <td>62.500 ± 2.544</td>
    <td>69.808 ± 5.400</td>
    <td>60.288 ± 5.830</td>
    <td>70.000</td>
  </tr>
  <tr>
    <td>dann</td>
    <td>86.219 ± 2.745</td>
    <td>76.594 ± 3.284</td>
    <td>75.219 ± 2.331</td>
    <td>71.875 ± 3.168</td>
    <td>79.969 ± 1.457</td>
    <td>69.844 ± 3.205</td>
    <td>77.000</td>
  </tr>
  <tr>
    <td>KMM</td>
    <td>70.673 ± 4.641</td>
    <td>67.981 ± 3.065</td>
    <td>59.808 ± 2.878</td>
    <td>35.962 ± 1.532</td>
    <td>67.596 ± 5.237</td>
    <td>44.712 ± 7.111</td>
    <td>57.788</td>
  </tr>
  <tr>
    <td>KLIEP</td>
    <td>74.231 ± 3.019</td>
    <td>70.385 ± 2.120</td>
    <td>64.038 ± 1.783</td>
    <td>44.327 ± 3.485</td>
    <td>68.462 ± 4.392</td>
    <td>47.308 ± 6.568</td>
    <td>61.458</td>
  </tr>
  <tr>
    <td>uLSIF</td>
    <td>72.788 ± 2.763</td>
    <td>71.154 ± 3.454</td>
    <td>65.673 ± 2.679</td>
    <td>49.135 ± 4.821</td>
    <td>67.981 ± 5.512</td>
    <td>44.615 ± 5.930</td>
    <td>61.891</td>
  </tr>
  <tr>
    <td>PCA</td>
    <td>75.096 ± 5.137</td>
    <td>69.904 ± 4.142</td>
    <td>63.269 ± 4.242</td>
    <td>43.558 ± 5.011</td>
    <td>63.558 ± 4.888</td>
    <td>37.308 ± 7.477</td>
    <td>58.782</td>
  </tr>
  <tr>
    <td>TCA</td>
    <td>51.442 ± 9.035</td>
    <td>52.308 ± 9.169</td>
    <td>49.712 ± 7.446</td>
    <td>39.231 ± 10.305</td>
    <td>55.962 ± 5.048</td>
    <td>39.519 ± 9.841</td>
    <td>48.029</td>
  </tr>
  <tr>
    <td>GFK</td>
    <td>65.577 ± 5.315</td>
    <td>59.615 ± 7.776</td>
    <td>57.596 ± 7.646</td>
    <td>38.077 ± 5.485</td>
    <td>52.115 ± 5.854</td>
    <td>33.846 ± 5.039</td>
    <td>51.138</td>
  </tr>
  <tr>
    <td>OTDA</td>
    <td><b>89.327 ± 0.360</b></td>
    <td>84.519 ± 3.803</td>
    <td><b>77.981 ± 2.810</b></td>
    <td>77.212 ± 2.288</td>
    <td>84.135 ± 2.995</td>
    <td>83.269 ± 1.113</td>
    <td>82.740</td>
  </tr>
  <tr>
    <td>Monge Mapping</td>
    <td>89.808 ± 1.676</td>
    <td>82.981 ± 2.862</td>
    <td>78.077 ± 2.076</td>
    <td>74.712 ± 2.518</td>
    <td><b>87.115 ± 2.605</b></td>
    <td>79.423 ± 2.777</td>
    <td>82.019</td>
  </tr>
  <tr>
    <td>JDOT</td>
    <td><b>89.327 ± 0.932</b></td>
    <td><b>85.673 ± 3.062</b></td>
    <td>77.596 ± 2.076</td>
    <td><b>77.692 ± 1.654</b></td>
    <td>82.692 ± 2.107</td>
    <td><b>84.519 ± 0.471</b></td>
    <td><b>82.917</b></td>
  </tr>
</tbody>
</table>


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
