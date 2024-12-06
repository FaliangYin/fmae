# Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms and Interface for XAI
This repository is the implementation of experiment section of the paper
["Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms and Interface for XAI"](https://ieeexplore.ieee.org/document/10731553), 
which has been accepted for publication in the IEEE Transactions on Fuzzy Systems.

In this work, Fuzzy Model-Agnostic Explanation (FMAE) is proposed as a post-hoc method to explain the behavior of 
black box models. First, we design the hierarchical FMAE framework including the levels of sample, local, domain and 
universe. Second, we introduce learning algorithms for explainers, as well as simplification and aggregation algorithms 
to downscale and upscale the explainers. Third, we present an explanation interface including semantic inference and 
feature salience to deliver explanations to users.

## Reference
The data sets used in the experiments are provided the following repositories:

[1] [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/)  
[2] [KEEL-dataset Repository](https://sci2s.ugr.es/keel/datasets.php)

The codes in the references folder are from the following works.

[1] Y. Liu, S. Khandagale, C. White, and W. Neiswanger, “Synthetic benchmarks for scientific research in 
explainable machine learning,” Adv. Neural 842 Inf. Process. Syst. Datasets Track, 2021.  
[2] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model 
predictions,” in Proc. Int. Conf. Neural Inf. Process. Syst., Red Hook, NY, 
USA, 2017, pp. 4768–4777.  
[3] M. T. Ribeiro, S. Singh, and C. Guestrin, ““why should I trust you?”: 
Explaining the predictions of any classifier,” in Proc. 22nd ACM SIGKDD 
Int. Conf. Knowl. Discov. Data Mining, 2016, pp. 1135–1144.  
[4] G.Plumb, D.Molitor, and A.Talwalkar, “Model agnostic supervised local 
explanations,” in Proc. Int. Conf. Neural Inf. Process. Syst., Montreal, 
Canada, 2018, pp. 2520–2529.  
[5] D. A. Melis and T. Jaakkola, “Towards robust interpretability with 
self-explaining neural networks,” in Advances in Neural Information 
Processing Systems, Red Hook, NY, USA: Curran Associates, 2018.  
[6] R. Luss et al., “Leveraging latent features for local explanations,” in Proc. 
27th ACM SIGKDD Conf. Knowl. Discov. Data Mining, 2021, pp. 1139–1149.  
[7] C.-K. Yeh, C.-Y. Hsieh, A. Suggala, D. I. Inouye, and P. K. Ravikumar, 
“On the (in)fidelity and sensitivity of explanations,” in Advances in Neural 
Information Processing Systems, RedHook, NY, USA: Curran Associates, 
2019, pp. 10967–10978.