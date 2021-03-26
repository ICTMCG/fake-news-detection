# Fake News Detection 虚假新闻检测
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

This repo is a collection of AWESOME things about fake news detection, including papers, code, etc. Feel free to star and fork.

## Contents
- [Papers](#paper)
  - [Survey](#survey) 综述
  - [Fact Checking](#fact) 真实性检验
  - [Multi-Modal](#multi-modal) 多模态

## <span id="paper">Papers</span>
### <span id="survey">Survey</span> 综述
- [False news detection on social media](). ARXIV 2019.
- [A survey on fake news and rumour detection techniques](https://www.sciencedirect.com/science/article/pii/S0020025519304372). Information Sciences, 2019, 497: 38-55.
- [Detection and resolution of rumours in social media: A survey](https://dl.acm.org/doi/abs/10.1145/3161603). ACM Computing Surveys (CSUR), 2018, 51(2): 1-36.
- [The Spread of True and False News Online](https://science.sciencemag.org/CONTENT/359/6380/1146.abstract). Science, 2018, 359(6380): 1146-1151.
- [Fake News Detection on Social Media: A Data Mining Perspective](https://dl.acm.org/doi/abs/10.1145/3137597.3137600?casa_token=Mf0tvofQf7kAAAAA:LgdXVmsJzYxVyrTgrhoFio_zxDXORoh6NNGP4__D64yam0rOKfwdbi__38Jg01U7pC-M19Tkb2NC_BU). ACM SIGKDD explorations newsletter, 2017, 19(1): 22-36.

### <span id="fact">Fact Checking</span> 真实性检验

- EMNLP-2020 [Where Are the Facts? Searching for Fact-checked Information to Alleviate the Spread of Fake News](https://www.aclweb.org/anthology/2020.emnlp-main.621.pdf).
  - To stop users from spreading fake news, this paper proposed a novel framework to search for fact-checking articles which is related to original poster. The search can directly warn fake news posters and online users about misinformation, discourage them from spreading fake news.
  - 这篇文章提出了一个新的框架，检索和原始帖子相关的经过真实性检验的文章，并且贴出这些文章来警告用户这可能是虚假新闻来组织虚假新闻的传播。
- SIGIR-2019 [Learning from fact-checkers: analysis and generation of fact-checking language](https://dl.acm.org/doi/abs/10.1145/3331184.3331248?casa_token=DUE-x5HZNywAAAAA:ZmQO0tm5NXc6ibbo6898F5f8wZcIFL2HDhy8KSjmnSy2k9ArFi5Y2Ew450lLXPMrLoIY9YfmeNCIkn0).
  - Fact-checkers refute the misinformation by replying to the original poster and provides a fact-checking article as a supporting evidence. This paper focuses on generating responses for fact-checkers with a GRU-based generative model.
  - fact-checker会在别人的帖子下回复原始真实文章作为证据来驳斥错误信息。这篇文章旨在为fact-checker生成回复的句子来鼓励fack-checker进行更多的活动。提出了一个基于GRU的生成模型。
- EMNLP-2018 [DeClarE : Debunking Fake News and False Claims using Evidence-Aware Deep Learning](https://www.aclweb.org/anthology/D18-1003.pdf).
  - This paper proposed a novel method named DeClarE for fact checking. Given an input claim, DeClarE searches for web articles related to the claim and utilizes contexts of both articles and claims to check the fact.
  - 这篇文章针对真实性检验提出了一种新的方法DeClarE。给定一个言论，DaClarE在网上搜索相关的文章，使用相关文章和原始言论预测真实性。
- SIGIR-2018 [The rise of guardians: Fact-checking URL recommendation to combat fake news](https://dl.acm.org/doi/abs/10.1145/3209978.3210037?casa_token=wY-498SHN8EAAAAA:_p67817oi5vgktXQMPr8AyJ3AAOrn7lRoP_UhJ6dCf3WtVUquG499062k2gFJ-lVwbTesFZHzbS5cK0).
  - Find guardians who are willing to spread verified news and stimulate them to disseminate fact-checked news/information with a matrix factorization model.
  - 找到愿意传播经过真实性检验的新闻的保护者，这类人通常会在评论中给出真实新闻的链接。激励他们传播真实新闻。本文提出了一种基于矩阵分解的方法。

### <span id="multi-modal">Multi-Modal</span> 多模态

- EMNLP-2020 [Detecting Cross-Modal Inconsistency to Defend Against Neural Fake News](https://www.aclweb.org/anthology/2020.emnlp-main.163.pdf).
  - This paper proposed a novel problem defence against neural fake news with images and captions. To circumvent this problem, they present DIDAN,which exploits possible semantic inconsistencies between the text and image/captions to detect machine-generated articles. 
  - 这篇文章提出了一个新的问题，如何防御带有图片和图片描述的虚假新闻。为了解决这个问题，他们提出了DIDAN，使用文本和图片、图片描述之间的语义一致性来检测机器生成的文章。
- ICDM-2019 [Exploiting Multi-domain Visual Information for Fake News Detection](https://ieeexplore.ieee.org/abstract/document/8970940/).
  - Fake-news images may have significantly different characteristics from real-news images at both physical and semantic levels, which can be clearly reflected in the frequency and pixel domain. This paper proposed a novel framework Multi-domain Visual Neural Network (MVNN) to fuse the visual information of frequency and pixel domains for detecting fake news.
  - 虚假新闻中的图片和真实新闻的图片在物理和语义级别有很大的不同，这可以反应在频域和像素域。这篇文章提出了MVNN来融合频域和像素域的视觉信息来帮助欺诈检测。
- WWW-2019 [MVAE : Multimodal Variational Autoencoder for Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3308558.3313552?casa_token=tzDWRQ_VVxEAAAAA:Fc3ubDseJsk05pUdznqtA4cDD9BGemHHh8A1T6Nzur8qa0SU8SQY8O7_KRoj8tE_Ah75p8sslKjo4bU).
  - A shortcoming of the current approaches for the detection of fake news is their inability to learn a shared representation of multimodal (textual + visual) information. This paper proposed Multimodal Variational Autoencoder (MVAE), which uses a bimodal variational autoencoder for the task of fake news detection.
  - 现在的虚假新闻检测的方法有一个缺点，针对多模态的信息，他们不能学习一个共享的表示。这篇文章提出了MVAE，使用一个双模态的VAE（变分自动编码机）。
- KDD-2018 [EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3219819.3219903?casa_token=_1c4Ao5K0moAAAAA:jUDfxV9HoKVeRQnGwXYI6oGEm1MMOusLGjPiZjEAOm94MVld0ntG44G4kwdK_qLtipJa32ngFQE995U).
  - One of the unique challenges for fake news detection on social media is how to identify fake news on newly emerged events. Most existing methods learn event-specific features that can not be transferred to unseen events. This paper proposed an end-to-end framework named Event Adversarial Neural Network (EANN), which can derive event-invariant features with adversarial learning and thus benefit the detection of fake news on newly arrived events.
  - 在虚假新闻检测问题中有一个特别挑战，如何识别新发生的事件的虚假新闻。大多数现有的方法学习事件特殊的特征，无法迁移到没见过的事件。这篇文章提出了EANN，使用对抗训练提出事件不变的特征，有利于检测新发生的事件上的虚假新闻。

## Distinguished Scholars in Fake News Detection (虚假新闻检测领域杰出学者)
- [Huan Liu](https://www.public.asu.edu/~huanliu/): Professor in Ira A. Fulton Schools of Engineering, Arizona State University. ACM/AAAI/AAAS/IEEE Fellow. [Google scholar](https://scholar.google.com.hk/citations?user=Dzf46C8AAAAJ&hl=zh-CN&oi=ao).
- [Kai Shu](http://www.cs.iit.edu/~kshu/):  Assistant Professor in the Department of Computer Science at Illinois Institute of Technology. [Google scholar](https://scholar.google.com.hk/citations?user=-6bAV2cAAAAJ&hl=zh-CN).
- [Juan Cao](http://people.ucas.ac.cn/~caojuan): Professor in Institute of Computing Technology, Chinese Academy of Science. [Google scholar](https://scholar.google.com.hk/citations?hl=zh-CN&user=fSBdNg0AAAAJ).
- [Kyumin Lee](http://web.cs.wpi.edu/~kmlee/): Associate Professor, Computer Science, Worcester Polytechnic Institute. [Google scholar](https://scholar.google.com.hk/citations?user=zQKRsSEAAAAJ&hl=zh-CN&oi=sra).
- [Reza Zafarani](http://reza.zafarani.net/): Assistant Professor, Department of Electrical Engineering and Computer Science, Syracuse University. [Google scholar](https://scholar.google.com.hk/citations?hl=zh-CN&user=l0h7wL0AAAAJ&view_op=list_works&sortby=pubdate).