# Fake News Detection 虚假新闻检测
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

This repo is a collection of AWESOME things about fake news detection, including papers, code, etc. Feel free to star and fork.

---

## Contents
- [Papers](#paper)
  - [Survey](#survey) 综述
  - [Social Context](#social) 社交上下文
  - [News Contents](#content-base) 新闻内容
    - [Multi-Modal](#multi-modal) 多模态
    - [Emotion](#emotion) 情感
    - [Style](#style) 风格
    - [Discourse Stucture](#discourse) 语篇结构
  - [Fact Checking](#fact) 真实性检验
  - [Explainable](#explainable) 可解释
  - [Transfer Learning](#transfer) 迁移学习
- [Datasets](#datasets)
- [Distinguished Scholars in Fake News Detection](#scholars)

---

## <span id="paper">Papers</span>
### <span id="survey">Survey</span> 综述
- [A survey on fake news and rumour detection techniques](https://www.sciencedirect.com/science/article/pii/S0020025519304372). Information Sciences, 2019, 497: 38-55.
- [Detection and resolution of rumours in social media: A survey](https://dl.acm.org/doi/abs/10.1145/3161603). ACM Computing Surveys (CSUR), 2018, 51(2): 1-36.
- [The Spread of True and False News Online](https://science.sciencemag.org/CONTENT/359/6380/1146.abstract). Science, 2018, 359(6380): 1146-1151.
- [Fake News Detection on Social Media: A Data Mining Perspective](https://dl.acm.org/doi/abs/10.1145/3137597.3137600?casa_token=Mf0tvofQf7kAAAAA:LgdXVmsJzYxVyrTgrhoFio_zxDXORoh6NNGP4__D64yam0rOKfwdbi__38Jg01U7pC-M19Tkb2NC_BU). ACM SIGKDD explorations newsletter, 2017, 19(1): 22-36.

<hr style="height:1px;border:none;border-top:1px dashed #DCDCDC;" />

### <span id="social">Social Context</span> 社交上下文
- KDD-2021 [Causal Understanding of Fake News Dissemination on Social Media](http://www.cs.iit.edu/~kshu/files/kdd_causal.pdf)
  - To mitigate negative impact of fake news, this paper argued that it is critical to understand what user attributes potentially cause users to share fake news.
  - 为了减轻虚假新闻的负面影响，本文认为了解哪些用户属性可能导致用户分享假新闻至关重要。
- SIGIR-2021 [User Preference-aware Fake News Detection](https://arxiv.org/pdf/2104.12259) [code](https://github.com/safe-graph/GNN-FakeNews)
  - A user is more likely to spread a piece of fake news when it confirms his/her existing beliefs/preferences. This paper studied the novel problem of exploiting user preference for fake news detection.
  - 用户更可能传播他感兴趣的虚假新闻。这篇文章研究一个新的问题，在虚假新闻检测中利用用户偏好信息。
- CIKM-2020 [FANG : Leveraging Social Context for Fake News Detection Using Graph Representation](https://dl.acm.org/doi/abs/10.1145/3340531.3412046?casa_token=33FpLHu6h20AAAAA:fc2L3COGdQCca7fS2l4rOjcP_LzmDMVI1fROs9Yxi0m7xTuyQUpec9sm6MZe0_Yli7Vo4tcDh6nURN8) [code](https://github.com/nguyenvanhoang7398/FANG).
  - This paper proposed Factual News Graph (FANG) for fake news detection, which is scalable in training as it does not have to maintain all nodes, and it is efficient at inference time, without the need to re-process the entire graph.
  - 这篇文章提出了FANG来解决虚假新闻检测问题，FANG在训练中具有可扩展性，不必维护所有节点，并且在推理时非常高效，而无需重新处理整个图。
- ICDM-2020 [Adversarial Active Learning based Heterogeneous Graph Neural Network for Fake News Detection](https://ieeexplore.ieee.org/abstract/document/9338358/).
  - This paper attempted to solve the fake news detection problem with the support of a news-oriented HIN and proposed a novel method AA-HGNN. AA-HGNN utilizes an active learning framework to enhance learning performance, especially when facing the paucity of labeled data.
  - 这篇文章尝试基于新闻导向的异构信息网络解决虚假新闻检测的方法，并且提出了一个新的方法AA-HGNN。AA-HGNN使用主动学习的框架来增强学习表现，特别是面对标签数据不足的情况。
- WSDM-2019 [Beyond News Contents : The Role of Social Context for Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3289600.3290994?casa_token=vzRcFcZbogkAAAAA:CgTc3CqhxgZ3JqqwPLrCAz_vVP2wShHGZvZnLZdeM2Evss5Uqu4-L1UUhLVB-G62_hfT-WcqZLW52gY)
  - This paper proposed a tri-relationship embedding framework (tensor factorization method) TriFN, which models publisher-news relations and user-news interactions simultaneously for fake news classification. 
  - 这篇文章提出了一种三者关系embedding的框架TriFN(一种tensor分解的方法)，同时建模新闻发布者和新闻的关系，用户和新闻的交互来检测虚假新闻。
- AAAI-2018 [Early Detection of Fake News on Social Media Through Propagation Path Classification with Recurrent and Convolutional Networks](https://ojs.aaai.org/index.php/AAAI/article/view/11268).
  - This paper first models the propagation path of each news story as a multivariate time series and builds a time series classifier that incorporates both recurrent and convolutional networks which capture the global and local variations of user characteristics along the propagation path respectively, to detect fake news.
  - 这篇文章首先将每个新闻故事的传播路径建模为一个多元时间序列，并建立了一个时间序列分类器，该分类器结合了RNN和CNN，分别捕获了沿传播路径的用户特征的全局和局部变化，以检测假新闻。
- CIKM-2017 [CSI : A Hybrid Deep Model for Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3132847.3132877?casa_token=qUOs7PlAOKYAAAAA:wvXMJ4nzbcW6CWGTJCREzIvR8vkxXe4rt7tlI1-k-_GANPG87nPv8Z2iaCQs0x_uVGlaPkbnLzMBuO4).
  - This paper proposed a hybrid model that combines the text of an article, the user response it receives, and the source users promoting it for a more accurate and automated prediction.
  - 这篇文章提出了一个混合模型，组合了文章的文本，用户的反馈，传播它的用户，来实现一个更准确自动的预测。
- AAAI-2016 [News Verification by Exploiting Conflicting Social Viewpoints in Microblogs](https://ojs.aaai.org/index.php/AAAI/article/download/10382/10241)
  - This paper proposed to exploit the conflicting viewpoints in microblogs to detect relations among news tweets and construct a credibility network of tweets with these relations. They detect fake news based on the graph.
  - 这篇文章提出利用微博中的冲突观点来获取tweet之间的关系，并且构造一个信用网络。基于这个图来实现虚假新闻检测。

<hr style="height:1px;border:none;border-top:1px dashed #DCDCDC;" />

### <span id="content-base">News Contents</span> 新闻内容

#### <span id="multi-modal">Multi-Modal</span> 多模态

- KDD-2021 [Multimodal Emergent Fake News Detection via Meta Neural Process Networks](https://arxiv.org/pdf/2106.13711.pdf)
  - Significant challenges are posed for existing detection approaches to detect fake news on emergent events, where large-scale labeled datasets are difficult to obtain. This paper proposed an end-to-end fake news detection framework named MetaFEND, which is able to learn quickly to detect fake news on emergent events with a few verified posts.
  - 现有的虚假新闻检测方法很难检测新出现的事件（很难获得样本）。这篇文章提出了一种MetaFEND的方法，可以基于少量验证过的帖子快速检测新出现的事件。
- EMNLP-2020 [Detecting Cross-Modal Inconsistency to Defend Against Neural Fake News](https://www.aclweb.org/anthology/2020.emnlp-main.163.pdf).
  - This paper proposed a novel problem defence against neural fake news with images and captions. To circumvent this problem, they present DIDAN,which exploits possible semantic inconsistencies between the text and image/captions to detect machine-generated articles. 
  - 这篇文章提出了一个新的问题，如何防御带有图片和图片描述的虚假新闻。为了解决这个问题，他们提出了DIDAN，使用文本和图片、图片描述之间的语义一致性来检测机器生成的文章。
- ICDM-2019 [Exploiting Multi-domain Visual Information for Fake News Detection](https://ieeexplore.ieee.org/abstract/document/8970940/).
  - Fake-news images may have significantly different characteristics from real-news images at both physical and semantic levels, which can be clearly reflected in the frequency and pixel domain. This paper proposed a novel framework Multi-domain Visual Neural Network (MVNN) to fuse the visual information of frequency and pixel domains for detecting fake news.
  - 虚假新闻中的图片和真实新闻的图片在物理和语义级别有很大的不同，这可以反应在频域和像素域。这篇文章提出了MVNN来融合频域和像素域的视觉信息来帮助欺诈检测。
- WWW-2019 [MVAE : Multimodal Variational Autoencoder for Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3308558.3313552?casa_token=tzDWRQ_VVxEAAAAA:Fc3ubDseJsk05pUdznqtA4cDD9BGemHHh8A1T6Nzur8qa0SU8SQY8O7_KRoj8tE_Ah75p8sslKjo4bU).
  - A shortcoming of the current approaches for the detection of fake news is their inability to learn a shared representation of multimodal (textual + visual) information. This paper proposed Multimodal Variational Autoencoder (MVAE), which uses a bimodal variational autoencoder for the task of fake news detection.
  - 现在的虚假新闻检测的方法有一个缺点，针对多模态的信息，他们不能学习一个共享的表示。这篇文章提出了MVAE，使用一个双模态的VAE（变分自动编码机）。
- CIKM-2018 [Rumor detection with hierarchical social attention network](https://dl.acm.org/doi/abs/10.1145/3269206.3271709?casa_token=5ODcLmf4aHkAAAAA:vyQxpSmSkaNDQbzuqUL81HeLeKncNpLa8wEOxANLqGzRXU4W1SZ05Wwgo7BZgegvtW_v5KMt0-UHcmI).
  - A news usually contains a source post and a set of related posts. This paper divided related posts into several subevent according to timestamp, and proposed a hierarchical structure to detect rumor.
  - 一个新闻通常包含原始帖子和一些相关回复转发帖子。这篇文章将相关帖子根据时间戳划分为多个子事件段，并且提出了一个层次结构来检测谣言。
- KDD-2018 [EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3219819.3219903?casa_token=_1c4Ao5K0moAAAAA:jUDfxV9HoKVeRQnGwXYI6oGEm1MMOusLGjPiZjEAOm94MVld0ntG44G4kwdK_qLtipJa32ngFQE995U).
  - One of the unique challenges for fake news detection on social media is how to identify fake news on newly emerged events. Most existing methods learn event-specific features that can not be transferred to unseen events. This paper proposed an end-to-end framework named Event Adversarial Neural Network (EANN), which can derive event-invariant features with adversarial learning and thus benefit the detection of fake news on newly arrived events.
  - 在虚假新闻检测问题中有一个特别挑战，如何识别新发生的事件的虚假新闻。大多数现有的方法学习事件特殊的特征，无法迁移到没见过的事件。这篇文章提出了EANN，使用对抗训练提出事件不变的特征，有利于检测新发生的事件上的虚假新闻。
- ACMMM-2017 [Multimodal fusion with recurrent neural networks for rumor detection on microblogs](https://dl.acm.org/doi/abs/10.1145/3123266.3123454?casa_token=YezZ9B--TsAAAAAA:-tXSt-GU3owhdAj1Oaa5PEqwCdO0kfIXfL_VDQJavItIcAswt_rAyxBDucIJxEAXxj5pdbMTT5lYH7Q).
  - This paper proposed a novel Recurrent Neural Network with an attention mechanism (att-RNN) to fuse multimodal features for effective rumor detection.
  - 这篇文章提出了一个新的基于RNN和attention的方法，来融合多模态特征来进行高效的谣言检测。

#### <span id="emotion">Emotion</span> 情感

- WWW-2021 [Mining Dual Emotion for Fake News Detection](https://arxiv.org/abs/1903.01728).
  - This paper explored the relationship between publisher emotion and social emotion in fake news and real news, and proposed a method to model dual emotion (publisher emotion, social emotion) from five aspects, including emotion category, emotional lexicon, emotional intensity, sentiment score, other auxiliary features.
  - 这篇文章探索了在虚假新闻和真实新闻中新闻发布者的情感和社交情感之间的联系，并且提出了一种方法从五个方面来建模对偶情感，情感类别，情感词典，情感强度，情感分数，其他辅助特征。
- SIGIR-2019 [Leveraging emotional signals for credibility detection](https://dl.acm.org/doi/abs/10.1145/3331184.3331285?casa_token=GvwK-IepIWIAAAAA:yJKIEOHNHSIB7XcmRH5y9S1-4GqCAgmI5eZ4iDrsm4_HZ2x_2O3T8Sk0s-prWzUBR45fq6BFuNOCZSI).
  - The role of emotional signals has not yet been explored in existing methods. This paper proposed an LSTM model that incorporates emotional signals (emotional intensity lexicon) extracted from the text of the claims to differentiate between credible and non-credible ones. 
  - 现有的虚假新闻检测方法还没有探索情感信号的使用。这篇文章提出了一个LSTM模型，使用从claim的文本信息中提取到的情感信号（情感强度词典）来区分可信和不可信的新闻。

#### <span id="style">Style</span> 风格
- ACL-2018 [A stylometric inquiry into hyperpartisan and fake news](https://www.aclweb.org/anthology/P18-1022.pdf).
  - This paper demonstrated the importance of style for fake news detection.
  - 这篇文章验证了风格对于虚假新闻检测的重要性

#### <span id="discourse">Discourse stucture</span> 语篇结构
- NAACL-2019 [Learning hierarchical discourse-level structure for fake news detection](https://arxiv.org/abs/1903.07389).
  - This paper proposed a novel method to study automatic document structure learning for fake news detection, and detect fake news with a hierarchical structure.
  - 这篇文章提出了一个新的方法可以自动抓获文档结构，并且使用一个层次结构检测虚假新闻。
  


<hr style="height:1px;border:none;border-top:1px dashed #DCDCDC;" />

### <span id="fact">Fact Checking</span> 真实性检验

- EMNLP-2020 [Where Are the Facts? Searching for Fact-checked Information to Alleviate the Spread of Fake News](https://www.aclweb.org/anthology/2020.emnlp-main.621.pdf).
  - To stop users from spreading fake news, this paper proposed a novel framework to search for fact-checking articles which is related to original poster. The search can directly warn fake news posters and online users about misinformation, discourage them from spreading fake news.
  - 这篇文章提出了一个新的框架，检索和原始帖子相关的经过真实性检验的文章，并且贴出这些文章来警告用户这可能是虚假新闻来组织虚假新闻的传播。
- ACL-2020 [Fine-grained Fact Verification with Kernel Graph Attention Network](https://www.aclweb.org/anthology/2020.acl-main.655/).
  - This paper proposed Kernel Graph Attention Network (KGAT). KGAT introduces node kernels, which better measure the importance of the evidence node, and edge kernels, which conduct fine-grained evidence propagation in the graph.
  - 本文提出了KGAT。 KGAT引入了可以更好地衡量证据节点重要性的节点核，以及可以在图中进行细粒度证据传播的边核。
- SIGIR-2019 [Learning from fact-checkers: analysis and generation of fact-checking language](https://dl.acm.org/doi/abs/10.1145/3331184.3331248?casa_token=DUE-x5HZNywAAAAA:ZmQO0tm5NXc6ibbo6898F5f8wZcIFL2HDhy8KSjmnSy2k9ArFi5Y2Ew450lLXPMrLoIY9YfmeNCIkn0).
  - Fact-checkers refute the misinformation by replying to the original poster and provides a fact-checking article as a supporting evidence. This paper focuses on generating responses for fact-checkers with a GRU-based generative model.
  - fact-checker会在别人的帖子下回复原始真实文章作为证据来驳斥错误信息。这篇文章旨在为fact-checker生成回复的句子来鼓励fack-checker进行更多的活动。提出了一个基于GRU的生成模型。
- EMNLP-2018 [DeClarE : Debunking Fake News and False Claims using Evidence-Aware Deep Learning](https://www.aclweb.org/anthology/D18-1003.pdf).
  - This paper proposed a novel method named DeClarE for fact checking. Given an input claim, DeClarE searches for web articles related to the claim and utilizes contexts of both articles and claims to check the fact.
  - 这篇文章针对真实性检验提出了一种新的方法DeClarE。给定一个言论，DaClarE在网上搜索相关的文章，使用相关文章和原始言论预测真实性。
- SIGIR-2018 [The rise of guardians: Fact-checking URL recommendation to combat fake news](https://dl.acm.org/doi/abs/10.1145/3209978.3210037?casa_token=wY-498SHN8EAAAAA:_p67817oi5vgktXQMPr8AyJ3AAOrn7lRoP_UhJ6dCf3WtVUquG499062k2gFJ-lVwbTesFZHzbS5cK0).
  - Find guardians who are willing to spread verified news and stimulate them to disseminate fact-checked news/information with a matrix factorization model.
  - 找到愿意传播经过真实性检验的新闻的保护者，这类人通常会在评论中给出真实新闻的链接。激励他们传播真实新闻。本文提出了一种基于矩阵分解的方法。
- WSDM-2018 [Leveraging the Crowd to Detect and Reduce the Spread of Fake News and Misinformation](https://dl.acm.org/doi/abs/10.1145/3159652.3159734?casa_token=ZIwfXFyA_GUAAAAA:4hCVbLCZe1O07SvK4ip688MeVzCDcgTVqlofgch_bmIvo0bfUGqli7mCG9tMRToO_6jPN5FFTYC2Aac).
  - This paper developed a scalable online algorithm, Curb, to select which stories to send for fact checking and when to do so to efficiently reduce the spread of misinformation.
  - 这篇文章开发了一种可扩展的在线算法Curb，来选择哪些故事进行事实检查以及何时进行检查，以有效地减少错误信息的传播。



<hr style="height:1px;border:none;border-top:1px dashed #DCDCDC;" />

### <span id="explainable">Explainable</span> 可解释

- ACL-2020 [GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media](https://www.aclweb.org/anthology/2020.acl-main.48.pdf).
  - Given the source short-text tweet and the corresponding sequence of retweet users without text comments, this paper proposed GCAN to predict whether the source tweet is fake or not, and generate explanation by highlighting the evidences on suspicious retweeters and the words they concern. 
  - 给定原始的短文本推特以及相关的转发用户序列，这篇文章提出了GCAN来预测原始推特是否为虚假的，并且通过标注出可以的转发者和词语来生成解释。
- KDD-2019 [Defend: Explainable Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3292500.3330935?casa_token=xfpltxWHUwEAAAAA:mifb7BpUrE-nAm4hHpAzW2Gozw8g_xmA2j6UXRzJKm0lAUS0Z8gNEXpE3FRWJnSpeIeKBE4cuB45tYc).
  - Explainability has been ignored by most existing fake news detection methods. This paper proposed dEFEND to detect fake news by jointly exploring explainable information from news contents and user comments.
  - 现在大多数虚假新闻检测的方法忽略了可解释的重要性。这篇文章提出了dEFEND来检测虚假新闻。同时考虑了新闻内容和用户评论中的可解释信息。

<hr style="height:1px;border:none;border-top:1px dashed #DCDCDC;" />

### <span id="transfer">Transfer Learning</span> 迁移学习
- SIGIR-2021 [Generalizing to the Future: Mitigating Entity Bias in Fake News Detection](https://arxiv.org/abs/2204.09484). [Code](https://github.com/ICTMCG/ENDEF-SIGIR2022)
  - Existing fake news detection methods overlooked the unintended entity bias in the real-world data, which seriously influences models' generalization ability to future data. They propose an entity debiasing framework (ENDEF) which generalizes fake news detection models to the future data by mitigating entity bias from a cause-effect perspective. Based on the causal graph among entities, news contents, and news veracity, they separately model the contribution of each cause (entities and contents) during training. In the inference stage, they remove the direct effect of the entities to mitigate entity bias.
  - 现有的方法忽视了训练数据集中存在的实体偏差（entity bias）造成的负面影响，导致其在未来数据上泛化效果不佳。作者提出了一种简单有效的实体去偏方法：建立实体、新闻内容与新闻真实性标签之间的因果联系，在训练阶段分别建模实体和内容对新闻真实性的影响，在测试阶段直接移除基于实体预测新闻真实性的部分以去除实体偏差。
- CIKM-2021 [MDFEND: Multi-domain Fake News Detection](https://dl.acm.org/doi/pdf/10.1145/3459637.3482139). [Code](https://github.com/kennqiang/MDFEND-Weibo21)
  - They first design a benchmark of fake news dataset for MFND with domain label annotated, namely Weibo21, which consists of 4,488 fake news and 4,640 real news from 9 different domains. They further propose an effective Multi-domain Fake News Detection Model (MDFEND) by utilizing a domain gate to aggregate multiple representations extracted by a mixture of experts. 
  - 作者首先贡献一个多领域虚假新闻检测的数据集，包含领域标签，叫做Weibo21，包含4488条假新闻和4640条真新闻，共9个不同的领域。他们进一步提出一种高效的多领域虚假新闻检测模型，通过使用领域门控机制来聚合混合专家提取的多个表示。
- AAAI-2021 [Embracing Domain Differences in Fake News: Cross-domain Fake News Detection using Multi-modal Data](Embracing Domain Differences in Fake News: Cross-domain Fake News Detection using Multi-modal Data).
  - The performance of fake news detection methods generally drops if news records are coming from different domains, especially for domains that are unseen or rarely-seen during training. Under the setting of unknown domain label, this paper proposed a method to discover domain label and a domain-specific and domain-shared structure to detect fake news.
  - 如果新的记录来自不同领域，现有的虚假新闻检测方法效果会下降，尤其是该领域在训练时没有见过。在不知道领域标签的设定下，这篇文章提出了一个发现domain label的方法，以及一个领域特定和领域共享的结构来检测虚假新闻。
- KDD-2020 [DETERRENT: Knowledge Guided Graph Attention Network for Detecting Healthcare Misinformation](https://dl.acm.org/doi/abs/10.1145/3394486.3403092?casa_token=xxQDHHtbbbMAAAAA:akPj6Pmd2_0HwWWZOtM3k3OeeUd829CDJwzDgElElvTqOCjHuRjM92haLZ3lFZ-vAR6w4f5BclyPf_s).
  - This paper leveraged on the additional information from medical knowledge graph to help detect healthcare misinformation. In addition, the proposed method is capable of providing useful explanations for the results of detection.
  - 这篇文章利用额外的医学知识图谱的信息来帮助检测医疗保健错误信息。另外，提出的方法可以为检测的结果提供解释。
- EMNLP-2019 [Different Absorption from the Same Sharing : Sifted Multi-task Learning for Fake News Detection](https://www.aclweb.org/anthology/D19-1471/).
  - This paper utilizes stance detection task to help fake news detection. They designed a sifted multi-task learning method with a selected sharing layer for fake news detection. The selected sharing layer adopts gate mechanism and attention mechanism to filter and select shared feature flows between tasks.
  - 本文利用观点检测任务来辅助虚假新闻检测。他们设计了一种筛选的多任务学习方法，并使用了可选共享层来检测虚假新闻。可选共享层采用门机制和注意力机制对任务之间的共享特征流进行过滤和选择。
- WWW-2019 [A topic-agnostic approach for identifying fake news pages](https://dl.acm.org/doi/abs/10.1145/3308560.3316739?casa_token=yADZC7xsZswAAAAA:WdPv6azaEkDV3rvkfdEtzm4_KBzTZgufpRKzR6OaXpo25eVI_75nPCg5JF2aHefyaDImxNl-30KSYYA).
  - An important challenge for existing approaches comes from the dynamic nature of news: as new political events are covered, topics and discourse constantly change and thus, a classifier trained using content from articles published at a given time is likely to become ineffective in the future. To address this challenge, this paper proposed a topic-agnostic (TAG) classification strategy that uses topic-agnostic (linguistic and web-markup) features to identify fake news pages.
  - 现有虚假新闻检测的方法的一个重要挑战来自新闻的动态性质：随着报道新的政治事件，话题和话语不断变化，因此，使用给定时间发表的文章内容进行训练的分类器预测将来可能会失效。为了解决这一挑战，本文提出了一种与主题无关的（TAG）分类策略，该策略使用与主题无关的（语言和Web标记）特征来识别假新闻页面。
- WWW-2018 [Detect Rumor and Stance Jointly by Neural Multi-task Learning](https://dl.acm.org/doi/abs/10.1145/3184558.3188729?casa_token=y_PooBtZLeEAAAAA%3AFc0eA5ID4GRn6KfmAcAR4bohcAIpYvKvlUC36l71-ub5pUgJOyfqTWke06i3H-Ux92CbIPGYqq-Klds).
  - This paper argues that rumor detection and stance classification should be treated as a joint, collaborative effort, considering the strong connections between the veracity of claim and the stances expressed in responsive posts. They proposed a joint framework which capture both task-specific and task-invariant features.
  - 本文认为，谣言检测与立场分类之间有着的紧密联系，因此将谣言发现和立场分类联合训练。 他们提出了一个联合框架，该框架同时捕获了任务特定和任务不变的特征。

---

## <span id="datasets">Datasets</span>

- [FacebookHoax](https://github.com/gabll/some-like-it-hoax) [Some like it hoax: Automated fake news detection in social networks](https://arxiv.org/abs/1704.07506)
- [BuzzFeedNews](https://github.com/BuzzFeedNews/2016-10-facebook-fact-check) [Hyperpartisan Facebook Pages Are Publishing False And Misleading Information At An Alarming Rate](https://www.buzzfeednews.com/article/craigsilverman/partisan-fb-pages-analysis)
- [LIAR (ACL2017)](https://www.cs.ucsb.edu/˜william/data/liar_dataset.zip) [Embracing Domain Differences in Fake News: Cross-domain Fake News Detection using Multi-modal Data](https://www.aclweb.org/anthology/P17-2067.pdf)
- [CoAID](https://github.com/cuilimeng/CoAID) [CoAID: COVID-19 Healthcare Misinformation Dataset](https://arxiv.org/pdf/2006.00885.pdf)
- [FakeNewsNet (Journal of Big Data 2020)](https://github.com/KaiDMML/FakeNewsNet) [FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media](https://arxiv.org/pdf/1809.01286)
- [FakeHealth (AAAI 2021)](https://doi.org/10.5281/zenodo.3606757) [Ginger cannot cure cancer: Battling fake health news with a comprehensive data repository](https://ojs.aaai.org/index.php/ICWSM/article/download/7350/7204/)

---

## <span id="scholars">Distinguished Scholars in Fake News Detection</span> (虚假新闻检测领域杰出学者)
- [Huan Liu](https://www.public.asu.edu/~huanliu/): Professor, Arizona State University. ACM/AAAI/AAAS/IEEE Fellow. [[Google Scholar]](https://scholar.google.com/citations?user=Dzf46C8AAAAJ)
- [Juan Cao](http://people.ucas.ac.cn/~caojuan): Professor, Institute of Computing Technology, Chinese Academy of Science. [[Google Scholar]](https://scholar.google.com/citations?user=fSBdNg0AAAAJ)
- [Preslav Nakov](https://www.hbku.edu.qa/en/staff/dr-preslav-nakov): Principal Scientist, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University. [[Google Scholar]](https://scholar.google.com/citations?user=DfXsKZ4AAAAJ)
- [Kyumin Lee](http://web.cs.wpi.edu/~kmlee/): Associate Professor, Computer Science, Worcester Polytechnic Institute. [Google Scholar](https://scholar.google.com/citations?user=zQKRsSEAAAAJ)
- [Kai Shu](http://www.cs.iit.edu/~kshu/):  Assistant Professor, Illinois Institute of Technology. [[Google Scholar]](https://scholar.google.com/citations?user=-6bAV2cAAAAJ)
- [Jing Ma](https://majingcuhk.github.io/): Assistant Professor, Hong Kong Baptist University. [[Google Scholar]](https://scholar.google.com/citations?user=78Jby0EAAAAJ)
- [Reza Zafarani](http://reza.zafarani.net/): Assistant Professor, Department of Electrical Engineering and Computer Science, Syracuse University. [[Google Scholar]](https://scholar.google.com/citations?user=l0h7wL0AAAAJ)
