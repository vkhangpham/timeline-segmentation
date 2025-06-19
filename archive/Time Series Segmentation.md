# **Time Series Segmentation of Scientific Literature: Identifying Evolving Research Fronts through Topic and Citation Dynamics**

## **Executive Summary**

This report provides a comprehensive overview of advanced methodologies for segmenting time series data derived from scientific publications. The primary objective is to delineate distinct temporal periods characterized by shifts in dominant research topics, prevailing methods, or emergent techniques within a given scientific domain. The analytical framework integrates diverse data points, including textual content (topics extracted from abstracts and keywords), publication year, citation counts, and intricate citation relationships.

The analysis reveals a progression from foundational time series segmentation techniques to sophisticated, integrated solutions that combine dynamic topic models with citation network analysis. A critical aspect of this evolution is the application of change point detection and burst detection algorithms, which are instrumental in identifying significant, abrupt shifts in research focus or impact. State-of-the-art solutions highlighted in this report encompass neural dynamic topic models, iterative frameworks that incorporate external time series feedback, and temporal graph neural networks designed for dynamic citation analysis.

For practical application, the report identifies several established open-source tools. These include sktime and ClaSPy for general time series segmentation, and Bibliometrix, VOSviewer, CiteSpace, pyBibX, and BERTrend for specialized bibliometric and topic analysis. These tools offer robust capabilities for data processing, visualization, and the identification of temporal patterns.

The methodologies and tools discussed herein offer powerful analytical lenses for various stakeholders, including individual researchers, funding agencies, and policymakers. By enabling a deeper understanding of the intellectual structure and temporal evolution of scientific fields, these approaches facilitate the identification of emerging research fronts and inform strategic decisions regarding research investment and direction.

## **1\. Introduction: Mapping the Temporal Landscape of Scientific Knowledge**

### **1.1 The Challenge of Dynamic Research Fronts**

Scientific knowledge is not a static entity; it is a continuously evolving landscape where new ideas emerge, established methodologies gain or lose prominence, and fundamental paradigms undergo transformative shifts over time.1 Understanding these inherent dynamics is crucial for effectively navigating the vast and complex body of scientific literature.1 The ability to map out the intellectual structure and temporal evolution of research fields allows for a clearer perception of how scientific progress unfolds. This dynamic nature necessitates sophisticated analytical approaches that can capture and interpret these temporal shifts, moving beyond static snapshots of research activity. The identification of "research fronts"—dynamic, knowledge-driven clusters of scholarly activity—is central to this endeavor, as these represent areas of intense current exploration and rapid discovery, often characterized by tightly interconnected networks of recent publications.4

### **1.2 Objectives of Time Series Segmentation in Publication Data**

Time series segmentation, in the context of scientific publication data, is a strategy employed to divide a chronological sequence of documents into discrete time chunks, or "segments".6 The primary goal is to extract temporal patterns by observing the characteristics of the data within these segments. This approach aims to uncover latent temporal evolution patterns and detect unexpected regularities or shifts in research regimes, thereby making the analysis of massive time series data more manageable.7

The applications of segmenting time series publication data are multifaceted and provide valuable insights into the underlying research landscape:

* **Trend Analysis:** Segmentation algorithms often use statistical methods to group data into similar segments, which can expose different trends, seasonality, or other temporal elements in research activity.6 This allows for the identification of areas of growing interest over time.2  
* **Forecasting:** Insights gained from analyzing smaller, homogeneous segments can be leveraged to forecast future trends in scientific output or topic development.6  
* **Noise Reduction:** By grouping noisy or distorted data (e.g., fluctuations due to data collection errors) into smoothed segments, segmentation can help in revealing clearer underlying patterns.6  
* **Anomaly Detection:** Characterizing the properties within each time segment enables the identification of unusual or anomalous patterns in the data, which could signify unexpected developments or shifts in research.6  
* **Semantic Segmentation:** Beyond mere statistical grouping, semantic segmentation aims to divide poorly understood time series into discrete and homogeneous segments that are semantically meaningful, revealing the underlying thematic evolution.7

### **1.3 Key Data Points for Analysis**

Effective time series segmentation of publication data relies on a rich set of data points that capture both the content and the relational dynamics of scientific literature. These data points provide the raw material for identifying periods characterized by dominant topics, methods, or techniques:

* **Topics:** Extracted from the textual content of publications (e.g., abstracts, titles, keywords), topics represent the thematic content of research. Topic modeling algorithms identify latent topics by analyzing word co-occurrence patterns, providing a higher-level understanding of the main subjects discussed in a corpus.8 The evolution of these topics over time is a primary indicator of shifting research fronts.1  
* **Publication Year:** This fundamental temporal attribute orders the data chronologically, forming the basis of the time series. It allows for the tracking of topic emergence, growth, and decline over specific periods.2  
* **Citation Count:** The total number of times a publication has been cited by other works serves as a quantitative measure of its impact and influence.5 Analyzing citation counts over time can reveal periods of heightened influence or "bursts" of attention for specific papers or research areas.1  
* **Citation Relationships:** Beyond simple counts, the relationships between citing and cited papers (e.g., co-citation, direct citation networks) provide a deeper understanding of the flow of ideas and the intellectual structure of a field.1 These networks can map out research fronts, identify influential works, and reveal the dynamics of scholarly connections.5 For instance, highly cited and co-cited publications often represent the core of a predominant scientific paradigm.11

The combined analysis of these data points allows for a robust characterization of temporal segments in scientific literature, offering a multifaceted view of research evolution.

## **2\. Methodological Approaches to Time Series Segmentation**

### **2.1 Foundational Time Series Segmentation Algorithms**

Time series segmentation is a crucial preprocessing technique that partitions a time series into sequences of discrete chunks, each exhibiting homogeneous characteristics. Several common algorithmic approaches form the bedrock of this field:

* **Top-down (Divide and Conquer/Binary Split):** This approach begins with the entire dataset and recursively breaks it down into smaller segments. The splitting criterion typically involves maximizing the differences between the two resulting segments. This process continues iteratively until a clear pattern emerges within each segment or a predefined stopping condition is met.6 It is effective for identifying major structural changes in the data.  
* **Bottom-up:** In contrast, the bottom-up approach starts by breaking the dataset into multiple, often very small, initial segments. An error function is then used to calculate the difference between the actual data and the data as represented by these segments. The goal is to find the segmentation that minimizes this error. Segments are iteratively merged by selecting those whose merger results in the smallest increase in the error function. This merging process continues until a desired number of segments is reached or an error threshold is met.6 This method is particularly useful for identifying subtle changes and building segments from local homogeneity.  
* **Sliding Window:** This algorithm defines a fixed or adaptive window (with a start and end boundary) that iteratively slides across the time series. The window extends as long as the data within it fits under some user-defined threshold of homogeneity. When the threshold is exceeded, a segment boundary is identified, and a new window begins. This method is computationally efficient and suitable for detecting local changes.6

It is important to note that these distinct approaches are not mutually exclusive and can often be combined to leverage their respective strengths. For example, the SWAB algorithm (Sliding Window And Bottom-up) integrates both techniques to achieve more robust segmentation.6 While these foundational methods are general, their application to complex publication data requires integration with domain-specific techniques to capture the nuances of scientific evolution.

### **2.2 Topic Modeling for Identifying Dominant Themes**

Topic modeling is a powerful technique in natural language processing (NLP) that automatically uncovers abstract themes or "topics" within large collections of textual documents. When applied to scientific literature, it is instrumental in identifying dominant research themes and their evolution over time.

#### **2.2.1 Core Concepts of Topic Modeling**

Topic modeling algorithms identify latent topics by analyzing patterns of word co-occurrence within documents.8 This process goes beyond individual words, providing a higher-level understanding of the main subjects discussed in a text corpus, which is particularly valuable for large volumes of text data where manual analysis is impractical.8

Key approaches to topic modeling include:

* **Latent Dirichlet Allocation (LDA):** One of the most widely used probabilistic topic models, LDA represents documents as probabilistic distributions over a predefined number of topics, and topics as probabilistic distributions over words.8 It assumes a generative process where each document is created by first randomly assigning a distribution of topics to the document, then choosing a topic from that distribution, and finally selecting a word from the chosen topic's word distribution.8 LDA is effective for text understanding, document clustering, organization, and information extraction.8  
* **Probabilistic Latent Semantic Analysis (pLSA):** A precursor to LDA, pLSA also discovers latent topics by modeling word-document co-occurrence. However, unlike LDA, pLSA models topics at the document level, which can lead to issues with scalability and overfitting.8  
* **Parallel Latent Dirichlet Allocation (pLDA):** This variant of LDA is designed for distributed computing environments, partitioning document collections into smaller subsets and performing LDA independently on each partition. This approach enhances scalability for very large datasets.8

These models provide a foundational layer for understanding the thematic content of scientific literature, which can then be analyzed temporally to identify shifts in research focus.

#### **2.2.2 Dynamic Topic Models (DTMs)**

Traditional topic models like static LDA evaluate documents in a corpus all at once, making them less suitable for analyzing chronological data and tracking topic evolution over time.9 Dynamic Topic Models (DTMs) address this limitation by explicitly modeling the temporal changes in topics and their prevalence within documents.9

DTMs are a variant of topic models specifically designed for temporal archives.9 They introduce a novel approach by creating topics defined by a set of words and associating them with a timestamp or time range.9 This allows for the observation of when topics emerge, grow, or decline.9 Early DTMs, such as those developed by Blei and Lafferty (2006), used Gaussian time series on the natural parameters of multinomial topics and logistic normal topic proportion models to enable the development of topics over time.9

Recent advancements in DTMs leverage neural topic modeling techniques, such as **BERTopic**, to capture time-aware features of evolving topics across different periods.1 BERTopic, for instance, can be fitted on documents from non-overlapping time periods and then merged based on pairwise cosine similarity between topics, enabling dynamic topic modeling in an online learning setting.15 This allows for the identification and tracking of emerging topics and trends over time.16

The output of DTMs typically includes a set of topics, each represented by a list of top words, but crucially, these topics are associated with a temporal dimension. This enables researchers to gain a qualitative understanding of the contents of a large document collection over time and provides quantitative, predictive models of sequential corpora.9

#### **2.2.3 Iterative Topic Modeling with Time Series Feedback (ITMTF)**

While DTMs inherently handle temporal aspects of topics, the Iterative Topic Modeling with Time Series Feedback (ITMTF) framework offers a novel approach by integrating probabilistic topic models with *external* time series variables.13 This framework is designed to discover "causal topics"—semantically coherent topics that exhibit strong, potentially lagged, associations with a non-textual time series variable, such as stock prices.13

The core mechanism of ITMTF involves an iterative refinement process where time series data provides feedback to the topic model.13 This feedback is injected by imposing prior distributions on topic model parameters in each iteration. The process unfolds as follows:

1. **Initial Topic Generation:** A standard probabilistic topic model (e.g., LDA) is applied to the time-stamped document collection to generate an initial set of topics.13  
2. **Topic-level Causality Analysis:** For each topic, a "topic curve" is generated by summing the topic's coverage across documents with the same timestamp. This topic curve is then treated as a time series and correlated with the external non-textual time series using standard causality measures (e.g., Pearson correlations, Granger tests).13 This step identifies topics that are significantly correlated with the external time series and determines their associated time lags, indicating potential causal relationships.13  
3. **Word-level Causality Analysis:** For the identified candidate causal topics, the causality measure is applied again to find the most significant causal words among their top words. The impact values of these words (e.g., word-level Pearson correlations with the external time series) are recorded.13 This step refines the understanding of which specific terms within a topic drive its correlation with the external variable.  
4. **Prior Generation:** A prior distribution on the topic model parameters is defined using the significant terms and their impact values from the previous step.13 This prior "steers" the topic modeling process, encouraging the formation of topics that are more likely to be correlated with the external time series.13 Words are separated into "positive" and "negative" impact terms based on their correlation, ensuring consistent impact relative to the external time series within a topic.13  
5. **Remodeling with Prior:** The topic modeling method is reapplied to the document collection, but this time incorporating the generated prior. This ensures that the newly discovered topics are dynamically adapted to fit the patterns of the external time series data.13  
6. **Iteration:** Steps 2-5 are repeated until predefined stopping criteria are met, such as achieving a desired topic quality or observing no further significant changes in topic correlation.13

The strength of ITMTF lies in its ability to integrate non-textual time series data directly into the topic modeling process, rather than merely as a post-processing filter. This iterative feedback mechanism allows for the discovery of topics that are not only semantically coherent but also strongly correlated with external dynamics, providing a more nuanced understanding of underlying trends.13 While the original examples focus on stock prices, the conceptual framework can be extended to bibliometric metrics like citation counts or topic prevalence as the "external time series variable" to identify periods of dominant topics or methods in scientific literature, by treating these metrics as the numerical time series to be explained by the textual content.13

### **2.3 Citation Network Analysis for Influence and Evolution**

Citation analysis is a fundamental bibliometric method used to measure the impact and influence of scholarly works by examining patterns and frequencies of citations.5 It provides a powerful lens for understanding the flow of ideas and the evolution of knowledge within and across different fields.5

#### **2.3.1 Principles of Citation Analysis**

At its core, citation analysis involves tracking citations to identify influential works, map research fronts, and gain insights into the structure and dynamics of scholarly networks.5 Key metrics include:

* **Citation Count:** The total number of times a publication has been cited, serving as a quantitative measure of its direct impact.5 Highly cited papers often provide a snapshot of a discipline, indicating acknowledged subfields, theories, and methodologies.1  
* **Journal Impact Factor:** A measure of the average number of citations received by articles in a particular journal over a two-year period.5  
* **h-index:** A metric that attempts to quantify both the productivity and citation impact of a scholar or a publication set.5

Citation analysis is invaluable for navigating the vast scientific literature, identifying significant and influential works, and mapping research trends over time.5 By analyzing citation patterns, researchers can track the emergence and evolution of topics, identify "hot spots" of activity, and even forecast future directions.5 This method also helps in identifying research gaps and opportunities, spotting emerging trends, and understanding connections between fields.5

#### **2.3.2 Co-citation Analysis and Research Fronts**

Co-citation analysis is a specific technique within citation analysis that measures the frequency with which two documents are cited together by a third document.11 The strength of co-citation indicates a conceptual relationship between the co-cited documents, even if they do not directly cite each other.

Co-citation clustering involves grouping documents that are frequently co-cited. These clusters are assumed to represent narrow subject matter groupings that reflect the social and cognitive structures of research specialties.11 A cluster of highly cited and co-cited scientific publications in a co-citation network is often considered to represent the core of a predominant scientific paradigm or a "research front".11

The use of co-citation clustering is particularly effective for:

* **Identifying Research Fronts:** By analyzing the frequency of co-citation, researchers can identify areas of intense, current research activity.11  
* **Detecting Emergent Paradigms:** The growth of a paradigm can be depicted and animated through the rise of citation rates and the movement of its core cluster towards the center of the co-citation network.11 This helps in understanding paradigmatic transformations in scientific activity.11  
* **Visualizing Scientific Domain Evolution:** Co-citation networks, especially when represented through social networks, offer a powerful technique for visualizing and analyzing large scientific domains, revealing their evolution over time and suggesting new lines for development.11

For instance, studies have used co-citation analysis of author or publication records across two-year time slices to identify cohesive subfields and track fluctuations in their membership, providing evidence for paradigm shifts.19

#### **2.3.3 Dynamic Citation Network Models**

Traditional citation analysis often provides a static view of influence. However, scientific literature and its underlying citation networks are inherently dynamic. Dynamic citation network models aim to capture this temporal evolution, recognizing that the impact and relevance of papers change as new publications emerge and new citation relationships are formed.

One advanced approach involves **Temporal Graph Neural Networks (TGNs)**, which are powerful tools for modeling dynamic interactions in networks.20 In the context of citation networks, TGNs can continuously update a paper's embedding (a numerical representation of its characteristics and context) as new citation relationships appear.20 This enhances the paper's relevance for future recommendations and allows for a more precise understanding of how the academic community's view of a paper evolves.20

The "Dynamic Egocentric Models for Citation Networks" framework, for example, models the cumulative number of citations papers receive over time using multivariate counting processes.22 This framework incorporates various network and nodal statistics, including textual information (e.g., LDA-based matching statistics on abstracts), to understand the mechanisms driving citation network evolution.22 By analyzing factors like preferential attachment (how likely a paper is to be cited based on its current citations), recency-based citation intensity (temporary elevation due to recent citations), and triangle statistics (patterns of relationships between citing and cited papers), these models provide interpretable insights into *why* certain papers become more influential over time.22 This allows for the identification of dynamic developments, such as trending papers or shifts in the structure of knowledge dissemination.22 The ability to scale to large networks makes these models practical for analyzing real-world, complex citation datasets.22

## **3\. Integrated Solutions for Research Front Detection**

Identifying evolving research fronts requires a holistic approach that combines the thematic insights from topic modeling with the relational dynamics captured by citation networks. Purely language-driven models or citation networks alone may not fully explain the emergence of new ideas or the evolution of scientific fields.23

### **3.1 Combining Topic Models and Citation Networks**

The integration of topic models and citation networks provides a powerful synergy for understanding the temporal evolution of scientific literature. This combined approach allows researchers to analyze both the content (what is being discussed) and the influence (how ideas propagate and connect) over time.

A notable approach is the "inheritance topic model" proposed by He et al. (2009).1 This framework adapts the Latent Dirichlet Allocation (LDA) model to the citation network to explicitly model topic evolution.1 Unlike previous methods that treated papers as a "bag of words" and largely ignored citation impact, the inheritance topic model leverages citations as inherent elements that naturally indicate linkages between topics.24

The core idea is that when a paper A cites paper B, paper A often uses content from B to extend its own content, implying a form of "topic inheritance".24 This model explicitly uses citations between documents to model topics, making the connection between new and old topics more easily captured compared to citation-unaware models.24 The framework considers papers not only in the current time period but also those cited by papers in the current period, allowing for a more comprehensive view of topic dependencies across time.24

While the full technical details of the "inheritance topic model" are complex, its essence lies in adapting the Bayesian framework of LDA to incorporate citation links, quantifying the uncertainty associated with citation parameters (e.g., influential weights on citing papers).24 This leads to a more robust understanding of topic evolution by directly inferring citations within the model.24 The results of such citation-aware approaches clearly demonstrate that citations significantly enhance the ability to understand how topics evolve.1

Another example of combining these approaches is seen in studies detecting emerging scientific fields, such as the nanocarbon field.27 These methodologies often involve:

1. **Citation Network Construction and Clustering:** Converting citation data into unweighted networks, filtering irrelevant papers, and then dividing the network into clusters using topological clustering methods like modularity maximization. These clusters represent groups of papers with dense citation relations.27  
2. **Feature Extraction:** Extracting features from the bibliographic information and citation network (e.g., network macro features, cluster features, network centrality features, and citing paper features) to predict emerging papers.27  
3. **Topic Analysis of Clusters:** Applying topic models like LDA to estimate the topics within these identified clusters. This allows for the analysis of emerging research areas at the granularity of terms and research areas, rather than just individual papers.27

This combined approach provides both micro-level (individual paper) and semi-macro level (research field/cluster) perspectives, enabling a comprehensive discussion of both qualitative (thematic content) and quantitative (citation impact, emergence prediction) aspects of evolving fields.27

### **3.2 Change Point and Burst Detection in Bibliometric Time Series**

Within the continuous flow of scientific publications, identifying discrete periods of significant change is crucial for understanding research evolution. Change point detection and burst detection algorithms serve this purpose by pinpointing abrupt shifts in data properties.

#### **3.2.1 Role of Change Point Detection**

Change point detection (CPD) is the problem of identifying abrupt changes in a time series where the underlying statistical properties of the data shift.28 In the context of scientific literature, these "change points" can signify the beginning or end of a dominant research period, the adoption of a new methodology, or a shift in a field's primary focus.

Common statistical formulations for CPD involve analyzing the probability distributions of data before and after a candidate change point.28 If these distributions are significantly different, a change point is identified. Methods include:

* **Likelihood Ratio Methods:** These monitor the logarithm of the likelihood ratio between two consecutive intervals in time-series data. A change point is detected if this ratio exceeds a threshold, indicating a significant difference in data distribution.28  
* **Cumulative Sum (CUSUM):** A widely used algorithm that accumulates deviations relative to a specified target. A change point is indicated when the cumulative sum exceeds a predefined threshold.28  
* **Change Finder:** This method transforms CPD into a time series-based outlier detection problem. It fits an AutoRegressive (AR) model to the data and incrementally updates its parameters, identifying change points when the dissimilarity measure between two samples is high.28

More flexible non-parametric variations have emerged that estimate the ratio of probability densities directly, without needing explicit density estimation, simplifying the process.28 The objective of CPD is to estimate the time of significant and abrupt changes in the dynamics of a system, which, for publication data, translates to shifts in topic prevalence, citation patterns, or methodological adoption.29

#### **3.2.2 Citation Burst Detection**

Citation burst detection is a specialized form of anomaly or change point detection applied to citation data. A "burst" refers to a period of uncharacteristically high frequency or surge of a particular type of event, such as a sudden increase in citations to a specific publication or a surge in the appearance of certain keywords.10 These bursts often indicate emerging research fronts or "hot topics" that are rapidly gaining attention within the scientific community.10

**Kleinberg's burst detection algorithm** is a prominent method for identifying such phenomena.10 It models the stream of events (e.g., citations, keyword occurrences) as a sequence of discrete batches and identifies time periods where the event rate is uncharacteristically frequent compared to a baseline.32 The algorithm returns an optimal state sequence, typically represented by 0s (baseline state) and 1s (bursty state), indicating when the system enters or exits a burst period.32 Parameters such as the multiplicative distance between states (s) and the difficulty of moving up a state (gamma) can be adjusted to control sensitivity.10 Python implementations of Kleinberg's algorithm are available, allowing for practical application.31

CiteSpace, a popular bibliometric analysis software, supports burst detection on various event types, including citation counts of cited references, frequencies of keyword appearances, and publication counts by authors or institutions over time.10 This capability is crucial for identifying dynamic shifts in research focus and impact.

#### **3.2.3 Application in Research Fronts**

The application of change point and burst detection methods to bibliometric time series is fundamental for identifying and characterizing research fronts and paradigm shifts.

* **Identifying "Hot" Topics:** Citation bursts directly pinpoint publications or keywords that are experiencing a rapid surge in attention, indicating "hot" or emerging research areas.4 These are often characterized by tightly interconnected networks of recent publications with dense citation links.4  
* **Detecting Paradigm Shifts:** A "paradigm shift," as conceptualized by Thomas Kuhn, involves a fundamental change in the theories or methods of a scientific field.3 Bibliometric characteristics illustrating such shifts can be identified through methods like author co-citation analysis across different time slices, which can show a widespread realignment of researchers and the emergence of new, previously unthinkable concepts becoming standard knowledge.3 The growth of a paradigm can be visualized through the rise of citation rates and the movement of core clusters in co-citation networks.11  
* **Tracking Evolution:** By applying these detection algorithms to time series of topics, methods, or citation patterns, researchers can segment the timeline of a scientific field into distinct periods, each representing a particular phase of development. This allows for tracking the evolution of ideas, understanding the lineage of topics, and objectively evaluating contributions.24

These detection mechanisms provide the temporal boundaries for segments, within which the dominant topics, methods, and techniques can then be characterized.

## **4\. State-of-the-Art Solutions and Practical Tools**

The field of time series segmentation for publication data is continuously evolving, with recent advancements focusing on integrating deep learning and graph-based approaches to capture complex temporal and relational dynamics.

### **4.1 Recent Advancements (2022-2025)**

#### **4.1.1 Neural Dynamic Topic Models with Graph Integration**

The cutting edge in dynamic topic modeling often involves neural networks and the integration of graph structures, particularly citation networks, to capture more nuanced relationships and temporal evolution. Recent papers from 2022-2025 demonstrate this trend:

* **"Variational Graph Author Topic Modeling" (2022)**: This work integrates graph structures, likely representing author relationships or citation networks, within a neural topic modeling framework.33  
* **"Topic Modeling on Document Networks with Adjacent-Encoder" (2022)**: This paper explicitly deals with topic modeling on document networks, using graph information (document relationships) for topic discovery through a neural network architecture.33  
* **"Dynamic Topic Models for Temporal Document Networks" (ICML 2022\)**: This highly relevant paper directly combines dynamic topic models with temporal document networks, explicitly using graph information in a dynamic context for scientific literature analysis.33  
* **"Neural Dynamic Focused Topic Model" (AAAI 2023\)**: While not explicitly mentioning graphs, its focus on dynamic topic modeling in a neural context is pertinent.33  
* **"Graph Neural Topic Model with Commonsense Knowledge" (2023)**: This paper combines graph neural networks with topic modeling, which can be applied to scientific literature where relationships like citations form a graph.33  
* **"Hyperbolic Graph Topic Modeling Network with Continuously Updated Topic Tree" (KDD 2023\)**: This paper combines a graph topic modeling network with a continuously updated topic tree, strongly implying a dynamic aspect to topic modeling. The use of hyperbolic embeddings suggests a specific way of representing graph structure, highly relevant for analyzing evolving scientific literature.33  
* **"ANTM: An Aligned Neural Topic Model for Exploring Evolving Topics" (2023)**: This model is designed for exploring evolving topics, a core aspect of dynamic topic modeling.33  
* **"GINopic: Topic Modeling with Graph Isomorphism Network" (NAACL 2024\)**: This work uses Graph Isomorphism Networks for topic modeling, which could be applied to track changes in scientific networks over time.33

These advancements signify a move towards more sophisticated models that can capture the intricate interplay between textual content and network structure as scientific fields evolve.

#### **4.1.2 BERTrend Framework**

BERTrend is a novel framework designed for detecting and monitoring "weak signals" and emerging trends in large, evolving text corpora, including scientific publications.15 It leverages neural topic modeling, specifically **BERTopic**, in an online learning setting to identify and track topic evolution over time.15

While BERTrend is not explicitly marketed for "time series segmentation" as a standalone function, its core functionality of tracking topic evolution directly supports this task. Its metrics contribute significantly to identifying segments in scientific literature based on topic evolution:

* **Dynamic Topic Modeling Analysis:** BERTrend's focus on dynamic topic modeling allows for the observation of shifts in dominant topics, the emergence of new topics, and the decline of others within a corpus. These changes in the topic landscape naturally define segments within the literature.16  
* **TEMPTopic Metrics:** BERTrend provides dedicated metrics called TEMPTopic to understand topic dynamics 16:  
  * **Stability Evaluation:** Measures how consistent and coherent topics remain over time. A period of high topic stability could indicate a mature research area, while a significant drop might signal a transition point, marking a segment boundary.16  
  * **Volatility Assessment:** Analyzes how much topics change over different time periods. High volatility may indicate rapid development or a paradigm shift, suggesting a distinct segment.16  
  * **Temporal Topic Embedding Stability** and **Temporal Topic Representation Stability:** These perspectives evaluate stability based on semantic shifts in topic discussion (embeddings) and changes in defining words/phrases (representations), respectively, further delineating segments.16  
* **Popularity-based Signal Classification:** BERTrend classifies topics as noise, weak signals, or strong signals based on their popularity trends, using a metric that considers both document count and update frequency.15 The identification of "weak signals"—early indicators of larger trends—is particularly relevant for marking the beginning of new research segments, even before they become widely recognized.16

It is important to note that, as described in the provided information, BERTrend's focus is solely on analyzing the textual content of the corpora, and it **does not mention any integration of citation data** within its framework.16

#### **4.1.3 Chain-Free Dynamic Topic Models**

Traditional dynamic topic models often rely on Markov chains to link topics across time slices, which can lead to issues such as repetitive topics (similar semantics within a time slice) and unassociated topics (topics not truly belonging to their time slices).34 The "Modeling Dynamic Topics in Chain-Free Fashion by Evolution-Tracking Contrastive Learning and Unassociated Word Exclusion" (ACL 2024 Findings) paper proposes a novel **Chain-Free Dynamic Topic Model (CFDTM)** to address these limitations.34

CFDTM breaks the tradition of chaining topics via Markov chains and introduces two key methods:

* **Evolution-Tracking Contrastive Learning (ETC):** This method builds positive relations among dynamic topics to track their evolution, pulling topic embeddings of consecutive time slices closer in semantic space. Simultaneously, it builds negative relations between different topics within a single time slice to prevent them from being too similar, thus enhancing topic diversity and mitigating the repetitive topic issue.34  
* **Unassociated Word Exclusion (UWE):** This method refines topic semantics by explicitly excluding unassociated words. It identifies words that do not exist in the vocabulary set of a specific time slice and models their embeddings as negative pairs with topic embeddings, pushing them away. This alleviates the unassociated topic issue and improves the model's robustness.34

The overall CFDTM combines these methods within a Variational Autoencoder (VAE) framework, optimizing for topic coherence, evolution tracking, and distinct topic representation.35 Similar to BERTrend, the provided information indicates that this paper **does not integrate citation networks or other graph structures** for scientific literature analysis, focusing on textual content and its temporal evolution.35

### **4.2 Established Open-Source Tools with Code/Links**

Several open-source tools and libraries are available for performing time series segmentation and bibliometric analysis, providing practical solutions for researchers.

#### **4.2.1 General Time Series Segmentation**

For general time series segmentation, robust Python libraries are available:

* **sktime:** This library offers a comprehensive framework for time series analysis, including segmentation algorithms. It provides implementations for various transformers and models, such as ClaSP (Classification Score Profile) for time series segmentation.36 ClaSP hierarchically splits a time series into parts, determining split points by training a binary time series classifier to identify subsequences from different partitions.36 It can output core change points (fmt="sparse") or interval series representing the segmentation (fmt="dense").36 sktime also includes functionality for selecting optimal window sizes, such as using the dominant frequency of the Fourier Transform.36  
  * **Code/Documentation:** [https://www.sktime.net/en/latest/examples/annotation/segmentation\_with\_clasp.html](https://www.sktime.net/en/latest/examples/annotation/segmentation_with_clasp.html) 36  
* **ClaSPy:** This is the official Python package implementation of the ClaSP algorithm, specifically designed for time series segmentation.38 It aims to partition a time series into semantically meaningful segments and is described as accurate, domain-agnostic, and parameter-free, though domain-specific knowledge can be used to guide it.38 ClaSPy works for both univariate and multivariate time series and generates a score profile indicating the probability of a "change point".38  
  * **Code/Documentation:** [https://github.com/ermshaua/claspy](https://github.com/ermshaua/claspy) 38

#### **4.2.2 Bibliometric Analysis & Science Mapping**

For comprehensive bibliometric analysis and science mapping, several widely used open-source software programs are available:

* **Bibliometrix (R Package) and Biblioshiny (Web App):** Bibliometrix is an R package that allows users to perform extensive bibliometric analyses on large scientific databases (e.g., Web of Science, Scopus, PubMed).39 It provides functions for data retrieval, cleaning, bibliometric indicators, co-citation and co-word networks, and co-authorship analysis.39 Biblioshiny is its web-based graphical user interface, making bibliometric analyses accessible without coding skills.39 It supports performance analysis and science mapping, including visualizations of temporal trends.39  
  * **Code/Documentation:** [https://www.bibliometrix.org/](https://www.bibliometrix.org/) 40  
* **VOSviewer:** A free and open-source software program for creating visual maps of bibliometric data, such as co-citation and co-word networks.39 It is particularly useful for analyzing large datasets and offers advanced features like cluster detection and visualization of temporal trends.39  
* **CiteSpace:** Another free and open-source software program for analyzing bibliometric data, including citation networks and temporal trends.39 CiteSpace is particularly effective for identifying patterns and trends in the literature, such as the emergence of new research areas, and for tracking field development over time.39 It supports burst detection for citations, keywords, authors, and institutions.10

These tools are essential for mapping the intellectual structure of scientific fields and identifying research fronts.

#### **4.2.3 Python Libraries for Bibliometrics & Topic Evolution**

Python offers libraries that integrate bibliometric analysis with AI capabilities, including topic modeling and temporal analysis:

* **pyBibX:** This Python library is designed for comprehensive bibliometric and scientometric analyses on raw data files from databases like Scopus, Web of Science, and PubMed.42 It integrates artificial intelligence tools into its core functionality.42 While it does not explicitly offer "time series segmentation" as a distinct function, its capabilities enable the analysis of temporal patterns and topic evolution:  
  * **Evolution Plots:** Creates interactive plots based on abstracts, titles, sources, author keywords, or keywords plus, visualizing how these elements change over time.44 This can indirectly help identify periods of dominant topics or methods.  
  * **Productivity Plots:** Generates plots for authors, countries, institutions, and journals, showing documents published per year, providing a time-series view of productivity.44  
  * **Citations per Year:** Provides a bar plot showing yearly citation counts.44  
  * **RPYS (Reference Publication Year Spectroscopy):** Visualizes citation patterns over the years, revealing peaks in reference publication years that may indicate influential works or shifts in research trends.44 This is a direct functionality for identifying influential works over time using citation metrics.  
  * **Topic Modeling with BERTopic:** Offers capabilities to visualize topics over time, cluster documents by topic, and find representative documents/words for topics.44  
  * **Citation Analysis:** Creates interactive plots of citation networks between documents and their references.42 It can also identify "Sleeping Beauties" (papers uncited for a long time that later receive sudden attention).44  
  * **Code/Documentation:** [https://github.com/Valdecy/pybibx](https://github.com/Valdecy/pybibx) 44

While pyBibX offers extensive visualization and analysis of topic evolution and citation patterns over time, its documentation does not explicitly state functions to automatically identify and output distinct time segments or change points based on these metrics, beyond presenting the data visually.44 The "RPYS" feature comes closest by highlighting shifts, but it is described as a visualization tool.

## **5\. Conclusions and Future Directions**

### **5.1 Synthesis of Findings**

The comprehensive analysis of time series segmentation for scientific publication data underscores its critical role in deciphering the dynamic evolution of research fronts. The report demonstrates that effective segmentation necessitates a multi-faceted approach, integrating both the thematic content of publications and the intricate network of citation relationships. Foundational time series segmentation algorithms, such as top-down, bottom-up, and sliding window methods, provide the structural basis for partitioning temporal data. However, their full potential in scientific literature is realized when combined with advanced techniques.

Topic modeling, particularly Dynamic Topic Models (DTMs) like BERTopic and Chain-Free DTMs, has emerged as a state-of-the-art method for identifying and tracking the evolution of dominant themes over time. These models capture the emergence, growth, and decline of research topics, providing a semantic segmentation of the literature. The Iterative Topic Modeling with Time Series Feedback (ITMTF) framework further enhances this by allowing external time series data to iteratively refine topic discovery, demonstrating a powerful mechanism for uncovering topics correlated with external dynamics, which could conceptually include bibliometric indicators.

Complementing textual analysis, citation network analysis offers invaluable insights into the influence and intellectual structure of scientific fields. Co-citation clustering effectively identifies research fronts and emergent paradigms by mapping the conceptual relationships between cited works. More recent advancements in dynamic citation network models, such as Temporal Graph Neural Networks, capture the continuous evolution of influence by updating paper embeddings as new citations appear.

The integration of these methodologies, exemplified by citation-aware topic models like the "inheritance topic model," provides a more holistic understanding of scientific evolution by explicitly modeling the linkages between topics through citations. Change point detection and burst detection algorithms serve as crucial mechanisms for identifying significant, abrupt shifts in research focus or impact, effectively delineating the boundaries of these temporal segments.

In essence, the synergy between textual topic analysis and relational citation network analysis, combined with robust time series segmentation techniques, offers a powerful framework for mapping the temporal landscape of scientific knowledge. This allows for a deeper understanding of how research fields develop, how new ideas gain traction, and how established paradigms shift.

### **5.2 Challenges and Opportunities**

Despite the significant advancements, several challenges and opportunities remain in the field of time series segmentation for publication data:

* **Data Quality and Granularity:** The accuracy of segmentation heavily relies on the quality and granularity of bibliographic data. Inconsistencies in author names, affiliations, and keyword indexing can introduce noise. Future work could focus on more robust data preprocessing and disambiguation techniques to enhance the reliability of analyses.  
* **Interpretability of Complex Models:** While advanced neural and graph-based models offer superior performance in capturing complex dynamics, their "black-box" nature can sometimes hinder interpretability. Developing methods that provide clearer explanations for identified segments and topic shifts, perhaps through integrated visualization tools, would be highly beneficial.  
* **Integration Gaps:** Although frameworks like ITMTF and the "inheritance topic model" demonstrate the power of integrating textual and non-textual or network data, there remain opportunities for deeper, more seamless integration. For instance, directly incorporating citation dynamics into the core architecture of dynamic topic models, beyond mere post-processing or external feedback, could yield more coherent and causally informed segments.  
* **Real-time Detection and Prediction:** While current methods excel at retrospective analysis, the ability to detect emerging trends and paradigm shifts in near real-time, and to predict future directions, is an ongoing challenge. Leveraging online learning settings, as seen in BERTrend, represents a promising avenue. Further development of predictive models that can forecast the trajectory of research fronts based on early signals would be transformative for strategic planning.  
* **Scalability for Massive Datasets:** The exponential growth of scientific literature demands highly scalable algorithms. While some tools and models address this, continuous innovation in computational efficiency and distributed processing will be crucial for analyzing ever-larger corpora.  
* **Cross-Disciplinary Analysis:** Applying these methods to identify interdisciplinary research fronts and understand how knowledge flows between disparate fields presents a complex but valuable opportunity. This requires models capable of bridging diverse vocabularies and citation cultures.

### **5.3 Recommendations for Researchers and Policymakers**

Based on the capabilities of current state-of-the-art solutions, several recommendations can be made for researchers and policymakers:

* **For Researchers:**  
  * **Adopt Integrated Methodologies:** Move beyond siloed analyses of topics or citations. Employ integrated frameworks that combine dynamic topic modeling with citation network analysis to gain a more comprehensive understanding of research evolution.  
  * **Leverage Open-Source Tools:** Utilize readily available open-source tools like sktime, ClaSPy, Bibliometrix/Biblioshiny, VOSviewer, CiteSpace, and pyBibX. These tools provide robust functionalities for data processing, analysis, and visualization, enabling efficient exploration of publication data.  
  * **Focus on Change Point and Burst Detection:** Actively incorporate change point and burst detection algorithms to pinpoint significant shifts in research. This allows for the precise identification of emerging trends, declining areas, and methodological transitions, which are critical for timely adaptation of research agendas.  
  * **Explore Recent Advancements:** Investigate and experiment with neural dynamic topic models and temporal graph neural networks. These cutting-edge approaches offer enhanced capabilities for capturing complex, time-aware patterns in textual and network data.  
* **For Funding Agencies and Policymakers:**  
  * **Inform Strategic Funding Decisions:** Utilize time series segmentation and research front detection to identify burgeoning research areas that warrant increased investment and to understand the impact of past funding initiatives. This data-driven approach can optimize resource allocation.  
  * **Monitor Scientific Progress:** Implement systematic monitoring of research fronts to track the health and vitality of various scientific disciplines. This can help in identifying areas of stagnation or rapid growth, informing policy adjustments.  
  * **Anticipate Technological and Societal Impacts:** By identifying weak signals and emerging topics early, policymakers can better anticipate future technological breakthroughs and societal challenges, allowing for proactive policy development and preparedness.  
  * **Foster Interdisciplinary Research:** Employ tools that can map connections across fields to identify and support emerging interdisciplinary research fronts, which are often sources of significant innovation.

By embracing these advanced analytical approaches, stakeholders can gain unprecedented clarity into the dynamics of scientific knowledge, fostering more informed decisions and accelerating the pace of discovery and innovation.

#### **Works cited**

1. Detecting topic evolution in scientific literature: How can citations help? \- ResearchGate, accessed May 31, 2025, [https://www.researchgate.net/publication/221615057\_Detecting\_topic\_evolution\_in\_scientific\_literature\_How\_can\_citations\_help](https://www.researchgate.net/publication/221615057_Detecting_topic_evolution_in_scientific_literature_How_can_citations_help)  
2. Analyzing Paper Citation Trend of Popular Research Fields \- Journal of Computing & Biomedical Informatics, accessed May 30, 2025, [https://jcbi.org/index.php/Main/article/download/401/313](https://jcbi.org/index.php/Main/article/download/401/313)  
3. Debunking revolutionary paradigm shifts: evidence of cumulative scientific progress across science \- Digital CSIC, accessed May 30, 2025, [https://digital.csic.es/bitstream/10261/384078/1/krauss-2024-debunking-revolutionary-paradigm-shifts-evidence-of-cumulative-scientific-progress-across-science.pdf](https://digital.csic.es/bitstream/10261/384078/1/krauss-2024-debunking-revolutionary-paradigm-shifts-evidence-of-cumulative-scientific-progress-across-science.pdf)  
4. Impactful research fronts in digital educational ecosystem: advancing Clarivate's approach with a new impact factor metric \- Frontiers, accessed May 31, 2025, [https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1557812/full](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1557812/full)  
5. Exploring Bibliometric Methods: Citation Analysis in Research \- Alfasoft, accessed May 30, 2025, [https://alfasoft.com/blog/products/scientific-writing-and-publishing/exploring-bibliometric-methods-citation-analysis-in-research/](https://alfasoft.com/blog/products/scientific-writing-and-publishing/exploring-bibliometric-methods-citation-analysis-in-research/)  
6. What Is Segmentation in Time- Series or Statistical Analysis? \- QuestDB, accessed May 30, 2025, [https://questdb.com/glossary/segmentation/](https://questdb.com/glossary/segmentation/)  
7. lzz19980125/awesome-time-series-segmentation-papers \- GitHub, accessed May 30, 2025, [https://github.com/lzz19980125/awesome-time-series-segmentation-papers](https://github.com/lzz19980125/awesome-time-series-segmentation-papers)  
8. Topic modeling in NLP: Approaches, implementation and use cases \- LeewayHertz, accessed May 30, 2025, [https://www.leewayhertz.com/topic-modeling-in-nlp/](https://www.leewayhertz.com/topic-modeling-in-nlp/)  
9. Dynamic Topic Models \- ResearchGate, accessed May 30, 2025, [https://www.researchgate.net/publication/221345245\_Dynamic\_Topic\_Models](https://www.researchgate.net/publication/221345245_Dynamic_Topic_Models)  
10. Burst Detection \- CiteSpace, accessed May 30, 2025, [https://citespace.podia.com/glossary-burstness](https://citespace.podia.com/glossary-burstness)  
11. The bibliometric approach to identify paradigms in knowledge domains \- ResearchGate, accessed May 30, 2025, [https://www.researchgate.net/publication/286859018\_The\_bibliometric\_approach\_to\_identify\_paradigms\_in\_knowledge\_domains](https://www.researchgate.net/publication/286859018_The_bibliometric_approach_to_identify_paradigms_in_knowledge_domains)  
12. Latent Dirichlet Allocation \- Journal of Machine Learning Research, accessed May 30, 2025, [https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  
13. Mining Causal Topics in Text Data: Iterative Topic Modeling with Time Series Feedback \- biz.uiowa.edu, accessed May 31, 2025, [https://www.biz.uiowa.edu/faculty/trietz/papers/ITMTF.pdf](https://www.biz.uiowa.edu/faculty/trietz/papers/ITMTF.pdf)  
14. Dynamic Topic Models \- David Mimno, accessed May 31, 2025, [https://mimno.infosci.cornell.edu/info6150/readings/dynamic\_topic\_models.pdf](https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf)  
15. BERTrend: Neural Topic Modeling for Emerging Trends Detection \- arXiv, accessed May 31, 2025, [https://arxiv.org/html/2411.05930v1](https://arxiv.org/html/2411.05930v1)  
16. rte-france/BERTrend \- GitHub, accessed May 30, 2025, [https://github.com/rte-france/BERTrend](https://github.com/rte-france/BERTrend)  
17. Dynamic insights into research trends and trajectories in early reading: an analytical exploration via dynamic topic modeling \- ResearchGate, accessed May 30, 2025, [https://www.researchgate.net/publication/378393590\_Dynamic\_insights\_into\_research\_trends\_and\_trajectories\_in\_early\_reading\_an\_analytical\_exploration\_via\_dynamic\_topic\_modeling](https://www.researchgate.net/publication/378393590_Dynamic_insights_into_research_trends_and_trajectories_in_early_reading_an_analytical_exploration_via_dynamic_topic_modeling)  
18. (PDF) Time Series Impact Driven by Topic Modeling \- ResearchGate, accessed May 31, 2025, [https://www.researchgate.net/publication/363079266\_Time\_Series\_Impact\_Driven\_by\_Topic\_Modeling](https://www.researchgate.net/publication/363079266_Time_Series_Impact_Driven_by_Topic_Modeling)  
19. Bibliometric Characteristics of a Paradigm Shift: the 2012 Nobel Prize in Medicine, accessed May 30, 2025, [https://www.researchgate.net/publication/273693819\_Bibliometric\_Characteristics\_of\_a\_Paradigm\_Shift\_the\_2012\_Nobel\_Prize\_in\_Medicine](https://www.researchgate.net/publication/273693819_Bibliometric_Characteristics_of_a_Paradigm_Shift_the_2012_Nobel_Prize_in_Medicine)  
20. \[2408.15371\] Temporal Graph Neural Network-Powered Paper Recommendation on Dynamic Citation Networks \- arXiv, accessed May 30, 2025, [https://arxiv.org/abs/2408.15371](https://arxiv.org/abs/2408.15371)  
21. Towards Ideal Temporal Graph Neural Networks \- arXiv, accessed May 31, 2025, [https://arxiv.org/pdf/2412.20256](https://arxiv.org/pdf/2412.20256)  
22. icml.cc, accessed May 30, 2025, [https://icml.cc/Conferences/2011/papers/464\_icmlpaper.pdf](https://icml.cc/Conferences/2011/papers/464_icmlpaper.pdf)  
23. The Evolution of Scientific Literature as Metastable Knowledge States \- National Center for Science and Engineering Statistics (NCSES), accessed May 31, 2025, [https://ncses.nsf.gov/299/assets/0/files/the-evolution-of-scientific-literature-as-metastable-knowledge-states.pdf](https://ncses.nsf.gov/299/assets/0/files/the-evolution-of-scientific-literature-as-metastable-knowledge-states.pdf)  
24. Detecting topic evolution in scientific literature: how can citations help? \- C. Lee Giles, accessed May 31, 2025, [https://clgiles.ist.psu.edu/pubs/CIKM2009-topic-evolution-citations.pdf](https://clgiles.ist.psu.edu/pubs/CIKM2009-topic-evolution-citations.pdf)  
25. Identifying Emergent Research Trends by Key Authors and Phrases \- NUS Computing \- National University of Singapore, accessed May 31, 2025, [https://www.comp.nus.edu.sg/\~kanmy/papers/identifying-emergent-research.pdf](https://www.comp.nus.edu.sg/~kanmy/papers/identifying-emergent-research.pdf)  
26. Identifying Emergent Research Trends by Key Authors and Phrases \- ACL Anthology, accessed May 31, 2025, [https://aclanthology.org/C18-1022.pdf](https://aclanthology.org/C18-1022.pdf)  
27. Emerging Scientific Field Detection Using Citation Networks and ..., accessed May 31, 2025, [https://www.mdpi.com/2571-5577/3/3/40](https://www.mdpi.com/2571-5577/3/3/40)  
28. A Survey of Methods for Time Series Change Point Detection \- PMC, accessed May 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5464762/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5464762/)  
29. Change Point Detection in Time Series via Multivariate Singular Spectrum Analysis Arwa Alanqary \- DSpace@MIT, accessed May 30, 2025, [https://dspace.mit.edu/bitstream/handle/1721.1/139610/alanqary-alanqary-sm-ccse-2021-thesis.pdf?sequence=1\&isAllowed=y](https://dspace.mit.edu/bitstream/handle/1721.1/139610/alanqary-alanqary-sm-ccse-2021-thesis.pdf?sequence=1&isAllowed=y)  
30. Parameters for burst detection \- PMC \- PubMed Central, accessed May 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3915237/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3915237/)  
31. Burst detection by kleinberg's algorithm \- GitHub Gist, accessed May 30, 2025, [https://gist.github.com/fb5fa79fde91ca0bbe98](https://gist.github.com/fb5fa79fde91ca0bbe98)  
32. burst\_detection \- PyPI, accessed May 30, 2025, [https://pypi.org/project/burst\_detection/](https://pypi.org/project/burst_detection/)  
33. Papers of Neural Topic Models (NTMs) \- GitHub, accessed May 31, 2025, [https://github.com/BobXWu/Paper-Neural-Topic-Models](https://github.com/BobXWu/Paper-Neural-Topic-Models)  
34. Modeling Dynamic Topics in Chain-Free Fashion by Evolution-Tracking Contrastive Learning and Unassociated Word Exclusion \- arXiv, accessed May 31, 2025, [https://arxiv.org/html/2405.17957v1](https://arxiv.org/html/2405.17957v1)  
35. Modeling Dynamic Topics in Chain-Free Fashion by Evolution ..., accessed May 31, 2025, [https://arxiv.org/pdf/2405.17957](https://arxiv.org/pdf/2405.17957)  
36. Time Series Segmentation with sktime and ClaSP, accessed May 30, 2025, [https://www.sktime.net/en/v0.23.0/examples/annotation/segmentation\_with\_clasp.html](https://www.sktime.net/en/v0.23.0/examples/annotation/segmentation_with_clasp.html)  
37. Time Series Segmentation with sktime and ClaSP, accessed May 30, 2025, [https://www.sktime.net/en/latest/examples/annotation/segmentation\_with\_clasp.html](https://www.sktime.net/en/latest/examples/annotation/segmentation_with_clasp.html)  
38. ClaSPy: A Python package for time series segmentation. \- GitHub, accessed May 31, 2025, [https://github.com/ermshaua/claspy](https://github.com/ermshaua/claspy)  
39. Main software to analyze bibliometric data \- Bibliometrix, accessed May 30, 2025, [https://www.bibliometrix.org/home/index.php/blog/135-main-software-to-analyze-bibliometric-data](https://www.bibliometrix.org/home/index.php/blog/135-main-software-to-analyze-bibliometric-data)  
40. Bibliometrix \- Home, accessed May 30, 2025, [https://www.bibliometrix.org/](https://www.bibliometrix.org/)  
41. Full article: Temporal analysis of motivation research trends in armed forces, accessed May 31, 2025, [https://www.tandfonline.com/doi/full/10.1080/14702436.2025.2498744?af=R](https://www.tandfonline.com/doi/full/10.1080/14702436.2025.2498744?af=R)  
42. PyBibX – a Python library for bibliometric and scientometric analysis powered with artificial intelligence tools | Emerald Insight, accessed May 30, 2025, [https://www.emerald.com/insight/content/doi/10.1108/dta-08-2023-0461/full/html](https://www.emerald.com/insight/content/doi/10.1108/dta-08-2023-0461/full/html)  
43. (PDF) PyBibX \-a Python library for bibliometric and scientometric analysis powered with artificial intelligence tools \- ResearchGate, accessed May 30, 2025, [https://www.researchgate.net/publication/388420680\_PyBibX\_-a\_Python\_library\_for\_bibliometric\_and\_scientometric\_analysis\_powered\_with\_artificial\_intelligence\_tools](https://www.researchgate.net/publication/388420680_PyBibX_-a_Python_library_for_bibliometric_and_scientometric_analysis_powered_with_artificial_intelligence_tools)  
44. Valdecy/pybibx: A Bibliometric and Scientometric Python ... \- GitHub, accessed May 31, 2025, [https://github.com/Valdecy/pybibx](https://github.com/Valdecy/pybibx)