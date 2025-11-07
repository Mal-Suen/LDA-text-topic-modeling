# LDA document topic distribution processing based on Python README | 基于Python的LDA文档主题分布处理

<div align="center">

**Language / 语言选择:**
[English](#english) | [中文](#中文)

---

</div>
<a id="english"></a>

## I. What is LDA?
**LDA (Latent Dirichlet Allocation)** is a topic model used for text analysis, which helps us automatically discover hidden **topics** from a large amount of text. I will explain this model in the simplest way possible.
Imagine you have a pile of articles (such as news, microblogs, papers, etc.). LDA is like a "topic detector" that can automatically identify the main topics being discussed in these articles. For example, if you analyze 100 sports news articles, LDA may find that these articles mainly revolve around topics such as "football", "basketball", and "tennis".

### The application value of LDA
In today's era of rapid development of large models, LDA (Latent Dirichlet Allocation) still holds its unique value and significance. As an unsupervised topic model, it is suitable for scenarios requiring topic discovery and interpretability, automatically discovering hidden topic structures from text data. Furthermore, the LDA model is simple, highly interpretable, with controllable parameter quantity, and is less prone to overfitting.
LDA is still widely used in multiple fields in practice:
- **Text classification**: Infer the distribution of document topics by training an LDA model for classification
- **Topic Modeling**: Discovering the hidden topic structure in text data and revealing underlying patterns
- **Public opinion analysis**: Identify popular topics and public concerns from social media
- **Academic research**: analyzing research trends and domain developments in a large number of papers
- **Policy analysis**: Conducting topic extraction and quantitative analysis on policy texts
  
### How does LDA work?
Its core logic can be simplified as:
1. **Topic refers to word distribution**
   Each topic is defined as a probabilistic combination of a set of related words (for example, the "education" topic includes words such as "student" and "course", with their respective probabilities indicated).
2. **Document generation process**
   Assuming that each document is generated through the following steps:
   - Randomly select multiple topics (e.g., 30% topic A, 70% topic B);
   - Extract words from the word list of each selected topic based on probability, and finally compose a document.
3. **Actual output**
   **Topic Analysis**: Display high-frequency words under each topic (e.g., Topic 1: "education", "learning", "teacher");
   **Document analysis**: Calculate the proportion of each topic in each document (e.g., Document 1: 40% Topic A, 60% Topic B).
Without the need for manual data annotation, LDA can automatically discover the underlying structure in text, but it requires the number of topics (K value) to be preset.

## II. Layered Logic of Code Implementation
The code employs **object-oriented design** to break down the entire process of LDA modeling into modular steps, facilitating understanding and reuse.
### 1. Initialization: Setting model parameters

```python
def __init__(self, num_topics=2, passes=10, random_state=42):
    self.num_topics = num_topics  # Number of topics (e.g., 2 topics)
    self.passes = passes          # training iterations (e.g. 10 times)
    self.random_state = random_state  # Random seed (ensure reproducibility of results)
    self.dictionary = None        # dictionary (mapping from word to ID)
    self.corpus = None            # corpus (document frequency statistics)
    self.lda_model = None         # LDA model object
    self.documents = []          # Document list (raw text)
```

### 2. Data loading: Reading text from a file

```python
def load_corpus(self, file_path):
    """Load corpus data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            self.documents = [line.strip().split() for line in f if line.strip()]
        logging.info(f"Successfully loaded {len(self.documents)} documents")
        return self.documents
    except FileNotFoundError:
        logging.error(f"File does not exist: {file_path}")
        return []
```

### 3.  Feature engineering: constructing dictionaries and corpora

```python
def build_dictionary_and_corpus(self):
    """Build dictionary and corpus"""
    if not self.documents:
        logging.error("No document data available")
        return None, None

    # Create a dictionary, dictionary: vocabulary → ID mapping
    self.dictionary = corpora.Dictionary(self.documents)
    logging.info(f"Dictionary size: {len(self.dictionary)}")

    # Build a corpus, corpus: document → word frequency statistics
    self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]
    return self.dictionary, self.corpus
```

- **Dictionary (`Dictionary`)**: Assigns unique IDs to all words in the document, such as `{"Machine learning": 1, "Topic modeling": 2, "LDA": 3}`;
- **Corpus**: Transform each document into a sparse representation of "word ID-word frequency", such as `[(1, 2), (2, 1), (3, 3)]` (indicating that "machine learning" appears 2 times, "topic modeling" appears 1 time, and "LDA" appears 3 times in the document).

### 4.  Model training: Constructing LDA model

```python
def train_model(self):
    """Train LDA model"""
    if self.corpus is None:
        self.build_dictionary_and_corpus()

    self.lda_model = models.LdaModel(
        corpus=self.corpus,
        id2word=self.dictionary,
        num_topics=self.num_topics,
        passes=self.passes,
        random_state=self.random_state
    )
    return self.lda_model
```

- **LDA model**: It learns the probability distributions of "document-topic" and "topic-word" through "word frequency statistics", ultimately generating a topic model;
- **Parameter Function**:
  - `num_topics`: Controls the number of topics (e.g., 2 topics);
  - `passes`: The number of training iterations (e.g., 10 times, affecting the degree of model convergence);
  - `random_state`: Random seed (ensures consistent results across different runs).

### 5.  Result analysis: Extracting topic keywords and document topic distribution

#### (1) Extracting topic keywords

```python
def get_topic_keywords(self, top_n=5):
    """Obtain subject keywords"""
    if self.lda_model is None:
        self.train_model()

    topic_keywords = {}
    for topic_id in range(self.num_topics):
        topic_terms = self.lda_model.show_topic(topic_id, topn=top_n)
        topic_keywords[topic_id] = topic_terms

    return topic_keywords
```

- **Function**: Obtain the `top_n` words (such as the top 5) with the highest probability under each topic to understand the content of the topic;
- **Example**: If the keywords for Topic 0 are `["machine learning", "algorithm", "data"]`, then the topic can be interpreted as "machine learning-related topics".

#### (2) Analyze document topic distribution

```python
def analyze_document_topics(self, top_n=3):
    """Analyze document topic distribution"""
    if self.lda_model is None:
        self.train_model()

    doc_topics = {}
    for doc_id in range(len(self.documents)):
        doc_bow = self.dictionary.doc2bow(self.documents[doc_id])
        topic_dist = self.lda_model.get_document_topics(
            doc_bow,
            minimum_probability=0.0
        )
        top_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:top_n]
        doc_topics[doc_id] = top_topics

    return doc_topics
```

- **Function**: Analyze the probability of each document belonging to each topic and output the association between "document-topic";

### 6.  Model evaluation: Coherence score

```python
def evaluate_model(self):
    """Evaluate model performance"""
    if self.lda_model is None:
        self.train_model()

    coherence_model = CoherenceModel(
        model=self.lda_model,
        texts=self.documents,
        dictionary=self.dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    return coherence_score
```

- **Consistency Score**: An indicator for measuring topic quality (the higher the score, the more "coherent" the topic);
- **Principle**: Evaluate the semantic relevance of words within the topic. A higher score indicates a more reasonable topic.

### 7. Generate visualization using pyLDAvis

```python
# Generate visualization using pyLDAvis
vis_data = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)  # Generate visualization data

# Save visualization as HTML file (optional)
pyLDAvis.save_html(vis_data, "lda_visualization.html")  # Save to current directory

# Direct visualization display (optional, requires an environment supporting HTML rendering)
pyLDAvis.display(vis_data) # Display directly in HTML-supported environments such as Jupyter Notebook
```

## III. Result Demonstration

**Test data**

```
New Year's Eve Gathering
New Year's program schedule, Spring Festival Gala, prosperity
The broader market is declining, with the stock market seeing a decline in retail investors
Falling stock market, making money
Golden Monkey, Spring Festival, prosperity, New Year
New car, New Year, New Year's goods, New Spring
stock market rebound decline
Stock market, retail investors, making money
New Year, watch, Spring Festival Gala
The market declines, with retail investors
```

**Output result**

```
--- Key Topics ---
Topic 0: Spring Festival, New Year, Gala, New Year's goods, Chinese New Year
Topic 1: Decline, stock market, retail investors, broad market, making money

--- Document Topic Distribution ---
Document 0: Topic 0 (0.915) Topic 1 (0.085)
Document 1: Topic 0 (0.914) Topic 1 (0.086)
Document 2: Topic 1 (0.899) Topic 0 (0.101)
Document 3: Topic 1 (0.874) Topic 0 (0.126)
Document 4: Topic 0 (0.897) Topic 1 (0.103)
Document 5: Topic 0 (0.897) Topic 1 (0.103)
Document 6: Topic 1 (0.873) Topic 0 (0.127)
Document 7: Topic 1 (0.874) Topic 0 (0.126)
Document 8: Topic 0 (0.914) Topic 1 (0.086)
Document 9: Topic 1 (0.874) Topic 0 (0.126)

--- Model Evaluation ---
Topic consistency score: 0.5332
```

<a id="中文"></a>

## 一、什么是LDA？

**LDA（Latent Dirichlet Allocation，隐含狄利克雷分布）**是一种用于文本分析的主题模型，它可以帮助我们从大量文本中自动发现隐藏的**主题**。下面我用最通俗的方式解释这个模型。

想象你有一堆文章（比如新闻、微博、论文等），LDA就像一个"主题探测器"，能自动找出这些文章都在讨论哪些主要话题。比如分析100篇体育新闻，LDA可能会发现这些文章主要围绕"足球"、"篮球"、"网球"等几个主题展开。

### LDA的应用价值

在大模型飞速发展的今天，LDA（潜在狄利克雷分布）仍然具有其独特的价值和意义，作为一种‌**无监督的主题模型**‌，它适合需要‌**主题发现**‌和‌**可解释性**‌的场景，从文本数据中自动发现隐藏的主题结构‌。而且LDA模型简单、可解释性强，参数量可控，不容易过拟合‌。

LDA在实际中仍被广泛应用于多个领域：

- ‌**文本分类**‌：通过训练LDA模型推断文档主题分布，进行分类‌
- ‌**主题建模**‌：发现文本数据中隐藏的主题结构，揭示潜在模式‌
- ‌**舆情分析**‌：从社交媒体中发现热门话题和公众关注点‌
- ‌**学术研究**‌：分析大量论文的研究趋势和领域发展‌
- ‌**政策分析**‌：对政策文本进行主题提取和量化分析

### LDA如何工作？

其核心逻辑可简化为：

1. ‌**主题即词分布**‌
   每个主题被定义为一组相关词的概率组合（如“教育”主题包含“学生”“课程”等词，并标注各自出现概率）。
2. ‌**文档生成过程**‌
   假设每篇文档通过以下步骤生成：
   - 随机选择多个主题（如30%主题A、70%主题B）；
   - 从每个选中主题的词表中按概率抽词，最终组成文档。
3. ‌**实际输出**‌
   - ‌**主题分析**‌：展示每个主题下的高频词（如主题1：“教育”“学习”“教师”）；
   - ‌**文档分析**‌：统计每篇文档中各主题的占比（如文档1：40%主题A、60%主题B）。

无需人工标注数据，LDA能自动发现文本中的潜在结构，但需预先设定主题数量（K值）。

## 二、代码实现的分层逻辑

代码通过‌**面向对象设计**‌，将LDA建模的全流程拆解为模块化步骤，方便理解与复用。

### 1.初始化：设置模型参数

```python
def __init__(self, num_topics=2, passes=10, random_state=42):
    self.num_topics = num_topics  # 主题数量（如2个主题）
    self.passes = passes          # 训练迭代次数（如10次）
    self.random_state = random_state  # 随机种子（确保结果可复现）
    self.dictionary = None        # 词典（词汇到ID的映射）
    self.corpus = None            # 语料库（文档的词频统计）
    self.lda_model = None         # LDA模型对象
    self.documents = []          # 文档列表（原始文本）
```

### 2.数据加载：从文件读取文本

```python
def load_corpus(self, file_path):
    """加载语料数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            self.documents = [line.strip().split() for line in f if line.strip()]
        logging.info(f"成功加载 {len(self.documents)} 篇文档")
        return self.documents
    except FileNotFoundError:
        logging.error(f"文件不存在: {file_path}")
        return []
```

### 3. 特征工程：构建词典与语料库

```python
def build_dictionary_and_corpus(self):
    """构建词典和语料库"""
    if not self.documents:
        logging.error("没有可用的文档数据")
        return None, None

    # 创建词典，词典：词汇→ID映射
    self.dictionary = corpora.Dictionary(self.documents)
    logging.info(f"词典大小: {len(self.dictionary)}")

    # 构建语料库，语料库：文档→词频统计
    self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]
    return self.dictionary, self.corpus
```

- ‌**词典（`Dictionary`）**‌：为所有文档中的词汇分配唯一ID，如`{"机器学习": 1, "主题建模": 2, "LDA": 3}`；
- ‌**语料库（`Corpus`）**‌：将每篇文档转化为“词ID-词频”的稀疏表示，如`[(1, 2), (2, 1), (3, 3)]`（表示文档中“机器学习”出现2次，“主题建模”出现1次，“LDA”出现3次）。

### 4. 模型训练：构建LDA模型

```python
def train_model(self):
    """训练LDA模型"""
    if self.corpus is None:
        self.build_dictionary_and_corpus()

    self.lda_model = models.LdaModel(
        corpus=self.corpus,
        id2word=self.dictionary,
        num_topics=self.num_topics,
        passes=self.passes,
        random_state=self.random_state
    )
    return self.lda_model
```

- ‌**LDA模型**‌：通过“词频统计”学习“文档-主题”“主题-词语”的概率分布，最终生成主题模型；
- **参数作用**：
  - `num_topics`：控制主题数量（如2个主题）；
  - `passes`：训练迭代次数（如10次，影响模型收敛程度）；
  - `random_state`：随机种子（确保不同运行结果一致）。

### 5. 结果分析：提取主题关键词与文档主题分布

#### （1）提取主题关键词

```python
def get_topic_keywords(self, top_n=5):
    """获取主题关键词"""
    if self.lda_model is None:
        self.train_model()

    topic_keywords = {}
    for topic_id in range(self.num_topics):
        topic_terms = self.lda_model.show_topic(topic_id, topn=top_n)
        topic_keywords[topic_id] = topic_terms

    return topic_keywords
```

- ‌**功能**‌：获取每个主题下概率最高的`top_n`个词语（如前5个），用于理解主题内容；
- ‌**示例**‌：若主题0的关键词为`["机器学习", "算法", "数据"]`，则该主题可解释为“机器学习相关主题”。

#### （2）分析文档主题分布

```python
def analyze_document_topics(self, top_n=3):
    """分析文档主题分布"""
    if self.lda_model is None:
        self.train_model()

    doc_topics = {}
    for doc_id in range(len(self.documents)):
        doc_bow = self.dictionary.doc2bow(self.documents[doc_id])
        topic_dist = self.lda_model.get_document_topics(
            doc_bow,
            minimum_probability=0.0
        )
        top_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:top_n]
        doc_topics[doc_id] = top_topics

    return doc_topics
```

- ‌**功能**‌：分析每篇文档属于各主题的概率，输出“文档-主题”的关联关系；

### 6. 模型评估：一致性得分（Coherence）

```python
def evaluate_model(self):
    """评估模型性能"""
    if self.lda_model is None:
        self.train_model()

    coherence_model = CoherenceModel(
        model=self.lda_model,
        texts=self.documents,
        dictionary=self.dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    return coherence_score
```

- ‌**一致性得分**‌：衡量主题质量的指标（值越高，主题越“连贯”）；
- ‌**原理**‌：评估主题内词语的语义相关性，分数越高说明主题更合理。

### 7.使用pyLDAvis生成可视化

```python
# 使用pyLDAvis生成可视化
vis_data = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)  # 生成可视化数据

# 保存可视化为HTML文件（可选）
pyLDAvis.save_html(vis_data, "lda_visualization.html")  # 保存到当前目录

# 直接展示可视化（可选，需支持HTML渲染的环境）
pyLDAvis.display(vis_data)  # 在Jupyter Notebook等支持HTML的环境中直接展示
```

## 三、结果演示

**测试数据**

```
新春 备 年货 新年 联欢晚会
新春 节目单 春节 联欢晚会 红火
大盘 下跌 股市 散户  
下跌 股市 赚钱  
金猴 新春 红火 新年  
新车 新年 年货 新春  
股市 反弹 下跌  
股市 散户 赚钱  
新年 看 春节 联欢晚会
大盘 下跌 散户
```

**输出结果**

```
--- 主题关键词 ---
主题0: 新春, 新年, 联欢晚会, 年货, 春节
主题1: 下跌, 股市, 散户, 大盘, 赚钱

--- 文档主题分布 ---
文档0: 主题0(0.915) 主题1(0.085) 
文档1: 主题0(0.914) 主题1(0.086) 
文档2: 主题1(0.899) 主题0(0.101) 
文档3: 主题1(0.874) 主题0(0.126) 
文档4: 主题0(0.897) 主题1(0.103) 
文档5: 主题0(0.897) 主题1(0.103) 
文档6: 主题1(0.873) 主题0(0.127) 
文档7: 主题1(0.874) 主题0(0.126) 
文档8: 主题0(0.914) 主题1(0.086) 
文档9: 主题1(0.874) 主题0(0.126) 

--- 模型评估 ---
主题一致性得分: 0.5332
```
