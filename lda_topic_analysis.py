# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA主题建模完整实现
"""

import logging
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis  # 用于适配gensim的LDA模型

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class LDATopicModel:
    """
    LDA主题建模类
    功能完整的主题分析实现
    """

    def __init__(self, num_topics=2, passes=10, random_state=42):
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.documents = []

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

    def build_dictionary_and_corpus(self):
        """构建词典和语料库"""
        if not self.documents:
            logging.error("没有可用的文档数据")
            return None, None

        # 创建词典
        self.dictionary = corpora.Dictionary(self.documents)
        logging.info(f"词典大小: {len(self.dictionary)}")

        # 构建语料库
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]
        return self.dictionary, self.corpus

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

    def get_topic_keywords(self, top_n=5):
        """获取主题关键词"""
        if self.lda_model is None:
            self.train_model()

        topic_keywords = {}
        for topic_id in range(self.num_topics):
            topic_terms = self.lda_model.show_topic(topic_id, topn=top_n)
            topic_keywords[topic_id] = topic_terms

        return topic_keywords

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

    def run_analysis(self, file_path):
        """执行完整分析流程"""
        logging.info("=== LDA主题建模分析开始 ===")

        # 1. 加载数据、特征工程、模型训练
        self.load_corpus(file_path)
        self.build_dictionary_and_corpus()
        self.train_model()

        # 2. 主题关键词、文档主题分布、模型评估
        topic_keywords = self.get_topic_keywords()
        doc_topics = self.analyze_document_topics()
        coherence_score = self.evaluate_model()

        # 3. 使用pyLDAvis生成可视化
        vis_data = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)  # 生成可视化数据

        # 4. 保存可视化为HTML文件（可选）
        pyLDAvis.save_html(vis_data, "lda_visualization.html")  # 保存到当前目录

        # 5. 直接展示可视化（可选，需支持HTML渲染的环境）
        pyLDAvis.display(vis_data)  # 在Jupyter Notebook等支持HTML的环境中直接展示

        # 6. 输出分析结果
        print("\n--- 主题关键词 ---")
        for topic_id, keywords in topic_keywords.items():
            words = [word for word, _ in keywords]
            print(f"主题{topic_id}: {', '.join(words)}")

        print("\n--- 文档主题分布 ---")
        for doc_id in range(min(5, len(self.documents))):
            topics = doc_topics[doc_id]
            print(f"文档{doc_id}: ", end="")
            for topic_id, prob in topics:
                print(f"主题{topic_id}({prob:.3f}) ", end="")
            print()

        print(f"\n--- 模型评估 ---")
        print(f"主题一致性得分: {coherence_score:.4f}")

        print("\n=== 分析完成 ===")

        return {
            'topic_keywords': topic_keywords,
            'doc_topics': doc_topics,
            'coherence_score': coherence_score
        }


def main():
    """主函数"""
    lda_model = LDATopicModel(num_topics=2, passes=15)
    results = lda_model.run_analysis('test.txt')
    if results:
        print("\n所有分析结果已成功生成！")

if __name__ == "__main__":
    main()