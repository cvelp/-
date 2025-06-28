import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
import gensim
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd


# 2. 数据预处理
# 定义正则表达式用于去除非字母字符
non_alpha_pattern = re.compile('[^a-zA-Z]')
# 加载停用词
stop_words = set(stopwords.words('english'))
# 初始化词形还原器
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # 去除URL、非字母字符并转换为小写
    text = re.sub(r'http\S+', '', text)  # 移除URL
    text = non_alpha_pattern.sub(' ', text).lower()
    # 按空格切分单词
    words = text.split()
    # 去除停用词和长度<=2的单词
    words = [word for word in words if word not in stop_words and len(word) > 2]
    # 词形还原
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# 1. 数据准备
file_path = 'twitter新闻.txt'  # 请根据实际路径调整
df = pd.read_csv(file_path, sep='\t')
news_data = df['post_text'].tolist()

preprocessed_data = [preprocess_text(news) for news in news_data]

# 3. 模型构建与训练
# 构建词典
dictionary = corpora.Dictionary(preprocessed_data)
# 生成语料库
corpus = [dictionary.doc2bow(text) for text in preprocessed_data]
# 训练LDA模型
num_topics = 3  # 可根据实际情况调整主题数量
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# 4. 可视化分析
# pyLDAvis交互图
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
# 保存为本地html文件，避免加载远程资源出错
pyLDAvis.save_html(vis, 'lda_visualization.html')
print("pyLDAvis交互图已保存到 lda_visualization.html，可在浏览器中打开查看")

# 词云图
for topic_id in range(num_topics):
    topic_words = lda_model.show_topic(topic_id)
    word_freq = {word: freq for word, freq in topic_words}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Topic {topic_id + 1} Word Cloud")
    plt.axis('off')
    plt.show()

# 文档-主题概率分布矩阵（可选）
doc_topic_matrix = np.zeros((len(news_data), num_topics))
for i, doc in enumerate(corpus):
    for topic, prob in lda_model.get_document_topics(doc):
        doc_topic_matrix[i][topic] = prob

print("文档-主题概率分布矩阵:")
print(doc_topic_matrix)

# 结合大模型分析各主题的内容（这里简单打印主题关键词，后续可真正调用大模型完善）
for topic_id in range(num_topics):
    print(f"主题 {topic_id + 1} 关键词:")
    print(lda_model.show_topic(topic_id))