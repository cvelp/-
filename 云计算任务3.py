import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
import numpy as np
import random
from tqdm import tqdm


# 1. 数据预处理与加载
class NewsDataset(Dataset):
    def __init__(self, texts, topic_features, emotion_features, labels):
        self.texts = texts
        self.topic_features = topic_features
        self.emotion_features = emotion_features
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        topic = torch.tensor(self.topic_features[idx], dtype=torch.float32)
        emotion = torch.tensor(self.emotion_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'topic_features': topic,
            'emotion_features': emotion,
            'label': label
        }


# 2. 早期融合模型（拼接特征）
class EarlyFusionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', topic_dim=10, emotion_dim=10, num_classes=2):
        super(EarlyFusionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size

        # 特征融合层：拼接BERT池化输出、主题特征、情感特征
        fusion_dim = self.bert_dim + topic_dim + emotion_dim
        self.fusion_fc = nn.Linear(fusion_dim, 128)
        self.classifier = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, topic_features, emotion_features):
        # BERT特征提取
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_pooled = outputs.pooler_output  # [batch_size, 768]

        # 特征拼接
        fused_features = torch.cat([bert_pooled, topic_features, emotion_features], dim=1)  # [batch_size, 768+10+10]

        # 融合层处理
        fused_features = self.fusion_fc(fused_features)
        fused_features = self.relu(fused_features)
        fused_features = self.dropout(fused_features)

        # 分类层
        logits = self.classifier(fused_features)
        return logits

    # 3. 注意力机制融合模型（跨模态对齐）
    class AttentionFusionModel(nn.Module):
        def __init__(self, bert_model_name='bert-base-uncased', topic_dim=10, emotion_dim=10, num_classes=2):
            super(AttentionFusionModel, self).__init__()
            self.bert = BertModel.from_pretrained(bert_model_name)
            self.bert_dim = self.bert.config.hidden_size

            # 主题/情感特征投影层
            self.topic_proj = nn.Linear(topic_dim, self.bert_dim)
            self.emotion_proj = nn.Linear(emotion_dim, self.bert_dim)

            # 跨模态注意力
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.bert_dim,
                num_heads=4,
                dropout=0.2,
                batch_first=True
            )

            # 分类层
            self.classifier = nn.Linear(self.bert_dim, num_classes)
            self.dropout = nn.Dropout(0.3)

        def forward(self, input_ids, attention_mask, topic_features, emotion_features):
            # BERT特征提取
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_hidden = outputs.last_hidden_state  # [batch_size, seq_len, 768]

            # 主题/情感特征投影
            topic_proj = self.topic_proj(topic_features).unsqueeze(1)  # [batch_size, 1, 768]
            emotion_proj = self.emotion_proj(emotion_features).unsqueeze(1)  # [batch_size, 1, 768]

            # 跨模态注意力（文本-主题-情感动态对齐）
            attn_output, _ = self.cross_attention(
                query=bert_hidden,
                key=topic_proj.repeat(1, bert_hidden.size(1), 1),
                value=emotion_proj.repeat(1, bert_hidden.size(1), 1)
            )  # [batch_size, seq_len, 768]

            # 全局平均池化
            pooled_output = torch.mean(attn_output, dim=1)  # [batch_size, 768]
            pooled_output = self.dropout(pooled_output)

            # 分类层
            logits = self.classifier(pooled_output)
            return logits

    # 4. 训练与评估函数
    def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                topic_features = batch['topic_features'].to(device)
                emotion_features = batch['emotion_features'].to(device)
                labels = batch['label'].to(device)

                # 前向传播
                optimizer.zero_grad()
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    topic_features=topic_features,
                    emotion_features=emotion_features
                )
                loss = criterion(logits, labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress.set_postfix(loss=loss.item())

            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader):.4f}")

    def evaluate_model(model, dataloader, device):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                topic_features = batch['topic_features'].to(device)
                emotion_features = batch['emotion_features'].to(device)
                labels = batch['label'].to(device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    topic_features=topic_features,
                    emotion_features=emotion_features
                )
                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(probs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy

    # 5. 示例运行（含数据模拟）
    def run_example():
        # 模拟数据（实际应用中需替换为真实数据）
        texts = [
            "The news highlights positive economic growth this quarter",
            "Recent events show negative impacts on global markets",
            "Experts predict mixed outcomes for upcoming policies",
            # ... 更多文本
        ]

        # 模拟主题特征（10维）
        topic_features = np.random.rand(len(texts), 10)
        # 模拟情感特征（10维，0-1表示情感强度）
        emotion_features = np.random.rand(len(texts), 10)
        # 模拟标签（二分类：0/1）
        labels = np.random.randint(0, 2, len(texts))

        # 数据加载
        dataset = NewsDataset(texts, topic_features, emotion_features, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)

        # 模型初始化与训练
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        early_fusion_model = EarlyFusionModel().to(device)
        attention_model = AttentionFusionModel().to(device)

        # 优化器与损失函数
        early_optimizer = AdamW(early_fusion_model.parameters(), lr=2e-5)
        attention_optimizer = AdamW(attention_model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()

        # 训练早期融合模型
        print("=== 训练早期融合模型 ===")
        train_model(early_fusion_model, train_loader, early_optimizer, criterion, device, epochs=2)
        early_acc = evaluate_model(early_fusion_model, val_loader, device)

        # 训练注意力机制模型
        print("\n=== 训练注意力机制融合模型 ===")
        train_model(attention_model, train_loader, attention_optimizer, criterion, device, epochs=2)
        attention_acc = evaluate_model(attention_model, val_loader, device)

        # 结果对比
        print(f"\n模型性能对比：\n早期融合模型准确率：{early_acc:.2f}%\n注意力机制模型准确率：{attention_acc:.2f}%")

    # 执行示例
    if __name__ == "__main__":
        run_example()