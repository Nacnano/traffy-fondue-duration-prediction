# model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class TraffyBertRegressor(nn.Module):
    def __init__(self, model_name: str, num_org_features: int):
        super(TraffyBertRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(self.bert.config.hidden_size + 5 + num_org_features, 64)
        self.relu = nn.ReLU()
        self.regressor = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask, specificity, comment_length,
                is_priority, creation_month, day_of_week, org_features):

        # Run BERT
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # 🔧 ป้องกัน pooled_output มี 3 มิติ (เช่น [1,1,768])
        if pooled_output.ndim == 3:
            pooled_output = pooled_output.squeeze(1)
        pooled_output = self.dropout(pooled_output)

        # 🔧 ป้องกัน feature อื่นมี 3 มิติ
        for feature_name in ['specificity', 'comment_length', 'is_priority', 'creation_month', 'day_of_week', 'org_features']:
            feature = locals()[feature_name]
            if feature.ndim == 3:
                locals()[feature_name] = feature.squeeze()

        combined = torch.cat(
            (
                pooled_output,
                specificity,
                comment_length,
                is_priority,
                creation_month,
                day_of_week,
                org_features
            ),
            dim=1
        )

        dense_output = self.relu(self.dense(combined))
        return self.regressor(dense_output)

def load_model(model_path: str, model_name: str, num_org_features: int, device: torch.device):
    model = TraffyBertRegressor(model_name, num_org_features)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
