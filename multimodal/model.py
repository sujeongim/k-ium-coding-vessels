import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoTokenizer

class MultiModalAneurysmClassifier(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased", image_model_name="resnet18", hidden_dim=512):
        super().__init__()
        
        # Image encoder
        image_model = models.__dict__[image_model_name](pretrained=True)
        self.image_encoder = nn.Sequential(*list(image_model.children())[:-1])  # remove FC layer
        self.image_feature_dim = image_model.fc.in_features
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_feature_dim = self.text_encoder.config.hidden_size
        
        # Fusion MLP
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.image_feature_dim + self.text_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Final classifier (22 binary outputs)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim * 8, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 22),
        #     nn.Sigmoid()
        # )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 15)
        )


    def forward(self, images, texts):
        """
        images: [B, 8, 3, H, W]
        texts: List[List[str]] — length B, each element is 8 texts
        """
        B = images.size(0)
        all_fused = []

        for i in range(8):
            # 각 i번째 이미지와 텍스트
            img = images[:, i, :, :, :]  # [B, 3, H, W]
            img_feat = self.image_encoder(img).squeeze(-1).squeeze(-1)  # [B, image_feature_dim]

            ith_texts = [sample[i] for sample in texts]  # B개의 i번째 텍스트 추출
            tokens = self.tokenizer(ith_texts, return_tensors="pt", padding=True, truncation=True).to(images.device)
            txt_feat = self.text_encoder(**tokens).last_hidden_state[:, 0, :]  # [B, text_feature_dim]

            fused = torch.cat([img_feat, txt_feat], dim=1)  # [B, image+text]
            fused = self.fusion_layer(fused)  # [B, hidden_dim]
            all_fused.append(fused)

        combined = torch.cat(all_fused, dim=1)  # [B, hidden_dim * 8]
        out = self.classifier(combined)  # [B, 22]
        return out

class ImageOnlyAneurysmClassifier(nn.Module):
    def __init__(self, image_model_name="resnet18", hidden_dim=512):
        super().__init__()
        
        # Image encoder
        image_model = models.__dict__[image_model_name](pretrained=True)
        self.image_encoder = nn.Sequential(*list(image_model.children())[:-1])  # remove FC layer
        self.image_feature_dim = image_model.fc.in_features
        
        # MLP after image features
        self.image_fusion_layer = nn.Sequential(
            nn.Linear(self.image_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Final classifier (15 outputs)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 15)
        )

    def forward(self, images, texts=None):
        """
        images: [B, 8, 3, H, W]
        """
        B = images.size(0)
        all_fused = []

        for i in range(8):
            img = images[:, i, :, :, :]  # [B, 3, H, W]
            img_feat = self.image_encoder(img).squeeze(-1).squeeze(-1)  # [B, image_feature_dim]
            fused = self.image_fusion_layer(img_feat)  # [B, hidden_dim]
            all_fused.append(fused)

        combined = torch.cat(all_fused, dim=1)  # [B, hidden_dim * 8]
        out = self.classifier(combined)  # [B, 15]
        return out
