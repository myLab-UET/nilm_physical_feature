import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

class Builder:
    def __init__(self, **kwargs):
        pass

def filter_kwargs_for_module(module, **kwargs):
    """
    Filter kwargs to match the arguments of the module's __init__ method.
    """
    if inspect.isclass(module):
        init_signature = inspect.signature(module.__init__)
    elif inspect.isfunction(module) or inspect.ismethod(module):
        init_signature = inspect.signature(module)
    else:
        raise ValueError("module must be a class or a function/method")
        
    valid_params = init_signature.parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

class FCN(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super(FCN, self).__init__()
        # input_shape: (C, L)
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=128, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, nb_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x # Return logits

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        
        self.expand = in_channels != out_channels
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        if self.expand:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.expand:
            res = self.shortcut(x)
        else:
            res = self.shortcut(x)
            
        out = F.relu(out + res)
        return out

class ResNet(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super(ResNet, self).__init__()
        n_feature_maps = 64
        
        self.block1 = ResNetBlock(input_shape[0], n_feature_maps)
        self.block2 = ResNetBlock(n_feature_maps, n_feature_maps * 2)
        self.block3 = ResNetBlock(n_feature_maps * 2, n_feature_maps * 2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_feature_maps * 2, nb_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Hugging Face Transformer
try:
    from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
except ImportError:
    print("Transformers library not found. HFTimeSeriesTransformer will not be available.")

class HFTimeSeriesTransformer(nn.Module):
    def __init__(self, input_shape, nb_classes, d_model=64, nhead=4, num_layers=2, context_length=None):
        super(HFTimeSeriesTransformer, self).__init__()
        # input_shape: (C, L)
        self.n_features = input_shape[0]
        self.seq_len = input_shape[1]
        
        if context_length is None:
            context_length = self.seq_len

        # Config for TimeSeriesTransformer
        # We use dummy values for lags_sequence and prediction_length as we are doing classification
        self.config = TimeSeriesTransformerConfig(  # ty:ignore[unresolved-attribute]
            prediction_length=1,
            context_length=context_length,
            input_size=self.n_features,
            num_time_features=1,
            d_model=d_model,
            encoder_layers=num_layers,
            decoder_layers=num_layers,
            encoder_attention_heads=nhead,
            decoder_attention_heads=nhead,
            lags_sequence=[1],
        )
        
        self.model = TimeSeriesTransformerModel(self.config)
        
        # Classification head
        # We will use the mean of the encoder's last hidden state
        self.class_head = nn.Linear(d_model, nb_classes)

    def forward(self, x):
        # x: (N, C, L)
        batch_size = x.shape[0]
        
        # Permute to (N, L, C) for HF model
        x = x.permute(0, 2, 1) # (N, L, C)
        
        # Max Lag
        lag = max(self.config.lags_sequence)
        
        # Pad input with zeros at the beginning to match context_length + lag
        # x shape (N, L, C) -> (N, L + lag, C)
        padding = torch.zeros((batch_size, lag, self.n_features), device=x.device)
        x = torch.cat([padding, x], dim=1)
        
        # New sequence length
        new_seq_len = x.shape[1]
        
        # Create dummy time features: shape (N, L+lag, num_time_features)
        time_features = torch.linspace(0, 1, new_seq_len, device=x.device).unsqueeze(0).unsqueeze(-1) # (1, L+lag, 1)
        time_features = time_features.expand(batch_size, -1, -1) # (N, L+lag, 1)
        
        # Create past_observed_mask
        past_observed_mask = torch.ones_like(x, device=x.device)
        
        # Forward pass through HF model
        outputs = self.model(
            past_values=x,
            past_time_features=time_features,
            past_observed_mask=past_observed_mask,
        )
        
        # Extract encoder last hidden state: (N, L, d_model)
        # Note: TimeSeriesTransformerModel output is Seq2SeqTSModelOutput or tuple
        # outputs[0] is usually last_hidden_state (decoder's), but if we want classification on input sequence,
        # we might look at encoder_last_hidden_state which is outputs.encoder_last_hidden_state
        
        encoder_hidden = outputs.encoder_last_hidden_state
        
        # Global Average Pooling
        pooled = encoder_hidden.mean(dim=1) # (N, d_model)
        
        logits = self.class_head(pooled)
        return logits

def get_model_by_name(model_name, input_shape, nb_classes):
    if model_name == 'fcn':
        return FCN(input_shape, nb_classes)
    elif model_name == 'resnet':
        return ResNet(input_shape, nb_classes)
    elif model_name == 'hf_transformer':
        return HFTimeSeriesTransformer(input_shape, nb_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported.")