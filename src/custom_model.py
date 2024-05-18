from pooling_layers import *
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from config import CFG
import warnings
warnings.filterwarnings("ignore")


class CustomModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CFG.backbone_model)
        if CFG.zero_dropout:
            self.backbone.attention_probs_dropout_prob = 0.0
            self.backbone.hidden_dropout_prob = 0.0
        self.backbone.resize_token_embeddings(len(tokenizer))
        self.pool = get_pooling_layer()
        self.fc = nn.Linear(self.pool.output_dim, 1)

        if CFG.use_initialization:
            self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=CFG.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=CFG.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs = self.backbone(**inputs, output_hidden_states=True)
        feature = self.pool(inputs, outputs)
        output = self.fc(feature)

        return SequenceClassifierOutputWithPast(
            loss=None,
            logits=output,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )


def Get_AutoModel(tokenizer):
    model = AutoModelForSequenceClassification(
        CFG.backbone_model,
        num_labels=1
    )
    model.resize_token_embeddings(len(tokenizer))

    if CFG.zero_dropout:
        model.config.attention_probs_dropout_prob = 0.0
        model.config.hidden_dropout_prob = 0.0

    return model
