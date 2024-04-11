import torch
import random
from torch import nn
from clip import clip


class TriMPL(nn.Module):
    def __init__(self, cfg, classes, clip_model):
        super(TriMPL, self).__init__()
        self.n_view = cfg.TriMPL.N_VIEW
        self.n_ctx = cfg.TriMPL.N_CTX
        self.n_cls = len(classes)
        self.dtype = clip_model.dtype
        self.logit_scale = clip_model.logit_scale
        self.image_encoder = clip_model.visual
        self.custom_text_encoder = CustomTextEncoder(cfg, classes, clip_model)

    def forward(self, image):
        im_features = self.image_encoder(image.type(self.dtype))
        im_features = im_features / im_features.norm(dim=-1, keepdim=True)

        prompts = self.custom_text_encoder()
        prompts = prompts / prompts.norm(dim=-1, keepdim=True)
        prompts = prompts.transpose(0, 1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (im_features @ prompts)
        logits = logits.reshape(-1, self.n_cls, self.n_view)

        if self.training:
            k = random.randint(0, self.n_view-1)
            indices = torch.randperm(self.n_view)[k:]
            logits = logits[:, :, indices]

        logits = torch.mean(logits, dim=-1)
        if not self.training:
            return logits, None
        return logits, self.custom_text_encoder.feature_descriptors


class CustomTextEncoder(nn.Module):
    def __init__(self, cfg, classes, clip_model):
        super(CustomTextEncoder, self).__init__()
        self.n_view = cfg.TriMPL.N_VIEW
        self.n_ctx = cfg.TriMPL.N_CTX
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.n_cls = len(classes)
        self.dtype = clip_model.dtype

        self.positional_embedding = clip_model.positional_embedding
        self.tsf_layers = clip_model.transformer.resblocks
        self.n_layer = len(self.tsf_layers)
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        template = "a photo of a"
        tokenized_general = clip.tokenize(template)
        tokenized_general = tokenized_general[0, 1:tokenized_general.argmax(dim=-1)]
        self.n_g = tokenized_general.shape[0]
        embedding = torch.empty(self.n_g, self.ctx_dim, dtype=self.dtype)
        nn.init.normal_(embedding, std=0.02)
        self.general_prompt = nn.Parameter(embedding)

        handcrafted = [template + " " + cls for cls in classes]
        tokenized_handcrafted = clip.tokenize(handcrafted)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_handcrafted).type(self.dtype)
        self.register_buffer("handcrafted_prompts", embedding)

        embedding = torch.empty(self.n_view, self.n_ctx, self.ctx_dim, dtype=self.dtype)
        nn.init.normal_(embedding, std=0.02)
        self.feature_descriptors = nn.Parameter(embedding)

        placeholder = " ".join(["X"] * self.n_ctx)
        pseudo = [template + " " + cls + " " + placeholder for cls in classes]
        tokenized_pseudo = clip.tokenize(pseudo)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_pseudo).type(self.dtype)
        self.register_buffer("pseudo_prompts", embedding)
        self.eots = tokenized_pseudo.argmax(dim=-1)

        alpha = torch.full((self.n_layer,), fill_value=0, dtype=self.dtype)
        beta = torch.full((self.n_layer,), fill_value=0, dtype=self.dtype)
        self.mix_alpha = nn.Parameter(alpha)
        self.mix_beta = nn.Parameter(beta)

    def forward(self):
        prompts = self.pseudo_prompts.clone()
        prompts[:, 1:1+self.n_g] = self.general_prompt

        prompts = prompts.unsqueeze(1)
        prompts = prompts.repeat(1, self.n_view, 1, 1)
        for i in range(self.n_cls):
            e = self.eots[i]
            prompts[i, :, e-self.n_ctx:e] = self.feature_descriptors

        prompts = prompts.reshape(self.n_cls*self.n_view, -1, self.ctx_dim)
        prompts = prompts + self.positional_embedding.type(self.dtype)
        prompts = prompts.permute(1, 0, 2)

        handcrafted_prompts = self.handcrafted_prompts.unsqueeze(1)
        handcrafted_prompts = handcrafted_prompts.repeat(1, self.n_view, 1, 1)
        handcrafted_prompts = handcrafted_prompts.reshape(self.n_cls*self.n_view, -1, self.ctx_dim)
        handcrafted_prompts = handcrafted_prompts + self.positional_embedding.type(self.dtype)
        handcrafted_prompts = handcrafted_prompts.permute(1, 0, 2)

        mix_alpha = torch.exp(self.mix_alpha)
        mix_beta = torch.exp(self.mix_beta)
        ratios_alpha = mix_alpha / (mix_alpha+mix_beta)
        ratios_beta = mix_beta / (mix_alpha+mix_beta)
        for ith, layer in enumerate(self.tsf_layers):
            ratio_alpha = ratios_alpha[ith]
            ratio_beta = ratios_beta[ith]
            # for i in range(self.n_cls):
            #     to = self.mix_to[i]
            #     s = i*self.n_view
            #     e = s + self.n_view
            #     prompts_new[1:to, s:e] = ratio_beta*prompts[1:to, s:e] + ratio_alpha*handcrafted_prompts[1:to, s:e]
            prompts_new = prompts.clone()
            prompts_new[1:1 + self.n_g] = ratio_beta*prompts[1:1 + self.n_g] + ratio_alpha * handcrafted_prompts[1:1 + self.n_g]
            prompts = prompts_new

            prompts = layer(prompts)
            handcrafted_prompts = layer(handcrafted_prompts)

        prompts = prompts.permute(1, 0, 2)
        prompts = self.ln_final(prompts).type(self.dtype)

        eots = self.eots.repeat_interleave(self.n_view)
        features = prompts[torch.arange(prompts.shape[0]), eots] @ self.text_projection
        return features


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


def build_model(cfg, categories, device):
    clip_model = load_clip_to_cpu(cfg)

    if cfg.TriMPL.PREC == "fp32" or cfg.TriMPL.PREC == "amp":
        clip_model.float()

    model = TriMPL(cfg, categories, clip_model)
    parameters = ["general_prompt", "feature_descriptors", "mix_alpha", "mix_beta"]

    for name, param in model.named_parameters():
        update = False
        for p in parameters:
            if p in name:
                update = True
                break
        param.requires_grad_(update)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)

    if device == torch.device("cuda"):
        device_count = torch.cuda.device_count()
        if device_count > 1:
            model = nn.DataParallel(model)

    model = model.to(device)
    return model


def load_model(model_path, cfg, categories):
    clip_model = load_clip_to_cpu(cfg)

    if cfg.TriMPL.PREC == "fp32" or cfg.TriMPL.PREC == "amp":
        clip_model.float()

    model = TriMPL(cfg, categories, clip_model)
    state_dict = torch.load(model_path)

    if "custom_text_encoder.handcrafted_prompts" in state_dict:
        del state_dict["custom_text_encoder.handcrafted_prompts"]

    if "custom_text_encoder.pseudo_prompts" in state_dict:
        del state_dict["custom_text_encoder.pseudo_prompts"]

    if "custom_text_encoder.token_prefix" in state_dict:
        del state_dict["custom_text_encoder.token_prefix"]

    if "custom_text_encoder.token_suffix" in state_dict:
        del state_dict["custom_text_encoder.token_suffix"]

    model.load_state_dict(state_dict, strict=False)
    return model
