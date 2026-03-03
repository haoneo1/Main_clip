import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision

from src.model_base import BasePromptModel, _to_key_list
from src.clip import clip


class FinetuneModel(BasePromptModel):
    def __init__(self, opts):
        super().__init__(opts)

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2
        )

        self.best_metric = self.opts.best_metric_init

        self.use_clip_cls = self.opts.use_clip_cls
        self.lambda_cls = self.opts.lambda_cls
        self.cls_tau = self.opts.cls_tau
        self.use_clip_logit_scale = self.opts.use_clip_logit_scale

        self.seen_class_names = [str(x) for x in self.opts.seen_class_names]

        self.cls_templates = list(
            getattr(
                self.opts,
                "cls_templates",
                [
                    "a photo of a {}",
                    "a photo of the {}",
                    "a sketch of a {}",
                    "a sketch of the {}",
                ],
            )
        )

        self.class_to_idx = {
            str(name): idx for idx, name in enumerate(self.seen_class_names)
        }

        self.register_buffer(
            "text_classifier",
            torch.empty(0, self.feat_dim, dtype=torch.float32),
            persistent=False,
        )

        if self.use_clip_cls:
            self.freeze_text_tower()

        self._val_query_feats = []
        self._val_gallery_feats = []
        self._val_categories = []

    def freeze_text_tower(self):
        if hasattr(self.clip, "transformer"):
            for p in self.clip.transformer.parameters():
                p.requires_grad_(False)

        if hasattr(self.clip, "token_embedding"):
            if hasattr(self.clip.token_embedding, "weight") and self.clip.token_embedding.weight is not None:
                self.clip.token_embedding.weight.requires_grad_(False)

        if hasattr(self.clip, "positional_embedding"):
            if isinstance(self.clip.positional_embedding, torch.Tensor):
                self.clip.positional_embedding.requires_grad_(False)

        if hasattr(self.clip, "ln_final"):
            for p in self.clip.ln_final.parameters():
                p.requires_grad_(False)

        if hasattr(self.clip, "text_projection"):
            if isinstance(self.clip.text_projection, torch.Tensor):
                self.clip.text_projection.requires_grad_(False)

    def build_text_classifier(self):
        if not self.use_clip_cls:
            return

        if len(self.seen_class_names) == 0:
            raise ValueError("use_clip_cls=True, but seen_class_names is empty.")

        text_features = []

        was_training = self.clip.training
        self.clip.eval()

        with torch.no_grad():
            for class_name in self.seen_class_names:
                cname = str(class_name).replace("_", " ")
                prompts = [tmpl.format(cname) for tmpl in self.cls_templates]
                tokens = clip.tokenize(prompts).to(self.device)

                txt_feat = self.clip.encode_text(tokens)
                txt_feat = F.normalize(txt_feat, dim=-1)

                class_feat = txt_feat.mean(dim=0)
                class_feat = F.normalize(class_feat, dim=0)
                text_features.append(class_feat)

        self.text_classifier = torch.stack(text_features, dim=0).to(self.device)

        if was_training:
            self.clip.train()

    def on_fit_start(self):
        if self.use_clip_cls:
            self.build_text_classifier()

    def _get_cls_logit_scale(self):
        if self.use_clip_logit_scale and hasattr(self.clip, "logit_scale"):
            return self.clip.logit_scale.exp().clamp(max=100.0)
        else:
            return torch.tensor(1.0 / self.cls_tau, device=self.device)

    def _category_to_label_tensor(self, category):
        keys = _to_key_list(category)
        labels = []

        for k in keys:
            if k not in self.class_to_idx:
                raise KeyError(f"Category '{k}' not found in seen_class_names.")
            labels.append(self.class_to_idx[k])

        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def _clip_text_cls_loss(self, feat, labels):
        if self.text_classifier.numel() == 0:
            raise RuntimeError("text_classifier is empty.")

        feat = F.normalize(feat, dim=-1)
        text_w = F.normalize(self.text_classifier, dim=-1)

        scale = self._get_cls_logit_scale()
        logits = scale * torch.matmul(feat, text_w.t())

        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, logits, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.clip.parameters(), "lr": self.opts.clip_LN_lr},
                {"params": [self.sk_prompt, self.img_prompt], "lr": self.opts.prompt_lr},
            ]
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]

        img_feat = self.forward(img_tensor, dtype="image")
        sk_feat = self.forward(sk_tensor, dtype="sketch")
        neg_feat = self.forward(neg_tensor, dtype="image")

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        total_loss = triplet_loss

        self.log("train_triplet", triplet_loss, on_step=True, on_epoch=True, prog_bar=False)

        if self.use_clip_cls:
            labels = self._category_to_label_tensor(category)

            sk_cls_loss, _, sk_acc = self._clip_text_cls_loss(sk_feat, labels)
            img_cls_loss, _, img_acc = self._clip_text_cls_loss(img_feat, labels)

            cls_loss = sk_cls_loss + img_cls_loss
            total_loss = triplet_loss + self.lambda_cls * cls_loss

            self.log("train_cls_sk", sk_cls_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_cls_img", img_cls_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_cls_total", cls_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_acc_sk", sk_acc, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_acc_img", img_acc, on_step=True, on_epoch=True, prog_bar=False)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def on_validation_epoch_start(self):
        self._val_query_feats = []
        self._val_gallery_feats = []
        self._val_categories = []

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]

        img_feat = self.forward(img_tensor, dtype="image")
        sk_feat = self.forward(sk_tensor, dtype="sketch")
        neg_feat = self.forward(neg_tensor, dtype="image")

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self._val_query_feats.append(sk_feat.detach().cpu())
        self._val_gallery_feats.append(img_feat.detach().cpu())
        self._val_categories.extend(_to_key_list(category))

        return loss

    def on_validation_epoch_end(self):
        if len(self._val_query_feats) == 0:
            return

        query_feat_all = torch.cat(self._val_query_feats, dim=0)
        gallery_feat_all = torch.cat(self._val_gallery_feats, dim=0)
        all_category = np.asarray(self._val_categories, dtype=str)

        query = F.normalize(query_feat_all, dim=1)
        gallery = F.normalize(gallery_feat_all, dim=1)

        ap = torch.zeros(len(query), dtype=torch.float32)

        for idx in range(len(query)):
            scores = torch.matmul(gallery, query[idx])
            target_np = (all_category == all_category[idx])
            target = torch.from_numpy(target_np).to(dtype=torch.bool)

            if target.sum().item() == 0:
                ap[idx] = 0.0
                continue

            ap[idx] = retrieval_average_precision(scores, target)

        mAP = ap.mean()
        self.log("mAP", mAP, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top10", mAP, on_step=False, on_epoch=True, prog_bar=False)

        if self.global_step > 0:
            self.best_metric = max(self.best_metric, float(mAP.item()))

        print(f"[VAL] mAP: {mAP.item():.6f}, Best mAP: {self.best_metric:.6f}")