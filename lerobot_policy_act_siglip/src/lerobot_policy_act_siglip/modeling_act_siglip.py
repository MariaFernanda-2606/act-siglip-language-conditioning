"""ACT conditioned on a SigLIP text embedding, with a subtask-completion head.

Changes with respect to lerobot's ACT (everything else is inherited unchanged):
    * one extra transformer-encoder token: SigLIP embedding -> (centre) -> Linear -> LayerNorm;
    * a completion head on the encoder output at that token, used by the planner to advance:
        - "done":     P(the given instruction is already completed);
        - "progress": progress (0..1) through the current subtask;
    * the embedding is looked up from the batch `index` during training, or set with
      `set_instruction(text)` at inference;
    * optional (off by default, so older checkpoints load unchanged):
        - FiLM: the language modulates the ResNet stages listed in `film_layers`;
        - state dropout: the robot-state token is sometimes replaced by a learned "no state" token;
        - state-free completion head (`completion_use_state=False`);
        - per-phase action-loss weights and extra weight on the first frames of each episode;
        - auxiliary grasp head: predicts where the current pick's grasp will be (training signal only).
"""

from __future__ import annotations

from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from .colour_swap import PERMS, protect_mask, recolour_normalised
from .configuration_act_siglip import ACTSiglipConfig
from .planner import COLORS, PHASES, subtask_text

OBS_LANGUAGE = "observation.language_embedding"
OBS_PROGRESS = "observation.progress_level"  # (B,) long: 0..progress_levels-1, or progress_levels = unknown


def _normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def _segment_targets(episode: np.ndarray, phase: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame progress through its subtask segment, whether its action chunk crosses a boundary, and the
    number of frames left in its segment."""
    n = len(episode)
    new_segment = np.r_[True, (episode[1:] != episode[:-1]) | (phase[1:] != phase[:-1])]
    new_episode = np.r_[True, episode[1:] != episode[:-1]]
    seg_id, ep_id = np.cumsum(new_segment) - 1, np.cumsum(new_episode) - 1
    seg_start = np.flatnonzero(new_segment)
    seg_len = np.diff(np.r_[seg_start, n])
    ep_start = np.flatnonzero(new_episode)
    ep_len = np.diff(np.r_[ep_start, n])

    pos_in_seg = np.arange(n) - seg_start[seg_id]
    length = seg_len[seg_id]
    progress = pos_in_seg / np.maximum(length - 1, 1)
    to_seg_end = length - 1 - pos_in_seg
    to_ep_end = ep_len[ep_id] - 1 - (np.arange(n) - ep_start[ep_id])
    crosses = (to_seg_end < chunk_size - 1) & (to_seg_end < to_ep_end)
    return progress.astype(np.float32), crosses, to_seg_end.astype(np.int64)


class ACTSiglip(ACT):
    """ACT network with an extra language token and an optional completion head."""

    def __init__(self, config: ACTSiglipConfig):
        super().__init__(config)
        self.use_language = config.language_mode != "none"
        n_1d_tokens = 1 + bool(config.robot_state_feature) + bool(config.env_state_feature)
        if self.use_language:
            self.language_token_index = n_1d_tokens
            n_1d_tokens += 1
            self.encoder_language_input_proj = nn.Sequential(
                nn.Linear(config.language_dim, config.dim_model),
                nn.LayerNorm(config.dim_model),
            )
        else:
            self.language_token_index = 0  # a progress head then reads the latent token
        # Re-created with room for the language token; order is [latent, state, env, language].
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if config.completion_head != "none":
            self.completion_head = nn.Sequential(
                nn.Linear(config.dim_model, config.dim_model),
                nn.ReLU(),
                nn.Linear(config.dim_model, 1),
            )
        # FiLM generators (language -> per-channel gamma, beta) for the selected ResNet stages.
        # Zero-initialised: the pretrained visual features are unchanged at the start of training.
        self.film = nn.ModuleDict()
        if config.film_layers:
            channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}  # resnet18/34
            if config.vision_backbone not in ("resnet18", "resnet34"):
                raise ValueError("film_layers is implemented for resnet18/resnet34 backbones")
            for layer in config.film_layers:
                gen = nn.Linear(config.language_dim, 2 * channels[layer])
                nn.init.zeros_(gen.weight)
                nn.init.zeros_(gen.bias)
                self.film[layer] = gen
        # Auxiliary "where will the grasp be?" head on the language token (training signal only).
        self.last_grasp: Tensor | None = None
        if config.grasp_dim > 0:
            self.grasp_head = nn.Sequential(
                nn.Linear(config.dim_model, config.dim_model),
                nn.ReLU(),
                nn.Linear(config.dim_model, config.grasp_dim),
            )
        # Learned "no robot state" token (state dropout / state-free completion head).
        if config.robot_state_feature and (config.state_dropout > 0 or not config.completion_use_state):
            self.state_null_embed = nn.Parameter(torch.zeros(config.dim_model))
        # Progress within the subtask (Bi-HIL): an encoder token after the language token, with its own position
        # embedding (separate parameters, so checkpoints without it still load), and a head that predicts it.
        # Zero-initialised: at the start of fine-tuning the token carries no information.
        self.last_progress: Tensor | None = None
        if config.progress_levels > 0:
            self.progress_embed = nn.Embedding(config.progress_levels + 1, config.dim_model)  # last row = unknown
            nn.init.zeros_(self.progress_embed.weight)
            self.progress_pos_embed = nn.Parameter(torch.zeros(1, config.dim_model))
            self.progress_head = nn.Sequential(
                nn.Linear(config.dim_model, config.dim_model),
                nn.ReLU(),
                nn.Linear(config.dim_model, 1),
            )

    def progress_level(self, progress: Tensor) -> Tensor:
        """Progress in [0, 1] -> level 0..progress_levels-1."""
        levels = self.config.progress_levels
        return (progress * levels).long().clamp(0, levels - 1)

    # -------------------------------------------------------------- pieces of ACT.forward
    def _latent(self, batch: dict[str, Tensor], batch_size: int):
        """VAE latent during training, zeros otherwise (unchanged from ACT)."""
        if not (self.config.use_vae and ACTION in batch and self.training):
            zeros = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, device=batch[OBS_STATE].device)
            return zeros, None, None
        cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
        action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
        if self.config.robot_state_feature:
            robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            vae_encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], axis=1)
        else:
            vae_encoder_input = torch.cat([cls_embed, action_embed], axis=1)
        pos_embed = self.vae_encoder_pos_enc.clone().detach()
        cls_joint_is_pad = torch.full(
            (batch_size, 2 if self.config.robot_state_feature else 1), False, device=batch[OBS_STATE].device
        )
        key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
        cls_token_out = self.vae_encoder(
            vae_encoder_input.permute(1, 0, 2), pos_embed=pos_embed.permute(1, 0, 2), key_padding_mask=key_padding_mask
        )[0]
        latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
        mu = latent_pdf_params[:, : self.config.latent_dim]
        log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
        return mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu), mu, log_sigma_x2

    def _backbone(self, img: Tensor, film_language: Tensor | None) -> Tensor:
        """ResNet feature map; with FiLM, the outputs of the selected stages are modulated by the language."""
        if not self.film or film_language is None:
            return self.backbone(img)["feature_map"]
        x = img
        for name, module in self.backbone.items():  # conv1, bn1, relu, maxpool, layer1..layer4
            x = module(x)
            if name in self.film:
                gamma, beta = self.film[name](film_language).to(x.dtype)[:, :, None, None].chunk(2, dim=1)
                x = x * (1 + gamma) + beta
            if name == "layer4":
                break
        return x

    def _image_tokens(self, batch: dict[str, Tensor], film_language: Tensor | None = None,
                      n: int | None = None) -> tuple[list[Tensor], list[Tensor]]:
        """Image tokens for the first n samples (default all), FiLM-conditioned on film_language (n, D)."""
        tokens, pos = [], []
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                img = img if n is None else img[:n]
                cam_features = self._backbone(img, film_language)
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                tokens.extend(list(einops.rearrange(cam_features, "b c h w -> (h w) b c")))
                pos.extend(list(einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")))
        return tokens, pos

    def _encode(self, latent: Tensor, context: list[Tensor], language: Tensor | None,
                image_tokens: list[Tensor], pos: Tensor, progress: Tensor | None = None) -> Tensor:
        tokens = [self.encoder_latent_input_proj(latent), *context]
        if language is not None:
            tokens.append(language)
        if progress is not None:
            tokens.append(progress)
        return self.encoder(torch.stack(tokens + image_tokens, axis=0), pos_embed=pos)

    # -------------------------------------------------------------- forward
    def forward(self, batch: dict[str, Tensor], extra_languages: tuple[Tensor, ...] = (), n_completion: int | None = None,
                infer_progress: bool = False, progress_floor: Tensor | None = None):
        """Returns (actions, (mu, log_sigma_x2), completion_logits).

        completion_logits is None without a completion head, otherwise a (1 + len(extra_languages), m)
        tensor for the first m = n_completion samples (default: whole batch): row 0 for the samples' own
        language embedding, then one row per extra embedding (each extra is (m, language_dim), centred).
        Completion rows are computed with a zero latent, as at inference; with FiLM the image features are
        recomputed for each extra language; with completion_use_state=False they never see the robot state.

        With progress_levels > 0 the action pass gets a progress token: the level in batch[OBS_PROGRESS] (training,
        offline evaluation) or, with infer_progress=True (robot), the level predicted by the progress head from the
        completion pass, which then runs first (progress_floor: optional (B,) lower bound on the predicted progress,
        to keep it non-decreasing within a subtask). The completion pass always sees the "unknown" level; the
        progress it predicts for its own-language rows is stored in self.last_progress, (m,) in [0, 1].
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions must be provided when using the variational objective in training mode."
        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        use_progress = self.config.progress_levels > 0
        if infer_progress and not use_progress:
            raise ValueError("infer_progress needs progress_levels > 0")

        latent, mu, log_sigma_x2 = self._latent(batch, batch_size)
        state_token = None
        if self.config.robot_state_feature:
            state_token = self.encoder_robot_state_input_proj(batch[OBS_STATE])
        env_token = self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]) if self.config.env_state_feature else None

        def context_for(state: Tensor | None, env: Tensor | None) -> list[Tensor]:
            return [t for t in (state, env) if t is not None]

        action_state = state_token
        if state_token is not None and self.training and self.config.state_dropout > 0:
            drop = torch.rand(batch_size, 1, device=state_token.device) < self.config.state_dropout
            action_state = torch.where(drop, self.state_null_embed.to(state_token.dtype).expand_as(state_token), state_token)

        language = self.encoder_language_input_proj(batch[OBS_LANGUAGE]) if self.use_language else None
        film_language = batch[OBS_LANGUAGE] if (self.film and self.use_language) else None
        image_tokens, image_pos = self._image_tokens(batch, film_language)
        pos_1d = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if use_progress:
            pos_1d.append(self.progress_pos_embed)
        pos = torch.stack(pos_1d + image_pos, axis=0)

        def progress_tokens(level: Tensor) -> Tensor:
            return self.progress_embed(level).to(pos.dtype)

        unknown = self.config.progress_levels

        def completion_pass() -> Tensor:
            """Completion logits (k, m); sets self.last_progress from the own-language rows."""
            state_free = state_token is not None and not self.config.completion_use_state
            if (not use_progress and mu is None and not extra_languages and not state_free
                    and action_state is state_token):
                # Latent is already zero (inference / no VAE) and the inputs are the same: reuse the main pass.
                return self.completion_head(encoder_out[self.language_token_index]).squeeze(-1)[None]
            # One batched encoder pass over [own language, *extra languages] with a zero latent,
            # on the first m samples only (memory).
            m = batch_size if n_completion is None else n_completion
            k = 1 + len(extra_languages)
            langs = None
            if self.use_language:
                langs = torch.cat([language[:m]] + [self.encoder_language_input_proj(e) for e in extra_languages], 0)
            if film_language is not None and extra_languages:
                per_lang = [[t[:m] for t in image_tokens]] + [self._image_tokens(batch, e, n=m)[0] for e in extra_languages]
                comp_images = [torch.cat(ts, 0) for ts in zip(*per_lang)]
            else:
                comp_images = [t[:m].repeat(k, 1) for t in image_tokens]
            comp_state = None
            if state_token is not None:
                comp_state = (self.state_null_embed.to(state_token.dtype).expand(k * m, -1) if state_free
                              else state_token[:m].repeat(k, 1))
            comp_env = env_token[:m].repeat(k, 1) if env_token is not None else None
            comp_progress = None
            if use_progress:
                comp_progress = progress_tokens(torch.full((k * m,), unknown, dtype=torch.long, device=pos.device))
            zero_latent = torch.zeros(k * m, self.config.latent_dim, device=pos.device, dtype=pos.dtype)
            out = self._encode(zero_latent, context_for(comp_state, comp_env), langs, comp_images, pos, comp_progress)
            lang_out = out[self.language_token_index]
            if use_progress:
                self.last_progress = torch.sigmoid(self.progress_head(lang_out[:m]).squeeze(-1).float())
            return self.completion_head(lang_out).squeeze(-1).view(k, m)

        completion_logits = None
        action_progress = None
        if infer_progress:
            completion_logits = completion_pass()
            progress = self.last_progress
            if progress_floor is not None:
                progress = torch.maximum(progress, progress_floor.to(progress))
                self.last_progress = progress
            action_progress = progress_tokens(self.progress_level(progress))
        elif use_progress:
            level = batch.get(OBS_PROGRESS)
            if level is None:
                level = torch.full((batch_size,), unknown, dtype=torch.long, device=pos.device)
            action_progress = progress_tokens(level.long())

        encoder_out = self._encode(latent, context_for(action_state, env_token), language, image_tokens, pos,
                                   action_progress)
        decoder_in = torch.zeros((self.config.chunk_size, batch_size, self.config.dim_model), dtype=pos.dtype, device=pos.device)
        decoder_out = self.decoder(decoder_in, encoder_out, encoder_pos_embed=pos,
                                   decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1))
        actions = self.action_head(decoder_out.transpose(0, 1))
        # Grasp prediction of the action pass (standardised pose), kept for the loss / offline evaluation.
        self.last_grasp = (self.grasp_head(encoder_out[self.language_token_index])
                           if self.config.grasp_dim > 0 else None)

        if self.config.completion_head != "none" and completion_logits is None:
            completion_logits = completion_pass()

        return actions, (mu, log_sigma_x2), completion_logits


class ACTSiglipPolicy(ACTPolicy):
    """ACT policy conditioned on SigLIP language embeddings (policy type `act_siglip`)."""

    config_class = ACTSiglipConfig
    name = "act_siglip"

    def __init__(self, config: ACTSiglipConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._load_language_tables()
        self.model = ACTSiglip(config)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self._instruction: tuple[str, Tensor] | None = None
        self._encoded_cache: dict[str, Tensor] = {}
        self._siglip = None
        self.last_completion: float | None = None
        self.last_progress: float | None = None
        self.progress_ratchet = True  # inference: predicted progress never decreases within a subtask
        self.reset()

    def reset(self):
        super().reset()
        self._progress_floor = 0.0  # restarts with each subtask (set_instruction calls reset on a change)

    # ------------------------------------------------------------------ language tables
    def _load_language_tables(self) -> None:
        cfg = self.config
        self._row_of_text: dict[str, int] = {}
        if cfg.language_mode != "none":
            path = Path(cfg.language_table_path) if cfg.language_table_path else None
            if path is not None and path.exists():
                table = np.load(path)
                cfg.language_texts = [str(t) for t in table["texts"]]
                embeddings = torch.from_numpy(table["embeddings"].astype(np.float32))
                cfg.language_dim = embeddings.shape[1]
            elif cfg.language_texts:  # loading a checkpoint: values come from the state dict
                embeddings = torch.zeros(len(cfg.language_texts), cfg.language_dim)
            else:
                raise ValueError("language_mode != 'none' requires --policy.language_table_path")
            self.register_buffer("language_table", embeddings)
            self.register_buffer("language_mean", embeddings.mean(0))
            self._row_of_text = {_normalise_text(t): i for i, t in enumerate(cfg.language_texts)}

        self._has_frame_index = False
        path = Path(cfg.frame_index_path) if cfg.frame_index_path else None
        if path is not None and path.exists():
            index = np.load(path)
            progress, crosses, to_seg_end = _segment_targets(index["episode_index"], index["phase"], cfg.chunk_size)
            # Non-persistent: training-only lookups, not stored in the checkpoint.
            def buf(name, array):
                self.register_buffer(name, torch.from_numpy(array), persistent=False)
            buf("frame_subtask_row", index["subtask_row"].astype(np.int64))
            buf("frame_task_row", index["task_row"].astype(np.int64))
            buf("frame_phase", index["phase"].astype(np.int64))
            buf("frame_in_episode", index["frame_index"].astype(np.int64))
            buf("frame_progress", progress)
            buf("frame_crosses_boundary", crosses)
            buf("frame_to_segment_end", to_seg_end)
            self._has_frame_index = True
            if cfg.completion_head == "done" or cfg.colour_swap_prob > 0:
                # Colour of each frame (from its task text) and canonical instruction rows per (colour, phase),
                # used to show a frame with the instruction of an earlier phase, or of another colour.
                task_texts = np.array([t.lower() for t in cfg.language_texts])
                colour_of_row = np.array([next((i for i, c in enumerate(COLORS) if c in t), -1) for t in task_texts])
                buf("frame_colour", colour_of_row[index["task_row"]].astype(np.int64))
                missing = [subtask_text(p, c) for c in COLORS for p in PHASES
                           if _normalise_text(subtask_text(p, c)) not in self._row_of_text]
                if missing:
                    raise ValueError(f"Canonical subtasks missing from the language table: {missing}")
                buf("canonical_rows", np.array([[self._row_of_text[_normalise_text(subtask_text(p, c))] for p in PHASES]
                                                for c in COLORS], dtype=np.int64))

        self._has_grasp_targets = False
        path = Path(cfg.grasp_target_path) if cfg.grasp_target_path else None
        if path is not None and path.exists() and (self._has_frame_index or cfg.grasp_dim == 0):
            # (A checkpoint loaded without the frame index, e.g. on the robot, only needs grasp_dim.)
            self._require_frame_index("grasp_target_path")
            grasp = np.load(path)
            if not np.array_equal(grasp["index"], index["index"]):
                raise ValueError(f"{path} was not built from the same frame index")
            cfg.grasp_dim = int(grasp["target"].shape[1])
            self.register_buffer("frame_grasp", torch.from_numpy(grasp["target"].astype(np.float32)), persistent=False)
            self._has_grasp_targets = True
        elif path is not None and cfg.grasp_dim == 0:
            raise FileNotFoundError(f"grasp_target_path not found: {path}")

    def _require_frame_index(self, what: str) -> None:
        if not self._has_frame_index:
            raise ValueError(f"{what} needs --policy.frame_index_path (output of build_frame_subtask_index.py)")

    def _centre(self, emb: Tensor) -> Tensor:
        return emb - self.language_mean if self.config.center_language else emb

    # ------------------------------------------------------------------ inference API
    def encode_text(self, text: str) -> Tensor:
        """SigLIP embedding of `text`: taken from the table when present, otherwise encoded on the fly."""
        key = _normalise_text(text)
        if key in self._row_of_text:
            return self.language_table[self._row_of_text[key]]
        if key not in self._encoded_cache:
            from transformers import AutoProcessor, SiglipModel

            if self._siglip is None:
                model_id = "google/siglip-base-patch16-224"
                self._siglip = (AutoProcessor.from_pretrained(model_id), SiglipModel.from_pretrained(model_id).eval())
            processor, model = self._siglip
            with torch.no_grad():
                feats = model.get_text_features(**processor(text=[text], padding="max_length", return_tensors="pt"))
            feats = feats if isinstance(feats, Tensor) else feats.pooler_output
            self._encoded_cache[key] = F.normalize(feats.float(), dim=1)[0].to(self.language_table.device)
        return self._encoded_cache[key]

    def set_instruction(self, text: str | None) -> None:
        """Set the instruction used at inference. Changing it flushes queued/ensembled actions."""
        if self.config.language_mode == "none":
            return
        if text is None:
            self._instruction = None
            return
        if self._instruction is None or self._instruction[0] != text:
            self._instruction = (text, self.encode_text(text))
            self.reset()

    # ------------------------------------------------------------------ batch preparation
    def _mask_image(self, key: str, img: Tensor) -> Tensor:
        """Blank the configured regions of a (B, C, H, W) normalised image (0 = dataset mean)."""
        boxes = self.config.image_masks.get(key)
        if not boxes:
            return img
        h, w = img.shape[-2:]
        keep = torch.ones(h, w, dtype=img.dtype, device=img.device)
        for x0, y0, x1, y1 in boxes:
            keep[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)] = 0
        return img * keep

    def _prepare(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        if self.config.image_features:
            unknown = set(self.config.image_masks) - set(self.config.image_features)
            if unknown:
                raise ValueError(f"image_masks for unknown cameras: {sorted(unknown)}")
            batch[OBS_IMAGES] = [self._mask_image(key, batch[key]) for key in self.config.image_features]
        if self.config.language_mode == "none":
            return batch
        batch_size = batch[OBS_STATE].shape[0]
        if self._instruction is not None:
            emb = self._instruction[1].to(batch[OBS_STATE].device).expand(batch_size, -1)
        elif "index" in batch:
            self._require_frame_index("Language lookup from the dataset index")
            rows_by_frame = self.frame_subtask_row if self.config.language_mode == "subtask" else self.frame_task_row
            emb = self.language_table[rows_by_frame[batch["index"].long()]]
        else:
            raise ValueError("No instruction available: call policy.set_instruction(text) before inference")
        batch[OBS_LANGUAGE] = self._centre(emb)
        if (self.config.progress_levels > 0 and self._instruction is None and "index" in batch
                and OBS_PROGRESS not in batch):
            # Dataset frame: true progress within its annotated segment (training / offline evaluation).
            batch[OBS_PROGRESS] = self.model.progress_level(self.frame_progress[batch["index"].long()])
        return batch

    @torch.no_grad()
    def _colour_swap(self, batch: dict[str, Tensor]) -> Tensor:
        """Permute the cube colours of some pick-phase samples (in place on the prepared batch).

        Returns (swapped, retargeted): (B,) bool masks of the samples whose colours were permuted, and of those whose
        target colour changed (a permutation may keep the target and only swap the other two cubes; the instruction
        and the whole action chunk then stay valid). See ACTSiglipConfig.colour_swap_prob.
        """
        cfg = self.config
        self._require_frame_index("colour_swap_prob")
        index = batch["index"].long()
        pick = self.frame_phase[index] == 0
        swap = pick & (torch.rand(pick.shape, device=pick.device) < cfg.colour_swap_prob)
        retarget = torch.zeros_like(swap)
        if not swap.any():
            return swap, retarget
        rows = swap.nonzero(as_tuple=True)[0]
        perm = PERMS.to(index.device)[torch.randint(len(PERMS), (len(rows),), device=index.device)]
        colour = self.frame_colour[index[rows]]
        new_colour = perm.gather(1, colour[:, None])[:, 0]
        retarget[rows] = new_colour != colour
        batch[OBS_LANGUAGE] = batch[OBS_LANGUAGE].clone()
        batch[OBS_LANGUAGE][rows] = self._centre(self.language_table[self.canonical_rows[new_colour, 0]]).to(
            batch[OBS_LANGUAGE].dtype)
        images = list(batch[OBS_IMAGES])
        for i, key in enumerate(self.config.image_features):
            if key in cfg.colour_swap_cameras:
                img = images[i].clone()
                protect = protect_mask(cfg.colour_swap_protect.get(key, []), *img.shape[-2:], img.device)
                img[rows] = recolour_normalised(img[rows], perm, protect)
                # Keep the masked regions blank (recolouring maps them back to ~0 anyway, this makes it exact).
                images[i] = self._mask_image(key, img)
            elif key in cfg.colour_swap_blank_cameras:
                img = images[i].clone()
                img[rows] = 0
                images[i] = img
        batch[OBS_IMAGES] = images
        return swap, retarget

    # ------------------------------------------------------------------ training / inference
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self._prepare(batch)
        if self.config.progress_levels > 0 and OBS_PROGRESS not in batch:
            # Robot: progress predicted from the images (state-free completion pass), then used by the action pass.
            n = batch[OBS_STATE].shape[0]
            floor = None
            if self.progress_ratchet:
                floor = torch.full((n,), self._progress_floor, device=batch[OBS_STATE].device)
            actions, _, completion_logits = self.model(batch, infer_progress=True, progress_floor=floor)
            self.last_progress = self.model.last_progress.mean().item()
            self._progress_floor = max(self._progress_floor, self.last_progress)
        else:
            actions, _, completion_logits = self.model(batch)
        if completion_logits is not None:
            self.last_completion = torch.sigmoid(completion_logits[0]).mean().item()
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self._prepare(batch)
        swapped = retargeted = None
        if self.training and self.config.colour_swap_prob > 0:
            swapped, retargeted = self._colour_swap(batch)
        head = self.config.completion_head
        extra, earlier_valid, m = (), None, None
        if head != "none":
            self._require_frame_index(f"completion_head={head!r}")
            batch_size = batch[OBS_STATE].shape[0]
            m = max(1, round(self.config.completion_batch_fraction * batch_size))
            index = batch["index"].long()[:m]  # the dataloader shuffles, so the first m samples are random
        if head == "done":
            # Show each frame with the canonical instruction of a random earlier phase (label "done").
            phase = self.frame_phase[index]
            earlier = (torch.rand(phase.shape, device=phase.device) * phase.float()).long()
            earlier_valid = phase > 0
            rows = self.canonical_rows[self.frame_colour[index], earlier]
            extra = (self._centre(self.language_table[rows]),)
        if self.config.progress_levels > 0 and self.training:
            # Robustness to the predicted progress: shift some levels by one, hide some ("unknown").
            levels = self.config.progress_levels
            level = batch[OBS_PROGRESS].clone()
            jitter = torch.rand(level.shape, device=level.device) < self.config.progress_jitter
            step = torch.where(torch.rand(level.shape, device=level.device) < 0.5, -1, 1)
            level = torch.where(jitter, (level + step).clamp(0, levels - 1), level)
            hide = torch.rand(level.shape, device=level.device) < self.config.progress_dropout
            batch[OBS_PROGRESS] = torch.where(hide, torch.full_like(level, levels), level)

        actions_hat, (mu_hat, log_sigma_x2_hat), completion_logits = self.model(batch, extra, n_completion=m)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        if self.config.drop_boundary_chunks:
            self._require_frame_index("drop_boundary_chunks")
            keep = ~self.frame_crosses_boundary[batch["index"].long()]
            valid_mask = valid_mask & keep[:, None, None]
        if retargeted is not None:
            # Samples whose target colour was changed: only the actions within the pick phase match the new
            # instruction (later ones head to the original colour's drawer).
            left = self.frame_to_segment_end[batch["index"].long()]
            in_pick = torch.arange(abs_err.shape[1], device=left.device)[None, :] <= left[:, None]
            valid_mask = valid_mask & (~retargeted[:, None] | in_pick)[:, :, None]
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}  # unweighted, comparable across runs
        if swapped is not None:
            loss_dict["colour_swapped"] = swapped.float().mean().item()
            loss_dict["colour_retargeted"] = retargeted.float().mean().item()
        loss = l1_loss
        sample_w = None  # per-sample action-loss weights
        if self.config.phase_loss_weights is not None:
            self._require_frame_index("phase_loss_weights")
            w = torch.tensor(self.config.phase_loss_weights, device=abs_err.device, dtype=abs_err.dtype)
            sample_w = w[self.frame_phase[batch["index"].long()]]
        if self.config.early_frames > 0 and self.config.early_frames_weight != 1:
            self._require_frame_index("early_frames")
            early = self.frame_in_episode[batch["index"].long()] < self.config.early_frames
            w = torch.where(early, self.config.early_frames_weight, 1.0).to(abs_err.dtype)
            sample_w = w if sample_w is None else sample_w * w
        if sample_w is not None:
            weighted = valid_mask * sample_w[:, None, None]
            loss = (abs_err * weighted).sum() / (weighted.sum() * abs_err.shape[-1]).clamp_min(1e-6)
            loss_dict["l1_weighted"] = loss.item()

        if self.config.grasp_dim > 0 and self.config.grasp_target_path:
            if not self._has_grasp_targets:
                raise ValueError("grasp head training needs grasp_target_path and frame_index_path")
            idx = batch["index"].long()
            pick = self.frame_phase[idx] == 0
            grasp_err = F.smooth_l1_loss(self.model.last_grasp, self.frame_grasp[idx], reduction="none").mean(-1)
            grasp_loss = (grasp_err * pick).sum() / pick.sum().clamp_min(1)
            loss_dict["grasp_loss"] = grasp_loss.item()
            loss = loss + self.config.grasp_loss_weight * grasp_loss
        if self.config.use_vae:
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss = loss + mean_kld * self.config.kl_weight

        if head == "done":
            own, other = completion_logits[0], completion_logits[1]
            loss_own = F.binary_cross_entropy_with_logits(own, torch.zeros_like(own))
            loss_other = F.binary_cross_entropy_with_logits(
                other[earlier_valid], torch.ones_like(other[earlier_valid])
            ) if earlier_valid.any() else own.sum() * 0
            completion_loss = 0.5 * (loss_own + loss_other)
            with torch.no_grad():
                correct = (own < 0).float().sum() + (other[earlier_valid] > 0).float().sum()
                loss_dict["done_acc"] = (correct / (own.numel() + earlier_valid.sum())).item()
        elif head == "progress":
            target = self.frame_progress[index]
            completion_loss = F.l1_loss(torch.sigmoid(completion_logits[0]), target)
        if head != "none":
            loss_dict["completion_loss"] = completion_loss.item()
            loss = loss + self.config.completion_loss_weight * completion_loss
        if self.config.progress_levels > 0:
            # Progress head, on the completion pass (no state, progress unknown) of the first m samples.
            target = self.frame_progress[index]
            progress_loss = F.l1_loss(self.model.last_progress, target)
            loss_dict["progress_loss"] = progress_loss.item()
            loss = loss + self.config.progress_loss_weight * progress_loss

        return loss, loss_dict
