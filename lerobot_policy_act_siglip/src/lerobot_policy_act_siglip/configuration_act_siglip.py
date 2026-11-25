from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

LANGUAGE_MODES = ("none", "task", "subtask")
COMPLETION_HEADS = ("auto", "none", "done", "progress")


@PreTrainedConfig.register_subclass("act_siglip")
@dataclass
class ACTSiglipConfig(ACTConfig):
    """ACT with an extra encoder token carrying a frozen SigLIP text embedding.

    Args (in addition to ACTConfig):
        language_mode: Which text conditions the policy.
            - "none":    plain ACT (baseline).
            - "task":    the episode-level command, constant over the episode.
            - "subtask": the subtask active at the current frame (main method).
        language_table_path: .npz from encode_subtask_vocabulary.py (texts + L2-normalised embeddings).
            The table is copied into the checkpoint, so it is only required to build a new policy.
        frame_index_path: .npz from build_frame_subtask_index.py. Maps the dataset's global `index`
            to table rows, phases and colours. Needed for training only.
        center_language: Subtract the table mean before projection. SigLIP text embeddings of the
            subtasks are very similar (cos("open the drawer", "close the drawer") = 0.96); removing
            the shared component leaves the part that distinguishes them.
        completion_head: Signal used by the planner to advance to the next subtask.
            - "done":     binary "is this instruction already completed?" read at the language token.
                          Trained with the frame's own instruction (label 0) and with the instruction of
                          an earlier phase (label 1). Requires language_mode="subtask".
            - "progress": regression of progress (0..1) through the current subtask.
            - "none":     no completion head.
            - "auto":     "done" for language_mode="subtask", "none" otherwise.
            Completion passes always use a zero latent, as at inference, so the head cannot read the
            future actions through the VAE latent.
        completion_loss_weight: Weight of the completion loss added to the ACT loss.
        completion_batch_fraction: Fraction of each batch used for the completion passes. The extra
            encoder pass over ~1200 tokens is memory-heavy (batch 8 with 4 cameras does not fit 16 GB at
            fraction 1.0); the completion task is simple, so a subset per step is enough.
        drop_boundary_chunks: Exclude from the action loss the samples whose action chunk crosses a
            subtask boundary (ablation; default keeps them so transitions are learned).
        film_layers: ResNet stages whose output is modulated by the (centred) language embedding with
            FiLM, feature * (1 + gamma) + beta, e.g. ["layer2", "layer3", "layer4"]. Makes the visual
            features depend on the instruction, so the policy can't route around the text (a single
            language token among ~1200 image tokens was ignored). Initialised to identity. Empty = off.
        state_dropout: Probability, per training sample, of replacing the robot-state token with a learned
            "no state" embedding in the action pass. Forces the target to be found in the images (+ text)
            instead of being read from the arm pose once the operator commits.
        completion_use_state: If False the completion head never sees the robot state (always the "no
            state" embedding), so "done" must be judged from the images (cube in gripper, drawer open)
            rather than from the arm pose.
        phase_loss_weights: Per-phase weight of the action loss (pick, move, open, place, close), e.g.
            [3, 1, 1, 1, 1]: pick frames are where only the text tells which cube is the target.
        early_frames: Frames with frame_index < early_frames (start of the episode, when the arm is still at
            the common start pose) get their action loss multiplied by early_frames_weight. 0 = off.
        early_frames_weight: See early_frames.
        grasp_target_path: .npz from build_grasp_targets.py. Turns on an auxiliary head that predicts, from
            the encoder output at the language token, where the grasp of the current pick will be (picking
            arm's joints at the end of the pick phase). Trained on pick-phase frames only; not used at
            inference. The cubes are shuffled per episode, so early in the pick this is only predictable by
            locating, in the images, the cube named in the text. Needed for training only.
        grasp_loss_weight: Weight of the grasp loss.
        image_masks: Regions blanked out of the camera images, {camera key: [[x0, y0, x1, y1], ...]} in
            fractions of width / height. Applied inside the policy (training, evaluation and robot alike), after
            normalisation, so masked pixels become the dataset mean. Used to hide the background (people,
            chairs, loose objects) that lets the network recognise and memorise individual episodes.
        colour_swap_prob: Probability, per pick-phase training sample, of permuting the cube colours (colour_swap.py):
            the cubes in `colour_swap_cameras` are recoloured, the instruction becomes the canonical pick text of the
            new target colour, and the actions are kept (same cube, now with the instructed colour). Breaks the
            "this scene -> target here" memorisation. Actions of the chunk past the end of the pick phase are left
            out of the loss for these samples (they head to the original drawer). Training only. 0 = off.
        colour_swap_cameras: Cameras whose images are recoloured (the static ones, where the drawers can be protected).
        colour_swap_protect: Regions never recoloured, {camera key: [[x0, y0, x1, y1], ...]} (the drawer cabinet). A
            5th value restricts a box to one colour (0 red, 1 green, 2 yellow), e.g. the band of the red drawer, so
            a cube of another colour in front of it is still recoloured.
        colour_swap_blank_cameras: Cameras blanked (dataset mean) in swapped samples, e.g. the wrist cameras, which
            move and may see the drawers anywhere; otherwise they would show the original colours.
        progress_levels: Progress within the current subtask as an extra encoder token (as in Bi-HIL, where the
            low-level policy is conditioned on the subtask progress rate, discretised into 10 levels). The token
            embeds one of `progress_levels` levels, or an extra "unknown" level. In training the level comes from the
            frame index (position within the annotated segment); at inference it comes from a progress head that
            reads the state-free completion pass (with progress "unknown"), run before the action pass, and is kept
            non-decreasing within a subtask. Gives the policy the notion of time it lacks when the scene stops
            changing (e.g. "wait, then release"). Requires a completion head. 0 = off.
        progress_dropout: Probability, per training sample, of giving the action pass the "unknown" progress level.
        progress_jitter: Probability, per training sample, of shifting the true level by one (either way), so the
            policy tolerates the error of the predicted progress at inference.
        progress_loss_weight: Weight of the progress head loss (L1 on the progress in [0, 1]).
        grasp_dim: Size of the grasp target (filled from the file; stored so the head can be rebuilt on load).
        language_texts: Filled automatically from the table; stored in the saved config so the
            language buffers can be allocated when loading a checkpoint.
        language_dim: Dimension of the text embeddings (filled from the table).
    """

    chunk_size: int = 50
    n_action_steps: int = 50

    language_mode: str = "subtask"
    language_table_path: str | None = None
    frame_index_path: str | None = None
    center_language: bool = True
    completion_head: str = "auto"
    completion_loss_weight: float = 1.0
    completion_batch_fraction: float = 0.25
    drop_boundary_chunks: bool = False
    film_layers: list[str] = field(default_factory=list)
    state_dropout: float = 0.0
    completion_use_state: bool = True
    phase_loss_weights: list[float] | None = None
    early_frames: int = 0
    early_frames_weight: float = 1.0
    grasp_target_path: str | None = None
    grasp_loss_weight: float = 1.0
    grasp_dim: int = 0
    image_masks: dict[str, list[list[float]]] = field(default_factory=dict)
    colour_swap_prob: float = 0.0
    colour_swap_cameras: list[str] = field(default_factory=list)
    colour_swap_protect: dict[str, list[list[float]]] = field(default_factory=dict)
    colour_swap_blank_cameras: list[str] = field(default_factory=list)
    progress_levels: int = 0
    progress_dropout: float = 0.2
    progress_jitter: float = 0.3
    progress_loss_weight: float = 1.0
    language_texts: list[str] = field(default_factory=list)
    language_dim: int = 768

    def __post_init__(self):
        super().__post_init__()
        if self.language_mode not in LANGUAGE_MODES:
            raise ValueError(f"language_mode must be one of {LANGUAGE_MODES}, got {self.language_mode!r}")
        if self.completion_head not in COMPLETION_HEADS:
            raise ValueError(f"completion_head must be one of {COMPLETION_HEADS}, got {self.completion_head!r}")
        if self.completion_head == "auto":
            self.completion_head = "done" if self.language_mode == "subtask" else "none"
        if not 0 < self.completion_batch_fraction <= 1:
            raise ValueError("completion_batch_fraction must be in (0, 1]")
        if self.completion_head == "done" and self.language_mode != "subtask":
            raise ValueError("completion_head='done' is conditioned on the subtask: use language_mode='subtask'")
        valid_layers = ("layer1", "layer2", "layer3", "layer4")
        if any(layer not in valid_layers for layer in self.film_layers):
            raise ValueError(f"film_layers must be a subset of {valid_layers}, got {self.film_layers}")
        if self.film_layers and self.language_mode == "none":
            raise ValueError("film_layers needs language: set language_mode to 'task' or 'subtask'")
        if not 0 <= self.state_dropout < 1:
            raise ValueError("state_dropout must be in [0, 1)")
        if self.phase_loss_weights is not None and len(self.phase_loss_weights) != 5:
            raise ValueError("phase_loss_weights needs 5 values (pick, move, open, place, close)")
        if self.early_frames < 0 or self.early_frames_weight <= 0:
            raise ValueError("early_frames must be >= 0 and early_frames_weight > 0")
        for name, regions in (("image_masks", self.image_masks), ("colour_swap_protect", self.colour_swap_protect)):
            for cam, boxes in regions.items():
                for box in boxes:
                    sizes = (4, 5) if name == "colour_swap_protect" else (4,)
                    if len(box) not in sizes or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                        raise ValueError(f"{name}[{cam!r}]: boxes are [x0, y0, x1, y1] fractions, got {box}")
                    if len(box) == 5 and box[4] not in (0, 1, 2):
                        raise ValueError(f"{name}[{cam!r}]: 5th value is a colour index (0 red, 1 green, 2 yellow)")
        if not 0 <= self.colour_swap_prob <= 1:
            raise ValueError("colour_swap_prob must be in [0, 1]")
        if self.colour_swap_prob > 0:
            if self.language_mode != "subtask":
                raise ValueError("colour_swap_prob swaps the pick instruction: use language_mode='subtask'")
            if not self.colour_swap_cameras:
                raise ValueError("colour_swap_prob > 0 needs colour_swap_cameras")
            if set(self.colour_swap_cameras) & set(self.colour_swap_blank_cameras):
                raise ValueError("a camera can't be both recoloured and blanked")
        if self.progress_levels < 0:
            raise ValueError("progress_levels must be >= 0")
        if self.progress_levels > 0:
            if self.completion_head == "none":
                raise ValueError("progress_levels > 0 predicts the progress from the completion pass: needs a completion head")
            if self.completion_head == "progress":
                raise ValueError("progress_levels > 0 adds its own progress head: use completion_head='done'")
            if not (0 <= self.progress_dropout < 1 and 0 <= self.progress_jitter <= 1):
                raise ValueError("progress_dropout must be in [0, 1) and progress_jitter in [0, 1]")
        if (self.grasp_target_path or self.grasp_dim) and self.language_mode == "none":
            raise ValueError("the grasp head reads the language token: set language_mode to 'task' or 'subtask'")
