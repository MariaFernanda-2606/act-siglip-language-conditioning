"""Run the language-conditioned ACT (act_siglip) + subtask planner on the Trossen AI Stationary robot.

Runs in the `act_siglip` env (LeRobot 0.5.2). Talks to the hardware directly with the same settings as the
trossen fork driver (lerobot/common/robot_devices/motors/trossen_arm_driver.py): follower arms only,
position mode, home pose before the episode, home -> sleep at the end, min_time_to_move = 3 / fps.

Modes (use them in this order the first time):
  check    cameras + arms only: goes to home, prints the state, saves live camera images next to dataset
           images (check_cameras.png) so placement/colour can be compared. No policy.
  dry-run  full loop with the policy, but actions are only printed/logged, never sent. Arms stay at home.
  run      full loop, actions are sent to the arms.
  eval     `--episodes` trials of the same --command in a row (operator resets the scene between trials and
           labels how far each trial got); metrics in <out-dir>/eval/<--eval-name>/ (trials.csv, summary.json);
           with --hf-repo the trials are also recorded as a LeRobot dataset and pushed to the Hub at the end.
           In eval, q+Enter ends the current trial (not the session); Ctrl+C ends the session.

While running, type in the terminal + Enter:   n = force next subtask,   q = stop (arms go home -> sleep).
Ctrl+C also stops safely.

Safety layers on every command sent:
  * clip to the dataset action range (meta/stats.json min/max) +- a small margin;
  * per-step change limit vs the present position (--max-step-rad for joints, --max-step-gripper);
  * --max-seconds episode limit.

Example:
  python scripts/run_act_siglip_robot.py --mode check
  python scripts/run_act_siglip_robot.py --mode dry-run \
      --policy-path outputs/train/act_siglip_subtask_v3/checkpoints/100000/pretrained_model \
      --command "place the green cube in the green drawer"
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

CAMERAS = {  # name -> RealSense serial (from the trossen fork configs.py, TrossenAIStationaryRobotConfig)
    "cam_high": "218622275519",
    "cam_low": "218622275952",
    "cam_left_wrist": "218622274511",
    "cam_right_wrist": "218622270778",
}
ARMS = {"left": "192.168.1.5", "right": "192.168.1.4"}  # follower arms; state/action order = left then right
HOME_POSE = [0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0, 0.0]
SLEEP_POSE = [0.0] * 7
GRIPPER_DIMS = (6, 13)
DEFAULT_ROOT = ("/home/trossen-ai/.cache/huggingface/lerobot/MariaFPG/"
                "place_the_cubes_in_the_drawers_v3_annotated")


# ----------------------------------------------------------------------------------------- hardware
class RealSenseCamera:
    """Colour stream (RGB, 640x480) read in a background thread; read() returns the latest frame."""

    def __init__(self, name: str, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        import pyrealsense2 as rs

        self.name = name
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipeline.start(cfg)
        self._frame, self._lock, self._stop = None, threading.Lock(), threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            color = frames.get_color_frame()
            if color:
                with self._lock:
                    self._frame = np.asanyarray(color.get_data()).copy()

    def read(self, timeout_s: float = 5.0) -> np.ndarray:
        t0 = time.time()
        while True:
            with self._lock:
                if self._frame is not None:
                    return self._frame
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"camera {self.name}: no frame after {timeout_s}s")
            time.sleep(0.01)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.pipeline.stop()


class FollowerArm:
    def __init__(self, name: str, ip: str, fps: int, min_time_multiplier: float):
        import trossen_arm as trossen

        self.trossen, self.name, self.ip = trossen, name, ip
        self.min_time = min_time_multiplier / fps
        print(f"connecting {name} arm at {ip} ...")
        self.driver = trossen.TrossenArmDriver()
        self.driver.configure(trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_follower, ip, True)
        self.driver.set_all_modes(trossen.Mode.position)

    def read(self) -> np.ndarray:
        return np.asarray(self.driver.get_all_positions(), dtype=np.float32)

    def write(self, q: np.ndarray):
        self.driver.set_all_positions(np.asarray(q, dtype=np.float32).tolist(), self.min_time, False)

    def go(self, pose, seconds: float = 2.0, blocking: bool = True):
        self.driver.set_all_positions(list(pose), seconds, blocking)

    def stop_motion(self):
        t = self.trossen
        self.driver.set_all_modes(t.Mode.velocity)
        self.driver.set_all_velocities([0.0] * self.driver.get_num_joints(), 0.0, False)
        self.driver.set_all_modes(t.Mode.position)


class Robot:
    def __init__(self, fps: int, min_time_multiplier: float, with_cameras: bool = True):
        self.arms, self.cameras = {}, {}
        try:
            for name, ip in ARMS.items():
                self.arms[name] = FollowerArm(name, ip, fps, min_time_multiplier)
            if with_cameras:
                for name, serial in CAMERAS.items():
                    # Cameras always stream at 30 fps (as recorded; RealSense only supports fixed rates);
                    # --fps only sets the control rate, the loop always takes the latest frame.
                    self.cameras[name] = RealSenseCamera(name, serial, fps=30)
        except Exception:
            self.close(park=False)
            raise

    def state(self) -> np.ndarray:
        return np.concatenate([self.arms[n].read() for n in ARMS])

    def images(self) -> dict[str, np.ndarray]:
        return {n: c.read() for n, c in self.cameras.items()}

    def send(self, action: np.ndarray):
        for i, n in enumerate(ARMS):
            self.arms[n].write(action[7 * i: 7 * (i + 1)])

    def home(self):
        for a in self.arms.values():
            a.go(HOME_POSE, 2.0, blocking=False)
        time.sleep(2.2)

    def close(self, park: bool = True):
        if park and self.arms:
            print("parking arms: home -> sleep")
            for a in self.arms.values():
                a.stop_motion()
            self.home()
            for a in self.arms.values():
                a.go(SLEEP_POSE, 2.0, blocking=False)
            time.sleep(2.2)
        for c in self.cameras.values():
            c.close()


# ----------------------------------------------------------------------------------------- helpers
class KeyboardCommands:
    """Reads lines from stdin in a thread: 'n' = next subtask, 'q' = quit."""

    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        for line in sys.stdin:
            self.q.put(line.strip().lower())

    def poll(self) -> list[str]:
        out = []
        while not self.q.empty():
            out.append(self.q.get_nowait())
        return out


class SafetyFilter:
    def __init__(self, stats_path: Path, margin_rad: float, margin_gripper: float,
                 max_step_rad: float, max_step_gripper: float):
        stats = json.loads(stats_path.read_text())["action"]
        lo, hi = np.array(stats["min"], np.float32), np.array(stats["max"], np.float32)
        margin = np.full(14, margin_rad, np.float32)
        margin[list(GRIPPER_DIMS)] = margin_gripper
        self.lo, self.hi = lo - margin, hi + margin
        self.max_step = np.full(14, max_step_rad, np.float32)
        self.max_step[list(GRIPPER_DIMS)] = max_step_gripper
        self.n_clipped_range = self.n_clipped_step = 0

    def __call__(self, action: np.ndarray, present: np.ndarray) -> np.ndarray:
        a = np.clip(action, self.lo, self.hi)
        self.n_clipped_range += int(not np.allclose(a, action))
        b = present + np.clip(a - present, -self.max_step, self.max_step)
        self.n_clipped_step += int(not np.allclose(a, b))
        return b


def check_mode(robot: Robot, root: Path, out_dir: Path):
    import cv2
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    np.set_printoptions(precision=3, suppress=True)
    state = robot.state()
    print(f"state left : {state[:7]}\nstate right: {state[7:]}")
    print(f"home pose  : {np.array(HOME_POSE)}  (all training episodes start exactly here)")
    print(f"max |state - home| = {np.abs(state - np.tile(HOME_POSE, 2)).max():.3f} rad")

    live = robot.images()
    ds = LeRobotDataset("local", root=root, episodes=[0])
    ref = ds[0]
    rows = []
    for name in CAMERAS:
        a = live[name]
        b = (ref[f"observation.images.{name}"].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).numpy()
        col = np.concatenate([a, b], 0)
        cv2.putText(col, f"{name} LIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(col, f"{name} DATASET ep0", (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        rows.append(col)
        print(f"{name:16s} live mean RGB {a.reshape(-1, 3).mean(0).round(1)}  dataset {b.reshape(-1, 3).mean(0).round(1)}")
    grid = np.concatenate(rows, 1)
    path = out_dir / "check_cameras.png"
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"saved {path}  (top: live, bottom: dataset episode 0 frame 0)")


def load_legacy_act(path: Path, device: str, n_action_steps: int):
    """Plain ACT checkpoint trained with the trossen fork (LeRobot v2.1 format: normalisation buffers inside
    model.safetensors). Loaded into the upstream ACT (identical module names) with the same pre/post interface,
    so a known-good policy can be run through this exact runner. No language: the instruction is ignored."""
    import json

    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from safetensors.torch import load_file

    raw = json.loads((path / "config.json").read_text())
    kw = {k: v for k, v in raw.items() if k in ACTConfig.__dataclass_fields__
          and k not in ("type", "input_features", "output_features", "normalization_mapping", "device")}
    cfg = ACTConfig(**kw)
    cfg.input_features = {k: PolicyFeature(FeatureType[v["type"]], tuple(v["shape"])) for k, v in raw["input_features"].items()}
    cfg.output_features = {k: PolicyFeature(FeatureType[v["type"]], tuple(v["shape"])) for k, v in raw["output_features"].items()}
    cfg.device, cfg.temporal_ensemble_coeff, cfg.n_action_steps = device, None, n_action_steps
    sd = load_file(str(path / "model.safetensors"))

    class LegacyACT(ACTPolicy):
        last_completion = None

        def set_instruction(self, text):
            pass

    policy = LegacyACT(cfg)
    policy.model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)
    policy.to(device).eval()
    norm = {k.split("buffer_")[1].rsplit(".", 1)[0] + "." + k.rsplit(".", 1)[1]: v.to(device)
            for k, v in sd.items() if k.startswith("normalize_inputs.") or k.startswith("unnormalize_outputs.")}

    def pre(obs):
        out = {}
        for k, v in obs.items():
            key = k.replace(".", "_")
            out[k] = (v - norm[key + ".mean"]) / (norm[key + ".std"] + 1e-8) if key + ".mean" in norm else v
        return out

    def post(action):
        return action * norm["action.std"] + norm["action.mean"]

    print(f"legacy ACT loaded: chunk {cfg.chunk_size}, n_action_steps {cfg.n_action_steps} (language ignored)")
    return policy, pre, post


def load_policy(path: Path, device: str, n_action_steps: int, legacy: bool = False):
    if legacy:
        return load_legacy_act(path, device, n_action_steps)
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    from lerobot_policy_act_siglip import ACTSiglipPolicy

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.language_table_path = cfg.frame_index_path = None  # everything needed is in the checkpoint
    cfg.device = device
    cfg.temporal_ensemble_coeff = None  # ensembling (if any) is done here, asynchronously
    cfg.n_action_steps = n_action_steps
    policy = ACTSiglipPolicy.from_pretrained(path, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def make_batch(pre, state: np.ndarray, images: dict[str, np.ndarray], device, command: str):
    """Same result as prepare_observation_for_inference + preprocessor, but images go to the GPU as uint8 (faster)."""
    import torch

    obs = {"observation.state": torch.from_numpy(state.astype(np.float32))[None].to(device)}
    for n, im in images.items():
        obs[f"observation.images.{n}"] = torch.from_numpy(im).to(device).permute(2, 0, 1)[None].float() / 255
    obs["task"], obs["robot_type"] = command, ""
    return pre(obs)


def smooth_chunk(chunk: np.ndarray, sigma: float) -> np.ndarray:
    """--smooth-chunk: Gaussian filter along time on the arm joints of one predicted chunk (grippers untouched,
    so the release pulse stays sharp). Centred over the planned future trajectory -> no lag. The v7 chunks zig-zag
    step to step (yellow eval 25/11: 26 velocity reversals/s over the 12 joints vs 0.3 in the demos); sigma 1 frame
    removes them, cuts the p99 acceleration 6x and moves the trajectory by 0.002 rad median (p99 0.013)."""
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    j = [i for i in range(chunk.shape[1]) if i not in GRIPPER_DIMS]
    x = chunk[:, j]
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)], 0)  # hold the end values
    out = chunk.copy()
    out[:, j] = sum(k[i] * pad[i:i + len(x)] for i in range(len(k)))
    return out


class ChunkCrossfader:
    """Chunk execution with a sigmoid crossfade between consecutive chunks (as in Flash ALOHA).

    A new chunk is predicted every `replan_every` frames (or at once after an instruction change). The old chunk
    still holds a plan for the next frames, so for the first `crossfade` frames of the new chunk the command is
    (1 - w) * old + w * new, with w going 0 -> 1 along a sigmoid: the arm moves from one plan to the other
    instead of jumping. replan_every + crossfade must fit in the chunk (the old plan has to cover the blend).
    """

    def __init__(self, predict, chunk_size: int, replan_every: int, crossfade: int, steepness: float = 10.0,
                 unblended: tuple[int, ...] = ()):
        if replan_every + crossfade > chunk_size:
            raise ValueError(f"--replan-every ({replan_every}) + --crossfade ({crossfade}) must be <= chunk size "
                             f"({chunk_size}), e.g. --replan-every {chunk_size - crossfade}")
        self.predict, self.replan_every = predict, replan_every  # predict(state) -> (chunk (T, D), completion score)
        # Dimensions taken from the new chunk only (no blend), e.g. the grippers: blending the old chunk (gripper
        # still closed) into the new one flattens the release ramp and the cube is never let go.
        self.unblended = list(unblended)
        t = np.arange(1, crossfade + 1) / (crossfade + 1)
        s, s0, s1 = (1 / (1 + np.exp(-steepness * (x - 0.5))) for x in (t, 0.0, 1.0))
        self.weights = (s - s0) / (s1 - s0)  # weight of the new chunk, increasing from ~0 to ~1
        self.cur = self.prev = None
        self.i = 0
        self.replan_now = False

    def request_replan(self):
        self.replan_now = True

    def action(self, state: np.ndarray):
        score = None
        if self.cur is None or self.i >= self.replan_every or self.replan_now:
            new, score = self.predict(state)
            self.prev = None if self.cur is None else self.cur[self.i:]  # old plan for the coming frames
            self.cur, self.i, self.replan_now = new, 0, False
        a = self.cur[self.i]
        if self.prev is not None and self.i < min(len(self.weights), len(self.prev)):
            w = self.weights[self.i]
            new = a
            a = (1 - w) * self.prev[self.i] + w * new
            if self.unblended:
                a[self.unblended] = new[self.unblended]
        self.i += 1
        return a, score


class ReleaseAssist:
    """--release-assist: execution heuristic for the place subtask (NOT learned by the policy).

    In the demos the operator first opens the left gripper slowly (command ~0.015 while the cube still holds the
    fingers at ~0.024), then opens it fully (~0.031) and the cube drops. On the robot the policy can get stuck in that
    first stage: right pose, command 0.010-0.015, never past 0.024. This latches the gripper open once the mean
    command over the last `hold_s` seconds is above `threshold` AND the left arm has settled (joint speed over the
    last 0.5 s below `max_speed` rad/s), so it can't fire while the arm is still carrying the cube to the drawer.
    Active only during the place subtask; reset at every subtask change.

    --release-timeout: second trigger (independent of the command): over the last `timeout_s` seconds every left-arm
    joint stayed within `still_tol` rad (max - min) and the gripper state never showed the cube released
    (> `released_thr`). A speed threshold was too loose: on the robot the arm can creep towards the drawer at
    0.03-0.09 rad/s for 5 s and still be 0.2 rad from the release pose. Replay of the 24/11 robot logs with 5 s /
    0.05 rad: fires in all 5 stuck runs, never before a release the policy made by itself. 25/11 sweep: 4 s keeps
    that (v7 runs) and fires 1-23 s sooner in the stuck runs; demos: 1/236 train eps fire early (5 s: 0); 3 s fires
    in 174825 4.6 s before the policy's own release with the arm 0.2 rad away, and in 13 demos.
    threshold 0 disables the command trigger, timeout_s 0 the timeout trigger.
    """

    GRIPPER, ARM = 6, slice(0, 6)  # left arm (the one carrying the cube)

    def __init__(self, fps: int, threshold: float, hold_s: float, max_speed: float, open_value: float,
                 timeout_s: float = 0.0, released_thr: float = 0.026, still_tol: float = 0.05):
        from collections import deque
        self.threshold, self.max_speed, self.open_value = threshold, max_speed, open_value
        self.n_timeout = round(timeout_s * fps) if timeout_s > 0 else 0
        self.released_thr, self.still_tol = released_thr, still_tol
        self.reason = None
        self.n_cmd, self.n_speed = max(1, round(hold_s * fps)), max(1, round(0.5 * fps))
        self.speed_dt = self.n_speed / fps
        self._deque = deque
        self.reset()

    def reset(self):
        self.cmds = self._deque(maxlen=self.n_cmd)
        self.states = self._deque(maxlen=self.n_speed + 1)
        self.open = False
        self.window = self._deque(maxlen=max(1, self.n_timeout))  # left-arm joints over the timeout window
        self.released = False  # the policy released the cube by itself

    def __call__(self, action: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, bool]:
        """Returns (action to send, True if it fired at this step)."""
        fired = False
        if not self.open:
            self.cmds.append(float(action[self.GRIPPER])); self.states.append(state[self.ARM].copy())
            self.released |= bool(state[self.GRIPPER] > self.released_thr)
            self.window.append(state[self.ARM].copy())
            if self.n_timeout and not self.released and len(self.window) == self.n_timeout:
                w = np.asarray(self.window)
                if (w.max(0) - w.min(0)).max() <= self.still_tol:
                    self.open = fired = True
                    self.reason = "timeout"
            if not self.open and len(self.states) == self.n_speed + 1:
                speed = np.linalg.norm(self.states[-1] - self.states[0]) / self.speed_dt
                if (self.threshold > 0 and len(self.cmds) == self.n_cmd and np.mean(self.cmds) > self.threshold
                        and speed < self.max_speed):
                    self.open = fired = True
                    self.reason = "command"
        if self.open:
            action = action.copy()
            action[self.GRIPPER] = self.open_value
        return action, fired


class AsyncEnsembler:
    """Runs the policy in a background thread on the latest observation, as fast as the GPU allows.

    Each predicted chunk is stored with the frame of the observation it was computed from; the action for
    frame t is the weighted mean of the chunks computed in the last `window` frames (chunk[t - start]), with
    ACT's temporal ensembling weights exp(-coeff * i), i = 0 for the oldest chunk. window = chunk_size is the
    original ACT scheme (smoothest, most lag); a short window trades a little smoothness for less lag. The control loop never waits for the GPU.
    Changing the instruction drops all chunks (the robot holds still until the first new chunk arrives).
    """

    def __init__(self, policy, pre, post, device, command: str, coeff: float, window: int):
        import torch

        self.torch, self.policy, self.pre, self.post = torch, policy, pre, post
        self.device, self.command, self.coeff = device, command, coeff
        self.chunk_size = policy.config.chunk_size
        self.window = min(window, self.chunk_size)  # only chunks computed in the last `window` frames are averaged
        self.lock = threading.Lock()
        self.new_obs = threading.Condition(self.lock)
        self.obs = None            # (frame, state, images)
        self.instruction = None
        self.version = 0           # bumped on instruction change; stale chunks are discarded
        self.chunks: list[tuple[int, np.ndarray]] = []
        self.completion = None     # latest completion score not yet consumed
        self.n_inferences, self.infer_ms = 0, []
        self.error = None
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set_instruction(self, text: str):
        with self.lock:
            self.instruction = text
            self.version += 1
            self.chunks.clear()
            self.completion = None

    def submit(self, frame: int, state: np.ndarray, images: dict[str, np.ndarray]):
        with self.lock:
            self.obs = (frame, state, images)
            self.new_obs.notify()

    def action(self, frame: int) -> np.ndarray | None:
        with self.lock:
            usable = [(s, c) for s, c in self.chunks if s <= frame < s + self.window]
        if not usable:
            return None
        acts = np.stack([c[frame - s] for s, c in usable])          # oldest first
        w = np.exp(-self.coeff * np.arange(len(usable)))[:, None]
        return (acts * w).sum(0) / w.sum()

    def pop_completion(self) -> float | None:
        with self.lock:
            c, self.completion = self.completion, None
        return c

    def stop(self):
        with self.lock:
            self._stop = True
            self.new_obs.notify()
        self.thread.join(timeout=2)

    def _loop(self):
        last_frame = -1
        try:
            while True:
                with self.lock:
                    while not self._stop and (self.obs is None or self.obs[0] == last_frame or self.instruction is None):
                        self.new_obs.wait(timeout=0.5)
                    if self._stop:
                        return
                    frame, state, images = self.obs
                    instruction, version = self.instruction, self.version
                last_frame = frame
                t0 = time.perf_counter()
                with self.torch.inference_mode():
                    self.policy.set_instruction(instruction)
                    batch = make_batch(self.pre, state, images, self.device, self.command)
                    chunk = self.post(self.policy.predict_action_chunk(batch))[0].float().cpu().numpy()
                score = self.policy.last_completion
                self.infer_ms.append((time.perf_counter() - t0) * 1000)
                with self.lock:
                    if version != self.version:
                        continue  # instruction changed meanwhile
                    self.chunks = [(s, c) for s, c in self.chunks if s + self.chunk_size > frame] + [(frame, chunk)]
                    self.completion = score
                    self.n_inferences += 1
        except Exception as e:  # surfaced in the control loop
            self.error = e


# ----------------------------------------------------------------------------------------- main loop
def prepare(args):
    """Load the policy once (eval mode reuses it for every trial)."""
    import torch

    ensemble = args.temporal_ensemble is not None
    policy, pre, post = load_policy(args.policy_path, args.device, args.replan_every, args.legacy_act)
    if hasattr(policy, "progress_ratchet"):
        policy.progress_ratchet = not args.no_progress_ratchet
    uses_progress = getattr(getattr(policy, "config", None), "progress_levels", 0) > 0
    safety = SafetyFilter(Path(args.root) / "meta/stats.json", args.margin_rad, args.margin_gripper,
                          args.max_step_rad, args.max_step_gripper)
    print("action generation:", f"async temporal ensembling (coeff {args.temporal_ensemble}, window {args.ensemble_window} frames)" if ensemble
          else f"chunks, re-plan every {args.replan_every} frames"
          + (f", sigmoid crossfade over {args.crossfade} frames" if args.crossfade > 0 else ""),
          f"| planner threshold {args.threshold}, patience {args.patience} completion scores")
    return dict(policy=policy, pre=pre, post=post, device=torch.device(args.device), safety=safety,
                uses_progress=uses_progress, ensemble=ensemble)


def run_trial(args, robot: Robot, ctx: dict, command: str, out_dir: Path, start, recorder=None,
              tag: str = "") -> dict:
    """One episode: planner + policy until the plan completes, q, Ctrl+C or --max-seconds.

    `start()` is called right before the loop (home, confirmation...) and returns the KeyboardCommands.
    `recorder` (eval mode) gets every executed frame. The arms are left where they are at the end.
    """
    import torch
    from lerobot_policy_act_siglip import SubtaskPlanner

    policy, pre, post, device = ctx["policy"], ctx["pre"], ctx["post"], ctx["device"]
    safety, uses_progress, ensemble = ctx["safety"], ctx["uses_progress"], ctx["ensemble"]
    safety.n_clipped_range = safety.n_clipped_step = 0
    send = args.mode in ("run", "eval")
    planner = SubtaskPlanner(command, threshold=args.threshold, patience=args.patience)
    print(f"plan: {planner.subtasks}")

    # Warm-up (CUDA kernels), then start clean.
    policy.set_instruction(planner.current)
    with torch.inference_mode():
        policy.predict_action_chunk(make_batch(pre, robot.state(), robot.images(), device, command))
    policy.reset()
    policy.last_completion = None

    def sync_action(state):
        """Chunk mode: query the policy only when its action queue is empty."""
        with torch.inference_mode():
            batch = None
            if len(policy._action_queue) == 0:
                batch = make_batch(pre, state, robot.images(), device, command)
            action = post(policy.select_action(batch))
        score = policy.last_completion
        policy.last_completion = None
        return action.squeeze(0).float().cpu().numpy(), score

    ens = AsyncEnsembler(policy, pre, post, device, command, args.temporal_ensemble, args.ensemble_window) if ensemble else None

    chunk_log = {k: [] for k in ["chunk_step", "chunk_subtask", "chunk_used", "chunks"]}

    def predict_chunk(state, used=True):
        with torch.inference_mode():
            chunk = post(policy.predict_action_chunk(make_batch(pre, state, robot.images(), device, command)))
        score = policy.last_completion
        policy.last_completion = None
        chunk = chunk[0].float().cpu().numpy()
        if args.smooth_chunk > 0:
            chunk = smooth_chunk(chunk, args.smooth_chunk)
        if args.log_chunks:  # every predicted chunk (executed or only scored), unnormalised, as the robot would get it
            chunk_log["chunk_step"].append(step); chunk_log["chunk_subtask"].append(planner.step)
            chunk_log["chunk_used"].append(used); chunk_log["chunks"].append(chunk.astype(np.float32))
        return chunk, score

    def completion_only(state):
        """--done-every: one extra policy pass between re-plans, only to read the completion score (chunk dropped)."""
        return predict_chunk(state, used=False)[1]

    # --log-chunks needs the chunks in hand: with --crossfade 0 the crossfader just executes them (same as the queue).
    fader = (ChunkCrossfader(predict_chunk, policy.config.chunk_size, args.replan_every, args.crossfade,
                             unblended=(6, 13) if args.crossfade_joints_only else ())
             if (args.crossfade > 0 or args.log_chunks or args.smooth_chunk > 0) and not ensemble else None)
    assist = (ReleaseAssist(args.fps, args.release_assist, args.release_hold, args.release_max_speed, args.release_open,
                            args.release_timeout, args.release_state_thr, args.release_still_tol)
              if args.release_assist > 0 or args.release_timeout > 0 else None)
    if assist is not None and args.release_assist > 0:
        print(f"release assist ON (heuristic): place subtask, mean left-gripper cmd over {args.release_hold}s > "
              f"{args.release_assist} and left arm speed < {args.release_max_speed} rad/s -> gripper {args.release_open}")
    if assist is not None and args.release_timeout > 0:
        print(f"release timeout ON (rule): place subtask, left arm still (every joint within {args.release_still_tol} "
              f"rad) for {args.release_timeout}s without releasing -> gripper {args.release_open}")

    def set_instruction(text):
        if assist is not None:
            assist.reset()
        if ensemble:
            ens.set_instruction(text)
        else:
            policy.set_instruction(text)
            if fader is not None:
                fader.request_replan()

    keys = start()
    print("type n+Enter = next subtask, q+Enter = stop" + (" this trial" if args.mode == "eval" else ""))
    set_instruction(planner.current)

    log = {k: [] for k in ["t", "state", "action", "sent", "subtask", "completion", "progress", "forced", "assist",
                           "release_switch", "loop_dt"]}
    dt = 1.0 / args.fps
    t_start = time.perf_counter()
    next_tick = t_start
    step = 0
    n_hold = 0
    n_forced = 0
    t_release = None  # --close-after-release: time the left gripper first opened during "place"
    close_by_release = False
    assist_fired = False
    stop_reason = "max_seconds"
    try:
        while time.perf_counter() - t_start < args.max_seconds:
            if ens is not None and ens.error is not None:
                raise RuntimeError("inference thread failed") from ens.error
            forced = False
            for k in keys.poll():
                if k == "q":
                    stop_reason = "user_quit"
                elif k == "n" and not planner.done:
                    planner.step += 1
                    planner._streak = 0
                    planner.done = planner.step == len(planner.subtasks)
                    forced = True
                    n_forced += 1
            if stop_reason == "user_quit":
                break
            if forced and not planner.done:
                set_instruction(planner.current)
                print(f"[{time.perf_counter() - t_start:6.1f}s] FORCED -> {planner.current!r}")
            if planner.done:
                stop_reason = "plan_complete"
                print(f"[{time.perf_counter() - t_start:6.1f}s] plano completo")
                break

            t0 = time.perf_counter()
            state = robot.state()
            release_switch = False
            if args.close_after_release > 0 and not planner.done and planner.current.startswith("place"):
                if t_release is None and state[6] > args.release_state_thr:
                    t_release = t0 - t_start
                    print(f"[{t_release:6.1f}s] left gripper opened (cube released) -> close in {args.close_after_release}s "
                          "unless the completion score advances first")
                if t_release is not None and t0 - t_start - t_release >= args.close_after_release:
                    planner.step += 1
                    planner._streak = 0
                    planner.done = planner.step == len(planner.subtasks)
                    release_switch = close_by_release = True
                    if not planner.done:
                        set_instruction(planner.current)
                        print(f"[{t0 - t_start:6.1f}s] RELEASE RULE -> {planner.current!r}")
            if ensemble:
                ens.submit(step, state, robot.images())
                action = ens.action(step)
                score = ens.pop_completion()
            elif fader is not None:
                action, score = fader.action(state)
            else:
                action, score = sync_action(state)
            if score is None and args.done_every > 0 and step % args.done_every == 0:
                score = completion_only(state)
            assisted = False
            if action is None:  # no chunk yet (start / just after an instruction change): hold position
                n_hold += 1
                action = sent = state
            else:
                command_out = action  # log["action"] keeps the policy output; the assist override shows in "sent"
                if assist is not None and planner.current.startswith("place"):
                    command_out, fired = assist(action, state)
                    assisted = assist.open
                    if fired:
                        assist_fired = True
                        print(f"[{t0 - t_start:6.1f}s] RELEASE ASSIST ({assist.reason}): left gripper -> {args.release_open}")
                sent = safety(command_out, state)
                if send:
                    robot.send(sent)
            if recorder is not None:
                recorder.add(state, sent, robot.images(), command, planner.step, score)

            log["t"].append(t0 - t_start); log["state"].append(state); log["action"].append(action)
            log["sent"].append(sent); log["subtask"].append(planner.step); log["assist"].append(assisted)
            log["completion"].append(np.nan if score is None else score); log["forced"].append(forced)
            prog = getattr(policy, "last_progress", None) if uses_progress else None
            log["progress"].append(np.nan if prog is None else prog)
            log["release_switch"].append(release_switch)
            log["loop_dt"].append(time.perf_counter() - t0)

            if step % args.print_every == 0 or (score is not None and score >= args.threshold):
                recent = np.asarray(log["loop_dt"][-args.print_every:])
                extra = f"  infer {np.mean(ens.infer_ms[-10:]):.0f} ms" if ensemble and ens.infer_ms else ""
                p = "  -  " if score is None else f"{score:.2f}"
                if uses_progress and prog is not None:
                    p += f" prog={prog:.2f}"
                print(f"[{t0 - t_start:6.1f}s] {planner.step + 1}/5 {planner.current!r:52s} P(done)={p}  "
                      f"max|a-s|={np.abs(action - state).max():.3f}  loop {recent.mean() * 1000:.0f}/"
                      f"{recent.max() * 1000:.0f} ms{extra}")
            if planner.update(score):
                set_instruction(planner.current)
                print(f"[{t0 - t_start:6.1f}s] -> {planner.current!r}")
            step += 1
            next_tick += dt
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        stop_reason = "ctrl_c"
    finally:
        if ens is not None:
            ens.stop()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"rollout_{args.mode}{tag}_{stamp}.npz"
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in log.items()}, command=command,
                            subtasks=np.array(planner.subtasks), stop_reason=stop_reason,
                            policy_path=str(args.policy_path), mode=args.mode, fps=args.fps,
                            replan_every=args.replan_every, threshold=args.threshold, patience=args.patience,
                            temporal_ensemble=np.nan if args.temporal_ensemble is None else args.temporal_ensemble,
                            min_time_multiplier=args.min_time_multiplier, ensemble_window=args.ensemble_window,
                            crossfade=args.crossfade, crossfade_joints_only=args.crossfade_joints_only,
                            smooth_chunk=args.smooth_chunk,
                            done_every=args.done_every,
                            close_after_release=args.close_after_release, release_state_thr=args.release_state_thr,
                            t_release=np.nan if t_release is None else t_release,
                            progress_ratchet=not args.no_progress_ratchet,
                            release_assist=args.release_assist, release_hold=args.release_hold,
                            release_max_speed=args.release_max_speed, release_open=args.release_open,
                            release_timeout=args.release_timeout, release_still_tol=args.release_still_tol,
                            **({k: np.asarray(v) for k, v in chunk_log.items()} if args.log_chunks else {}),
                            infer_ms=np.asarray(ens.infer_ms if ens is not None else []))
        loop = np.asarray(log["loop_dt"]) if log["loop_dt"] else np.zeros(1)
        total = time.perf_counter() - t_start
        info = (f" | inference {np.mean(ens.infer_ms):.0f} ms, {ens.n_inferences / total:.1f} chunks/s"
                if ens is not None and ens.infer_ms else "")
        reached = planner.step + (not planner.done)
        print(f"stop: {stop_reason} | {step} steps in {total:.1f}s ({step / max(total, 1e-6):.1f} Hz) | "
              f"subtask reached {reached}/5 | loop mean {loop.mean() * 1000:.0f} ms, "
              f"max {loop.max() * 1000:.0f} ms (budget {dt * 1000:.0f}){info} | holds {n_hold} | "
              f"safety clips: range {safety.n_clipped_range}, step {safety.n_clipped_step}\nlog: {path}")
    # time at which the planner entered each subtask (auto or forced), NaN if never reached
    sub = np.asarray(log["subtask"]) if log["subtask"] else np.zeros(0, int)
    ts = np.asarray(log["t"]) if log["t"] else np.zeros(0)
    t_enter = [float(ts[np.argmax(sub >= i)]) if (sub >= i).any() else np.nan for i in range(1, len(planner.subtasks))]
    return dict(stop_reason=stop_reason, duration_s=round(total, 2), steps=step, planner_reached=reached,
                plan_complete=planner.done, n_forced=n_forced, assist_fired=assist_fired, holds=n_hold,
                clips_range=safety.n_clipped_range, clips_step=safety.n_clipped_step,
                loop_ms_mean=round(float(loop.mean() * 1000), 1), loop_ms_max=round(float(loop.max() * 1000), 1),
                t_enter=[round(t, 2) for t in t_enter], rollout=str(path), close_by_release=close_by_release,
                t_release=None if t_release is None else round(t_release, 2))


def control_loop(args, robot: Robot, out_dir: Path):
    ctx = prepare(args)
    command = args.command or input("comando da tarefa: ").strip()

    def start():
        robot.home()
        if args.mode == "run":
            input(">>> MODO RUN: os bracos VAO se mover. Mao no botao de emergencia. Enter para comecar...")
        else:
            print("dry-run: actions are computed and logged but NOT sent")
        return KeyboardCommands()

    run_trial(args, robot, ctx, command, out_dir, start)


# ----------------------------------------------------------------------------------------- eval mode
STAGES_PT = ["nada", "pegou o cubo", "levou o cubo ate a gaveta", "abriu a gaveta", "cubo dentro da gaveta",
             "gaveta fechada (SUCESSO)"]
# label 'p': the pick failed, the operator skipped it with n+Enter and the robot went on without the cube;
# then `stage` = how far the later phases went (1 = nothing after the pick)
STAGES_SKIP_PT = ["-", "nada depois do pick", "foi ate a gaveta", "abriu a gaveta", "fez o place",
                  "fechou a gaveta"]
PHASES_PT = ["", "", "move (ate a gaveta)", "open (abriu a gaveta)", "place", "close (fechou a gaveta)"]
TRIAL_FIELDS = ["trial", "layout", "command", "policy", "stage", "success", "pick_skipped", "wrong_cube", "autonomous",
                "n_forced", "assist_fired", "close_by_release", "t_release", "stop_reason", "duration_s", "planner_reached", "t_enter", "loop_ms_mean",
                "clips_range", "clips_step", "hf_episode_index", "note", "rollout", "time"]


def ask(keys: KeyboardCommands, prompt: str) -> str:
    """Blocking question through the same stdin reader the trial uses (input() would race with it)."""
    keys.poll()  # drop anything typed during the trial
    print(prompt, end=" ", flush=True)
    while True:
        try:
            return keys.q.get(timeout=0.5)
        except queue.Empty:
            continue


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(float(c - h), 3), round(float(c + h), 3))


def _true(v) -> bool:
    return str(v) in ("True", "1")


def summarise(trials: list[dict]) -> dict:
    """Headline numbers = the task as a whole: a trial labelled 'p' (pick failed, skipped with n) counts as
    stage 0 / failure. `after_pick` measures the later phases (move, open, place, close) in every trial that got
    past the pick: picked by the robot, or skipped with n (then without the cube)."""
    n = len(trials)
    stage = np.array([int(t["stage"]) for t in trials]) if n else np.zeros(0, int)
    skip = np.array([_true(t.get("pick_skipped")) for t in trials]) if n else np.zeros(0, bool)
    task_stage = np.where(skip, 0, stage)
    succ = task_stage == 5
    auto = np.array([_true(t["autonomous"]) for t in trials]) if n else np.zeros(0, bool)
    dur = np.array([float(t["duration_s"]) for t in trials]) if n else np.zeros(0)

    def rate(mask, m):
        return round(float(mask.sum() / m), 3) if m else None

    s = dict(n_trials=n,
             success=int(succ.sum()), success_rate=rate(succ, n), success_ci95=wilson(int(succ.sum()), n),
             success_autonomous=int((succ & auto).sum()), success_autonomous_rate=rate(succ & auto, n),
             stage_reached_rate={STAGES_PT[k]: rate(task_stage >= k, n) for k in range(1, 6)},
             pick_skipped=int(skip.sum()),
             wrong_cube=int(sum(_true(t["wrong_cube"]) for t in trials)),
             duration_success_mean_s=round(float(dur[succ].mean()), 1) if succ.any() else None,
             duration_success_sd_s=round(float(dur[succ].std()), 1) if succ.any() else None,
             stop_reasons={r: sum(t["stop_reason"] == r for t in trials) for r in sorted({t["stop_reason"] for t in trials})},
             interventions_trials=int((~auto).sum()))

    def phases(mask):
        m = int(mask.sum())
        return dict(n=m, **{PHASES_PT[k]: rate(mask & (stage >= k), m) for k in range(2, 6)})

    picked = ~skip & (stage >= 1)
    s["after_pick"] = dict(all=phases(picked | skip), pick_by_robot=phases(picked), pick_skipped_no_cube=phases(skip))
    te = [json.loads(t["t_enter"]) for t in trials if t.get("t_enter")]
    if te and len({len(x) for x in te}) == 1:
        arr = np.array(te, float)
        s["planner_t_enter_median_s"] = [None if np.isnan(v) else round(float(v), 1) for v in np.nanmedian(arr, 0)] \
            if np.isfinite(arr).any() else None
    return s


class EvalRecorder:
    """Records the executed trials as a LeRobot v3 dataset in a staging folder and pushes it to the Hub.

    Frames = what the robot saw and the command actually sent (after safety), at the control rate (--fps).
    Extra per-frame columns: subtask_index (planner step), completion (done-head score, -1 when not scored).
    """

    def __init__(self, repo_id: str, staging: Path, fps: int, root_ref: str):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id, self.staging = repo_id, staging
        threads = 4 * len(CAMERAS)
        if staging.exists() and not any((staging / "meta" / "episodes").glob("*/*.parquet")):
            # no complete episode (session interrupted before / during the first save): set aside, start fresh
            aside = staging.with_name(f"{staging.name}_incompleto_{time.strftime('%Y%m%d_%H%M%S')}")
            staging.rename(aside)
            print(f"[hf] {staging} had no complete episode -> moved to {aside} (delete it when not needed)")
        if not staging.exists():
            from huggingface_hub import HfApi, snapshot_download

            if HfApi().repo_exists(repo_id, repo_type="dataset"):
                # already pushed and the local copy deleted: continue from the Hub copy, otherwise the next push
                # would overwrite its episodes with a new dataset starting at episode 0. New episodes are appended
                # to the last data/video files, so the whole dataset (videos included) is needed locally.
                print(f"[hf] {repo_id} already on the Hub -> downloading it to continue ({staging})...")
                snapshot_download(repo_id, repo_type="dataset", local_dir=staging)
        if (staging / "meta" / "info.json").exists():
            print(f"[hf] resuming staged dataset {staging}")
            self.ds = LeRobotDataset.resume(repo_id, root=staging, image_writer_threads=threads)
            if self.ds.fps != fps:
                raise SystemExit(f"staged dataset has fps {self.ds.fps}, this run uses --fps {fps}")
        else:
            ref = json.loads((Path(root_ref) / "meta" / "info.json").read_text())["features"]
            keys = ["action", "observation.state"] + [f"observation.images.{c}" for c in CAMERAS]
            features = {k: {f: ref[k][f] for f in ("dtype", "shape", "names")} for k in keys}
            features["subtask_index"] = {"dtype": "int64", "shape": [1], "names": None}
            features["completion"] = {"dtype": "float32", "shape": [1], "names": None}
            self.ds = LeRobotDataset.create(repo_id, fps, features, root=staging, robot_type="trossen_ai_stationary",
                                            use_videos=True, image_writer_threads=threads)

    @property
    def n_episodes(self) -> int:
        return self.ds.meta.total_episodes

    def add(self, state, sent, images, task, subtask, score):
        frame = {"observation.state": np.asarray(state, np.float32), "action": np.asarray(sent, np.float32),
                 "subtask_index": np.array([subtask], np.int64),
                 "completion": np.array([-1.0 if score is None else score], np.float32), "task": task}
        for c in CAMERAS:
            frame[f"observation.images.{c}"] = images[c]
        self.ds.add_frame(frame)

    def save(self) -> int:
        """Encodes the 4 videos one after the other in this process (~15 s per camera per 2 min of trial).
        LeRobot's default encodes them in forked worker processes; forking this process (CUDA + RealSense
        threads) hung on the robot (24/11: >5 min, never finished). Ctrl+C is deferred until the save ends."""
        import signal

        idx = self.n_episodes
        interrupted = []

        def defer(signum, frame):
            interrupted.append(True)
            print("\n[hf] aguarde: terminando de salvar os videos, a sessao encerra logo depois...", flush=True)

        old = signal.signal(signal.SIGINT, defer)
        try:
            self.ds.save_episode(parallel_encoding=False)
        finally:
            signal.signal(signal.SIGINT, old)
        if interrupted:
            raise KeyboardInterrupt
        return idx

    def discard(self):
        try:
            if self.ds.has_pending_frames():
                self.ds.clear_episode_buffer()
        except Exception as e:  # buffer left half-written by an interrupted save
            print(f"[hf] could not clear the episode buffer cleanly ({e!r}); resetting it")
            self.ds.writer.episode_buffer = self.ds.writer._create_episode_buffer()

    def finalize(self):
        """Always at the end of a session (pushed or not): writes the parquet footers so the staged
        dataset is valid and can be resumed next session."""
        self.discard()
        self.ds.finalize()

    def push(self, private: bool, extra_files: list[Path]) -> bool:
        import shutil

        for f in extra_files:  # trial labels/metrics travel with the videos
            if f.exists():
                shutil.copy(f, self.staging / "meta" / f"eval_{f.name}")
        try:
            self.ds.push_to_hub(private=private, tags=["eval", "act_siglip"])
        except Exception as e:  # keep the staged copy so nothing is lost
            print(f"[hf] push FAILED ({e!r}); staged data kept in {self.staging}")
            return False
        print(f"[hf] pushed {self.n_episodes} episodes to https://huggingface.co/datasets/{self.repo_id}")
        return True


def release_and_home(robot: Robot, seconds: float = 1.0):
    """After the trial is labelled: open both grippers where the arms are, then go home."""
    s = robot.state()
    s[6], s[13] = 0.031, 0.03
    for i, n in enumerate(ARMS):
        robot.arms[n].go(s[7 * i: 7 * (i + 1)], seconds, blocking=False)
    time.sleep(seconds + 0.2)
    robot.home()


def layout_check(robot: Robot, area: dict | None, layout: int):
    """Text-only placement check (cam_high): cube positions vs the demo area and the layout targets."""
    if area is None:
        return
    try:
        from cube_area_overlay import camera_shift, detect_cubes, NAMES_PT
        import cv2
    except ImportError as e:
        print(f"(layout check unavailable: {e})")
        return
    rgb = robot.cameras["cam_high"].read() if hasattr(robot, "cameras") and "cam_high" in robot.cameras \
        else robot.images()["cam_high"]
    s = area["cube_side_px"]
    x0, y0, x1, y1 = area["rect_96"]
    targets = area["layouts"][layout - 1] if layout else {}
    cubes = detect_cubes(rgb)
    ref = cv2.imread(str(Path(__file__).resolve().parents[1] / "docs/teste_robo/cam_high_referencia.png"), 0)
    if ref is not None:
        dx, dy = camera_shift(ref, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        print(f"  camera vs gravacao: dx {dx:+.1f} dy {dy:+.1f} px" + ("  <-- CAMERA MEXEU" if max(abs(dx), abs(dy)) > 4 else ""))
    for c in ("red", "yellow", "green"):
        if c not in cubes:
            print(f"  {NAMES_PT[c]:9s} NAO DETECTADO")
            continue
        x, y = cubes[c]
        msg = "dentro da area" if x0 <= x <= x1 and y0 <= y <= y1 else "FORA DA AREA"
        if c in targets:
            msg += f", {np.hypot(x - targets[c][0], y - targets[c][1]) / s:.1f} cubo do alvo"
        print(f"  {NAMES_PT[c]:9s} {msg}")


def eval_loop(args, robot: Robot, out_dir: Path):
    import csv

    ctx = prepare(args)
    command = args.command
    eval_dir = out_dir / "eval" / args.eval_name
    (eval_dir / "rollouts").mkdir(parents=True, exist_ok=True)
    csv_path, summary_path = eval_dir / "trials.csv", eval_dir / "summary.json"
    trials = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
    if trials and trials[0]["command"] != command:
        raise SystemExit(f"{csv_path} is for {trials[0]['command']!r}; use another --eval-name")
    area_path = Path(__file__).resolve().parents[1] / "docs/teste_robo/area_cubos.json"
    area = json.loads(area_path.read_text()) if area_path.exists() else None
    n_layouts = len(area["layouts"]) if (area and args.layouts) else 0
    recorder = EvalRecorder(args.hf_repo, eval_dir / "hf_staging", args.fps, args.root) if args.hf_repo else None
    n_rec = sum(str(t.get("hf_episode_index", "")) != "" for t in trials)
    if recorder is not None and recorder.n_episodes != n_rec:
        print(f"[hf] ATENCAO: o dataset {args.hf_repo} tem {recorder.n_episodes} episodios, mas {csv_path} tem {n_rec} "
              "tentativas gravadas -> confira se --eval-name e --hf-repo sao da mesma avaliacao")
    keys = KeyboardCommands()
    print(f"\n=== AVALIACAO '{args.eval_name}': {command!r} | {len(trials)}/{args.episodes} tentativas feitas | "
          f"resultados em {eval_dir}" + (f" | dataset -> {args.hf_repo}" if recorder else ""))
    ask(keys, ">>> os bracos VAO se mover. Mao no botao de emergencia. Enter para continuar...")

    def write(trials):
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
            w.writeheader()
            w.writerows(trials)
        summary = summarise(trials)
        summary.update(command=command, policy=str(args.policy_path), legacy_act=args.legacy_act, fps=args.fps,
                       replan_every=args.replan_every, crossfade=args.crossfade, max_seconds=args.max_seconds)
        summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        return summary

    try:
        while len(trials) < args.episodes:
            k = len(trials) + 1
            layout = (k - 1) % n_layouts + 1 if n_layouts else 0
            robot.home()
            while True:
                print(f"\n--- tentativa {k}/{args.episodes}" + (f" | LAYOUT {layout} (docs/teste_robo/layouts_teste.png)" if layout else "")
                      + " | gaveta fechada, cubos no lugar")
                layout_check(robot, area, layout)
                a = ask(keys, "Enter = comecar | c = checar de novo | q = encerrar a sessao:")
                if a in ("", "q"):
                    break
            if a == "q":
                break
            res = run_trial(args, robot, ctx, command, eval_dir / "rollouts", lambda: keys, recorder, tag=f"_t{k:02d}")
            if res["stop_reason"] == "ctrl_c":
                print("Ctrl+C: tentativa descartada, sessao encerrada")
                if recorder:
                    recorder.discard()
                break
            print("bracos parados. Avalie a cena ANTES de mexer em algo.")
            opts = " | ".join(f"{i} {s}" for i, s in enumerate(STAGES_PT))
            while True:
                a = ask(keys, f"Ate onde deu certo? [{opts} | p pick deu errado, passei com n | "
                              "d descartar (erro de montagem)]:")
                if a in ("d", "p") or (a.isdigit() and 0 <= int(a) <= 5):
                    break
            if a == "d":
                if recorder:
                    recorder.discard()
                print("descartada (nao conta).")
                release_and_home(robot)
                continue
            skipped = a == "p"
            if skipped:
                opts = " | ".join(f"{i} {s}" for i, s in enumerate(STAGES_SKIP_PT) if i >= 1)
                while True:
                    a = ask(keys, f"Depois do pick, ate onde foi (sem o cubo)? [{opts}]:")
                    if a.isdigit() and 1 <= int(a) <= 5:
                        break
            stage = int(a)
            wrong = ask(keys, "Pegou/mexeu no cubo de OUTRA cor? [s/N]:") == "s"
            note = ask(keys, "Observacao (Enter = nenhuma):")
            release_and_home(robot)
            trials.append(dict(trial=k, layout=layout, command=command, policy=str(args.policy_path), stage=stage,
                               success=stage == 5 and not skipped, pick_skipped=skipped, wrong_cube=wrong,
                               autonomous=res["n_forced"] == 0 and not res["assist_fired"],
                               n_forced=res["n_forced"], assist_fired=res["assist_fired"],
                               close_by_release=res["close_by_release"], t_release=res["t_release"],
                               stop_reason=res["stop_reason"], duration_s=res["duration_s"],
                               planner_reached=res["planner_reached"], t_enter=json.dumps(res["t_enter"]),
                               loop_ms_mean=res["loop_ms_mean"], clips_range=res["clips_range"],
                               clips_step=res["clips_step"], hf_episode_index="", note=note,
                               rollout=res["rollout"], time=time.strftime("%Y-%m-%d %H:%M:%S")))
            s = write(trials)  # the label is safe on disk before the (slow) video encoding
            if recorder:
                n_cams, dur = len(CAMERAS), res["duration_s"]
                print(f"salvando videos: ~{max(10, round(n_cams * 16 * dur / 114))} s (pode arrumar a cena; "
                      "NAO precisa apertar nada)...", flush=True)
                t_save = time.time()
                try:
                    trials[-1]["hf_episode_index"] = recorder.save()
                    print(f"videos salvos em {time.time() - t_save:.0f} s (episodio {trials[-1]['hf_episode_index']})")
                except KeyboardInterrupt:
                    trials[-1]["hf_episode_index"] = recorder.n_episodes - 1
                    write(trials)
                    raise
                except Exception as e:
                    print(f"[hf] FALHA ao salvar os videos ({e!r}). O resultado da tentativa esta salvo; a gravacao "
                          "fica desligada no resto da sessao.")
                    recorder = None
                write(trials)
            print(f"=> {s['success']}/{s['n_trials']} sucessos ({s['success_rate']:.0%}, IC95 {s['success_ci95']})"
                  f" | pick pulado com n: {s['pick_skipped']}")
    except KeyboardInterrupt:
        print("\nCtrl+C: sessao encerrada")
        if recorder:
            recorder.discard()
    finally:
        s = write(trials)
        print(f"\n=== RESUMO {args.eval_name} ({eval_dir}) ===\n{json.dumps(s, indent=1, ensure_ascii=False)}")
        if recorder:
            recorder.finalize()
        if recorder and recorder.n_episodes:
            a = ask(keys, f"Enviar {recorder.n_episodes} episodios para o Hugging Face ({args.hf_repo}, "
                          f"{'publico' if args.hf_public else 'privado'})? [s/N]:")
            if a == "s" and recorder.push(not args.hf_public, [csv_path, summary_path]):
                if len(trials) >= args.episodes and not args.hf_keep_local:
                    import shutil
                    shutil.rmtree(recorder.staging)
                    print(f"[hf] avaliacao completa: copia local dos videos apagada ({recorder.staging})")
                else:
                    print(f"[hf] copia local mantida em {recorder.staging} para continuar a avaliacao depois "
                          "(rode de novo com o mesmo --eval-name); apagada quando completar --episodes")
            elif a != "s":
                print(f"[hf] nao enviado; dados em {recorder.staging}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["check", "dry-run", "run", "eval"], required=True)
    ap.add_argument("--episodes", type=int, default=30, help="eval: number of counted trials (same --command)")
    ap.add_argument("--eval-name", default=None,
                    help="eval: results go to <out-dir>/eval/<name>/ (trials.csv, summary.json, rollouts/); "
                         "running again with the same name continues where it stopped")
    ap.add_argument("--no-layouts", dest="layouts", action="store_false",
                    help="eval: don't cycle the fixed cube layouts of docs/teste_robo/area_cubos.json")
    ap.add_argument("--hf-repo", default=None,
                    help="eval: also record the trials (4 cameras + state + sent action) as a LeRobot dataset and "
                         "push it to this Hub repo at the end (asks before pushing), e.g. MariaFPG/eval_v7_red")
    ap.add_argument("--hf-public", action="store_true", help="eval: create the Hub dataset public (default private)")
    ap.add_argument("--hf-keep-local", action="store_true",
                    help="eval: keep the staged dataset after a successful push (default: deleted once all "
                         "--episodes are done)")
    ap.add_argument("--policy-path", type=Path, help="…/checkpoints/<step>/pretrained_model")
    ap.add_argument("--legacy-act", action="store_true",
                    help="--policy-path is a plain ACT checkpoint from the trossen fork (e.g. the per-colour models); "
                         "language is ignored. Use --replan-every 100 to execute chunks like the fork did")
    ap.add_argument("--command", default=None, help="full task command (asked interactively if omitted)")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="dataset root (stats for the safety clip, reference images)")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/robot"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min-time-multiplier", type=float, default=3.0, help="as in the fork: min_time_to_move = m / fps")
    ap.add_argument("--replan-every", type=int, default=10, help="chunk mode: frames between policy queries")
    ap.add_argument("--crossfade", type=int, default=0, metavar="N",
                    help="chunk mode: blend the old and new chunk over N frames with a sigmoid (Flash ALOHA) instead "
                         "of jumping at each re-plan; needs --replan-every + N <= chunk size (e.g. 80 + 20). 0 = off")
    ap.add_argument("--smooth-chunk", type=float, default=0.0, metavar="SIGMA",
                    help="chunk mode: Gaussian-smooth every predicted chunk along time (sigma in frames, arm joints "
                         "only, no lag) to remove the step-to-step zig-zag of the predictions. Suggested: 1. 0 = off")
    ap.add_argument("--crossfade-joints-only", action="store_true",
                    help="with --crossfade: blend only the arm joints; both grippers follow the new chunk directly "
                         "(blending them flattens the release ramp: the cube was not let go with --crossfade 20)")
    ap.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                    help="async temporal ensembling (ACT paper: 0.01): the policy runs in a background thread on "
                         "the latest observation and overlapping chunks are averaged; control stays at --fps")
    ap.add_argument("--ensemble-window", type=int, default=10,
                    help="temporal ensembling: average only chunks computed in the last N frames (50 = original ACT, smoothest but laggy; 10 measured best: chunk-mode accuracy, 4x smoother)")
    ap.add_argument("--done-every", type=int, default=0, metavar="N",
                    help="chunk mode: also score completion every N frames between re-plans (one extra policy pass, "
                         "~30 ms; the chunk is dropped), so the planner advances soon after a subtask ends instead of "
                         "waiting --patience re-plans. With N=5 and patience 3 the switch needs ~15 frames. 0 = off "
                         "(score only at re-plans, old behaviour)")
    ap.add_argument("--no-progress-ratchet", action="store_true",
                    help="models with progress_levels > 0: let the predicted subtask progress decrease (by default it "
                         "never decreases within a subtask and restarts at each new subtask)")
    ap.add_argument("--log-chunks", action="store_true",
                    help="chunk mode: save every predicted chunk (step, subtask, executed or only scored by "
                         "--done-every) in the rollout npz, e.g. to see if a release pulse is predicted but never reached")
    ap.add_argument("--release-assist", type=float, default=0.0, metavar="THR",
                    help="HEURISTIC, off by default. Place subtask only: open the left gripper fully (--release-open) "
                         "once its mean command over --release-hold s exceeds THR and the left arm has settled "
                         "(speed < --release-max-speed). Suggested: 0.010")
    ap.add_argument("--release-timeout", type=float, default=0.0, metavar="S",
                    help="RULE, off by default. Place subtask only: open the left gripper (--release-open) once "
                         "every left-arm joint stayed within --release-still-tol rad over the last S s and the cube "
                         "was not released (gripper state <= --release-state-thr). Suggested: 4")
    ap.add_argument("--release-still-tol", type=float, default=0.05,
                    help="release timeout: max joint range (rad) over the window to count the arm as stopped")
    ap.add_argument("--release-hold", type=float, default=1.0, help="release assist: averaging window (s)")
    ap.add_argument("--release-max-speed", type=float, default=0.10,
                    help="release assist: max left-arm joint speed (rad/s, over 0.5 s) to count as settled")
    ap.add_argument("--release-open", type=float, default=0.031,
                    help="release assist: gripper command sent once fired (demo peak median 0.031)")
    ap.add_argument("--close-after-release", type=float, default=0.0, metavar="S",
                    help="RULE, off by default. During \"place\": once the left gripper state first exceeds "
                         "--release-state-thr (cube released), advance to \"close\" S seconds later unless the "
                         "completion score advanced first. In the demos the right arm starts closing ~0.5 s after the "
                         "release, the auto labels start \"close\" ~2.2 s after it, manual labels ~1.7 s. Suggested: 1.5")
    ap.add_argument("--release-state-thr", type=float, default=0.026,
                    help="left gripper state counted as released (demos: holding the cube <= 0.0239, open > 0.027)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=None,
                    help="consecutive completion scores >= threshold (default: 5 with temporal ensembling, "
                         "one score per inference ~ every 1-2 frames; 3 in chunk mode, one score per chunk)")
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--max-step-rad", type=float, default=0.1, help="max joint change per command (rad)")
    ap.add_argument("--max-step-gripper", type=float, default=0.05,
                    help="max gripper change per command (m); default > full range = off, so a grasp can squeeze "
                         "as during data collection")
    ap.add_argument("--margin-rad", type=float, default=0.05)
    ap.add_argument("--margin-gripper", type=float, default=0.005)
    ap.add_argument("--print-every", type=int, default=60)
    args = ap.parse_args()
    if args.patience is None:
        args.patience = 5 if args.temporal_ensemble is not None else 3
    if args.crossfade < 0 or (args.crossfade > 0 and args.temporal_ensemble is not None):
        ap.error("--crossfade must be >= 0 and can't be combined with --temporal-ensemble")
    if args.done_every < 0 or (args.done_every > 0 and (args.temporal_ensemble is not None or args.legacy_act)):
        ap.error("--done-every must be >= 0; it is for chunk mode (temporal ensembling already scores every "
                 "inference) and needs a completion head (not --legacy-act)")
    if args.crossfade_joints_only and args.crossfade == 0:
        ap.error("--crossfade-joints-only needs --crossfade > 0")
    if args.smooth_chunk < 0 or (args.smooth_chunk > 0 and args.temporal_ensemble is not None):
        ap.error("--smooth-chunk must be >= 0 and is for chunk mode (not --temporal-ensemble)")
    if args.log_chunks and args.temporal_ensemble is not None:
        ap.error("--log-chunks is for chunk mode (not --temporal-ensemble)")
    if args.close_after_release < 0 or (args.close_after_release > 0 and args.legacy_act):
        ap.error("--close-after-release must be >= 0 and needs the subtask planner (not --legacy-act)")
    if args.release_timeout < 0:
        ap.error("--release-timeout must be >= 0")
    if args.release_assist < 0 or args.release_hold <= 0 or args.release_max_speed <= 0:
        ap.error("--release-assist must be >= 0, --release-hold and --release-max-speed > 0")
    if args.mode != "check" and args.policy_path is None:
        ap.error("--policy-path is required for dry-run/run/eval")
    if args.mode == "eval" and (not args.command or not args.eval_name or args.episodes < 1):
        ap.error("--mode eval needs --command, --eval-name and --episodes >= 1")
    if args.hf_repo and args.mode != "eval":
        ap.error("--hf-repo is for --mode eval")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    robot = Robot(args.fps, args.min_time_multiplier)
    try:
        if args.mode == "check":
            robot.home()
            check_mode(robot, Path(args.root), args.out_dir)
        elif args.mode == "eval":
            eval_loop(args, robot, args.out_dir)
        else:
            control_loop(args, robot, args.out_dir)
    finally:
        robot.close(park=True)


if __name__ == "__main__":
    main()
