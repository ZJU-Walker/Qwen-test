# Human-Video-Prompted VLA — First-Version Implementation Plan

## 1. Goal

Replace the language instruction with a short human demonstration video.

At deployment:

1. A human demonstrates the task.
2. The system samples the demonstration at 3 FPS.
3. The sampled video is used as the task prompt.
4. The robot observes the current scene and state.
5. The VLA predicts robot actions conditioned on the human demonstration.

The first version uses the full prompt-video tokens directly. It does not use query tokens or a prompt resampler.

---

## 2. Policy Input

Each policy input contains:

```text
Human prompt video
+ robot top-camera observation history
+ current right-wrist image
+ current left-wrist image
+ robot state history
+ current robot state
+ RTC action prefix
```

Recommended sequence order:

```text
<human_demo>
human prompt video
</human_demo>

<robot_history>
top-camera observation-history video
</robot_history>

<right_wrist>
current right-wrist image

<left_wrist>
current left-wrist image

Past states: ...
State: ...
```

The labels are only structural markers. There is no natural-language task instruction.

The human demonstration and robot observation history must be passed as two separate video inputs. Do not concatenate them into one video.

---

## 3. Human Prompt Video Sampling

The raw human demonstration is recorded at:

```text
30 FPS
3–5 seconds
90–150 raw frames
```

For Qwen3-VL, sample the prompt video at:

```text
3 frames per second
```

Expected sampled-frame counts:

```text
3-second demo → about 9 frames
4-second demo → about 12 frames
5-second demo → about 15 frames
```

Recommended configuration:

```yaml
human_prompt:
  sample_fps: 3
  min_frames: 8
  max_frames: 16
  sampling: uniform
  include_first_frame: true
  include_last_frame: true
  preserve_timestamps: true
```

Requirements:

- Keep frames in temporal order.
- Preserve the real timestamps.
- Do not shuffle or reverse frames.
- End the demonstration after the human has successfully lifted the object.
- Do not include the scene-reset motion.

Store the original 30 FPS video and perform sampling during data loading and deployment.

---

## 4. Training Data

Each paired data item contains:

```text
one human demonstration video
+
one successful robot execution of the same task
```

For every robot timestep `t`, construct:

```python
sample = {
    "human_prompt_video": sampled_human_demo,
    "human_prompt_timestamps": human_prompt_timestamps,

    "robot_history_video": robot_top_history,
    "right_wrist": robot_right_wrist[t],
    "left_wrist": robot_left_wrist[t],

    "state_history": robot_state_history,
    "state": robot_state[t],

    "robot_action_chunk": robot_actions[t:t + 50],

    "pair_id": pair_id,
    "task_id": task_id,
}
```

The human prompt stays fixed for all samples from the paired robot episode.

The training targets remain entirely robot-side:

- Flow target: robot EE6D actions.
- FAST target: robot EE6D action tokens.
- Subtask target: a text description of the demonstrated task.

Do not convert human hand motion into robot actions.

Do not fit the FAST tokenizer on human motion.

---

## 5. Model Flow

The first-version model flow is:

```text
human prompt video
+ robot observation history
+ wrist images
+ robot states
        ↓
Qwen3-VL layers 0–17
        ↓
detach VLM KV cache
        ↓
18-layer flow-matching action expert
        ↓
50-step EE6D action chunk
        ↓
Gram-Schmidt rotation projection
        ↓
IK
        ↓
robot joint actions
```

No query tokens are added.

No prompt resampler is added.

The action expert directly attends to:

```text
human prompt-video tokens
+ robot observation-history tokens
+ wrist-image tokens
+ state tokens
```

Knowledge insulation remains unchanged:

```python
expert_context = vlm_kv[:18].detach()
```

The flow-matching loss trains the action expert but does not backpropagate into Qwen.

---

## 6. Losses

Keep the current three-loss setup:

\[
\mathcal{L}
=
\mathcal{L}_{\text{flow}}
+
\lambda_{\text{FAST}}\mathcal{L}_{\text{FAST}}
+
\lambda_{\text{subtask}}\mathcal{L}_{\text{subtask}}
\]

### Flow loss

Predict the paired robot EE6D action chunk.

### FAST cross-entropy loss

Qwen predicts robot FAST action tokens conditioned on:

```text
human prompt video
+ current robot observations
+ robot states
```

Keep:

```yaml
fast_loss_weight: 1.0
```

### Subtask cross-entropy loss

Qwen predicts a training-only description such as:

```text
pick up the green block
```

This helps Qwen learn the semantics of the human demonstration.

At inference:

```text
no language instruction
no subtask decoding
no FAST decoding
```

The action expert acts directly from the detached video-conditioned Qwen context.

---

## 7. Dataset Design

The dataset must force the model to use the human prompt.

For the first green/yellow experiment:

```text
human picks green → paired robot picks green
human picks yellow → paired robot picks yellow
```

Both objects should be visible in the robot scene.

Vary their positions between the human and robot demonstrations:

```text
Human scene:
green on the left
yellow on the right

Robot scene:
green on the right
yellow on the left
```

This prevents the model from learning to copy screen coordinates.

Never create mismatched positive pairs such as:

```text
human picks green
robot target picks yellow
```

---

## 8. Augmentation

Apply augmentation independently to the human video and robot observations.

For the human prompt clip:

```text
same crop across all prompt frames
same rotation across all prompt frames
optional color jitter
small temporal sampling variation
```

Do not use a different crop or rotation for every frame, because that creates artificial camera motion.

The existing robot observation augmentation can remain unchanged.

---

## 9. Deployment Flow

### Phase 1: Record the human demonstration

```text
1. Start recording.
2. Human demonstrates the pickup.
3. Stop recording after the object is lifted.
4. Reset the object outside the recorded clip.
5. Sample the demonstration at 3 FPS.
6. Upload the sampled prompt video to the server.
```

Store the prompt under:

```text
session_id + prompt_id
```

### Phase 2: Run the robot

Each inference request sends:

```text
prompt_id
new robot history frame
current wrist images
state history
current state
RTC action prefix
```

The server combines the stored human prompt with the live robot inputs and predicts the next action chunk.

For the first implementation, the server may re-encode the prompt video on every request.

After correctness is verified, cache the fixed human prompt's ViT output and Qwen KV cache to reduce latency.

When a new prompt is recorded:

```text
clear the previous RTC action queue
clear the old prompt association
start a new rollout
```

---

## 10. Minimal Repository Changes

### Dataset

Add:

```python
human_prompt_video
human_prompt_timestamps
pair_id
task_id
```

### Processor and collator

Support two separate videos:

```python
videos = [
    human_prompt_video,
    robot_history_video,
]
```

Keep their frame metadata separate.

### Prompt builder

Replace:

```text
Instruction: Pick up the green block
```

with:

```text
Human demonstration:
<video>

Current robot observation:
<video>
```

### Model

No major architecture change is required beyond accepting the additional video.

Keep:

```python
num_vlm_layers_for_expert = 18
expert_attends_subtask = False
knowledge_insulation = True
prompt_query_tokens = 0
```

### Server

Add:

```text
human-prompt upload
prompt_id storage
prompt retrieval during inference
```

### Checkpoint configuration

Store:

```json
{
  "instruction_mode": "human_video",
  "human_prompt_fps": 3,
  "human_prompt_max_frames": 16,
  "human_prompt_first": true,
  "expert_attends_raw_prompt": true,
  "prompt_query_tokens": 0
}
```

---

## 11. First Success Test

Use the exact same robot observation and state with two different human prompts.

### Test A

```text
Prompt: human picks green
Robot scene: green and yellow are both visible
```

Expected:

```text
predicted EE6D trajectory moves toward green
```

### Test B

```text
Prompt: human picks yellow
Robot scene and robot state remain unchanged
```

Expected:

```text
predicted EE6D trajectory moves toward yellow
```

Also compare:

```text
correct prompt
wrong prompt
blank prompt
temporally shuffled prompt
```

The main success criterion is:

> Changing only the human prompt video changes which object the robot approaches.

---

## 12. Final First-Version Configuration

```yaml
instruction_mode: human_video

human_prompt:
  sample_fps: 3
  min_frames: 8
  max_frames: 16
  sampling: uniform
  include_first_frame: true
  include_last_frame: true
  preserve_timestamps: true

model:
  qwen_layers_for_expert: 18
  prompt_query_tokens: 0
  expert_attends_raw_prompt: true
  expert_attends_subtask: false
  knowledge_insulation: true

training:
  action_space: ee6d
  action_horizon: 50
  fast_loss_weight: 1.0
  subtask_loss_enabled: true

deployment:
  use_prompt_id: true
  clear_rtc_on_prompt_change: true
  prompt_cache_enabled: false
```

The complete first-version pipeline is:

```text
3 FPS human prompt video
+ robot observation history
+ current wrist images
+ state history and current state
+ RTC prefix
        ↓
Qwen3-VL first 18 layers
        ↓
detached full multimodal context
        ↓
flow-matching action expert
        ↓
EE6D action chunk
        ↓
IK
        ↓
robot execution
```
