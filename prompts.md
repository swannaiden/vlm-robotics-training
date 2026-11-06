# Prompts

Every prompt, with its ground truth, is in `data/<run>/tasks.jsonl`. Both runs use the same user prompts.

## System prompt

`panda` run:

```
You are an expert in robot kinematics. You have no tools and cannot run code. Do not calculate or work through the kinematics step by step; give your best immediate estimate. Reply with only the answer, as JSON in the requested format, with no explanation.
```

`panda_strict` run:

```
You are an expert in robot kinematics. You have no tools and cannot run code. Do not calculate or work through the kinematics step by step; give your best immediate estimate. Reply with only the answer, as JSON in the requested format, with no explanation. DO NOT USE ANY REASONING. JUST REPLY WITH AN ANSWER.
```

## Example: forward kinematics, real Panda with its DH table

```
Robot: Franka Emika Panda, a 7-joint serial arm.
Kinematics: modified Denavit-Hartenberg parameters (Craig convention). Row i is the transform from frame i-1 to frame i:
T_i = RotX(alpha_{i-1}) * TransX(a_{i-1}) * RotZ(theta_i) * TransZ(d_i)
The end-effector frame is frame F, the last row. Its pose in the base frame (frame 0) is T_1 * T_2 * ... * T_F.

| i | a_{i-1} (m) | alpha_{i-1} (rad) | d_i (m) | theta_i (rad) |
|---|---|---|---|---|
| 1 | 0 | 0 | 0.333 | q1 |
| 2 | 0 | -pi/2 | 0 | q2 |
| 3 | 0 | pi/2 | 0.316 | q3 |
| 4 | 0.0825 | pi/2 | 0 | q4 |
| 5 | -0.0825 | -pi/2 | 0.384 | q5 |
| 6 | 0 | pi/2 | 0 | q6 |
| 7 | 0.088 | pi/2 | 0 | q7 |
| F | 0 | 0 | 0.107 | 0 |

Joint limits (rad): q1 [-2.8973, 2.8973], q2 [-1.7628, 1.7628], q3 [-2.8973, 2.8973], q4 [-3.0718, -0.0698], q5 [-2.8973, 2.8973], q6 [-0.0175, 3.7525], q7 [-2.8973, 2.8973]

Joint angles (rad): q = [0.180, 0.029, 0.137, -2.128, -0.156, 1.973, 0.453]

What is the position of the end-effector frame origin in the base frame, in meters?
Answer with only this JSON: {"x": <number>, "y": <number>, "z": <number>}
```

Ground truth: (0.502, 0.154, 0.382) m. Joint angles are from DROID episode 643, frame 196.

## Other conditions

- **Fake name** (`disguised`), **made-up lengths** (`madeup`, `conflict`), **novel arm** (`twisted`, `alien`): the same prompt with a different robot name and/or DH table. The novel arm swaps the twist angles of joints 5 and 7.
- **Name only** (`real`, condition `name`): the DH table is replaced by:

```
Robot: Franka Emika Panda, a 7-joint serial arm.
Use Franka's published kinematics (the modified DH parameters in the Franka documentation), with frame 0 at the robot base and the end-effector frame at the flange (no gripper attached).
```

- **Joint angles only** (`real`, condition `anon`):

```
Robot: a 7-joint serial arm. Its model and kinematic parameters are not given.
```

- **Multiple choice** (`fk_mc`) ends with:

```
Which of these is the position (x, y, z) of the end-effector frame origin in the base frame, in meters?
A: (0.502, 0.154, 0.382)
B: (-0.056, -0.055, 0.074)
C: (0.600, 0.184, 0.464)
D: (0.578, -0.210, 0.178)

Answer with only this JSON: {"answer": "<letter>"}
```

- **Inverse kinematics** (`ik`) ends with:

```
Find joint angles, within the joint limits, that put the end-effector frame origin at (x, y, z) = (0.502, 0.154, 0.382) m in the base frame. Orientation does not matter.
Answer with only this JSON, angles in radians: {"q1": <number>, "q2": <number>, "q3": <number>, "q4": <number>, "q5": <number>, "q6": <number>, "q7": <number>}
```
