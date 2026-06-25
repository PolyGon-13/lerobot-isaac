# Transformation Matrices - Issac_sim_6DOF_ARM

Homogeneous transformation matrices between consecutive frames.
Convention: URDF RPY (XYZ extrinsic / ZYX intrinsic).

## Notation

### Frames

| Index | Link |
|-------|------|
| $L_{0}$ | base_link |
| $L_{1}$ | J5 |
| $L_{2}$ | J1 |
| $L_{3}$ | gripper |
| $L_{4}$ | J2 |
| $L_{5}$ | J3 |
| $L_{6}$ | J4 |

### Joint Variables

| Variable | Joint | Type | From | To |
|----------|-------|------|------|----|
| $q_{1}$ | base | revolute (rad) | $L_{0}$ | $L_{2}$ |
| $q_{2}$ | gripper | revolute (rad) | $L_{1}$ | $L_{3}$ |
| $q_{3}$ | shoulder | revolute (rad) | $L_{2}$ | $L_{4}$ |
| $q_{4}$ | elbow | revolute (rad) | $L_{4}$ | $L_{5}$ |
| $q_{5}$ | wrist_pitch | revolute (rad) | $L_{5}$ | $L_{6}$ |

Shorthand: $c_i = \cos(q_i)$, $s_i = \sin(q_i)$

### Kinematic Tree

```
L0: base_link
  |-- [fixed] base_base_xl430
  |   L1: J5
  |     +-- [revolute] gripper (q2)
  |         L3: gripper
  +-- [revolute] base (q1)
      L2: J1
        +-- [revolute] shoulder (q3)
            L4: J2
              +-- [revolute] elbow (q4)
                  L5: J3
                    +-- [revolute] wrist_pitch (q5)
                        L6: J4
```

## Transforms

## base_base_xl430

$L_{0}$ **base_link** -> $L_{1}$ **J5** (fixed)

- **origin xyz**: (0.074455, 0.150022, 0.144295) m
- **origin rpy**: (0, 0, 0) rad

### Local Transform

$$
T^{0}_{1} = \begin{bmatrix}
1 & 0 & 0 & 0.074455 \\
0 & 1 & 0 & 0.150022 \\
0 & 0 & 1 & 0.144295 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## base

$L_{0}$ **base_link** -> $L_{2}$ **J1** (revolute)
  Variable: $q_{1}$

- **origin xyz**: (0.093965, 0.007519, 0.0639) m
- **origin rpy**: (1.570796, 0, 1.570796) rad
- **axis**: (0, 1, 0)
- **limits**: [1.570796, 4.712389] rad ([90deg, 270deg])

### Local Transform

$T^{0}_{2}(q_{1}) = T_{fixed} \cdot R_{axis}(q_{1})$ where:

$$
T_{fixed} = \begin{bmatrix}
0 & 0 & 1 & 0.093965 \\
1 & 0 & 0 & 0.007519 \\
0 & 1 & 0 & 0.0639 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{1}) = \begin{bmatrix}
c_{1} & 0 & s_{1} & 0 \\
0 & 1 & 0 & 0 \\
-s_{1} & 0 & c_{1} & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## gripper

$L_{1}$ **J5** -> $L_{3}$ **gripper** (revolute)
  Variable: $q_{2}$

- **origin xyz**: (0, 0, 0.02221) m
- **origin rpy**: (0, 0, -3.141593) rad
- **axis**: (0, 0, 1)
- **limits**: [-1.570796, 0.087266] rad ([-90deg, 5deg])

### Local Transform

$T^{1}_{3}(q_{2}) = T_{fixed} \cdot R_{axis}(q_{2})$ where:

$$
T_{fixed} = \begin{bmatrix}
-1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & 1 & 0.02221 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{2}) = \begin{bmatrix}
c_{2} & -s_{2} & 0 & 0 \\
s_{2} & c_{2} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## shoulder

$L_{2}$ **J1** -> $L_{4}$ **J2** (revolute)
  Variable: $q_{3}$

- **origin xyz**: (0.0465, 0.0451, 0.0221) m
- **origin rpy**: (0, 0, 0) rad
- **axis**: (0, 0, 1)
- **limits**: [-1.570796, 1.570796] rad ([-90deg, 90deg])

### Local Transform

$$
T^{2}_{4}(q_{3}) = \begin{bmatrix}
c_{3} & -s_{3} & 0 & 0.0465 \\
s_{3} & c_{3} & 0 & 0.0451 \\
0 & 0 & 1 & 0.0221 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## elbow

$L_{4}$ **J2** -> $L_{5}$ **J3** (revolute)
  Variable: $q_{4}$

- **origin xyz**: (-0.093581, 0.03525, 0) m
- **origin rpy**: (0, 0, -1.570796) rad
- **axis**: (0, 0, 1)
- **limits**: [-1.570796, 1.570796] rad ([-90deg, 90deg])

### Local Transform

$T^{4}_{5}(q_{4}) = T_{fixed} \cdot R_{axis}(q_{4})$ where:

$$
T_{fixed} = \begin{bmatrix}
0 & 1 & 0 & -0.093581 \\
-1 & 0 & 0 & 0.03525 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{4}) = \begin{bmatrix}
c_{4} & -s_{4} & 0 & 0 \\
s_{4} & c_{4} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## wrist_pitch

$L_{5}$ **J3** -> $L_{6}$ **J4** (revolute)
  Variable: $q_{5}$

- **origin xyz**: (-0.000045, 0.099984, -0.00001) m
- **origin rpy**: (-1.570796, 0, 0) rad
- **axis**: (0, -1, 0)
- **limits**: [-3.141593, 0] rad ([-180deg, 0deg])

### Local Transform

$T^{5}_{6}(q_{5}) = T_{fixed} \cdot R_{axis}(q_{5})$ where:

$$
T_{fixed} = \begin{bmatrix}
1 & 0 & 0 & -0.000045 \\
0 & 0 & 1 & 0.099984 \\
0 & -1 & 0 & -0.00001 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{5}) = \begin{bmatrix}
c_{5} & 0 & -s_{5} & 0 \\
0 & 1 & 0 & 0 \\
s_{5} & 0 & c_{5} & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## Global Transform Chains

Transform from root $L_0$ to any link, as product of local transforms along the kinematic chain.

$$T^{0}_{3} = T^{0}_{1} \cdot T^{1}_{3}(q_{2})\quad (L_0 \to L_{3}: \text{gripper})$$

$$T^{0}_{4} = T^{0}_{2}(q_{1}) \cdot T^{2}_{4}(q_{3})\quad (L_0 \to L_{4}: \text{J2})$$

$$T^{0}_{5} = T^{0}_{2}(q_{1}) \cdot T^{2}_{4}(q_{3}) \cdot T^{4}_{5}(q_{4})\quad (L_0 \to L_{5}: \text{J3})$$

$$T^{0}_{6} = T^{0}_{2}(q_{1}) \cdot T^{2}_{4}(q_{3}) \cdot T^{4}_{5}(q_{4}) \cdot T^{5}_{6}(q_{5})\quad (L_0 \to L_{6}: \text{J4})$$

