# 数据收集流程说明

## 概述

`data_record.sh` 脚本用于收集机器人演示数据，包括视觉数据、机器人状态和动作数据。

## 启动方式

```bash
bash data_record.sh
```

脚本会：
1. 激活 `twist2` conda 环境
2. 切换到 `deploy_real` 目录
3. 运行 `server_data_record.py` 数据收集脚本

## 配置参数

在 `data_record.sh` 中可以配置：
- `robot_ip`: 机器人 IP 地址（默认: `192.168.123.164`）
- `data_frequency`: 数据收集频率（默认: `30 Hz`）

## 数据收集流程

### 1. 初始化阶段

#### 1.1 Redis 连接
- 连接到本地 Redis 服务器（`localhost:6379`）
- 使用连接池提高性能
- 创建 Redis pipeline 用于批量读取操作

#### 1.2 视觉数据接收
- 创建共享内存用于存储图像数据
- 图像尺寸：`(360, 1280, 3)` - 双相机拼接（每个相机 640x360）
- 启动 `VisionClient` 线程，通过 ZeroMQ 从机器人接收 JPEG 压缩图像
- 接收端口：`5555`
- 图像数据写入共享内存，供主循环读取

**图像数据来源**：
- 图像来自机器人上的 **ZED 相机服务器**
- ZED 服务器运行在机器人上（通过 `bash ~/g1-onboard/docker_zed.sh` 启动）
- ZED 服务器使用 **ZeroMQ PUB socket** 在端口 `5555` 上发布 JPEG 压缩图像
- 数据格式：`[4字节宽度][4字节高度][4字节JPEG长度][JPEG数据]`
- `VisionClient` 作为 **ZeroMQ SUB 客户端**连接到 `tcp://{robot_ip}:5555` 接收图像

#### 1.3 数据记录器初始化
- 创建 `EpisodeWriter` 实例
- 数据保存目录：`{data_folder}/{task_name}/`
- 默认任务名称：当前时间戳（格式：`YYYYMMDD_HHMM`）
- 支持的数据键：`['rgb']`

### 2. 主循环

#### 2.1 控制器输入处理
从 Redis 读取控制器数据（`controller_data`）：
- **开始/停止录制**：按下 `LeftController.key_two` 按钮
  - 第一次按下：开始录制新 episode
  - 第二次按下：停止录制并保存当前 episode
- **退出程序**：按下 `LeftController.axis_click` 按钮

#### 2.2 录制状态管理
- **未录制状态**：只显示图像，不收集数据
- **录制状态**：收集并保存所有数据

#### 2.3 数据收集（仅在录制状态下）

每次循环收集以下数据：

##### 视觉数据
- **RGB 图像**：从共享内存读取（`image_array.copy()`）
- **图像时间戳**：`t_img`（毫秒）

##### 从 Redis 读取的状态数据
使用 Redis pipeline 批量读取（一次网络往返）：
- `state_body_unitree_g1_with_hands` → `state_body`
- `state_hand_left_unitree_g1_with_hands` → `state_hand_left`
- `state_hand_right_unitree_g1_with_hands` → `state_hand_right`
- `state_neck_unitree_g1_with_hands` → `state_neck`
- `t_state` → `t_state`（状态时间戳）

##### 从 Redis 读取的动作数据
- `action_body_unitree_g1_with_hands` → `action_body`
- `t_action` → `t_action`（动作时间戳）

#### 2.4 数据保存
- 将收集的数据字典添加到 `EpisodeWriter` 的队列中
- `EpisodeWriter` 使用后台线程异步处理数据：
  - 保存 RGB 图像为 JPG 文件（`{episode_dir}/rgb/{idx:06d}.jpg`）
  - 将状态和动作数据保存到 JSON 文件（`{episode_dir}/data.json`）

### 3. Episode 管理

#### 3.1 创建新 Episode
- 按下控制器按钮开始录制时：
  - 创建新的 episode 目录：`episode_{episode_id:04d}`
  - 重置数据缓冲区
  - 创建 RGB 图像保存目录

#### 3.2 保存 Episode
- 按下控制器按钮停止录制时：
  - 等待队列中的所有数据处理完成
  - 将所有数据保存到 JSON 文件：
    ```json
    {
      "info": {
        "version": "1.0.0",
        "date": "2025-12-09",
        "author": "YanjieZe",
        "image": {"width": 1280, "height": 360, "fps": 30}
      },
      "text": {
        "goal": "...",
        "desc": "...",
        "steps": "..."
      },
      "data": [
        {
          "idx": 0,
          "rgb": "rgb/000000.jpg",
          "t_img": 1234567890,
          "state_body": [...],
          "state_hand_left": [...],
          "state_hand_right": [...],
          "state_neck": [...],
          "t_state": 1234567890,
          "action_body": [...],
          "t_action": 1234567890
        },
        ...
      ]
    }
    ```

## 数据格式

### 状态数据格式
- `state_body`: 机器人身体状态（29 个关节的位置、速度等）
- `state_hand_left`: 左手状态
- `state_hand_right`: 右手状态
- `state_neck`: 颈部状态

### 动作数据格式
- `action_body`: 机器人身体动作（29 个关节的目标位置）

## 数据保存位置

默认保存路径：
```
{data_folder}/{task_name}/episode_{episode_id:04d}/
├── rgb/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── data.json
```

默认 `data_folder` 在代码中硬编码为：
```
/home/ANT.AMAZON.COM/yanjieze/projects/TWIST2/TWIST2-clean/deploy_real/twist2_demonstration
```

## 性能优化

1. **Redis Pipeline**：使用批量读取减少网络往返次数
2. **共享内存**：图像数据通过共享内存传递，避免复制开销
3. **异步处理**：使用后台线程处理图像保存，不阻塞主循环
4. **频率控制**：按照设定的频率（默认 30 Hz）控制数据收集节奏

## 依赖关系

数据收集脚本依赖于以下组件：

1. **Redis 服务器**：必须运行在 `localhost:6379`
   - 其他脚本（如 `server_low_level_g1_real.py`）负责写入状态和动作数据

2. **机器人 ZED 相机服务器**：必须运行在机器人上（`robot_ip:5555`）
   - 通过 ZeroMQ PUB-SUB 模式发送 JPEG 压缩图像
   - 启动方式：在机器人上运行 `bash ~/g1-onboard/docker_zed.sh`
   - 使用 ZED 立体相机捕获图像（双相机拼接为 1280x360）
   - 图像通过 ZeroMQ PUB socket 发布，格式为：`[宽度][高度][JPEG长度][JPEG数据]`

3. **控制器**：用于控制录制开始/停止
   - 控制器数据通过 Redis 的 `controller_data` key 传递

## 使用示例

1. **启动数据收集**：
   ```bash
   bash data_record.sh
   ```

2. **开始录制**：
   - 按下控制器上的 `key_two` 按钮
   - 听到语音提示："episode recording started."

3. **执行演示任务**：
   - 操作机器人完成目标任务
   - 数据会自动收集并保存

4. **停止录制**：
   - 再次按下 `key_two` 按钮
   - 听到语音提示："episode saved."
   - 数据会保存到对应的 episode 目录

5. **退出程序**：
   - 按下 `axis_click` 按钮
   - 或按 `Ctrl+C`

## 注意事项

1. 确保 Redis 服务器正在运行
2. 确保机器人视觉服务器正在运行
3. 确保网络连接正常（能够访问机器人 IP）
4. 录制前检查磁盘空间是否充足
5. 每个 episode 的数据会保存在独立的目录中

