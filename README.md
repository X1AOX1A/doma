# doma 🔥

> A smart GPU resource manager that automatically holds idle GPU memory and maintains controlled utilization to prevent resource preemption.

**doma** (DOg in the MAnager) is a lightweight daemon tool designed to intelligently occupy idle GPU resources. It monitors GPU usage patterns and automatically claims memory and maintains specified utilization levels when GPUs become idle, preventing resource preemption because of low utilization.

## ✨ Features

- **🤖 Automatic GPU Detection**: Monitors all available CUDA GPUs automatically
- **⏱️ Smart Idle Detection**: Waits for configurable idle periods before claiming resources
- **🎛️ Precise Utilization Control**: Maintains target GPU utilization using adaptive algorithms
- **💾 Memory Management**: Configurable memory holding with automatic cleanup
- **🔧 Daemon Architecture**: Runs as a background service with socket-based control
- **📊 Real-time Monitoring**: Continuous tracking of GPU memory and utilization metrics
- **🛡️ Safe Resource Handling**: Graceful cleanup and release of GPU resources

## 🚀 Quick Start

### Installation

```bash
# Install using uv (recommended)
git clone <repository-url>
cd doma
uv tool install ./
```

### Basic Usage

1. **Launch the doma server**:
   ```bash
   doma launch
   ```

2. **Start holding idle GPUs**:
   ```bash
   doma start
   ```

3. **Check server status**:
   ```bash
   doma status
   ```

4. **Stop holding GPUs** (keeps server running):
   ```bash
   doma stop
   ```

5. **Shutdown the server**:
   ```bash
   doma shutdown
   ```

## 📋 Commands

### `doma launch`
Starts the doma daemon server in the background.

**Options:**
- `--log-path`: Path to log file (default: `/tmp/doma/doma.log`)

### `doma start`
Begins monitoring and holding idle GPUs with specified configuration.

**Options:**
- `--wait-minutes`: Minutes to wait before holding GPU (default: 10)
- `--mem-threshold`: Memory threshold in GB for idle detection (default: 0.5)
- `--hold-mem`: Memory to hold in GB (default: 50% of free memory)
- `--hold-util`: Target GPU utilization to maintain (0-1, default: 0.5)

**Algorithm Options:**
- `--operator-gb`: Operator size in GB for control precision (default: 1.0)
- `--util-eps`: Utilization epsilon for convergence (default: 0.01)
- `--max-sleep-time`: Init maximum sleep time in seconds of binary search (default: 1)
- `--min-sleep-time`: Init minimum sleep time in seconds of binary search (default: 0)
- `--inspect-interval`: Interval in seconds to inspect GPU utilization during binary search (default: 1)
- `--util-samples-num`: Number of samples to take for utilization during binary search (default: 5)

### `doma restart`
Releases all GPUs and restarts with new configuration.

### `doma stop`
Stops holding GPUs and releases all resources (server continues running).

### `doma shutdown`
Completely shuts down the doma server.

### `doma status`
Shows current server status.

## 🎯 How It Works

### 1. Idle Detection
Doma continuously monitors each GPU's memory usage and utilization. A GPU is considered "idle" when:
- Memory usage stays below the configured threshold (`--mem-threshold`) 
- This condition persists for the specified waiting period (`--wait-minutes`)

### 2. Resource Holding
When a GPU becomes idle, doma:
- Allocates the specified amount of memory (`--hold-mem`)
- Maintains target utilization (`--hold-util`) through controlled compute operations
- Uses adaptive algorithms to precisely control utilization levels

### 3. Smart Release
Resources are automatically released when:
- The `stop` command is issued
- The server is shut down

## ⚙️ Configuration Examples

### Conservative Setup (Light Resource Usage)
```bash
doma start --wait-minutes 15 --hold-util 0.3 --hold-mem 2.0
```

### Aggressive Setup (Maximum Resource Claiming)
```bash
doma start --wait-minutes 5 --hold-util 0.8 --mem-threshold 0.1
```

### High Precision Control
```bash
doma start --util-eps 0.005 --operator-gb 0.5 --util-samples-num 10
```

## 🔧 Advanced Usage

### Custom Log Location
```bash
doma launch --log-path /var/log/doma/doma.log
```

### Dynamic Configuration Updates
```bash
# Change configuration without restarting server
doma restart --hold-util 0.7 --wait-minutes 5
```

### Production Deployment
```bash
# Launch with custom log path
doma launch --log-path /opt/doma/logs/doma.log

# Start with production settings
doma start --wait-minutes 20 --hold-util 0.6 --mem-threshold 0.5
```

## 🛠️ Development

### Requirements
- Python ≥ 3.11
- CUDA-capable GPU(s)
- PyTorch with CUDA support
- NVIDIA drivers


### Development Installation
```bash
git clone <repository-url>
cd doma
uv sync --group dev
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

- **Resource Management**: Doma is designed for responsible resource sharing. Always ensure you have permission to use GPU resources in shared environments.
- **Memory Safety**: The tool includes automatic cleanup mechanisms, but system crashes may require manual GPU memory cleanup.
- **Compatibility**: Requires NVIDIA GPUs with CUDA support. AMD GPUs are not currently supported.
- **Performance Impact**: Holding operations use minimal resources but may slightly impact system performance.

## 🆘 Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check if socket file exists
ls -la /tmp/doma/
# Remove if necessary
rm -f /tmp/doma/doma.sock
```

**GPU memory not released:**
```bash
# Force shutdown and restart
doma shutdown
# Wait a moment, then relaunch
doma launch
```

**Permission issues:**
```bash
# Ensure proper CUDA permissions
nvidia-smi
# Check if user has access to CUDA devices
```

---

**Author**: TideDra (gearyzhang@outlook.com)  
**Version**: 0.1.0