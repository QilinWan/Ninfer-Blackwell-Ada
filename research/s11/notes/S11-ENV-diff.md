# 环境/现场差异（2026-10-18 22:xx）

- 计划书假设两张 5060 Ti 空闲；现场两卡被本地 Qwen 栈占满：
  `/tank/Strata/bin/strata --serve ... --pipeline-windows 2`（PID 558171，15.4/15.7 GiB）。
- 该 server 有控制面：`POST /unload`（需 Origin: http://127.0.0.1:8084 + JSON body），
  返回 {"status":"unloaded"} 后两卡各降到 2 MiB；下个请求或 `POST /load` 会自动重载。
  本会话只在双卡窗口前 unload、窗口后 load，不动其文件与配置。
- 文件沙箱曾把 /dev/nvidia* 也挡住（landlock），已确认 policy=danger-full-access 下正常。
- deps：libcurl 8.11.0（无 OpenSSL，系统无头文件，按设计仅本地下载用）、FFmpeg 6.1.1
  （**--disable-x86asm**：nasm 不在且 nasm.us 经代理 SSL 失败；C 回退，功能完整）。
- 权重：HF 单连接 ~1 MB/s；aria2c 16 连接经 127.0.0.1:7890 可达 ~5 MB/s。
- 本会话接手全部包（notes/ 原为空，本地模型尚未落任何记录）。
