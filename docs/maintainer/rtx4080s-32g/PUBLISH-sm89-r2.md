# sm_89（4080S）引擎包的打包与发布

**状态：`engine-sm89-cu128-r2` 已于 2026-09-26 从 4080S 真机打包并发布。**
`engine-sm89-cu128-r1` 的附件已下架，因为 **r1 的 `rk4v4-e8` 会输出乱码**
（缺注意力输出的逆旋转，见 `docs/FEATURES-4bit-kv-and-yarn.zh-CN.md` §三）。

r2 的发布说明在 [`RELEASE-sm89-r2.md`](RELEASE-sm89-r2.md)，包内文档模板在
[`bundle/`](bundle/)（`start.sh`、`README.md`、`THIRD_PARTY_NOTICES.md`）。
本文件留下的是**复现步骤**与**这台机器上的两个坑**。

## 一、打包前：确认二进制确实是修复版

源码树 `/root/ninfer-e8`（HEAD `b767a17c`，树哈希
`1589d092f74db6d65a7eef95222d101cd57941a5`，与已发布的 `381f43d1` 内容一致），
构建目录 `/root/ninfer-e8/build-sm89-full`
（`-DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_BUILD_APPS=ON -DNINFER_ENABLE_VISION=ON`）。

```bash
# 二进制必须是"源码改动之后"链接出来的
stat -c '%y %n' /root/ninfer-e8/src/models/qwen3_5/program/planning/startup.cpp \
                /root/ninfer-e8/build-sm89-full/apps/ninfer-serve
ninja -n -C /root/ninfer-e8/build-sm89-full apps/ninfer-serve   # 应无待重建项
```

**必须在有卡启动时冒烟**（无卡模式跑不了）：

```bash
LD_LIBRARY_PATH=/root/ninfer/ff6/lib:/root/curl-install/lib \
  /root/ninfer-e8/build-sm89-full/apps/ninfer-serve <artifact> \
  --host 127.0.0.1 --port 8099 --max-context 4096 --kv-capacity 4096 \
  --kv-dtype rk4v4-e8 --no-cuda-graph
curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"12+7 等于几？"}],"max_tokens":64}'
```

r1 会输出乱码，r2 必须答对；同时确认启动日志里有 `capacity | KV ..., rk4v4-e8, auto`。

## 二、打包

包的形态与 r1 一致：**只含服务端引擎**（`apps/ninfer` CLI 的参数解析器不接受
`rk4v4-e8`，见 `README.md` 支持矩阵脚注）。

```bash
P=/root/rel89/ninfer-sm89-cu128
rm -rf /root/rel89 && mkdir -p $P/bin $P/lib
cp -a /root/ninfer-e8/build-sm89-full/apps/ninfer-serve $P/bin/
cp -aL /root/ninfer/ff6/lib/lib{avformat.so.60*,avcodec.so.60*,avutil.so.58*,swscale.so.7*,swresample.so.4*} $P/lib/
cp -aL /root/curl-install/lib/libcurl.so.4* $P/lib/
cp -aL /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12* $P/lib/
cp -aL /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 $P/lib/
cp -a /root/ninfer-e8/LICENSE $P/LICENSE
cp -a /usr/share/common-licenses/LGPL-2.1 $P/LGPL-2.1.txt
# 从本机上传 bundle/start.sh bundle/README.md bundle/THIRD_PARTY_NOTICES.md，start.sh 置 755
```

`libstdc++`/`libgcc_s` 必须一起打：conda 环境的 `libstdc++` 是 3.4.29，会把宿主的
`GLIBCXX_3.4.30` 顶掉。

## 三、这台机器上的两个坑

1. **无卡模式的容器被限成 0.5 CPU / 2 GiB**（`/sys/fs/cgroup/cpu.max` = `50000 100000`，
   `memory.max` = 2 GiB）。在这里编译基本不可行：`ninja -j32` 会连环 `c++: fatal error:
   Killed signal terminated program cc1plus`（`memory.events` 的 `oom_kill` 一路涨）。
   实测 `ninja -j1 apps/ninfer` 这种"只重链接"的活能过，全量重编不要试。
2. **xz 也一样被掐死**：`xz -6 -T4` 跑 10 分钟只出 33 MB。正确做法是
   **把 tar 流出来、在本机压缩**：

```bash
ssh -p <port> root@<host> 'cd /root/rel89 && tar cf - ninfer-sm89-cu128' \
  | xz -9 -T16 -c > ninfer-sm89-cu128-r2-x86_64.tar.xz
sha256sum ninfer-sm89-cu128-r2-x86_64.tar.xz
xz -t ninfer-sm89-cu128-r2-x86_64.tar.xz && tar tJf ninfer-sm89-cu128-r2-x86_64.tar.xz
```

## 四、上传（在本机执行，`gh` 凭证在本机）

```bash
sha256sum ninfer-sm89-cu128-r2-x86_64.tar.xz \
  | sed 's#  .*/#  #' > ninfer-sm89-cu128-r2-x86_64.tar.xz.sha256
gh release create engine-sm89-cu128-r2 -R QilinWan/Ninfer-Blackwell-Ada \
  --title "NInfer engine sm_89 / CUDA 12.8 · vision ON (r2)" \
  --notes-file docs/maintainer/rtx4080s-32g/RELEASE-sm89-r2.md \
  ninfer-sm89-cu128-r2-x86_64.tar.xz ninfer-sm89-cu128-r2-x86_64.tar.xz.sha256
```

## 五、备用产物

5090 上曾交叉编译过一份 sm_89 包（sha256 `ada8805a…`）。**它从未在 sm_89 上运行过，不要发布**；
只有 4080S 无法恢复时才考虑，且必须先按第一节冒烟。那个 5090 实例已停机，产物大概也取不到了。
