# 如何发布 sm_89（4080S）r2 引擎包

`engine-sm89-cu128-r1` 的附件已删除，因为 **r1 的 `rk4v4-e8` 会输出乱码**
（缺注意力输出的逆旋转，见 `docs/FEATURES-4bit-kv-and-yarn.zh-CN.md` §三）。
替换它的 r2 需要**从真机打包**——不要用交叉编译的产物，那些二进制从未在 sm_89 上运行过。

## 打包（在 4080S 云机上执行）

引擎树是维护前那台机器上的 `/root/ninfer-e8`，构建目录 `/root/ninfer-e8/build-sm89-full`
（`NINFER_ENABLE_VISION=ON`，`CMAKE_CUDA_ARCHITECTURES=89`），它产出的二进制就是
全部 Ada 实测数据的来源。打包结构照 r1：

```bash
P=/tmp/rel89/ninfer-sm89-cu128
rm -rf /tmp/rel89 && mkdir -p $P/bin $P/lib
cd /root/ninfer-e8
cp build-sm89-full/apps/ninfer-serve build-sm89-full/apps/ninfer $P/bin/
# 引擎链接的 FFmpeg 6 / libcurl
cp -a /root/ninfer/ff6/lib/*.so*   $P/lib/
cp -a /root/curl-install/lib/*.so* $P/lib/ 2>/dev/null
# CUDA 运行时
cp -a /usr/local/cuda-12.8/lib64/libcudart.so.12* $P/lib/
# 用户态兜底（老 glibc 环境的 GLIBCXX 问题）
for l in libstdc++.so.6 libgcc_s.so.1; do
  src=$(LD_LIBRARY_PATH=$P/lib ldd $P/bin/ninfer-serve | grep -oE "/[^ ]*/$l" | head -1)
  [ -n "$src" ] && cp -aL "$src" $P/lib/
done
# README：写明 r2 修复清单、可用 kv-dtype、建议参数、实测上限
cd /tmp/rel89 && tar cf - ninfer-sm89-cu128 | xz -9 -T0 > /root/ninfer-sm89-cu128-r2-x86_64.tar.xz
sha256sum /root/ninfer-sm89-cu128-r2-x86_64.tar.xz
```

**打包前先冒烟验证**（确认这个二进制确实是修复版）：

```bash
LD_LIBRARY_PATH=/root/ninfer/ff6/lib:/root/curl-install/lib \
  /root/ninfer-e8/build-sm89-full/apps/ninfer-serve <artifact> \
  --host 127.0.0.1 --port 8099 --max-context 4096 --kv-capacity 4096 \
  --kv-dtype rk4v4-e8 --no-cuda-graph
# 然后问一句 "12+7 等于几？" —— r1 会输出乱码，r2 必须答对
```

## 上传（在本机执行，`gh` 凭证在本机）

```bash
scp -P <port> root@<host>:/root/ninfer-sm89-cu128-r2-x86_64.tar.xz /tmp/
cd <repo>   # QilinWan/Ninfer-Blackwell-Ada 的克隆
gh release create engine-sm89-cu128-r2 -R QilinWan/Ninfer-Blackwell-Ada \
  --title "NInfer engine sm_89 / CUDA 12.8 · vision ON (r2)" \
  --notes-file docs/maintainer/rtx4080s-32g/RELEASE-sm89-r2.md \
  /tmp/ninfer-sm89-cu128-r2-x86_64.tar.xz
sha256sum /tmp/ninfer-sm89-cu128-r2-x86_64.tar.xz | awk '{print $1}' > /tmp/sum.txt
gh release upload engine-sm89-cu128-r2 /tmp/sum.txt -R QilinWan/Ninfer-Blackwell-Ada
```

## 发布说明要点（写进 `RELEASE-sm89-r2.md`）

* 本版与 r1 同架构（sm_89 / Ada），但**带全部 E8 修复**——r1 的 E8 路线会输出乱码；
* **sm_89 的 4-bit KV**：`rk4v4-e8`，130 B/token/kv-head，比 int8 小约 2.5×；
* **上下文外推**：`--rope-yarn-factor 1..4` + `--rope-original-max-position 262144`，
  实测裸跑约 820K token、带多模态+投机+2 并发 512K 已验证（上限约 544K）；
* `nvfp4` / `k8v4` 在 Ada 上是显式桩（需 Blackwell），可用档位为
  `rk4v4-e8` / `fp8` / `int8` / `bf16`；
* 要求：sm_89 显卡、支持 CUDA 12.8 的驱动、glibc ≥ Ubuntu 22.04
  （已捆绑 `libstdc++.so.6` / `libgcc_s.so.1`）。

## 备用产物

5090 上曾交叉编译过一份 sm_89 包（`/root/autodl-tmp/ninfer-sm89-cu128-r2-x86_64.tar.xz`，
295 MiB，sha256 `ada8805a…`）。**未在 sm_89 上运行过，不要直接发布**；仅当 4080S 无法恢复时，
先按上面的冒烟步骤验证再考虑。
