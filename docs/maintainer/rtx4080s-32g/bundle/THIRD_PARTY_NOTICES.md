# Third-party notices

This archive is a prebuilt NInfer engine. NInfer itself is licensed under the Apache License 2.0
(see `LICENSE`). The archive additionally redistributes the following components.

## FFmpeg 6.1 — LGPL-2.1-or-later

`lib/libavformat.so.60`, `lib/libavcodec.so.60`, `lib/libavutil.so.58`,
`lib/libswscale.so.7`, `lib/libswresample.so.4`

Built with the default (LGPL) configuration, i.e. without `--enable-gpl` or `--enable-nonfree`.
Source: <https://ffmpeg.org/releases/ffmpeg-6.1.tar.xz>. The license text is in `LGPL-2.1.txt`.

These libraries are **dynamically linked and shipped as separate replaceable shared objects**
under `lib/`. They are not statically linked into `bin/ninfer-serve`, so they can be replaced
without relinking the engine — including with a build of your own:

```sh
cp /your/ffmpeg/lib/libav*.so.60* ./lib/
./start.sh /path/to/model.ninfer --vision
```

## NVIDIA CUDA Runtime — NVIDIA Software License Agreement

`lib/libcudart.so.12` (CUDA 12.8.90)

Redistributed under the NVIDIA CUDA Toolkit End User License Agreement, which permits
redistribution of the runtime libraries with an application built using the toolkit.
Source: <https://developer.nvidia.com/cuda-downloads>.

## libcurl 8.x — curl license (MIT/X derivate)

`lib/libcurl.so.4`

Used for HTTPS model/artifact fetching. Source: <https://curl.se/download/>.

## GNU C++ runtime — GPL-3.0-or-later with the GCC Runtime Library Exception

`lib/libstdc++.so.6`, `lib/libgcc_s.so.1`

Redistributed under the GCC Runtime Library Exception, which permits this without affecting the
license of the engine. Bundling them keeps the engine's `GLIBCXX_3.4.30` requirement satisfied
from inside the archive: a conda environment that puts an older `libstdc++` (3.4.29) on
`LD_LIBRARY_PATH` would otherwise shadow the host copy and the engine would fail to load.
