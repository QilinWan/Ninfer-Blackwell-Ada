// Stand-in for the FFmpeg-backed decoder when the project is configured with
// NINFER_ENABLE_VISION=OFF. It keeps the media decode contract linkable for
// text-only builds and reports the missing capability at request time instead
// of at configure time.
#include "media/decode/decode.h"

namespace ninfer::media::decode {

namespace {

[[noreturn]] void unavailable() {
    throw Error(ErrorKind::InvalidInput,
                "this NInfer build was configured with NINFER_ENABLE_VISION=OFF, so image and "
                "video decoding is unavailable; reconfigure with -DNINFER_ENABLE_VISION=ON and "
                "the FFmpeg development packages installed");
}

} // namespace

ImageInfo inspect_image(std::span<const std::uint8_t>, const Policy&) { unavailable(); }

VideoInfo inspect_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unavailable();
}

Image decode_image(std::span<const std::uint8_t>, const Policy&) { unavailable(); }

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unavailable();
}

} // namespace ninfer::media::decode
