// Standalone reference for the Splash Q4 projection packing.
//
// Verbatim port of `parameter(n, g)` and `packSlab` from
//   refs/splash-src/dev/tests/engine/moe_metal_test.mm:98-142
// (including the identical LCG for input generation). Metal headers removed;
// logic unchanged. Used as an independent cross-check for converter/q4.py.
//
// Usage:
//   pack_slab_ref <seed> <out> <in> <outfile>
//     Generate inputs with the LCG below (nibbles, then scales, then biases),
//     pack, write section bytes.
//   pack_slab_ref @<inputfile> <out> <in> <outfile>
//     Read inputs from file: [nibbles u8 (out*in)] [scales f32 LE (out*in/64)]
//     [biases f32 LE (out*in/64)]; bf16 conversion done here via __bf16;
//     pack, write section bytes.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

class Random final {
 public:
  explicit Random(uint64_t seed) : state_(seed) {}
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }
  float unit() { return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F; }

 private:
  uint64_t state_;
};

float bf16(float value) { return float(__bf16(value)); }

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <seed|@inputfile> <out> <in> <outfile>\n", argv[0]);
    return 2;
  }
  const uint32_t out = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
  const uint32_t inputSize = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
  const uint32_t kStorageN = 256;
  if (out % kStorageN != 0 || inputSize % 64 != 0) {
    std::fprintf(stderr, "bad shape: out %u in %u\n", out, inputSize);
    return 2;
  }
  const uint64_t elements = uint64_t(out) * inputSize;
  const uint64_t parameters = elements / 64;
  const uint32_t groups = inputSize / 64;

  std::vector<uint8_t> nibbles(elements);
  std::vector<float> scales(parameters), biases(parameters);
  if (argv[1][0] == '@') {
    FILE *in = std::fopen(argv[1] + 1, "rb");
    if (!in) return 2;
    if (std::fread(nibbles.data(), 1, nibbles.size(), in) != nibbles.size()) return 2;
    if (std::fread(scales.data(), 1, scales.size() * 4, in) != scales.size() * 4) return 2;
    if (std::fread(biases.data(), 1, biases.size() * 4, in) != biases.size() * 4) return 2;
    std::fclose(in);
  } else {
    Random random(std::strtoull(argv[1], nullptr, 10));
    for (uint8_t &v : nibbles) v = static_cast<uint8_t>(random.next() & 15U);
    for (uint64_t i = 0; i < parameters; ++i) {
      scales[i] = bf16(0.01F + 0.01F * (random.unit() + 1.0F));
      biases[i] = bf16(-0.2F + 0.05F * random.unit());
    }
  }

  auto parameter = [inputSize](uint32_t n, uint32_t g) -> uint64_t {
    const uint32_t groups = inputSize / 64;
    return (uint64_t(n / kStorageN) * groups + g) * kStorageN + n % kStorageN;
  };

  // packSlab (moe_metal_test.mm:124-142), unchanged.
  std::vector<uint8_t> destination(elements * 9 / 16);
  auto *sc = reinterpret_cast<__bf16 *>(destination.data() + elements / 2);
  auto *bi = reinterpret_cast<__bf16 *>(destination.data() + elements / 2 + elements / 32);
  for (uint32_t n = 0; n < out; ++n) {
    for (uint32_t k = 0; k < inputSize; ++k) {
      const uint64_t nibble = parameter(n, k / 64) * 64 + k % 64;
      uint8_t &byte = destination[nibble / 2];
      const uint8_t value = nibbles[uint64_t(n) * inputSize + k];
      byte = (nibble & 1) ? (byte & 0x0F) | (value << 4) : (byte & 0xF0) | value;
    }
    for (uint32_t g = 0; g < groups; ++g) {
      sc[parameter(n, g)] = __bf16(scales[parameter(n, g)]);
      bi[parameter(n, g)] = __bf16(biases[parameter(n, g)]);
    }
  }

  FILE *o = std::fopen(argv[4], "wb");
  if (!o) return 2;
  if (std::fwrite(destination.data(), 1, destination.size(), o) != destination.size()) return 2;
  std::fclose(o);
  std::printf("packed %llu bytes (out=%u in=%u)\n",
              (unsigned long long)destination.size(), out, inputSize);
  return 0;
}
