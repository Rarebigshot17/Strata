// src/glm/cpu_dot.cpp - faster CPU dot products, bitwise ggml-cpu's (cpu_dot.hpp). Built with -mavx2 -mfma -mf16c
// (CMakeLists.txt); fast_vec_dot only hands them out on a CPU that has those.
#include "strata/glm/cpu_dot.hpp"

#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <cstring>

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#define STRATA_CPU_DOT 1
#endif

namespace strata::glm {

#if STRATA_CPU_DOT
namespace {

// ggml-cpu's hsum_float_8 (arch/x86/quants.c), the same adds in the same order
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// ggml_vec_dot_iq3_s_q8_K's AVX2 path with the grid values gathered instead of set from scalar loads
void vec_dot_iq3_s_q8_K(int n, float* s, size_t, const void* vx, size_t, const void* vy, size_t, int) {
    const block_iq3_s* x = static_cast<const block_iq3_s*>(vx);
    const block_q8_K* y = static_cast<const block_q8_K*>(vy);
    const int nb = n / QK_K;
    static const uint8_t k_mask1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    static const uint8_t k_mask2[32] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
                                        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const __m256i mask1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_mask1));
    const __m256i mask2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_mask2));
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask = _mm256_set1_epi32(256);
    const int* grid = reinterpret_cast<const int*>(iq3s_grid);

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = _cvtsh_ss(x[i].d) * y[i].d;
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint16_t* signs = reinterpret_cast<const uint16_t*>(x[i].signs);
        const int8_t* q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
            q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(qs)));
            qs += 16;
            __m256i idx0 = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32 + 0]), idx_shift), idx_mask);
            __m256i idx1 = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32 + 1]), idx_shift), idx_mask);
            idx0 = _mm256_or_si256(idx0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx1 = _mm256_or_si256(idx1, _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_i32gather_epi32(grid, idx0, 4);
            const __m256i q2_2 = _mm256_i32gather_epi32(grid, idx1, 4);

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32 / 2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32 / 2] >> 4;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

}  // namespace
#endif

VecDotFn fast_vec_dot(int type) {
#if STRATA_CPU_DOT
    static const bool cpu_ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                               __builtin_cpu_supports("f16c");
    if (cpu_ok && type == GGML_TYPE_IQ3_S) return vec_dot_iq3_s_q8_K;
#else
    (void) type;
#endif
    return nullptr;
}

}  // namespace strata::glm
