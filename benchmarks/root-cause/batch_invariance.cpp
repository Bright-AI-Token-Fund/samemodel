// Does ggml's CPU backend give the same result for an activation row no matter how many
// other rows are multiplied in the same call?
//
// For each weight type, W * x is computed for every activation vector on its own, and then
// for the first n vectors together (n = 2 ... 96). Each column of the batched result is
// compared with the result for the same vector alone. The weights are placed in the CPU
// "repack" buffer type, as llama.cpp does when a repacked kernel exists, and in the plain
// CPU buffer type.
//
// Only the CPU backend runs here. The BLAS backend, which llama.cpp also uses on macOS for
// batches of 32 rows or more, is not part of this test.
//
// Build against a llama.cpp release (headers of the same commit, release libraries):
//   clang++ -std=c++17 -O2 -I $SRC/ggml/include batch_invariance.cpp \
//       -L $REL -lggml -lggml-base -lggml-cpu -Wl,-rpath,$REL -o batch_invariance
// Run:
//   ./batch_invariance [n_in] [n_out] [threads] [seed]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// y = W * X on the CPU backend. Returns an empty vector if `repack` is asked for and this
// CPU has no repacked kernel for the type.
static std::vector<float> mul(ggml_backend_t cpu, ggml_backend_buffer_type_t wbuft, bool repack, ggml_type type,
                              const std::vector<uint8_t> & wq, int n_in, int n_out, const float * x, int n_tok) {
    ggml_init_params ip = {ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, type, n_in, n_out);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_tok);
    ggml_set_input(X);

    ggml_backend_buffer_t wbuf = ggml_backend_buft_alloc_buffer(wbuft, ggml_backend_buft_get_alloc_size(wbuft, W));
    ggml_backend_tensor_alloc(wbuf, W, ggml_backend_buffer_get_base(wbuf));
    std::vector<float> y;
    if (!repack || W->extra != nullptr) {
        ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
        ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
        ggml_set_output(Y);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, Y);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(X, x, 0, (size_t) n_in * n_tok * sizeof(float));
        ggml_backend_graph_compute(cpu, gf);
        y.resize((size_t) n_out * n_tok);
        ggml_backend_tensor_get(Y, y.data(), 0, y.size() * sizeof(float));
        ggml_gallocr_free(ga);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    return y;
}

int main(int argc, char ** argv) {
    const int n_in = argc > 1 ? atoi(argv[1]) : 4864;
    const int n_out = argc > 2 ? atoi(argv[2]) : 896;
    const int threads = argc > 3 ? atoi(argv[3]) : 4;
    const int seed = argc > 4 ? atoi(argv[4]) : 1;
    const int max_tok = 96;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, threads);
    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t repack = nullptr;
    if (get_extra) {
        for (ggml_backend_buffer_type_t * b = get_extra(dev); b && *b; b++) {
            if (std::string(ggml_backend_buft_name(*b)).find("REPACK") != std::string::npos) repack = *b;
        }
    }
    ggml_backend_buffer_type_t plain = ggml_backend_get_default_buffer_type(cpu);
    printf("neon=%d dotprod=%d matmul_int8=%d avx2=%d avx512=%d  n_in=%d n_out=%d threads=%d seed=%d\n", ggml_cpu_has_neon(),
           ggml_cpu_has_dotprod(), ggml_cpu_has_matmul_int8(), ggml_cpu_has_avx2(), ggml_cpu_has_avx512(), n_in, n_out, threads, seed);
    if (repack) { // which weight types have a repacked kernel on this CPU, for this shape
        const ggml_type all[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K,
                                 GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
                                 GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
                                 GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32};
        std::string yes, no;
        for (ggml_type t : all) {
            if (n_in % ggml_blck_size(t) != 0) continue;
            ggml_init_params ip = {ggml_tensor_overhead() * 4, nullptr, true};
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * W = ggml_new_tensor_2d(ctx, t, n_in, n_out);
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(repack, ggml_backend_buft_get_alloc_size(repack, W));
            ggml_backend_tensor_alloc(b, W, ggml_backend_buffer_get_base(b));
            (W->extra ? yes : no) += std::string(" ") + ggml_type_name(t);
            ggml_backend_buffer_free(b);
            ggml_free(ctx);
        }
        printf("repacked kernel for this shape:%s\nno repacked kernel:%s\n", yes.c_str(), no.c_str());
    }
    printf("every column of a batch is compared with the same vector multiplied alone\n\n");

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::vector<float> wf((size_t) n_in * n_out), xs((size_t) n_in * max_tok);
    for (auto & v : wf) v = 0.05f * nd(rng);
    for (auto & v : xs) v = nd(rng) * (ud(rng) < 0.01f ? 20.0f : 1.0f); // 1% outliers, as real activations have

    const ggml_type types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_F16};
    printf("%-6s %-7s %5s  %-22s %s\n", "type", "buffer", "rows", "max |batch - alone|", "columns that differ");
    for (ggml_type type : types) {
        if (n_in % ggml_blck_size(type) != 0) {
            printf("%-6s skipped: n_in is not a multiple of its block size\n\n", ggml_type_name(type));
            continue;
        }
        std::vector<uint8_t> wq(ggml_row_size(type, n_in) * (size_t) n_out);
        ggml_quantize_chunk(type, wf.data(), wq.data(), 0, n_out, n_in, nullptr);
        for (int pass = 0; pass < 2; pass++) {
            const bool want_repack = pass == 0;
            ggml_backend_buffer_type_t buft = want_repack ? repack : plain;
            const char * name = want_repack ? "repack" : "plain";
            if (!buft) continue;
            std::vector<std::vector<float>> alone(max_tok);
            for (int t = 0; t < max_tok; t++) {
                alone[t] = mul(cpu, buft, want_repack, type, wq, n_in, n_out, xs.data() + (size_t) t * n_in, 1);
                if (alone[t].empty()) break;
            }
            if (alone[0].empty()) {
                printf("%-6s %-7s     -  no repacked kernel for this type on this CPU\n", ggml_type_name(type), name);
                continue;
            }
            for (int n_tok : {2, 3, 4, 5, 8, 9, 16, 17, 31, 32, 33, 64, 65, 96}) {
                std::vector<float> y = mul(cpu, buft, want_repack, type, wq, n_in, n_out, xs.data(), n_tok);
                double worst = 0;
                int n_diff = 0;
                for (int t = 0; t < n_tok; t++) {
                    double rms = 0, m = 0;
                    for (int i = 0; i < n_out; i++) {
                        rms += (double) alone[t][i] * alone[t][i];
                        m = std::fmax(m, std::fabs((double) y[(size_t) t * n_out + i] - alone[t][i]));
                    }
                    m /= std::sqrt(rms / n_out); // relative to the RMS of the column
                    worst = std::fmax(worst, m);
                    n_diff += m > 0;
                }
                printf("%-6s %-7s %5d  %-22.3e %d of %d\n", ggml_type_name(type), name, n_tok, worst, n_diff, n_tok);
            }
        }
        printf("\n");
    }
    ggml_backend_free(cpu);
    return 0;
}
