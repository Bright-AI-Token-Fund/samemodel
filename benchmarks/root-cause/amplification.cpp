// How much does one matrix multiplication in ggml's CPU backend amplify a tiny perturbation
// of its input?
//
// The activations x are multiplied by W as they are, and again after a relative perturbation
// of size eps: x_j * (1 + eps * u_j), u uniform in [-1, 1]. With float weights the output
// moves by about eps. With quantised weights ggml first rounds the activations to 8 bits; a
// perturbation that pushes one value across a rounding boundary moves it by a whole step.
// The last column is the distance between that kernel and the same product computed with
// the activations left as float, which is what the BLAS backend does for large batches.
//
// Build against a llama.cpp release (headers of the same commit, release libraries):
//   clang++ -std=c++17 -O2 -I $SRC/ggml/include amplification.cpp \
//       -L $REL -lggml -lggml-base -lggml-cpu -Wl,-rpath,$REL -o amplification
// Run:
//   ./amplification [n_in] [n_out] [seed]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<float> mul(ggml_backend_t cpu, ggml_backend_buffer_type_t wbuft, bool repack, ggml_type type,
                              const std::vector<uint8_t> & wq, int n_in, int n_out, const float * x) {
    ggml_init_params ip = {ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, type, n_in, n_out);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, 1);
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
        ggml_backend_tensor_set(X, x, 0, (size_t) n_in * sizeof(float));
        ggml_backend_graph_compute(cpu, gf);
        y.resize(n_out);
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
    const int seed = argc > 3 ? atoi(argv[3]) : 1;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
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

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::vector<float> wf((size_t) n_in * n_out), x(n_in), u(n_in);
    for (auto & v : wf) v = 0.05f * nd(rng);
    for (auto & v : x) v = nd(rng);
    for (auto & v : u) v = ud(rng);

    printf("n_in=%d n_out=%d seed=%d. Relative RMS change of W*x when x is perturbed by eps.\n\n", n_in, n_out, seed);
    const double epss[] = {1e-7, 4e-7, 1e-6, 1e-5, 1e-4};
    printf("%-6s", "type");
    for (double e : epss) printf("  eps=%-8.0e", e);
    printf("  activations         error against float activations\n");
    const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K};
    for (ggml_type type : types) {
        if (n_in % ggml_blck_size(type) != 0) continue;
        std::vector<uint8_t> wq(ggml_row_size(type, n_in) * (size_t) n_out);
        if (type == GGML_TYPE_F32) {
            memcpy(wq.data(), wf.data(), wq.size());
        } else {
            ggml_quantize_chunk(type, wf.data(), wq.data(), 0, n_out, n_in, nullptr);
        }
        // the repack buffer when the type has a repacked kernel here, as llama.cpp does
        bool use_repack = repack != nullptr;
        std::vector<float> base = use_repack ? mul(cpu, repack, true, type, wq, n_in, n_out, x.data()) : std::vector<float>();
        if (base.empty()) {
            use_repack = false;
            base = mul(cpu, plain, false, type, wq, n_in, n_out, x.data());
        }
        printf("%-6s", ggml_type_name(type));
        for (double e : epss) {
            std::vector<float> xp(n_in);
            for (int j = 0; j < n_in; j++) xp[j] = (float) ((double) x[j] * (1.0 + e * (double) u[j]));
            std::vector<float> y = mul(cpu, use_repack ? repack : plain, use_repack, type, wq, n_in, n_out, xp.data());
            double num = 0, den = 0;
            for (int i = 0; i < n_out; i++) {
                num += ((double) y[i] - base[i]) * ((double) y[i] - base[i]);
                den += (double) base[i] * base[i];
            }
            printf("  %-12.2e", std::sqrt(num / den));
        }
        // the same product with the weights as stored and the activations left as float, in double
        std::vector<float> wd((size_t) n_in * n_out);
        if (type == GGML_TYPE_F32) {
            wd = wf;
        } else {
            ggml_get_type_traits(type)->to_float(wq.data(), wd.data(), (int64_t) n_in * n_out);
        }
        double num = 0, den = 0;
        for (int i = 0; i < n_out; i++) {
            double e = 0;
            for (int j = 0; j < n_in; j++) e += (double) wd[(size_t) i * n_in + j] * (double) x[j];
            num += ((double) base[i] - e) * ((double) base[i] - e);
            den += e * e;
        }
        printf("  %-18s  %.2e\n", type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ? "kept as float" : "rounded to 8 bits",
               std::sqrt(num / den));
    }
    ggml_backend_free(cpu);
    return 0;
}
