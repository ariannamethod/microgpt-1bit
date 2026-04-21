/*
 * train_microgpt_1bit.c — BitNet b1.58 GPT on Janus Sonar, via notorch (pure C).
 *
 * Architecture:
 *   N_LAYER=6, N_EMBD=192, N_HEAD=6, HEAD_DIM=32, HIDDEN=512 (8/3*192), CTX=256
 *   Char-level tokenizer (vocab built from dataset_clean.txt, typically ~85)
 *   RoPE + MHA + RMSNorm + BitLinear (QKV/O + gate/up/down) + SwiGLU
 *   Output head stays full-precision (BitNet paper convention).
 *   Chuck optimizer, cosine LR w/ 10% warmup.
 *
 * Target: ~2.7M params, ternary {-1,0,+1} weights (absmean) + int8 activations (absmax).
 *
 *   cc -O2 -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK ...
 *   ./train_microgpt_1bit [steps] [lr]
 *   ./train_microgpt_1bit --resume [steps] [lr]
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define N_LAYER    6
#define N_EMBD     192
#define N_HEAD     6
#define HEAD_DIM   32          /* N_EMBD / N_HEAD */
#define HIDDEN     512         /* 8/3 * N_EMBD, rounded to 64 */
#define CTX        256

#define LOG_EVERY   25
#define CKPT_EVERY  500
#define EVAL_SEQS   16
#define CKPT_PREFIX "weights/microgpt_1bit_ckpt"
#define SAVE_PREFIX "weights/microgpt_1bit"

/* ── char tokenizer ── */
typedef struct {
    int vocab;
    int char_to_id[256];
    unsigned char id_to_char[256];
} CharTok;

static int build_char_tokenizer(CharTok* tok, const char* data, long n) {
    for (int i = 0; i < 256; i++) tok->char_to_id[i] = -1;
    int v = 0;
    for (long i = 0; i < n; i++) {
        unsigned char c = (unsigned char)data[i];
        if (tok->char_to_id[c] < 0) {
            tok->char_to_id[c] = v;
            tok->id_to_char[v] = c;
            v++;
        }
    }
    tok->vocab = v;
    return v;
}

static int encode_chars(const CharTok* tok, const char* data, long n, int* out) {
    for (long i = 0; i < n; i++) {
        unsigned char c = (unsigned char)data[i];
        int id = tok->char_to_id[c];
        if (id < 0) id = 0;
        out[i] = id;
    }
    return (int)n;
}

/* ── model ── */
typedef struct {
    int vocab;
    nt_tensor *wte;                                  /* [V, DIM] */
    struct {
        nt_tensor *rms1;                             /* [DIM] */
        nt_tensor *wq, *wk, *wv, *wo;                /* [DIM, DIM] — BitLinear */
        nt_tensor *rms2;                             /* [DIM] */
        nt_tensor *w_gate, *w_up;                    /* [HIDDEN, DIM] — BitLinear */
        nt_tensor *w_down;                           /* [DIM, HIDDEN] — BitLinear */
    } L[N_LAYER];
    nt_tensor *rms_f;                                /* [DIM] */
    nt_tensor *head;                                 /* [V, DIM] — full precision */
} Model;

static long count_params(Model* m) {
    long n = m->wte->len + m->rms_f->len + m->head->len;
    for (int l = 0; l < N_LAYER; l++) {
        n += m->L[l].rms1->len + m->L[l].rms2->len;
        n += m->L[l].wq->len + m->L[l].wk->len + m->L[l].wv->len + m->L[l].wo->len;
        n += m->L[l].w_gate->len + m->L[l].w_up->len + m->L[l].w_down->len;
    }
    return n;
}

static Model* model_new(int vocab) {
    Model* m = (Model*)calloc(1, sizeof(Model));
    m->vocab = vocab;
    m->wte = nt_tensor_new2d(vocab, N_EMBD); nt_tensor_xavier(m->wte, vocab, N_EMBD);
    float rs = 0.02f / sqrtf(2.0f * N_LAYER);
    float out_scale = rs / 0.1f;
    for (int l = 0; l < N_LAYER; l++) {
        m->L[l].rms1 = nt_tensor_new(N_EMBD); nt_tensor_fill(m->L[l].rms1, 1.0f);
        m->L[l].wq   = nt_tensor_new2d(N_EMBD, N_EMBD); nt_tensor_xavier(m->L[l].wq, N_EMBD, N_EMBD);
        m->L[l].wk   = nt_tensor_new2d(N_EMBD, N_EMBD); nt_tensor_xavier(m->L[l].wk, N_EMBD, N_EMBD);
        m->L[l].wv   = nt_tensor_new2d(N_EMBD, N_EMBD); nt_tensor_xavier(m->L[l].wv, N_EMBD, N_EMBD);
        m->L[l].wo   = nt_tensor_new2d(N_EMBD, N_EMBD); nt_tensor_xavier(m->L[l].wo, N_EMBD, N_EMBD);
        for (int i = 0; i < m->L[l].wo->len; i++) m->L[l].wo->data[i] *= out_scale;
        m->L[l].rms2 = nt_tensor_new(N_EMBD); nt_tensor_fill(m->L[l].rms2, 1.0f);
        m->L[l].w_gate = nt_tensor_new2d(HIDDEN, N_EMBD); nt_tensor_xavier(m->L[l].w_gate, N_EMBD, HIDDEN);
        m->L[l].w_up   = nt_tensor_new2d(HIDDEN, N_EMBD); nt_tensor_xavier(m->L[l].w_up,   N_EMBD, HIDDEN);
        m->L[l].w_down = nt_tensor_new2d(N_EMBD, HIDDEN); nt_tensor_xavier(m->L[l].w_down, HIDDEN, N_EMBD);
        for (int i = 0; i < m->L[l].w_down->len; i++) m->L[l].w_down->data[i] *= out_scale;
    }
    m->rms_f = nt_tensor_new(N_EMBD); nt_tensor_fill(m->rms_f, 1.0f);
    m->head  = nt_tensor_new2d(vocab, N_EMBD); nt_tensor_xavier(m->head, N_EMBD, vocab);
    return m;
}

static void model_free(Model* m) {
    nt_tensor_free(m->wte);
    for (int l = 0; l < N_LAYER; l++) {
        nt_tensor_free(m->L[l].rms1); nt_tensor_free(m->L[l].rms2);
        nt_tensor_free(m->L[l].wq); nt_tensor_free(m->L[l].wk);
        nt_tensor_free(m->L[l].wv); nt_tensor_free(m->L[l].wo);
        nt_tensor_free(m->L[l].w_gate); nt_tensor_free(m->L[l].w_up);
        nt_tensor_free(m->L[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head); free(m);
}

/* 9 tensors per layer: rms1 + 4 attn (wq,wk,wv,wo) + rms2 + 3 ffn (gate,up,down) */
static int model_n_tensors(void) { return 1 + N_LAYER * 9 + 2; }

static nt_tensor** model_param_array(Model* m) {
    int n = model_n_tensors();
    nt_tensor** p = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int i = 0;
    p[i++] = m->wte;
    for (int l = 0; l < N_LAYER; l++) {
        p[i++] = m->L[l].rms1;
        p[i++] = m->L[l].wq; p[i++] = m->L[l].wk;
        p[i++] = m->L[l].wv; p[i++] = m->L[l].wo;
        p[i++] = m->L[l].rms2;
        p[i++] = m->L[l].w_gate; p[i++] = m->L[l].w_up; p[i++] = m->L[l].w_down;
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

static void save_model(Model* m, const char* prefix) {
    char path[256]; snprintf(path, sizeof(path), "%s.bin", prefix);
    nt_tensor** p = model_param_array(m);
    nt_save(path, p, model_n_tensors());
    free(p);
}

static void save_checkpoint(Model* m, int step, float best) {
    /* Write .meta FIRST (tiny, fast, atomic via temp+rename). If crash happens during
     * .bin write, .meta still shows last-known-good state. Atomic rename avoids
     * partial-write races. */
    char mp[256], mp_tmp[256];
    snprintf(mp,     sizeof(mp),     "%s.meta",     CKPT_PREFIX);
    snprintf(mp_tmp, sizeof(mp_tmp), "%s.meta.tmp", CKPT_PREFIX);
    FILE* f = fopen(mp_tmp, "w");
    if (f) {
        fprintf(f, "%d\n%.6f\n%d\n", step, best, m->vocab);
        fflush(f); fsync(fileno(f)); fclose(f);
        rename(mp_tmp, mp);
    }
    /* Now write .bin (10MB, slower). If we crash here, meta points to last-good bin. */
    save_model(m, CKPT_PREFIX);
}

static int load_checkpoint(Model* m, float* best_loss) {
    char wp[256], mp[256];
    snprintf(wp, sizeof(wp), "%s.bin", CKPT_PREFIX);
    snprintf(mp, sizeof(mp), "%s.meta", CKPT_PREFIX);
    int n = 0;
    nt_tensor** loaded = nt_load(wp, &n);
    if (!loaded) return -1;
    int expected = model_n_tensors();
    if (n != expected) {
        for (int i = 0; i < n; i++) nt_tensor_free(loaded[i]);
        free(loaded); return -1;
    }
    nt_tensor** dst = model_param_array(m);
    for (int i = 0; i < expected; i++) {
        memcpy(dst[i]->data, loaded[i]->data, dst[i]->len * sizeof(float));
        nt_tensor_free(loaded[i]);
    }
    free(loaded); free(dst);
    int step = 0; *best_loss = 99.0f;
    FILE* f = fopen(mp, "r");
    if (f) { int vv = 0; fscanf(f, "%d\n%f\n%d\n", &step, best_loss, &vv); fclose(f); }
    return step;
}

/* ── forward ── */
static int forward(Model* m, int* tokens, int* targets) {
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    struct {
        int rms1;
        int wq, wk, wv, wo;
        int rms2;
        int w_gate, w_up, w_down;
    } li[N_LAYER];
    for (int l = 0; l < N_LAYER; l++) {
        li[l].rms1   = nt_tape_param(m->L[l].rms1);
        li[l].wq     = nt_tape_param(m->L[l].wq);
        li[l].wk     = nt_tape_param(m->L[l].wk);
        li[l].wv     = nt_tape_param(m->L[l].wv);
        li[l].wo     = nt_tape_param(m->L[l].wo);
        li[l].rms2   = nt_tape_param(m->L[l].rms2);
        li[l].w_gate = nt_tape_param(m->L[l].w_gate);
        li[l].w_up   = nt_tape_param(m->L[l].w_up);
        li[l].w_down = nt_tape_param(m->L[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_f);
    int head_i = nt_tape_param(m->head);

    nt_tensor* tok_t = nt_tensor_new(CTX);
    nt_tensor* tgt_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) { tok_t->data[i] = (float)tokens[i]; tgt_t->data[i] = (float)targets[i]; }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t); nt_tensor_free(tgt_t);

    int h = nt_seq_embedding(wte_i, -1, tok_i, CTX, N_EMBD);

    for (int l = 0; l < N_LAYER; l++) {
        int xn = nt_seq_rmsnorm(h, li[l].rms1, CTX, N_EMBD);

        int q = nt_bit_seq_linear(li[l].wq, xn, CTX);
        int k = nt_bit_seq_linear(li[l].wk, xn, CTX);
        int v = nt_bit_seq_linear(li[l].wv, xn, CTX);

        q = nt_rope(q, CTX, HEAD_DIM);
        k = nt_rope(k, CTX, HEAD_DIM);

        int a = nt_mh_causal_attention(q, k, v, CTX, HEAD_DIM);
        int proj = nt_bit_seq_linear(li[l].wo, a, CTX);
        h = nt_add(h, proj);

        xn = nt_seq_rmsnorm(h, li[l].rms2, CTX, N_EMBD);
        int g = nt_bit_seq_linear(li[l].w_gate, xn, CTX);
        int u = nt_bit_seq_linear(li[l].w_up,   xn, CTX);
        int fused = nt_swiglu(g, u);
        int d = nt_bit_seq_linear(li[l].w_down, fused, CTX);
        h = nt_add(h, d);
    }

    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, N_EMBD);
    int logits = nt_seq_linear(head_i, hf, CTX);
    return nt_seq_cross_entropy(logits, tgt_i, CTX, m->vocab);
}

/* ── eval ── */
static float eval_loss(Model* m, int* encoded, int n_tokens) {
    float total = 0; int count = 0;
    int stride = n_tokens / EVAL_SEQS;
    if (stride < 1) stride = 1;
    for (int s = 0; s < EVAL_SEQS; s++) {
        int off = s * stride;
        if (off + CTX + 1 > n_tokens) break;
        nt_tape_start();
        nt_train_mode(0);
        int loss_idx = forward(m, encoded + off, encoded + off + 1);
        total += nt_tape_get()->entries[loss_idx].output->data[0];
        count++;
        nt_tape_clear();
        nt_train_mode(1);
    }
    return count > 0 ? total / count : 99.0f;
}

static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0 + tv.tv_usec/1000.0; }

int main(int argc, char** argv) {
    int resume = 0, ao = 1;
    if (argc > 1 && strcmp(argv[1], "--resume") == 0) { resume = 1; ao = 2; }
    int steps = ao < argc ? atoi(argv[ao]) : 10000;
    float base_lr = (ao+1) < argc ? (float)atof(argv[ao+1]) : 3e-4f;

    printf("══════════════════════════════════════════════════════════\n");
    printf("  microgpt-1bit — BitNet b1.58 training on Janus Sonar\n");
    printf("  notorch v2.1 + BLAS-patched BitSeqLinear (pure C, STE backward)\n");
    printf("══════════════════════════════════════════════════════════\n");

    const char* path = "dataset_clean.txt";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    char* raw = (char*)malloc(fsize + 1);
    fread(raw, 1, fsize, f); raw[fsize] = 0; fclose(f);

    CharTok tok;
    int vocab = build_char_tokenizer(&tok, raw, fsize);
    printf("  corpus: %.1f KB, vocab %d (char-level)\n", fsize/1024.0, vocab);

    int* encoded = (int*)malloc(fsize * sizeof(int));
    int n_tokens = encode_chars(&tok, raw, fsize, encoded);
    free(raw);
    printf("  tokens: %d\n", n_tokens);

    nt_seed(42);
    Model* model = model_new(vocab);
    long np = count_params(model);
    printf("  model: %ld params (%.2f MB f32)\n", np, np*4.0f/1048576.0f);
    printf("  config: N_LAYER=%d N_EMBD=%d N_HEAD=%d HEAD_DIM=%d HIDDEN=%d CTX=%d\n",
           N_LAYER, N_EMBD, N_HEAD, HEAD_DIM, HIDDEN, CTX);
    printf("  karpathy: %.1f epochs over %d steps\n", (float)steps * CTX / n_tokens, steps);

    float best_loss = 99.0f;
    int start_step = 0;
    if (resume) {
        int loaded = load_checkpoint(model, &best_loss);
        if (loaded >= 0) { start_step = loaded; printf("  RESUMED from step %d (best=%.4f)\n", loaded, best_loss); }
        else printf("  resume requested but no checkpoint — starting fresh\n");
    }

    nt_schedule sched = nt_schedule_cosine(base_lr, steps/10, steps, base_lr*0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    printf("\n  training: steps=%d lr=%.2e  ternary W (absmean) + int8 x (absmax)\n", steps, base_lr);
    printf("──────────────────────────────────────────────────────────\n");
    double t0 = now_ms();
    float first_loss = 0;

    for (int step = start_step; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);
        int off = rand() % (n_tokens - CTX - 1);

        nt_tape_start();
        int loss_idx = forward(model, encoded + off, encoded + off + 1);
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == start_step) first_loss = lv;
        if (lv < best_loss) best_loss = lv;

        nt_tape_backward(loss_idx);
        if (!nt_nan_guard_check(&guard)) { nt_tape_clear(); continue; }
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, lv);
        nt_tape_clear();

        if ((step+1) % LOG_EVERY == 0 || step == start_step) {
            printf("  step %5d/%d | loss %.4f | best %.4f | lr %.2e | %.1fs\n",
                   step+1, steps, lv, best_loss, lr, (now_ms()-t0)/1000.0);
            fflush(stdout);
        }

        if ((step+1) % CKPT_EVERY == 0) {
            float val = eval_loss(model, encoded, n_tokens);
            printf("  ──── ckpt %d | val %.4f | saving…\n", step+1, val);
            save_checkpoint(model, step+1, best_loss);
            fflush(stdout);
        }
    }

    float final_val = eval_loss(model, encoded, n_tokens);
    double total_s = (now_ms()-t0)/1000.0;

    printf("──────────────────────────────────────────────────────────\n");
    printf("  train: %.4f → best %.4f\n", first_loss, best_loss);
    printf("  val:   %.4f\n", final_val);
    printf("  time:  %.0fs (%.1f min) | %.2f steps/s\n", total_s, total_s/60.0, steps/total_s);
    printf("  nans:  %d\n", guard.total_nan_count);

    printf("\n  saving final…\n");
    save_model(model, SAVE_PREFIX);
    save_checkpoint(model, steps, best_loss);
    printf("  %s.bin\n", SAVE_PREFIX);

    model_free(model); free(encoded);
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  microgpt-1bit done. %d steps. No PyTorch. No Python.\n", steps);
    printf("══════════════════════════════════════════════════════════\n");
    return 0;
}
