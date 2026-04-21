/* Standalone eval: compute val loss on dataset using loaded weights.
 * Expected: ~2.03 if load_model + forward_logits are correct. */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_LAYER    6
#define N_EMBD     192
#define N_HEAD     6
#define HEAD_DIM   32
#define HIDDEN     512
#define CTX        256
#define EVAL_SEQS  16

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
        if (tok->char_to_id[c] < 0) { tok->char_to_id[c] = v; tok->id_to_char[v] = c; v++; }
    }
    tok->vocab = v; return v;
}

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *wo, *rms2, *w_gate, *w_up, *w_down;
    } L[N_LAYER];
    nt_tensor *rms_f, *head;
} Model;

static int model_n_tensors(void) { return 1 + N_LAYER * 9 + 2; }

static Model* load_model(const char* path) {
    int n = 0;
    nt_tensor** loaded = nt_load(path, &n);
    if (!loaded) return NULL;
    if (n != model_n_tensors()) {
        printf("load: got %d expected %d\n", n, model_n_tensors());
        for (int i = 0; i < n; i++) nt_tensor_free(loaded[i]);
        free(loaded); return NULL;
    }
    Model* m = (Model*)calloc(1, sizeof(Model));
    int i = 0;
    m->wte = loaded[i++];
    for (int l = 0; l < N_LAYER; l++) {
        m->L[l].rms1 = loaded[i++];
        m->L[l].wq = loaded[i++]; m->L[l].wk = loaded[i++];
        m->L[l].wv = loaded[i++]; m->L[l].wo = loaded[i++];
        m->L[l].rms2 = loaded[i++];
        m->L[l].w_gate = loaded[i++]; m->L[l].w_up = loaded[i++]; m->L[l].w_down = loaded[i++];
    }
    m->rms_f = loaded[i++]; m->head = loaded[i++];
    free(loaded); return m;
}

static int forward_ce(Model* m, int* tokens, int* targets, int vocab) {
    int wte_i = nt_tape_param(m->wte);
    struct { int rms1, wq, wk, wv, wo, rms2, w_gate, w_up, w_down; } li[N_LAYER];
    for (int l = 0; l < N_LAYER; l++) {
        li[l].rms1 = nt_tape_param(m->L[l].rms1);
        li[l].wq = nt_tape_param(m->L[l].wq);
        li[l].wk = nt_tape_param(m->L[l].wk);
        li[l].wv = nt_tape_param(m->L[l].wv);
        li[l].wo = nt_tape_param(m->L[l].wo);
        li[l].rms2 = nt_tape_param(m->L[l].rms2);
        li[l].w_gate = nt_tape_param(m->L[l].w_gate);
        li[l].w_up = nt_tape_param(m->L[l].w_up);
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
        int u = nt_bit_seq_linear(li[l].w_up, xn, CTX);
        int fused = nt_swiglu(g, u);
        int d = nt_bit_seq_linear(li[l].w_down, fused, CTX);
        h = nt_add(h, d);
    }
    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, N_EMBD);
    int logits = nt_seq_linear(head_i, hf, CTX);
    return nt_seq_cross_entropy(logits, tgt_i, CTX, vocab);
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "weights/microgpt_1bit.bin";
    FILE* f = fopen("dataset_clean.txt", "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END); long fs = ftell(f); fseek(f, 0, SEEK_SET);
    char* raw = (char*)malloc(fs + 1);
    fread(raw, 1, fs, f); raw[fs] = 0; fclose(f);
    CharTok tok;
    int vocab = build_char_tokenizer(&tok, raw, fs);
    int* encoded = (int*)malloc(fs * sizeof(int));
    for (long i = 0; i < fs; i++) {
        unsigned char c = (unsigned char)raw[i];
        encoded[i] = tok.char_to_id[c] >= 0 ? tok.char_to_id[c] : 0;
    }
    free(raw);

    Model* m = load_model(path);
    if (!m) { printf("load failed\n"); return 1; }

    printf("Eval on %d seqs, CTX=%d, vocab=%d\n", EVAL_SEQS, CTX, vocab);
    float total = 0; int count = 0;
    int stride = fs / EVAL_SEQS;
    for (int s = 0; s < EVAL_SEQS; s++) {
        int off = s * stride;
        if (off + CTX + 1 > fs) break;
        nt_tape_start();
        nt_train_mode(0);
        int ce = forward_ce(m, encoded + off, encoded + off + 1, vocab);
        float lv = nt_tape_get()->entries[ce].output->data[0];
        total += lv; count++;
        printf("  seq %2d | loss %.4f\n", s, lv);
        nt_tape_clear();
    }
    printf("avg loss: %.4f (training's final val was 2.0314)\n", total / count);
    free(encoded); return 0;
}
