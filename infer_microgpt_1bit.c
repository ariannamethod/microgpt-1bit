/*
 * infer_microgpt_1bit.c — Generation for BitNet 1.58 char-level Sonar model.
 *
 *   ./infer_microgpt_1bit weights/microgpt_1bit.bin "prompt"  [N] [temp] [topk] [mode]
 *     N    = tokens to generate (default 300)
 *     temp = temperature (default 0.9)
 *     topk = top-k cutoff (0 = off, default 40)
 *     mode = "base" | "spa" (default "base"). "spa" feeds dataset sentence
 *            embeddings as context history, modulates logits by SPA connectedness.
 *
 * Match-config with train_microgpt_1bit.c: same tensor count / layout, so
 * model_n_tensors() and model_param_array() agree with save format.
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define N_LAYER    6
#define N_EMBD     192
#define N_HEAD     6
#define HEAD_DIM   32
#define HIDDEN     512
#define CTX        256

#define MAX_SPA_SENTENCES 32
#define SPA_ALPHA   0.85f
#define SPA_STRENGTH 0.3f

/* Dario field (corpus statistics as additive logit bias), char-level adapt
 * from ariannamethod/janus.sonar. Weights calibrated for vocab=89 (smaller
 * than BPE 2048 — boost Dario weights slightly to compensate for higher
 * per-token distribution entropy). */
#define MAX_VOCAB       96
#define MW_HEBB_WINDOW  16
#define MW_BIGRAM_W     3.0f
#define MW_TRIGRAM_W    1.5f
#define MW_HEBB_W       0.5f
#define MW_UNIGRAM_FLOOR 1e-7f

static float g_mw_unigram[MAX_VOCAB];
static float g_mw_bigram[MAX_VOCAB][MAX_VOCAB];
static float g_mw_trigram[MAX_VOCAB][MAX_VOCAB][MAX_VOCAB];
static int   g_mw_ready = 0;
static int   g_mw_vocab = 0;

static void build_metaweights(const int* ids, int n, int vocab) {
    if (vocab > MAX_VOCAB) return;
    g_mw_vocab = vocab;
    memset(g_mw_unigram, 0, sizeof(g_mw_unigram));
    memset(g_mw_bigram,  0, sizeof(g_mw_bigram));
    memset(g_mw_trigram, 0, sizeof(g_mw_trigram));

    /* Unigram */
    for (int i = 0; i < n; i++) if (ids[i] >= 0 && ids[i] < vocab) g_mw_unigram[ids[i]] += 1.0f;
    float tot = 0; for (int i = 0; i < vocab; i++) tot += g_mw_unigram[i];
    if (tot > 0) for (int i = 0; i < vocab; i++) g_mw_unigram[i] /= tot;

    /* Bigram: P(b | a) */
    for (int i = 1; i < n; i++) {
        int a = ids[i-1], b = ids[i];
        if (a >= 0 && a < vocab && b >= 0 && b < vocab) g_mw_bigram[a][b] += 1.0f;
    }
    for (int a = 0; a < vocab; a++) {
        float s = 0; for (int b = 0; b < vocab; b++) s += g_mw_bigram[a][b];
        if (s > 0) for (int b = 0; b < vocab; b++) g_mw_bigram[a][b] /= s;
    }

    /* Trigram: P(c | a,b) — dense (vocab^3 = 704969 floats for vocab=89) */
    for (int i = 2; i < n; i++) {
        int a = ids[i-2], b = ids[i-1], c = ids[i];
        if (a >= 0 && a < vocab && b >= 0 && b < vocab && c >= 0 && c < vocab)
            g_mw_trigram[a][b][c] += 1.0f;
    }
    for (int a = 0; a < vocab; a++)
        for (int b = 0; b < vocab; b++) {
            float s = 0; for (int c = 0; c < vocab; c++) s += g_mw_trigram[a][b][c];
            if (s > 0) for (int c = 0; c < vocab; c++) g_mw_trigram[a][b][c] /= s;
        }

    g_mw_ready = 1;
}

static void mw_hebb_query(const int* ctx, int clen, float* out, int vocab) {
    memset(out, 0, vocab * sizeof(float));
    int lo = clen - MW_HEBB_WINDOW; if (lo < 0) lo = 0;
    /* out[t] = sum over recent ctx tokens of co-occurrence weight with t.
     * Simplified: use direct corpus-bigram co-occurrence as proxy. */
    for (int i = lo; i < clen; i++) {
        int c = ctx[i];
        if (c < 0 || c >= vocab) continue;
        float decay = 1.0f / (float)(clen - i + 1);
        for (int t = 0; t < vocab; t++) out[t] += decay * g_mw_bigram[c][t];
    }
}

static void apply_dario_field(float* logits, int vocab, const int* history, int hist_n) {
    if (!g_mw_ready || !history || hist_n < 1) return;
    int prev  = history[hist_n - 1];
    int prev2 = hist_n >= 2 ? history[hist_n - 2] : -1;
    float hebb[MAX_VOCAB];
    mw_hebb_query(history, hist_n, hebb, vocab);
    for (int i = 0; i < vocab; i++) {
        float bg = (prev >= 0 && prev < vocab) ? g_mw_bigram[prev][i] : 0;
        float tg = (prev2 >= 0 && prev >= 0) ? g_mw_trigram[prev2][prev][i] : 0;
        logits[i] += MW_BIGRAM_W * bg + MW_TRIGRAM_W * tg + MW_HEBB_W * hebb[i];
        if (g_mw_unigram[i] < MW_UNIGRAM_FLOOR) logits[i] = -1e9f;
    }
}

/* Age-based repetition penalty + count-crush (Q-style, char-level).
 * Window = last 30 chars. Recent repeat penalized more than old. */
static void apply_rep_penalty(float* logits, int vocab, const int* history, int hist_n) {
    if (!history || hist_n < 1) return;
    int lo = hist_n - 30; if (lo < 0) lo = 0;
    for (int i = lo; i < hist_n; i++) {
        int tok = history[i];
        if (tok < 0 || tok >= vocab) continue;
        float age = (float)(hist_n - i);
        float pen = 0.4f + 0.02f * age;
        if (pen > 1.0f) pen = 1.0f;
        if (logits[tok] > 0) logits[tok] *= pen;
        else                 logits[tok] *= (2.0f - pen);
    }
    /* Count-crush: char seen >= 8 times in last 50 chars → kill (anti-loop) */
    int counts[MAX_VOCAB]; memset(counts, 0, sizeof(counts));
    int clo = hist_n - 50; if (clo < 0) clo = 0;
    for (int i = clo; i < hist_n; i++) {
        int t = history[i];
        if (t >= 0 && t < vocab) counts[t]++;
    }
    for (int t = 0; t < vocab; t++)
        if (counts[t] >= 8) {
            if (logits[t] > 0) logits[t] *= 0.05f;
            else               logits[t] *= 3.0f;
        }
}

/* Char-level hard filters. Arg: char mapping table (id → unsigned char). */
static void apply_char_filters(float* logits, int vocab, const unsigned char* id_to_char,
                               const int* history, int hist_n) {
    int prev  = hist_n >= 1 ? history[hist_n - 1] : -1;
    int prev2 = hist_n >= 2 ? history[hist_n - 2] : -1;
    int prev3 = hist_n >= 3 ? history[hist_n - 3] : -1;
    for (int i = 0; i < vocab; i++) {
        unsigned char c  = id_to_char[i];
        unsigned char cp = prev  >= 0 ? id_to_char[prev]  : 0;
        unsigned char c2 = prev2 >= 0 ? id_to_char[prev2] : 0;
        unsigned char c3 = prev3 >= 0 ? id_to_char[prev3] : 0;
        /* 3x same char in a row = suppress 4th */
        if (prev >= 0 && prev2 >= 0 && c == cp && c == c2) logits[i] = -1e9f;
        /* double-space */
        if (c == ' ' && cp == ' ') logits[i] = -1e9f;
        /* space after newline (or vice versa) redundancy */
        if (c == ' ' && cp == '\n') logits[i] = -1e9f;
        /* triple newline kill */
        if (c == '\n' && cp == '\n' && c2 == '\n') logits[i] = -1e9f;
        (void)c3;
    }
}

/* ── char tokenizer (rebuilt from corpus so ids match training) ── */
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

/* ── model ── */
typedef struct {
    int vocab;
    nt_tensor *wte;
    struct {
        nt_tensor *rms1;
        nt_tensor *wq, *wk, *wv, *wo;
        nt_tensor *rms2;
        nt_tensor *w_gate, *w_up, *w_down;
    } L[N_LAYER];
    nt_tensor *rms_f;
    nt_tensor *head;
} Model;

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

static Model* load_model(const char* path, int vocab) {
    int n = 0;
    nt_tensor** loaded = nt_load(path, &n);
    if (!loaded) { printf("cannot load %s\n", path); return NULL; }
    int expected = model_n_tensors();
    if (n != expected) {
        printf("tensor count mismatch: got %d expected %d\n", n, expected);
        for (int i = 0; i < n; i++) nt_tensor_free(loaded[i]);
        free(loaded); return NULL;
    }
    Model* m = (Model*)calloc(1, sizeof(Model));
    m->vocab = vocab;
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
    free(loaded);
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

/* ── forward returning logits[CTX, vocab] idx on tape ── */
static int forward_logits(Model* m, int* tokens) {
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
    for (int i = 0; i < CTX; i++) tok_t->data[i] = (float)tokens[i];
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t);

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
    return logits;
}

/* ── sampling ── */
static int cmp_desc(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa < fb) - (fa > fb);
}

static int sample_token(const float* logits_row, int vocab, float temp, int topk) {
    float* scaled = (float*)malloc(vocab * sizeof(float));
    if (temp < 1e-6f) temp = 1e-6f;
    float maxv = -1e30f;
    for (int i = 0; i < vocab; i++) {
        scaled[i] = logits_row[i] / temp;
        if (scaled[i] > maxv) maxv = scaled[i];
    }
    if (topk > 0 && topk < vocab) {
        float* tmp = (float*)malloc(vocab * sizeof(float));
        memcpy(tmp, scaled, vocab * sizeof(float));
        qsort(tmp, vocab, sizeof(float), cmp_desc);
        float threshold = tmp[topk - 1];
        free(tmp);
        for (int i = 0; i < vocab; i++)
            if (scaled[i] < threshold) scaled[i] = -1e30f;
    }
    float sum = 0;
    for (int i = 0; i < vocab; i++) {
        scaled[i] = expf(scaled[i] - maxv);
        sum += scaled[i];
    }
    float r = (float)rand() / (float)RAND_MAX * sum;
    float acc = 0; int pick = 0;
    for (int i = 0; i < vocab; i++) {
        acc += scaled[i];
        if (acc >= r) { pick = i; break; }
    }
    free(scaled);
    return pick;
}

/* ── SPA: sentence embeddings from corpus ── */
static int build_sentence_history(const int* encoded, int n_tokens, int vocab,
                                  const float* W_embed, int dim,
                                  float* history_emb /* [MAX * dim] */) {
    /* Split encoded into sentences by periods ('.' id), take last MAX sentences,
     * compute embedding of each via nt_spa_embed_sentence. Needs id of '.'. */
    /* Reverse-scan: find periods, collect last MAX_SPA_SENTENCES sentence slices. */
    int starts[MAX_SPA_SENTENCES + 1];
    int n_sent = 0;
    int end = n_tokens;
    for (int i = n_tokens - 1; i >= 0 && n_sent < MAX_SPA_SENTENCES; i--) {
        /* Use space or newline as soft boundary since encoded is char-ids */
        if (encoded[i] == encoded[0]) { /* no real marker — use chunking */
            /* skip */
        }
    }
    /* Simpler: chunk encoded into windows of ~64 tokens, take last MAX of those. */
    (void)starts; (void)end;
    int chunk = 64;
    int total_chunks = n_tokens / chunk;
    int start_chunk = total_chunks > MAX_SPA_SENTENCES ? total_chunks - MAX_SPA_SENTENCES : 0;
    n_sent = total_chunks - start_chunk;
    if (n_sent < 0) n_sent = 0;
    for (int s = 0; s < n_sent; s++) {
        int off = (start_chunk + s) * chunk;
        nt_spa_embed_sentence(encoded + off, chunk, W_embed, vocab, dim,
                              SPA_ALPHA, history_emb + s * dim);
    }
    return n_sent;
}

/* ── main ── */
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s weights.bin \"prompt\" [N] [temp] [topk] [base|spa]\n", argv[0]);
        return 1;
    }
    const char* weights_path = argv[1];
    const char* prompt = argv[2];
    int N_gen = argc > 3 ? atoi(argv[3]) : 300;
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.9f;
    int topk = argc > 5 ? atoi(argv[5]) : 40;
    const char* mode = argc > 6 ? argv[6] : "base";
    int use_spa   = strcmp(mode, "spa")   == 0;
    int use_dario = strcmp(mode, "dario") == 0 || strcmp(mode, "full") == 0;
    int use_full  = strcmp(mode, "full")  == 0;  /* dario + SPA both */
    if (use_full) use_spa = 1;

    /* Build tokenizer from dataset (must be same as training) */
    const char* ds_path = "dataset_clean.txt";
    FILE* f = fopen(ds_path, "rb");
    if (!f) { printf("cannot open %s\n", ds_path); return 1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    char* raw = (char*)malloc(fsize + 1);
    fread(raw, 1, fsize, f); raw[fsize] = 0; fclose(f);
    CharTok tok;
    int vocab = build_char_tokenizer(&tok, raw, fsize);
    int* encoded = (int*)malloc(fsize * sizeof(int));
    for (long i = 0; i < fsize; i++) {
        unsigned char c = (unsigned char)raw[i];
        encoded[i] = tok.char_to_id[c] >= 0 ? tok.char_to_id[c] : 0;
    }
    free(raw);

    /* Build Dario metaweights if requested */
    if (use_dario) build_metaweights(encoded, (int)fsize, vocab);

    /* Load model */
    Model* m = load_model(weights_path, vocab);
    if (!m) return 1;

    printf("══════════════════════════════════════════════════════════\n");
    printf("  microgpt-1bit inference  |  mode=%s  |  vocab=%d  ctx=%d\n",
           mode, vocab, CTX);
    printf("  temp=%.2f  topk=%d  N=%d%s%s\n", temp, topk, N_gen,
           use_dario ? "  +Dario-field" : "",
           use_spa ? "  +SPA" : "");
    printf("══════════════════════════════════════════════════════════\n");

    /* SPA history from corpus (if mode == spa) */
    float* spa_hist = NULL;
    int spa_n = 0;
    if (use_spa) {
        spa_hist = (float*)calloc(MAX_SPA_SENTENCES * N_EMBD, sizeof(float));
        spa_n = build_sentence_history(encoded, (int)fsize, vocab,
                                        m->wte->data, N_EMBD, spa_hist);
        printf("  SPA: %d history chunks built\n", spa_n);
    }

    srand((unsigned)time(NULL));

    /* Seed tokens */
    int plen = (int)strlen(prompt);
    int* tokens = (int*)calloc(CTX + N_gen + 4, sizeof(int));
    int n = 0;
    for (int i = 0; i < plen; i++) {
        unsigned char c = (unsigned char)prompt[i];
        int id = tok.char_to_id[c] >= 0 ? tok.char_to_id[c] : 0;
        tokens[n++] = id;
    }
    if (n == 0) { tokens[n++] = 0; }

    /* Print prompt */
    printf("\n%s", prompt);
    fflush(stdout);

    /* Generate N_gen tokens */
    int ctx_window[CTX];
    int corpus_len = (int)fsize;
    for (int gen = 0; gen < N_gen; gen++) {
        /* Fill ctx_window: right-aligned tokens, left-pad with tail of corpus
         * (so model sees in-distribution context rather than repeated id=0). */
        if (n >= CTX) {
            for (int i = 0; i < CTX; i++) ctx_window[i] = tokens[n - CTX + i];
        } else {
            /* Pad from BEGINNING of dataset (known opening). This gives the
             * model an in-distribution prefix whose natural continuation
             * lands exactly at the prompt. Still a jump, but dataset's opening
             * ends in a newline near natural boundaries, minimizing shift. */
            int pad_len = CTX - n;
            for (int i = 0; i < pad_len; i++) {
                ctx_window[i] = (i < corpus_len) ? encoded[i] : 0;
            }
            for (int i = 0; i < n; i++) ctx_window[pad_len + i] = tokens[i];
        }
        int last_pos = CTX - 1;

        nt_tape_start();
        nt_train_mode(0);
        int logits_idx = forward_logits(m, ctx_window);
        float* logits_all = nt_tape_get()->entries[logits_idx].output->data;

        /* Copy last-position row */
        float row[512]; /* vocab < 512 */
        for (int v = 0; v < vocab; v++) row[v] = logits_all[last_pos * vocab + v];

        /* Dario field: bigram + trigram + hebbian + unigram floor */
        if (use_dario) apply_dario_field(row, vocab, tokens, n);

        /* Rep penalty + count-crush (always on for anti-loop) */
        apply_rep_penalty(row, vocab, tokens, n);

        /* Char-level hard filters */
        apply_char_filters(row, vocab, tok.id_to_char, tokens, n);

        /* SPA modulation */
        if (use_spa && spa_n > 0) {
            float query_emb[N_EMBD];
            int q_off = n > 64 ? n - 64 : 0;
            int q_len = n - q_off; if (q_len < 1) q_len = 1;
            nt_spa_embed_sentence(tokens + q_off, q_len, m->wte->data, vocab, N_EMBD,
                                  SPA_ALPHA, query_emb);
            float conn = nt_spa_connectedness(query_emb, N_EMBD, spa_hist, spa_n);
            nt_spa_modulate_logits(row, vocab, conn, SPA_STRENGTH);
        }

        nt_tape_clear();
        nt_train_mode(1);

        int next = sample_token(row, vocab, temp, topk);
        tokens[n++] = next;
        unsigned char c = tok.id_to_char[next];
        putchar((int)c);
        fflush(stdout);
    }
    printf("\n══════════════════════════════════════════════════════════\n");

    free(tokens); free(encoded); if (spa_hist) free(spa_hist);
    model_free(m);
    return 0;
}
