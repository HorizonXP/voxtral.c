/*
 * main.c - CLI entry point for voxtral.c
 *
 * Usage: voxtral -d <model_dir> -i <input.wav> [options]
 */

#include "voxtral.h"
#include "voxtral_kernels.h"
#include "voxtral_audio.h"
#ifdef USE_CUDA
#include "voxtral_cuda.h"
#endif
#ifdef USE_METAL
#include "voxtral_metal.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static void usage(const char *prog) {
    fprintf(stderr, "voxtral.c — Voxtral Realtime 4B speech-to-text\n\n");
    fprintf(stderr, "Usage: %s -d <model_dir> (-i <input.wav> | --stdin) [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d <dir>    Model directory (with consolidated.safetensors, tekken.json)\n");
    fprintf(stderr, "  -i <file>   Input WAV file (16-bit PCM, any sample rate)\n");
    fprintf(stderr, "  --stdin     Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --debug     Debug output (per-layer, per-chunk details)\n");
    fprintf(stderr, "  --silent    No status output (only transcription on stdout)\n");
    fprintf(stderr, "  -h          Show this help\n");
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *input_wav = NULL;
    int verbosity = 1; /* 0=silent, 1=normal, 2=debug */
    int use_stdin = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_wav = argv[++i];
        } else if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            verbosity = 2;
        } else if (strcmp(argv[i], "--silent") == 0) {
            verbosity = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || (!input_wav && !use_stdin)) {
        usage(argv[0]);
        return 1;
    }
    if (input_wav && use_stdin) {
        fprintf(stderr, "Error: -i and --stdin are mutually exclusive\n");
        return 1;
    }

    vox_verbose = verbosity;
    vox_verbose_audio = (verbosity >= 2) ? 1 : 0;
    vox_stream_output = stdout;

#ifdef USE_METAL
    vox_metal_init();
#endif

    const char *force_timing_env = getenv("VOX_PRINT_TIMINGS");
    int force_timing = (force_timing_env && force_timing_env[0] && force_timing_env[0] != '0');

    /* Load model */
    struct timeval t0_load, t1_load;
    gettimeofday(&t0_load, NULL);
    vox_ctx_t *ctx = vox_load(model_dir);
    gettimeofday(&t1_load, NULL);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }
    if (force_timing) {
        double load_ms = (t1_load.tv_sec - t0_load.tv_sec) * 1000.0 +
                         (t1_load.tv_usec - t0_load.tv_usec) / 1000.0;
        fprintf(stderr, "Model load: %.0f ms\n", load_ms);
    }

    struct timeval t0_run, t1_run;
    gettimeofday(&t0_run, NULL);
    if (use_stdin) {
        /* Stdin: vox_transcribe_stdin handles streaming output */
        char *text = vox_transcribe_stdin(ctx);
        if (!text) {
            fprintf(stderr, "Transcription failed\n");
            vox_free(ctx);
            return 1;
        }
        free(text);
    } else {
        /* File: use stream API for incremental output */
        int n_samples = 0;
        float *samples = vox_load_wav(input_wav, &n_samples);
        if (!samples) {
            fprintf(stderr, "Failed to load %s\n", input_wav);
            vox_free(ctx);
            return 1;
        }
        if (vox_verbose >= 1)
            fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                    n_samples, (float)n_samples / VOX_SAMPLE_RATE);

        vox_stream_t *s = vox_stream_init(ctx);
        vox_stream_feed(s, samples, n_samples);
        vox_stream_finish(s);
        free(samples);

        /* In silent mode, finish() normally doesn't print the newline. */
        if (vox_verbose < 1 && !force_timing) {
            fputs("\n", stdout);
            fflush(stdout);
        }

        vox_stream_free(s);
    }
    gettimeofday(&t1_run, NULL);
    if (force_timing) {
        double run_ms = (t1_run.tv_sec - t0_run.tv_sec) * 1000.0 +
                        (t1_run.tv_usec - t0_run.tv_usec) / 1000.0;
        fprintf(stderr, "Wall transcribe: %.0f ms\n", run_ms);
    }

    vox_free(ctx);
#ifdef USE_METAL
    vox_metal_shutdown();
#endif
#ifdef USE_CUDA
    vox_cuda_shutdown();
#endif
    return 0;
}
