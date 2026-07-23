/* Standalone C-ABI smoke test: link libaccprover, prove sample bytes, print the
 * result. Confirms the C ABI works end to end (verified==true, 32-byte commit),
 * and that two different inputs yield two different real commitments. */
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "accprover.h"

static int prove_and_print(const char *label, const uint8_t *in, size_t len,
                           uint8_t commit_out[32]) {
    uint8_t commit[32];
    uint8_t *proof = NULL, *pub = NULL;
    size_t proof_len = 0, pub_len = 0;
    int verified = -1;
    uint32_t input_vars = 0;

    int rc = accprover_prove(in, len, commit, &proof, &proof_len, &pub, &pub_len,
                             &verified, &input_vars);
    if (rc != ACCPROVER_OK) {
        const char *err = accprover_last_error();
        fprintf(stderr, "[%s] accprover_prove failed rc=%d: %s\n", label, rc,
                err ? err : "(no message)");
        return 1;
    }

    printf("[%s] verified      = %s\n", label, verified ? "true" : "false");
    printf("[%s] input_vars    = %u\n", label, input_vars);
    printf("[%s] proof bytes   = %zu\n", label, proof_len);
    printf("[%s] public_value  = ", label);
    for (size_t i = 0; i < pub_len; i++) printf("%02x", pub[i]);
    printf("\n[%s] commitment    = ", label);
    for (int i = 0; i < 32; i++) printf("%02x", commit[i]);
    printf(" (%zu bytes)\n", (size_t)32);

    if (commit_out) memcpy(commit_out, commit, 32);
    accprover_free_buf(proof, proof_len);
    accprover_free_buf(pub, pub_len);
    return verified ? 0 : 2;
}

int main(void) {
    const uint8_t a[] = "accidental-computer-smoke";
    const uint8_t b[] = "a-different-blob";
    uint8_t ca[32], cb[32];

    printf("=== libaccprover C-ABI smoke test ===\n");
    int ra = prove_and_print("A", a, sizeof(a) - 1, ca);
    int rb = prove_and_print("B", b, sizeof(b) - 1, cb);
    if (ra || rb) return 1;

    int differ = memcmp(ca, cb, 32) != 0;
    printf("\ndistinct inputs -> distinct commitments: %s\n", differ ? "YES" : "NO");
    printf("C-ABI GATE: %s\n", (ra == 0 && rb == 0 && differ) ? "PASS" : "FAIL");
    return (ra == 0 && rb == 0 && differ) ? 0 : 1;
}
