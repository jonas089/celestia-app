/*
 * accprover.h — C ABI for libaccprover.
 *
 * A thin C facade over the verified riscv-stf Expander-GKR prover
 * (riscv_stf::prove_execution), reusing rsema1d as the GKR input PCS. Statically
 * embeds the prover; dynamically links librsema1d (the Go DA encoder).
 *
 * Hand-authored (cbindgen not available in this environment); kept in sync with
 * src/lib.rs.
 */
#ifndef ACCPROVER_H
#define ACCPROVER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes. */
#define ACCPROVER_OK             0
#define ACCPROVER_ERR_NULL_ARG  -1
#define ACCPROVER_ERR_PROVE     -2
#define ACCPROVER_ERR_PANIC     -3

/*
 * Prove one RV32IM execution over the array derived from `input`, reuse rsema1d
 * as the GKR input commitment, and self-verify.
 *
 * On ACCPROVER_OK:
 *   out_commit      caller-provided 32-byte buffer, filled with the rsema1d
 *                   commitment (== independent Go/DA commit of the same rows);
 *   *out_proof      library-owned proof buffer (free via accprover_free_buf);
 *   *out_proof_len  its length;
 *   *out_pub        library-owned public value: final sum ++ loop count ++
 *                   mem[result], each little-endian uint32 (12 bytes);
 *   *out_pub_len    its length;
 *   *out_verified   1 iff the Expander verifier accepted;
 *   *out_input_vars GKR num_vars.
 *
 * On error, a negative code is returned; see accprover_last_error(). No buffers
 * are allocated on error.
 */
int accprover_prove(const uint8_t *input,
                    size_t input_len,
                    uint8_t *out_commit,
                    uint8_t **out_proof,
                    size_t *out_proof_len,
                    uint8_t **out_pub,
                    size_t *out_pub_len,
                    int *out_verified,
                    uint32_t *out_input_vars);

/*
 * Prove a rollup block's REAL STF as a CONTINUATION span of RV32IM chunks over
 * the fixed guest ELF at `guest_elf_path`, driving riscv_stf::prove_elf
 * (Expander-GKR + rsema1d). `input` is the serialized EthClientExecutorInput
 * blob for the block.
 *
 * First runs the emulator to HALT for the golden (block_number, state_root),
 * then proves a span of chunks: max_chunks==0 composes the whole trace to HALT;
 * otherwise a bounded max_chunks prefix of chunk_len-cycle chunks is proven (a
 * PARTIAL span; out_composed stays 0 unless the trace end was reached).
 * chunk_len==0 defaults to 32.
 *
 * On ACCPROVER_OK:
 *   *out_block_number     golden block number;
 *   out_state_root        caller-provided 32-byte buffer := golden state root;
 *   *out_num_chunks       chunks actually proven;
 *   *out_all_verified     1 iff every proven chunk's Expander verifier accepted;
 *   *out_all_go_match     1 iff every chunk's GKR-PCS root == Go/DA rsema1d Encode;
 *   *out_chain_valid      1 iff pc+reg chain and memory-product handoff hold;
 *   out_first_commitment  caller-provided 32-byte buffer := first chunk commitment;
 *   *out_composed         1 iff span reached HALT and closed + bound to golden;
 *   *out_proof            library-owned proof blob (free via accprover_free_buf):
 *                         u32-LE chunk count, then per chunk (u32-LE len ++ bytes);
 *   *out_proof_len        its length.
 *
 * On error, a negative code is returned; see accprover_last_error(). No buffers
 * are allocated on error.
 */
int accprover_prove_elf(const char *guest_elf_path,
                        const uint8_t *input,
                        size_t input_len,
                        uint32_t chunk_len,
                        uint32_t max_chunks,
                        uint64_t *out_block_number,
                        uint8_t *out_state_root,
                        uint32_t *out_num_chunks,
                        int *out_all_verified,
                        int *out_all_go_match,
                        int *out_chain_valid,
                        uint8_t *out_first_commitment,
                        int *out_composed,
                        uint8_t **out_proof,
                        size_t *out_proof_len);

/* Free a buffer produced by accprover_prove (out_proof / out_pub). */
void accprover_free_buf(uint8_t *ptr, size_t len);

/*
 * NUL-terminated description of the last error on the calling thread, or NULL.
 * Valid until the next accprover_* call on the same thread.
 */
const char *accprover_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* ACCPROVER_H */
