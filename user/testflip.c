//
// testflip.c — comprehensive test suite for the flip_display system call.
//
// To build: add  $U/_testflip  to the UPROGS list in Makefile.
// To run in xv6: testflip
//
// Tests 1-6:  required specification tests
// Tests 7-10: additional edge-case / stress tests
//
//  1. Happy flow           — valid 300-page aligned buffer
//  2. Unaligned address    — buf+N where N % PGSIZE != 0
//  3. Invalid pointers     — NULL, kernel PA 0x80000000, MAXVA
//  4. Partial mapping      — buffer smaller than 300 pages
//  5. Concurrency / stress — NCHILD forked processes flip in a tight loop
//  6. Process-exit lifecycle — child flips then exits immediately
//  7. Off-by-one           — exactly 299 pages (one short of 300)
//  8. Double flip          — three consecutive flips on the same buffer
//  9. Heap-top unmapped    — sbrk(0) is page-aligned but nothing is mapped
// 10. Sequential isolation — each process writes a unique pixel pattern
//

#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

#define PGSIZE   4096
#define FB_PAGES 300                    // must match GPU_FB_PAGES in kernel
#define FB_BYTES (FB_PAGES * PGSIZE)    // 640 x 480 x 4 = 1,228,800 bytes

// ── Test harness ──────────────────────────────────────────────────────

static int runs   = 0;
static int passed = 0;
static int failed = 0;

static void
check(const char *name, int got, int want)
{
    runs++;
    if (got == want) {
        printf("  [PASS] %s\n", name);
        passed++;
    } else {
        printf("  [FAIL] %s: want %d  got %d\n", name, want, got);
        failed++;
    }
}

// Grow the heap by n pages and return the base; returns 0 on failure.
static char *
alloc_pages(int n)
{
    char *p = sbrk(n * PGSIZE);
    return (p == (char *)-1) ? 0 : p;
}

// ── Test 1: Happy flow ────────────────────────────────────────────────
//
// The golden path: allocate exactly FB_PAGES pages with sbrk(), which
// returns a page-aligned address and eagerly maps all pages in xv6.
// flip_display must return 0.

static void
test_happy_flow(void)
{
    printf("Test 1: Happy flow\n");
    char *buf = alloc_pages(FB_PAGES);
    if (!buf) { printf("  [SKIP] sbrk failed\n"); return; }
    check("flip_display on full 300-page aligned buffer", flip_display(buf), 0);
}

// ── Test 2: Unaligned address ─────────────────────────────────────────
//
// The kernel's first check is  buf % PGSIZE != 0 → return -1.
// Shifting the base pointer by any non-zero sub-page offset must fail.
// We also verify that the base pointer itself is still accepted.

static void
test_unaligned(void)
{
    printf("Test 2: Unaligned address\n");
    // One extra page so buf+4095 never reaches unmapped territory.
    char *buf = alloc_pages(FB_PAGES + 1);
    if (!buf) { printf("  [SKIP] sbrk failed\n"); return; }
    check("flip_display(buf + 1)",    flip_display(buf + 1),    -1);
    check("flip_display(buf + 4)",    flip_display(buf + 4),    -1);
    check("flip_display(buf + 2048)", flip_display(buf + 2048), -1);
    check("flip_display(buf + 4095)", flip_display(buf + 4095), -1);
    check("flip_display(buf) baseline still passes", flip_display(buf), 0);
}

// ── Test 3: Invalid / kernel pointers ────────────────────────────────
//
// NULL (VA 0): page-aligned, but the process address space is only a few
// pages of code+data followed by a guard page (PTE_U cleared).  walkaddr
// returns 0 for the guard page, so sys_flip_display returns -1.
//
// 0x80000000: the physical DRAM base; page-aligned but never mapped in
// any user page table → walkaddr returns 0 on the very first page.
//
// MAXVA (1L<<38 = 0x4000000000): walkaddr rejects va >= MAXVA
// unconditionally, even before a page-table walk.

static void
test_invalid_ptrs(void)
{
    printf("Test 3: Invalid / kernel pointers\n");
    check("flip_display(NULL)",
          flip_display((void *)0), -1);
    check("flip_display(0x80000000)",
          flip_display((void *)0x80000000), -1);
    check("flip_display(MAXVA = 0x4000000000)",
          flip_display((void *)(uint64)0x4000000000), -1);
}

// ── Test 4: Partial mapping ────────────────────────────────────────────
//
// Only 10 of the required 300 pages are allocated.  The kernel's loop
// reaches page index 10 (the first unmapped page), walkaddr returns 0,
// and the syscall returns -1 — without crashing or reading beyond the
// mapped region.

static void
test_partial_mapping(void)
{
    printf("Test 4: Partial mapping (10 pages, need 300)\n");
    char *buf = alloc_pages(10);
    if (!buf) { printf("  [SKIP] sbrk failed\n"); return; }
    check("flip_display on 10-page buffer", flip_display(buf), -1);
}

// ── Test 5: Concurrency / stress ──────────────────────────────────────
//
// NCHILD processes each call flip_display NITER times while running
// concurrently.  flip_lock must serialise the detach→attach sequence so
// the shared static attach_buf[] and GPU virtqueue are never corrupted.
// A clean run (no kernel panic, all children exit 0) confirms the lock.
//
// Each child allocates its own buffer so VA ranges do not overlap.
// NOTE: every flip sends two virtio commands; raise NITER to 1000 for a
// heavier soak test at the cost of longer runtime.

#define NCHILD 4
#define NITER  50

static void
test_concurrency(void)
{
    printf("Test 5: Concurrency / stress (%d children x %d flips)\n",
           NCHILD, NITER);

    for (int i = 0; i < NCHILD; i++) {
        int pid = fork();
        if (pid < 0) {
            printf("  [FAIL] fork failed for child %d\n", i);
            failed++;
            return;
        }
        if (pid == 0) {
            char *buf = alloc_pages(FB_PAGES);
            if (!buf) exit(2);
            for (int j = 0; j < NITER; j++) {
                if (flip_display(buf) != 0)
                    exit(1);
            }
            exit(0);
        }
    }

    int ok = 1;
    for (int i = 0; i < NCHILD; i++) {
        int st = 0;
        wait(&st);
        if (st != 0) ok = 0;
    }
    runs++;
    if (ok) {
        printf("  [PASS] all %d children completed %d flips without error\n",
               NCHILD, NITER);
        passed++;
    } else {
        printf("  [FAIL] at least one child reported a flip error\n");
        failed++;
    }
}

// ── Test 6: Process-exit lifecycle ────────────────────────────────────
//
// sys_flip_display calls virtio_gpu_commit() before returning, so the
// current frame is pushed to the display before user space gets control
// back.  However, the GPU backing list still holds the physical addresses
// of the child's pages.  Once the child exits and the pages are freed,
// any subsequent display refresh by the virtio daemon reads stale memory.
//
// This test verifies that the flip call itself succeeds and the child
// exits cleanly.  The stale-memory hazard is observable on screen (the
// 0xDEADBEEF sentinel pattern) but cannot be detected programmatically
// from user space.

static void
test_exit_lifecycle(void)
{
    printf("Test 6: Process-exit lifecycle\n");
    char *buf = alloc_pages(FB_PAGES);
    if (!buf) { printf("  [SKIP] sbrk failed\n"); return; }

    // Fill with a sentinel; any GPU read of the freed pages produces
    // visually obvious corruption (magenta/red colour wash on screen).
    for (int i = 0; i < (int)(FB_BYTES / 4); i++)
        ((uint *)buf)[i] = 0xDEADBEEF;

    int pid = fork();
    if (pid < 0) {
        printf("  [FAIL] fork failed\n");
        failed++;
        return;
    }
    if (pid == 0) {
        exit(flip_display(buf) == 0 ? 0 : 1);
    }

    int st = 0;
    wait(&st);
    runs++;
    if (st == 0) {
        printf("  [PASS] child flipped and exited cleanly\n");
        printf("  [NOTE] GPU backing pages now reference freed physical memory.\n");
        printf("         A display-daemon refresh before the next flip reads\n");
        printf("         the 0xDEADBEEF sentinel (visible corruption on screen).\n");
        passed++;
    } else {
        printf("  [FAIL] child exited with status %d\n", st);
        failed++;
    }
}

// ── Test 7: Off-by-one — exactly 299 pages ────────────────────────────
//
// The validation loop runs for FB_PAGES (300) iterations (i = 0..299).
// Allocating FB_PAGES-1 = 299 pages means page index 299 is unmapped:
// walkaddr returns 0 and the syscall must return -1.
//
// This catches an off-by-one in the loop bound (< vs <=) or a count
// that starts from 1 instead of 0.

static void
test_off_by_one(void)
{
    printf("Test 7: Off-by-one — exactly 299 pages\n");
    char *buf = alloc_pages(FB_PAGES - 1);
    if (!buf) { printf("  [SKIP] sbrk failed\n"); return; }
    check("flip_display on 299-page buffer (one page short)", flip_display(buf), -1);
}

// ── Test 8: Double flip — idempotency & GPU-state consistency ─────────
//
// flip_display issues a RESOURCE_DETACH_BACKING followed by a
// RESOURCE_ATTACH_BACKING on every call.  If the detach is missing or
// incomplete, a second flip would leave the GPU resource with duplicate
// or mixed mem-entry records, potentially triggering a virtio protocol
// error or a kernel panic.  Three consecutive flips all returning 0
// confirm that the round-trip is self-consistent.

static void
test_double_flip(void)
{
    printf("Test 8: Double flip — idempotency\n");
    char *buf = alloc_pages(FB_PAGES);
    if (!buf) { printf("  [SKIP] sbrk failed\n"); return; }
    check("1st flip", flip_display(buf), 0);
    check("2nd flip", flip_display(buf), 0);
    check("3rd flip", flip_display(buf), 0);
}

// ── Test 9: Page-aligned but entirely unmapped address ────────────────
//
// sbrk(0) returns the current heap top without growing it.  xv6 keeps
// p->sz page-aligned, so this value IS page-aligned — but nothing is
// mapped at or above it.  The very first walkaddr call in the kernel
// loop must return 0, so flip_display must return -1.
//
// This guards against a bug where the kernel performs the alignment check
// but then skips or short-circuits the page-mapping validation loop.

static void
test_unmapped_aligned(void)
{
    printf("Test 9: Page-aligned but unmapped address (heap top)\n");
    char *top = sbrk(0);
    if (top == (char *)-1) { printf("  [SKIP] sbrk(0) failed\n"); return; }
    check("flip_display(sbrk(0))", flip_display(top), -1);
}

// ── Test 10: Sequential multi-process isolation ───────────────────────
//
// The kernel uses a single static attach_buf[] array.  Every call to
// virtio_gpu_flip() must overwrite ALL FB_PAGES entries with the new
// process's pages.  If any entry is left stale from a prior flip, the GPU
// resource receives a mix of old and new physical pages, silently
// corrupting the display (visible as colour bands from a previous
// process's pattern in the QEMU window).
//
// Each of NSEQ sequential child processes fills its buffer with a unique
// 32-bit pattern (PID-derived) and then flips.  We verify each child
// exits 0; visual inspection reveals stale entries if isolation is broken.

#define NSEQ 3

static void
test_sequential_isolation(void)
{
    printf("Test 10: Sequential multi-process isolation\n");

    int all_ok = 1;
    for (int i = 0; i < NSEQ; i++) {
        int pid = fork();
        if (pid < 0) {
            printf("  [FAIL] fork failed for iteration %d\n", i);
            failed++;
            return;
        }
        if (pid == 0) {
            char *buf = alloc_pages(FB_PAGES);
            if (!buf) exit(2);
            // Unique pattern per process; stale entries produce colour
            // discontinuities on screen.
            uint pat = (uint)getpid() * 0x01010101u;
            for (int j = 0; j < (int)(FB_BYTES / 4); j++)
                ((uint *)buf)[j] = pat;
            exit(flip_display(buf) == 0 ? 0 : 1);
        }
        // Wait for this child before spawning the next so flips are serial.
        int st = 0;
        wait(&st);
        if (st != 0) all_ok = 0;
    }

    runs++;
    if (all_ok) {
        printf("  [PASS] all %d sequential flips accepted by GPU\n", NSEQ);
        passed++;
    } else {
        printf("  [FAIL] at least one sequential process failed to flip\n");
        failed++;
    }
}

// ── main ──────────────────────────────────────────────────────────────

int
main(void)
{
    printf("\n=== testflip: flip_display test suite ===\n\n");

    test_happy_flow();           printf("\n");
    test_unaligned();            printf("\n");
    test_invalid_ptrs();         printf("\n");
    test_partial_mapping();      printf("\n");
    test_concurrency();          printf("\n");
    test_exit_lifecycle();       printf("\n");
    test_off_by_one();           printf("\n");
    test_double_flip();          printf("\n");
    test_unmapped_aligned();     printf("\n");
    test_sequential_isolation();

    printf("\n=== Results: %d/%d passed", passed, runs);
    if (failed > 0)
        printf(", %d FAILED", failed);
    printf(" ===\n");

    exit(failed > 0 ? 1 : 0);
}
