#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

#define PGSIZE    4096
#define SCREEN_W  640
#define SCREEN_H  480
#define FB_PIXELS (SCREEN_W * SCREEN_H)   // 307200 pixels
#define FB_BYTES  (FB_PIXELS * 4)         // 1228800 bytes = 300 pages
#define GPU_FB_PAGES 300

static int passed = 0, failed = 0;

static void
check(const char *name, int ok)
{
    if (ok) {
        printf("[OK]     %s\n", name);
        passed++;
    } else {
        printf("[FAILED] %s\n", name);
        failed++;
    }
}

int
main(void)
{
    printf("--- Task 1: sys_map_display test suite ---\n\n");

    // ── Rejection tests (no mapping established yet) ───────────────────

    // Completely unaligned
    check("unaligned address (0x10000005) rejected",
          (uint64)map_display((void*)0x10000005) == (uint64)-1);

    // Off-by-one from a page boundary
    check("address off-by-one from page boundary rejected",
          (uint64)map_display((void*)0x10000001) == (uint64)-1);

    // Page-aligned but inside the current heap — collision must be caught
    void *heap_top = sbrk(0);
    check("collision with heap rejected",
          (uint64)map_display((void*)((uint64)heap_top - PGSIZE)) == (uint64)-1);

    // ── Explicit valid VA — success path ──────────────────────────────

    void *want_va = (void*)0x20000000;
    uint32 *fb = (uint32*)map_display(want_va);

    check("explicit page-aligned VA accepted",
          fb != 0 && (uint64)fb != (uint64)-1);
    check("returned VA matches requested VA",
          fb == (uint32*)want_va);

    if (fb == 0 || (uint64)fb == (uint64)-1) {
        printf("Cannot continue — framebuffer mapping failed.\n");
        goto summary;
    }

    // ── Access tests: verify all 300 pages are mapped ─────────────────

    // Page 1 — first pixel
    fb[0] = 0xDEADBEEF;
    check("read-back fb[0]  (page 1 of 300)",
          fb[0] == 0xDEADBEEF);
    fb[0] = 0;

    // Page 2 — first pixel on the second 4 KB page (index = PGSIZE/4 = 1024)
    uint32 *page2_start = (uint32*)((uint8*)fb + PGSIZE);
    *page2_start = 0xCAFEBABE;
    check("read-back page 2 boundary (index 1024)",
          *page2_start == 0xCAFEBABE);
    *page2_start = 0;

    // Page 150 — a page in the middle of the backing list
    uint32 *page150_start = (uint32*)((uint8*)fb + 149 * PGSIZE);
    *page150_start = 0x11223344;
    check("read-back page 150 mid-range (index 149*1024)",
          *page150_start == 0x11223344);
    *page150_start = 0;

    // Page 300 — last pixel of the entire framebuffer
    fb[FB_PIXELS - 1] = 0xBEEFCAFE;
    check("read-back fb[FB_PIXELS-1] (last pixel, page 300)",
          fb[FB_PIXELS - 1] == 0xBEEFCAFE);
    fb[FB_PIXELS - 1] = 0;

    // ── Double-mapping prevention ──────────────────────────────────────

    // auto-select must fail: process already has a mapping
    check("double-map via auto-select (addr=0) rejected",
          (uint64)map_display(0) == (uint64)-1);

    // different explicit VA must also fail
    check("double-map via different explicit VA rejected",
          (uint64)map_display((void*)0x30000000) == (uint64)-1);

    // same explicit VA must also fail
    check("double-map via same explicit VA rejected",
          (uint64)map_display(want_va) == (uint64)-1);

    // ── Auto-select path, tested in a child ───────────────────────────
    // Fork gives the child a fresh fb_map_va = 0 and a page table that
    // contains only [0, p->sz) — the parent's fb mapping above p->sz
    // is NOT copied by uvmcopy, so the child can map independently.
    {
        int pid = fork();
        if (pid == 0) {
            int ok = 1;

            void *auto_va = map_display(0);
            if (auto_va == (void*)-1 || auto_va == 0) {
                printf("[FAILED] child: auto-select returned %p\n", auto_va);
                ok = 0;
                exit(1);
            }
            printf("[OK]     child: auto-select at %p\n", auto_va);

            // Must be page-aligned
            if ((uint64)auto_va % PGSIZE != 0) {
                printf("[FAILED] child: auto-selected VA is not page-aligned\n");
                ok = 0;
            } else {
                printf("[OK]     child: auto-selected VA is page-aligned\n");
            }

            // Write + read-back to prove all pages are accessible
            uint32 *cfa = (uint32*)auto_va;
            cfa[0] = 0xABCD1234;
            if (cfa[0] != 0xABCD1234) {
                printf("[FAILED] child: read-back on auto-mapped buffer\n");
                ok = 0;
            } else {
                printf("[OK]     child: read-back on auto-mapped buffer\n");
            }
            cfa[0] = 0;

            cfa[FB_PIXELS - 1] = 0x5A5A5A5A;
            if (cfa[FB_PIXELS - 1] != 0x5A5A5A5A) {
                printf("[FAILED] child: read-back last pixel on auto-mapped buffer\n");
                ok = 0;
            } else {
                printf("[OK]     child: read-back last pixel on auto-mapped buffer\n");
            }
            cfa[FB_PIXELS - 1] = 0;

            // Double-map inside child must also be blocked
            if ((uint64)map_display(0) != (uint64)-1) {
                printf("[FAILED] child: double-map not blocked\n");
                ok = 0;
            } else {
                printf("[OK]     child: double-map blocked inside child\n");
            }

            exit(ok ? 0 : 1);
        }
        int status = 0;
        wait(&status);
        check("child (auto-select + access + double-map guard)", status == 0);
    }

    // ── Fork isolation: framebuffer mapping is not inherited ──────────
    // If the child inherits p->fb_map_va from the parent, map_display
    // would fail the double-map guard even though the page table has no
    // framebuffer PTEs. If it inherits the PTEs themselves (it should not,
    // since uvmcopy stops at p->sz), the collision check would fire.
    // Either way the child could not re-map — so success here proves
    // neither kind of inheritance occurred.
    {
        int pid = fork();
        if (pid == 0) {
            // Parent mapped at want_va = 0x20000000; child should be able
            // to map there too, since it has no PTEs there and fb_map_va=0.
            void *r = map_display(want_va);
            if ((uint64)r == (uint64)-1) {
                printf("[FAILED] child: parent mapping was inherited (fb_map_va or PTEs)\n");
                exit(1);
            }
            printf("[OK]     child: parent's explicit VA is available in child\n");
            exit(0);
        }
        int status = 0;
        wait(&status);
        check("framebuffer mapping not inherited across fork", status == 0);
    }

summary:
    printf("\n--- %d passed, %d failed ---\n", passed, failed);
    exit(failed > 0 ? 1 : 0);
}
