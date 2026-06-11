//
// Driver for QEMU's virtio-gpu device.
// Allocates a 640x480 framebuffer, draws "Hello World", and flushes
// it to the QEMU display window.
//
// Run with:
//   qemu ... -device virtio-gpu-device,bus=virtio-mmio-bus.1
//            (remove -nographic; add -serial mon:stdio)
//

#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"
#include "virtio.h"

// VirtIO GPU MMIO base (virtio-mmio-bus.1 = VIRTIO1 = 0x10002000)
#define R1(r) ((volatile uint32 *)(VIRTIO1 + (r)))

// VirtIO device ID for GPU
#define VIRTIO_ID_GPU 16

// VirtIO GPU control command types
#define VIRTIO_GPU_CMD_RESOURCE_CREATE_2D 0x0101
#define VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING 0x0106
#define VIRTIO_GPU_CMD_SET_SCANOUT 0x0103
#define VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D 0x0105
#define VIRTIO_GPU_CMD_RESOURCE_FLUSH 0x0104
#define VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING 0x0107

// VirtIO GPU response types
#define VIRTIO_GPU_RESP_OK_NODATA 0x1100

// Pixel format: B8G8R8X8 (blue at lowest byte address)
// uint32 color value on LE RISC-V: bits [23:16]=R, [15:8]=G, [7:0]=B
#define VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM 2

// Screen dimensions and resource/scanout IDs
#define SCREEN_W 640
#define SCREEN_H 480
#define RESOURCE_ID 1
#define SCANOUT_ID 0

// Framebuffer total bytes and number of 4 KB pages required
#define FB_BYTES (SCREEN_W * SCREEN_H * 4)
#define FB_PAGES ((FB_BYTES + PGSIZE - 1) / PGSIZE) // 300

// Font scale: each 8x8 glyph pixel is drawn as SCALExSCALE screen pixels
#define SCALE 4

// Colors  (uint32 stored little-endian -> memory layout [B, G, R, X])
#define COLOR_BG 0x00000000u // black
#define COLOR_FG 0x00FFFFFFu // white  (R=FF G=FF B=FF)

// ── VirtIO GPU structures (from VirtIO spec §5.7) ─────────────────────

struct virtio_gpu_ctrl_hdr
{
    uint32 type;
    uint32 flags;
    uint64 fence_id;
    uint32 ctx_id;
    uint32 padding;
}; // 24 bytes

struct virtio_gpu_rect
{
    uint32 x, y, width, height;
}; // 16 bytes

struct virtio_gpu_resource_create_2d
{
    struct virtio_gpu_ctrl_hdr hdr;
    uint32 resource_id;
    uint32 format;
    uint32 width;
    uint32 height;
}; // 40 bytes

struct virtio_gpu_mem_entry
{
    uint64 addr;
    uint32 length;
    uint32 padding;
}; // 16 bytes

struct virtio_gpu_resource_attach_backing
{
    struct virtio_gpu_ctrl_hdr hdr;
    uint32 resource_id;
    uint32 nr_entries;
}; // 32 bytes (entries follow immediately in memory)

struct virtio_gpu_set_scanout
{
    struct virtio_gpu_ctrl_hdr hdr;
    struct virtio_gpu_rect r;
    uint32 scanout_id;
    uint32 resource_id;
}; // 48 bytes

struct virtio_gpu_transfer_to_host_2d
{
    struct virtio_gpu_ctrl_hdr hdr;
    struct virtio_gpu_rect r;
    uint64 offset;
    uint32 resource_id;
    uint32 padding;
}; // 56 bytes

struct virtio_gpu_resource_flush
{
    struct virtio_gpu_ctrl_hdr hdr;
    struct virtio_gpu_rect r;
    uint32 resource_id;
    uint32 padding;
}; // 48 bytes

struct virtio_gpu_resource_detach_backing
{
    struct virtio_gpu_ctrl_hdr hdr;
    uint32 resource_id;
    uint32 padding;
}; // 32 bytes

// ── Virtqueue state ──────────────────────────────────────────────────

#define GPU_NUM 8

// Serialises all virtqueue operations so the daemon and user syscalls
// can coexist on multi-CPU systems.
static struct spinlock gpu_lock;

// Serialises the full detach→attach sequence in virtio_gpu_flip so
// the static entries[] array is not overwritten by a concurrent flip.
// Must never be acquired while gpu_lock is already held.
static struct spinlock flip_lock;

static struct
{
    struct virtq_desc *desc;
    struct virtq_avail *avail;
    struct virtq_used *used;
    char free[GPU_NUM];
    uint16 used_idx;
} gq;

// ── Framebuffer page pointers ────────────────────────────────────────

static void *fb[FB_PAGES];

// ── RESOURCE_ATTACH_BACKING command buffer (header + all entries) ────

static struct
{
    struct virtio_gpu_resource_attach_backing backing;
    struct virtio_gpu_mem_entry entries[FB_PAGES];
} attach_buf;

// ── Shared response buffer (reused for every command) ────────────────

static struct virtio_gpu_ctrl_hdr cmd_resp;

// ── 8x8 bitmap font — full printable ASCII 0x20-0x7E ─────────────────
// Bit 7 of each byte = leftmost pixel in that row.
// Based on the public-domain font8x8_basic table.

static const uint8 font8x8[256][8] = {
    [0x20] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // ' '
    [0x21] = {0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00}, // '!'
    [0x22] = {0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // '"'
    [0x23] = {0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00}, // '#'
    [0x24] = {0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00}, // '$'
    [0x25] = {0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00}, // '%'
    [0x26] = {0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00}, // '&'
    [0x27] = {0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00}, // '\''
    [0x28] = {0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00}, // '('
    [0x29] = {0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00}, // ')'
    [0x2A] = {0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00}, // '*'
    [0x2B] = {0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00}, // '+'
    [0x2C] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06}, // ','
    [0x2D] = {0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00}, // '-'
    [0x2E] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00}, // '.'
    [0x2F] = {0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00}, // '/'
    [0x30] = {0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00}, // '0'
    [0x31] = {0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00}, // '1'
    [0x32] = {0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00}, // '2'
    [0x33] = {0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00}, // '3'
    [0x34] = {0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00}, // '4'
    [0x35] = {0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00}, // '5'
    [0x36] = {0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00}, // '6'
    [0x37] = {0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00}, // '7'
    [0x38] = {0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00}, // '8'
    [0x39] = {0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00}, // '9'
    [0x3A] = {0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00}, // ':'
    [0x3B] = {0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06}, // ';'
    [0x3C] = {0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00}, // '<'
    [0x3D] = {0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00}, // '='
    [0x3E] = {0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00}, // '>'
    [0x3F] = {0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00}, // '?'
    [0x40] = {0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00}, // '@'
    [0x41] = {0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00}, // 'A'
    [0x42] = {0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00}, // 'B'
    [0x43] = {0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00}, // 'C'
    [0x44] = {0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00}, // 'D'
    [0x45] = {0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00}, // 'E'
    [0x46] = {0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00}, // 'F'
    [0x47] = {0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00}, // 'G'
    [0x48] = {0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00}, // 'H'
    [0x49] = {0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // 'I'
    [0x4A] = {0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00}, // 'J'
    [0x4B] = {0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00}, // 'K'
    [0x4C] = {0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00}, // 'L'
    [0x4D] = {0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00}, // 'M'
    [0x4E] = {0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00}, // 'N'
    [0x4F] = {0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00}, // 'O'
    [0x50] = {0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00}, // 'P'
    [0x51] = {0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00}, // 'Q'
    [0x52] = {0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00}, // 'R'
    [0x53] = {0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00}, // 'S'
    [0x54] = {0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // 'T'
    [0x55] = {0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00}, // 'U'
    [0x56] = {0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00}, // 'V'
    [0x57] = {0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00}, // 'W'
    [0x58] = {0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00}, // 'X'
    [0x59] = {0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00}, // 'Y'
    [0x5A] = {0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00}, // 'Z'
    [0x5B] = {0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00}, // '['
    [0x5C] = {0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00}, // '\'
    [0x5D] = {0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00}, // ']'
    [0x5E] = {0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00}, // '^'
    [0x5F] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF}, // '_'
    [0x60] = {0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00}, // '`'
    [0x61] = {0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00}, // 'a'
    [0x62] = {0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00}, // 'b'
    [0x63] = {0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00}, // 'c'
    [0x64] = {0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00}, // 'd'
    [0x65] = {0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00}, // 'e'
    [0x66] = {0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00}, // 'f'
    [0x67] = {0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F}, // 'g'
    [0x68] = {0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00}, // 'h'
    [0x69] = {0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // 'i'
    [0x6A] = {0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E}, // 'j'
    [0x6B] = {0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00}, // 'k'
    [0x6C] = {0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // 'l'
    [0x6D] = {0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00}, // 'm'
    [0x6E] = {0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00}, // 'n'
    [0x6F] = {0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00}, // 'o'
    [0x70] = {0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F}, // 'p'
    [0x71] = {0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78}, // 'q'
    [0x72] = {0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00}, // 'r'
    [0x73] = {0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00}, // 's'
    [0x74] = {0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00}, // 't'
    [0x75] = {0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00}, // 'u'
    [0x76] = {0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00}, // 'v'
    [0x77] = {0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00}, // 'w'
    [0x78] = {0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00}, // 'x'
    [0x79] = {0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F}, // 'y'
    [0x7A] = {0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00}, // 'z'
    [0x7B] = {0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00}, // '{'
    [0x7C] = {0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00}, // '|'
    [0x7D] = {0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00}, // '}'
    [0x7E] = {0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // '~'
};

// ── Virtqueue helpers ────────────────────────────────────────────────

static int
alloc_desc(void)
{
    for (int i = 0; i < GPU_NUM; i++)
        if (gq.free[i])
        {
            gq.free[i] = 0;
            return i;
        }
    panic("virtio_gpu: no free descriptors");
}

static void
free_desc(int i)
{
    gq.desc[i].addr = 0;
    gq.desc[i].len = 0;
    gq.desc[i].flags = 0;
    gq.desc[i].next = 0;
    gq.free[i] = 1;
}

// Forward declaration — defined below after the virtqueue state is set up.
static void gpu_send(void *req, int req_len);

// ── GPU command helpers ───────────────────────────────────────────────

// Send RESOURCE_DETACH_BACKING for the display resource.
static void __attribute__((unused))
gpu_cmd_detach(void)
{
    static struct virtio_gpu_resource_detach_backing detach;
    memset(&detach, 0, sizeof(detach));
    detach.hdr.type = VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING;
    detach.resource_id = RESOURCE_ID;
    gpu_send(&detach, sizeof(detach));
}

// Send RESOURCE_ATTACH_BACKING for the display resource.
// entries[] must contain n physical-address/length pairs describing the
// backing pages; n must equal FB_PAGES.
static void
gpu_cmd_attach(struct virtio_gpu_mem_entry *entries, int n)
{
    attach_buf.backing.hdr.type = VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING;
    attach_buf.backing.resource_id = RESOURCE_ID;
    attach_buf.backing.nr_entries = n;
    for (int i = 0; i < n; i++)
        attach_buf.entries[i] = entries[i];
    gpu_send(&attach_buf, sizeof(attach_buf));
}

// Transfer the current resource backing to the host GPU and blit to the
// display window.  This is the only place TRANSFER_TO_HOST_2D and
// RESOURCE_FLUSH are ever sent.
static void
gpu_transfer_flush(void)
{
    static struct virtio_gpu_transfer_to_host_2d xfer;
    memset(&xfer, 0, sizeof(xfer));
    xfer.hdr.type = VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D;
    xfer.r.x = 0;
    xfer.r.y = 0;
    xfer.r.width = SCREEN_W;
    xfer.r.height = SCREEN_H;
    xfer.resource_id = RESOURCE_ID;
    gpu_send(&xfer, sizeof(xfer));

    static struct virtio_gpu_resource_flush flush;
    memset(&flush, 0, sizeof(flush));
    flush.hdr.type = VIRTIO_GPU_CMD_RESOURCE_FLUSH;
    flush.r.x = 0;
    flush.r.y = 0;
    flush.r.width = SCREEN_W;
    flush.r.height = SCREEN_H;
    flush.resource_id = RESOURCE_ID;
    gpu_send(&flush, sizeof(flush));
}

// ── Pixel writer ─────────────────────────────────────────────────────

// Write one 32-bit BGRX pixel into the scatter-gather framebuffer.
static void
putpixel(int x, int y, uint32 color)
{
    int byte_off = (y * SCREEN_W + x) * 4;
    int pg = byte_off / PGSIZE;
    int off = byte_off % PGSIZE;
    uint32 *p = (uint32 *)((uint8 *)fb[pg] + off);
    *p = color;
}

// Draw one glyph from font8x8, scaled SCALExSCALE.
static void
draw_char(int cx, int cy, unsigned char ch)
{
    const uint8 *rows = font8x8[ch];
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 8; col++)
        {
            uint32 color = (rows[row] & (1u << col)) ? COLOR_FG : COLOR_BG;
            for (int dy = 0; dy < SCALE; dy++)
                for (int dx = 0; dx < SCALE; dx++)
                    putpixel(cx + col * SCALE + dx, cy + row * SCALE + dy, color);
        }
    }
}

// ── Command submission (polling, no interrupts) ───────────────────────

// Submit a 2-descriptor command (request + shared response) and block
// until the device completes it by advancing the used ring.
static void
gpu_send(void *req, int req_len)
{
    acquire(&gpu_lock);
    int d0 = alloc_desc();
    int d1 = alloc_desc();

    gq.desc[d0].addr = (uint64)req;
    gq.desc[d0].len = (uint32)req_len;
    gq.desc[d0].flags = VRING_DESC_F_NEXT;
    gq.desc[d0].next = d1;

    gq.desc[d1].addr = (uint64)&cmd_resp;
    gq.desc[d1].len = sizeof(cmd_resp);
    gq.desc[d1].flags = VRING_DESC_F_WRITE;
    gq.desc[d1].next = 0;

    // Place head descriptor index in the available ring.
    gq.avail->ring[gq.avail->idx % GPU_NUM] = d0;
    __sync_synchronize();
    gq.avail->idx++;
    __sync_synchronize();

    // Notify device (queue index 0 = controlq).
    *R1(VIRTIO_MMIO_QUEUE_NOTIFY) = 0;

    // Poll until the device advances the used ring.
    while (1)
    {
        __sync_synchronize();
        if (gq.used->idx != gq.used_idx)
            break;
    }
    gq.used_idx++;

    free_desc(d0);
    free_desc(d1);
    release(&gpu_lock);
}

// ── Public init ───────────────────────────────────────────────────────

void virtio_gpu_init(void)
{
    uint32 status = 0;
    initlock(&gpu_lock, "vgpu");
    initlock(&flip_lock, "vgpu_flip");

    // ── 1. VirtIO device handshake ──────────────────────────────────────
    if (*R1(VIRTIO_MMIO_MAGIC_VALUE) != 0x74726976 ||
        *R1(VIRTIO_MMIO_VERSION) != 2 ||
        *R1(VIRTIO_MMIO_DEVICE_ID) != VIRTIO_ID_GPU ||
        *R1(VIRTIO_MMIO_VENDOR_ID) != 0x554d4551)
    {
        printf("virtio_gpu_init: GPU not found\n");
        return;
    }

    // Reset
    *R1(VIRTIO_MMIO_STATUS) = status;

    status |= VIRTIO_CONFIG_S_ACKNOWLEDGE;
    *R1(VIRTIO_MMIO_STATUS) = status;

    status |= VIRTIO_CONFIG_S_DRIVER;
    *R1(VIRTIO_MMIO_STATUS) = status;

    // No optional features required.
    *R1(VIRTIO_MMIO_DRIVER_FEATURES) = 0;

    status |= VIRTIO_CONFIG_S_FEATURES_OK;
    *R1(VIRTIO_MMIO_STATUS) = status;
    if (!(*R1(VIRTIO_MMIO_STATUS) & VIRTIO_CONFIG_S_FEATURES_OK))
        panic("virtio_gpu: FEATURES_OK not set");

    // ── 2. Initialise control queue (queue 0) ────────────────────────────
    *R1(VIRTIO_MMIO_QUEUE_SEL) = 0;
    if (*R1(VIRTIO_MMIO_QUEUE_READY))
        panic("virtio_gpu: queue already ready");
    if (*R1(VIRTIO_MMIO_QUEUE_NUM_MAX) < GPU_NUM)
        panic("virtio_gpu: queue too small");

    gq.desc = kalloc();
    gq.avail = kalloc();
    gq.used = kalloc();
    if (!gq.desc || !gq.avail || !gq.used)
        panic("virtio_gpu: kalloc failed for queue");
    memset(gq.desc, 0, PGSIZE);
    memset(gq.avail, 0, PGSIZE);
    memset(gq.used, 0, PGSIZE);

    *R1(VIRTIO_MMIO_QUEUE_NUM) = GPU_NUM;
    *R1(VIRTIO_MMIO_QUEUE_DESC_LOW) = (uint64)gq.desc;
    *R1(VIRTIO_MMIO_QUEUE_DESC_HIGH) = (uint64)gq.desc >> 32;
    *R1(VIRTIO_MMIO_DRIVER_DESC_LOW) = (uint64)gq.avail;
    *R1(VIRTIO_MMIO_DRIVER_DESC_HIGH) = (uint64)gq.avail >> 32;
    *R1(VIRTIO_MMIO_DEVICE_DESC_LOW) = (uint64)gq.used;
    *R1(VIRTIO_MMIO_DEVICE_DESC_HIGH) = (uint64)gq.used >> 32;
    *R1(VIRTIO_MMIO_QUEUE_READY) = 1;

    for (int i = 0; i < GPU_NUM; i++)
        gq.free[i] = 1;

    status |= VIRTIO_CONFIG_S_DRIVER_OK;
    *R1(VIRTIO_MMIO_STATUS) = status;

    // ── 3. Allocate framebuffer pages ───────────────────────────────────
    for (int i = 0; i < FB_PAGES; i++)
    {
        fb[i] = kalloc();
        if (!fb[i])
            panic("virtio_gpu: kalloc failed for framebuffer");
        memset(fb[i], 0, PGSIZE); // fill with COLOR_BG (0 = black)
    }

    // ── 4. RESOURCE_CREATE_2D ───────────────────────────────────────────
    static struct virtio_gpu_resource_create_2d create_req;
    memset(&create_req, 0, sizeof(create_req));
    create_req.hdr.type = VIRTIO_GPU_CMD_RESOURCE_CREATE_2D;
    create_req.resource_id = RESOURCE_ID;
    create_req.format = VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM;
    create_req.width = SCREEN_W;
    create_req.height = SCREEN_H;
    gpu_send(&create_req, sizeof(create_req));

    // ── 5. RESOURCE_ATTACH_BACKING ──────────────────────────────────────
    // Build the initial kernel framebuffer backing entries and attach.
    // gpu_cmd_attach() will also record them in attach_buf.entries[] so they
    // can be reused to restore the backing after a flip.
    static struct virtio_gpu_mem_entry fb_entries[FB_PAGES];
    for (int i = 0; i < FB_PAGES; i++) {
        fb_entries[i].addr   = (uint64)fb[i];
        fb_entries[i].length = PGSIZE;
    }
    gpu_cmd_attach(fb_entries, FB_PAGES);

    // ── 6. Draw "Hello World" into the framebuffer ──────────────────────
    {
        const char *msg = "Hello World";
        int nchars = 11;
        int char_w = 8 * SCALE;
        int char_h = 8 * SCALE;
        int text_w = nchars * char_w;
        int x0 = (SCREEN_W - text_w) / 2;
        int y0 = (SCREEN_H - char_h) / 2;
        for (int i = 0; msg[i]; i++)
            draw_char(x0 + i * char_w, y0, (unsigned char)msg[i]);
    }

    // ── 7. SET_SCANOUT (attach resource to display output) ──────────────
    static struct virtio_gpu_set_scanout scanout_req;
    memset(&scanout_req, 0, sizeof(scanout_req));
    scanout_req.hdr.type = VIRTIO_GPU_CMD_SET_SCANOUT;
    scanout_req.r.x = 0;
    scanout_req.r.y = 0;
    scanout_req.r.width = SCREEN_W;
    scanout_req.r.height = SCREEN_H;
    scanout_req.scanout_id = SCANOUT_ID;
    scanout_req.resource_id = RESOURCE_ID;
    gpu_send(&scanout_req, sizeof(scanout_req));

    // ── 8. TRANSFER_TO_HOST_2D (upload guest memory -> host GPU) ─────────
    gpu_transfer_flush();
    printf("virtio_gpu: \"Hello World\" displayed on 640x480 window\n");
}

// ── Public: flush the kernel fb[] to the display ─────────────────────
// Called by display_daemon.  Sends TRANSFER_TO_HOST_2D + RESOURCE_FLUSH.
void virtio_gpu_commit(void)
{
    gpu_transfer_flush();
}

// ── Public: map the kernel framebuffer pages into a user page table ───
// Maps all GPU_FB_PAGES pages from fb[] at virtual address va in pt,
// with PTE_U|PTE_R|PTE_W.  On partial failure, unmaps what was installed
// and returns -1.  Returns 0 on success.
int
virtio_gpu_map_fb(pagetable_t pt, uint64 va)
{
    for (int i = 0; i < GPU_FB_PAGES; i++) {
        if (mappages(pt, va + (uint64)i * PGSIZE, PGSIZE,
                     (uint64)fb[i], PTE_U | PTE_R | PTE_W) != 0) {
            if (i > 0)
                uvmunmap(pt, va, i, 0);
            return -1;
        }
    }
    return 0;
}

// ── Public: zero-copy page flip ──────────────────────────────────────
// Re-points the device backing list to the physical pages of a user
// buffer.  pt is the calling process's page table; va is a page-aligned
// user virtual address for a region of GPU_FB_PAGES pages that have
// already been validated as user-accessible by the caller.
void
virtio_gpu_flip(pagetable_t pt, uint64 va)
{
    static struct virtio_gpu_mem_entry entries[GPU_FB_PAGES];

    acquire(&flip_lock);
    for (int i = 0; i < GPU_FB_PAGES; i++) {
        entries[i].addr    = walkaddr(pt, va + (uint64)i * PGSIZE);
        entries[i].length  = PGSIZE;
        entries[i].padding = 0;
    }
    gpu_cmd_detach();
    gpu_cmd_attach(entries, GPU_FB_PAGES);
    release(&flip_lock);
}

// ── Public: restore the kernel framebuffer as the device backing ──────
// Called when a process that used flip_display exits, so the GPU stops
// reading from its (now-freed) pages and goes back to the kernel fb[].
void
virtio_gpu_restore_fb(void)
{
    static struct virtio_gpu_mem_entry entries[GPU_FB_PAGES];
    acquire(&flip_lock);
    for (int i = 0; i < GPU_FB_PAGES; i++) {
        entries[i].addr    = (uint64)fb[i];
        entries[i].length  = PGSIZE;
        entries[i].padding = 0;
    }
    gpu_cmd_detach();
    gpu_cmd_attach(entries, GPU_FB_PAGES);
    release(&flip_lock);
}

// ── GPU daemon ────────────────────────────────────────────────────────
// Kernel process started by kproc_create().  Wakes every DISPLAY_DAEMON_TICKS
// timer ticks and issues TRANSFER_TO_HOST_2D + RESOURCE_FLUSH so that
// writes made through map_display() automatically appear on the display
// without any user-space flush call.
//
// Commit period: DISPLAY_DAEMON_TICKS ticks.  xv6's timer fires every
// ~1/10th of a second at QEMU's default rate, giving ~10fps.
#define DISPLAY_DAEMON_TICKS 1

void display_daemon(void)
{
    // The scheduler holds p->lock across swtch into a new process.
    // Release it here, just like forkret does for user processes.
    struct proc *p = myproc();
    release(&p->lock);

    acquire(&tickslock);
    for (;;)
    {
        // Sleep until DISPLAY_DAEMON_TICKS ticks have elapsed.
        uint deadline = ticks + DISPLAY_DAEMON_TICKS;
        while (ticks < deadline)
            sleep(&ticks, &tickslock);

        release(&tickslock);
        virtio_gpu_commit();
        acquire(&tickslock);
    }
}
