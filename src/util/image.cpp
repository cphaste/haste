#include "image.h"

void image::Targa(float3 *buffer, ushort2 size, const char *file) {
    // add .tga extension to the file base name
    size_t file_len = strlen(file);
    char *filename = (char *)malloc(sizeof(char) * (file_len + 5));
    strncpy(filename, file, file_len);
    strncpy(filename + file_len, ".tga", 4);
    filename[file_len + 4] = '\0';
    
    // open the file for writing
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        free(filename);
        perror("Failed to open targa file for writing!");
        exit(EXIT_FAILURE);
    }

    // write 24-bit uncompressed targa header
    // thanks to Paul Bourke (http://local.wasp.uwa.edu.au/~pbourke/dataformats/tga/)
    putc(0, fp);
    putc(0, fp);

    putc(2, fp); // type is uncompressed RGB

    putc(0, fp);
    putc(0, fp);
    putc(0, fp);
    putc(0, fp);
    putc(0, fp);

    putc(0, fp); // x origin, low byte
    putc(0, fp); // x origin, high byte

    putc(0, fp); // y origin, low byte
    putc(0, fp); // y origin, high byte

    putc(size.x & 0xff, fp); // width, low byte
    putc((size.x & 0xff00) >> 8, fp); // width, high byte

    putc(size.y & 0xff, fp); // height, low byte
    putc((size.y & 0xff00) >> 8, fp); // height, high byte

    putc(24, fp); // 24-bit color depth

    putc(0, fp);

    // write out raw pixel data in groups of 3 bytes (BGR order)
    for (uint16_t y = 0; y < size.y; y++) {
        for (uint16_t x = 0; x < size.x; x++) {
            // get pixel value
            uint64_t offset = (x + y * size.x) * sizeof(float3);
            float3 *pixel = (float3 *) ((uint64_t)buffer + offset);

            // clamp rgb components to 1.0
            float r = (pixel->x > 1.0f) ? 1.0f : pixel->x;
            float g = (pixel->y > 1.0f) ? 1.0f : pixel->y;
            float b = (pixel->z > 1.0f) ? 1.0f : pixel->z;

            // convert to bytes
            uint8_t rbyte = (uint8_t)(r * 255);
            uint8_t gbyte = (uint8_t)(g * 255);
            uint8_t bbyte = (uint8_t)(b * 255);

            // write out the bytes in BGR order
            putc(bbyte, fp);
            putc(gbyte, fp);
            putc(rbyte, fp);
        }
    }

    fclose(fp);
    
    free(filename);
}
