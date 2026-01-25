import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
    // Base path for GitHub Pages - will be the repo name
    base: '/Vib3-CORE-Documented01-/',

    build: {
        outDir: 'dist',
        emptyOutDir: true,
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                kirigami: resolve(__dirname, 'index-kirigami.html'),
            }
        }
    },

    // Dev server configuration
    server: {
        port: 3000,
        open: true
    },

    // Resolve aliases for cleaner imports
    resolve: {
        alias: {
            '@': resolve(__dirname, 'src'),
            '@math': resolve(__dirname, 'src/math'),
            '@geometry': resolve(__dirname, 'src/geometry'),
            '@render': resolve(__dirname, 'src/render'),
            '@holograms': resolve(__dirname, 'src/holograms'),
            '@kirigami': resolve(__dirname, 'src/holograms/kirigami')
        }
    },

    // Optimizations
    optimizeDeps: {
        include: []
    }
});
