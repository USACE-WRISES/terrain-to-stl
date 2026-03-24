import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const repoBase = '/terrain-to-stl/';

export default defineConfig({
  plugins: [react()],
  base: process.env.GITHUB_ACTIONS ? repoBase : '/',
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  worker: {
    format: 'es',
  },
});
