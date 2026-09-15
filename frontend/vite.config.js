import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/run': 'http://localhost:7863',
      '/api': 'http://localhost:7863',
      '/queue': 'http://localhost:7863',
      '/info': 'http://localhost:7863',
    },
  },
});
