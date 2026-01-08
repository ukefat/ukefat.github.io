// @ts-check
import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import github from '@astrojs/github';

// https://astro.build/config
export default defineConfig({
  output: 'static',
  site: 'https://ukefat.github.io/',
  adapter: github(),
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});
