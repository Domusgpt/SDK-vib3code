export default [
  {
    ignores: [
      'node_modules/**',
      'dist/**',
      'dist-ci/**',
      'wasm/**',
      'tests/artifacts/**',
      'src/schemas/index.js',
      '**/*.ts'
    ]
  },
  {
    files: ['src/**/*.js'],
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: 'module'
    },
    rules: {}
  }
];
