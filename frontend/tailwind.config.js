/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#0D1117',
        surface: '#161B22',
        elevated: '#21262D',
        border: '#30363D',
        foreground: '#E6EDF3',
        muted: '#7D8590',
        accent: '#58A6FF',
        'status-green': '#3FB950',
        'status-yellow': '#D29922',
        'status-red': '#F85149',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
