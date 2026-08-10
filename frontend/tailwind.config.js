/** @type {import('tailwindcss').Config} */
const colors = require("tailwindcss/colors");

module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#000000", 
        surface: "#0d0d0d",    
        border: "#1f1f23",     
        foreground: "#f4f4f5", 
        muted: "#a1a1aa",      
        primary: {
          DEFAULT: "#16a34a",  
          hover: "#15803d",
        },
        accent: {
          emerald: "#16a34a",
          rose: "#f43f5e",
        },
        zinc: colors.zinc,
        emerald: colors.emerald,
        rose: colors.rose,
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      }
    },
  },
  plugins: [],
};
